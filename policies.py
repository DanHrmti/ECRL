from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from gym import spaces

from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.td3.policies import BasePolicy
from stable_baselines3.common.policies import BaseModel

from utils import batch_pairwise_dist


"""
Modules
"""


class ParticleAttention(nn.Module):
    """
    particle-based multi-head masked attention layer with output projection
    """

    def __init__(self, embed_dim, n_head, attn_pdrop=0.0, resid_pdrop=0.0, att_type='hybrid', linear_bias=False):
        super().__init__()
        assert embed_dim % n_head == 0
        assert att_type in ['hybrid', 'cross', 'self']
        self.att_type = att_type
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim, bias=linear_bias)
        self.query = nn.Linear(embed_dim, embed_dim, bias=linear_bias)
        self.value = nn.Linear(embed_dim, embed_dim, bias=linear_bias)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(embed_dim, embed_dim, bias=linear_bias)
        self.n_head = n_head

    def forward(self, x, c=None, mask=None, return_attention=False):
        B, N, T, C = x.size()  # batch size, n_particles, sequence length, embedding dimensionality (n_embd)

        query_input = x
        if self.att_type == 'hybrid':
            key_value_input = torch.cat([x, c], dim=1)
            key_value_N = key_value_input.shape[1]
        elif self.att_type == 'cross':
            key_value_input = c
            key_value_N = key_value_input.shape[1]
        else:   # self.att_type == 'self'
            key_value_input = x
            key_value_N = N

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key_value_input).view(B, key_value_N * T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, key_value_N * T, hs)
        q = self.query(query_input).view(B, N * T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, N * T, hs)
        v = self.value(key_value_input).view(B, key_value_N * T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, key_value_N * T, hs)
        # causal self-attention; Self-attend: (B, nh, N * T, hs) x (B, nh, hs, N  *T) -> (B, nh, N * T, N *T )
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, N * T, key_value_N * T)
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.n_head, -1, -1)
            att.masked_fill_(mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        if return_attention:
            attention_matrix = att
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, N*T, key_value_N*T) x (B, nh, key_value_N*T, hs) -> (B, nh, N*T, hs)
        y = y.transpose(1, 2).contiguous().view(B, N * T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        y = y.view(B, N, T, -1)

        # return
        if return_attention:
            return y, attention_matrix
        else:
            return y


class EITBlock(nn.Module):
    def __init__(self, embed_dim, h_dim, n_head, attn_pdrop=0.1, resid_pdrop=0.1, att_type='self'):
        super().__init__()
        self.att_type = att_type

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        if self.att_type != 'self':
            self.ln_c = nn.LayerNorm(embed_dim)

        self.attn = ParticleAttention(embed_dim, n_head, attn_pdrop, resid_pdrop, att_type)

        self.mlp = nn.Sequential(nn.Linear(embed_dim, h_dim),
                                 nn.ReLU(True),
                                 nn.Linear(h_dim, h_dim),
                                 nn.ReLU(True),
                                 nn.Linear(h_dim, embed_dim),
                                 nn.Dropout(resid_pdrop))

    def forward(self, x_in, c=None, x_mask=None, c_mask=None, return_attention=False):

        mask = x_mask

        if self.att_type != 'self':
            c = self.ln_c(c)

        if return_attention:
            x, attention_matrix = self.attn(self.ln1(x_in), c, mask, return_attention)
            x = x + x_in
        else:
            x = x_in + self.attn(self.ln1(x_in), c, mask)

        x = x + self.mlp(self.ln2(x))

        if return_attention:
            return x, attention_matrix
        else:
            return x


"""
SB3 Parent Policy
"""


class CustomTD3Policy(BasePolicy):
    """
    General TD3 Policy class.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param actor_class: architecture class to be used for actor and target networks
    :param actor_kwargs: actor kwargs
    :param critic_class: architecture class to be used for critic and target networks
    :param critic_kwargs: critic kwargs
    :param n_critics: Number of critic networks to create.
    :param optimizer_class: The optimizer to use, ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments, excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        actor_class,
        actor_kwargs: Dict[str, Any],
        critic_class,
        critic_kwargs: Optional[Dict[str, Any]] = {},
        n_critics: int = 2,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super(CustomTD3Policy, self).__init__(
            observation_space,
            action_space,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
        )

        # actor
        self.actor_class = actor_class
        self.actor_kwargs = actor_kwargs
        self.actor_kwargs.update({
            "observation_space": self.observation_space,
            "action_space": self.action_space,
        })
        self.actor, self.actor_target = None, None

        # critic
        self.critic_class = critic_class
        self.critic_kwargs = actor_kwargs.copy()
        self.critic_kwargs.update(critic_kwargs)
        self.critic_kwargs.update({"n_critics": n_critics})
        self.critic, self.critic_target = None, None

        # create networks and optimizers
        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        # Create actor and target
        self.actor = self.actor_class(**self.actor_kwargs).to(self.device)
        self.actor_target = self.actor_class(**self.actor_kwargs).to(self.device)
        # Initialize the target to have the same weights as the actor
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        # Create critic and target
        self.critic = self.critic_class(**self.critic_kwargs).to(self.device)
        self.critic_target = self.critic_class(**self.critic_kwargs).to(self.device)
        # Initialize the target to have the same weights as the actor
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic.optimizer = self.optimizer_class(self.critic.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        # Target networks should always be in eval mode
        self.actor_target.set_training_mode(False)
        self.critic_target.set_training_mode(False)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
            )
        )
        return data

    def forward(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self._predict(observation, deterministic=deterministic)

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        # Note: the deterministic parameter is ignored in the case of TD3. Predictions are always deterministic.
        return self.actor(observation)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode


"""
Entity Interaction Transformer Policy
"""


class EITActor(BasePolicy):
    def __init__(self, observation_space, action_space,
                 embed_dim=64, h_dim=128, n_head=1, dropout=0.0,
                 masking=False, goal_cond=False, **kwargs):
        super(EITActor, self).__init__(observation_space, action_space, squash_output=True)

        action_dim = get_action_dim(self.action_space)
        observation_shape = observation_space["achieved_goal"].shape
        particle_fdim = observation_shape[-1]

        self.masking = masking
        self.multiview = observation_shape[0] > 1 and len(observation_shape) == 3
        self.goal_cond = goal_cond

        particle_dim = particle_fdim - 1 if self.masking else particle_fdim
        self.particle_projection = nn.Linear(particle_dim, embed_dim)
        self.particle_self_att1 = EITBlock(embed_dim, h_dim, n_head,
                                           attn_pdrop=dropout, resid_pdrop=dropout, att_type='self')

        if self.goal_cond:
            self.goal_particle_projection = nn.Linear(particle_dim, embed_dim)
            self.particle_cross_att = EITBlock(embed_dim, h_dim, n_head,
                                               attn_pdrop=dropout, resid_pdrop=dropout, att_type='cross')

        self.particle_self_att2 = EITBlock(embed_dim, h_dim, n_head,
                                           attn_pdrop=dropout, resid_pdrop=dropout, att_type='self')


        self.particle_pool_att = EITBlock(embed_dim, h_dim, n_head,
                                          attn_pdrop=dropout, resid_pdrop=dropout, att_type='cross')

        self.ln = nn.LayerNorm(embed_dim)
        self.linear_out = nn.Linear(embed_dim, embed_dim, bias=True)

        self.output_mlp = nn.Sequential(nn.Linear(embed_dim, h_dim),
                                        nn.ReLU(True),
                                        nn.Linear(h_dim, h_dim),
                                        nn.ReLU(True),
                                        nn.Linear(h_dim, h_dim),
                                        nn.ReLU(True),
                                        nn.Linear(h_dim, action_dim))

        # particle encoding
        if self.multiview:
            self.view1_encoding = nn.Parameter(0.02 * torch.randn(1, 1, embed_dim))
            self.view2_encoding = nn.Parameter(0.02 * torch.randn(1, 1, embed_dim))

        # special particle
        self.out_particle = nn.Parameter(0.02 * torch.randn(1, 1, embed_dim))

    def forward(self, obs, return_attention=False):
        particles = obs["achieved_goal"]
        goal_particles = obs["desired_goal"]

        if len(particles.shape) == 4:
            bs, n_views, n_particles, feature_dim = particles.shape
        else:
            bs, n_particles, feature_dim = particles.shape
            n_views = 1

        if return_attention:
            attention_dict = {}

        # preprocess particles and produce masks
        state_mask, goal_mask = None, None
        if self.masking:
            # prepare attention masks (based on obj_on)
            particles_obj_on = particles[..., -1].view(bs, -1)
            particles = particles[..., :-1]  # remove obj_on from features
            state_mask = torch.where(particles_obj_on.unsqueeze(-1) < 0, True, False).transpose(1, 2)

            if self.goal_cond:
                goal_particles_obj_on = goal_particles[..., -1].view(bs, -1)
                goal_particles = goal_particles[..., :-1]  # remove obj_on from goal features
                goal_mask = torch.where(goal_particles_obj_on.unsqueeze(-1) < 0, True, False).transpose(1, 2)

        # project particle features
        particles = self.particle_projection(particles)
        if self.multiview:
            # add view identifying encoding
            particles_view1 = particles[:, 0] + self.view1_encoding.repeat(bs, n_particles, 1)
            particles_view2 = particles[:, 1] + self.view2_encoding.repeat(bs, n_particles, 1)
            particles = torch.cat([particles_view1, particles_view2], dim=1)
        else:
            particles = particles.squeeze(1)

        # forward through self-attention block1
        x = particles.unsqueeze(2)  # [bs, n_particles + 1, 1, embed_dim]
        if return_attention:
            x, attention_matrix = self.particle_self_att1(x, x_mask=state_mask, return_attention=True)
            attention_dict["self_1"] = attention_matrix
        else:
            x = self.particle_self_att1(x, x_mask=state_mask)

        if self.goal_cond:
            # project goal particle features
            goal_particles = self.goal_particle_projection(goal_particles)
            if self.multiview:
                # add goal view identifying encoding
                goal_particles_view1 = goal_particles[:, 0] + self.view1_encoding.repeat(bs, n_particles, 1)
                goal_particles_view2 = goal_particles[:, 1] + self.view2_encoding.repeat(bs, n_particles, 1)
                goal_particles = torch.cat([goal_particles_view1, goal_particles_view2], dim=1)
            else:
                goal_particles = goal_particles.squeeze(1)

            # forward through cross-attention block
            c = goal_particles.unsqueeze(2)  # [bs, n_particles, 1, embed_dim]
            if return_attention:
                x, attention_matrix = self.particle_cross_att(x, c, x_mask=goal_mask, return_attention=True)
                attention_dict["cross"] = attention_matrix
            else:
                x = self.particle_cross_att(x, c, x_mask=goal_mask)

        # forward through self-attention block2
        if return_attention:
            x, attention_matrix = self.particle_self_att2(x, x_mask=state_mask, return_attention=True)
            attention_dict["self_2"] = attention_matrix
        else:
            x = self.particle_self_att2(x, x_mask=state_mask)

        # pool using special output particle
        out_particle = self.out_particle.repeat(bs, 1, 1)
        out_particle = out_particle.unsqueeze(2)  # [bs, 1, 1, embed_dim]
        if return_attention:
            x_agg, attention_matrix = self.particle_pool_att(out_particle, x, x_mask=state_mask, return_attention=True)
            attention_dict["agg"] = attention_matrix
        else:
            x_agg = self.particle_pool_att(out_particle, x, x_mask=state_mask)
        x_agg = x_agg.squeeze(1, 2)  # [bs, embed_dim]
        # final layer norm
        x_agg = self.linear_out(self.ln(x_agg))

        # forward through output MLP
        action = torch.tanh(self.output_mlp(x_agg))  # [bs, action_dim]

        if return_attention:
            return action, attention_dict
        else:
            return action

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        # Note: the deterministic parameter is ignored in the case of TD3. Predictions are always deterministic.
        return self(observation)


class EITCriticNetwork(BaseModel):
    def __init__(self, observation_space, action_space,
                 embed_dim=64, h_dim=128, n_head=1, dropout=0.0,
                 masking=False, goal_cond=False, action_particle=True):
        super(EITCriticNetwork, self).__init__(observation_space, action_space)

        action_dim = get_action_dim(self.action_space)
        observation_shape = observation_space["achieved_goal"].shape
        particle_fdim = observation_shape[-1]

        self.masking = masking
        self.multiview = observation_shape[0] > 1 and len(observation_shape) == 3
        self.goal_cond = goal_cond
        self.action_particle = action_particle

        self.action_projection = nn.Sequential(nn.Linear(action_dim, h_dim),
                                               nn.ReLU(True),
                                               nn.Linear(h_dim, embed_dim))

        particle_dim = particle_fdim - 1 if self.masking else particle_fdim
        self.particle_projection = nn.Linear(particle_dim, embed_dim)
        self.particle_self_att1 = EITBlock(embed_dim, h_dim, n_head,
                                           attn_pdrop=dropout, resid_pdrop=dropout, att_type='self')

        if self.goal_cond:
            self.goal_particle_projection = nn.Linear(particle_dim, embed_dim)
            self.particle_cross_att = EITBlock(embed_dim, h_dim, n_head,
                                               attn_pdrop=dropout, resid_pdrop=dropout, att_type='cross')

        self.particle_self_att2 = EITBlock(embed_dim, h_dim, n_head,
                                           attn_pdrop=dropout, resid_pdrop=dropout, att_type='self')

        self.particle_pool_att = EITBlock(embed_dim, h_dim, n_head,
                                          attn_pdrop=dropout, resid_pdrop=dropout, att_type='cross')

        self.ln = nn.LayerNorm(embed_dim)
        self.linear_out = nn.Linear(embed_dim, embed_dim, bias=True)

        self.output_mlp = nn.Sequential(nn.Linear(2 * embed_dim, h_dim),
                                        nn.ReLU(True),
                                        nn.Linear(h_dim, h_dim),
                                        nn.ReLU(True),
                                        nn.Linear(h_dim, h_dim),
                                        nn.ReLU(True),
                                        nn.Linear(h_dim, 1))

        # particle encoding
        if self.multiview:
            self.view1_encoding = nn.Parameter(0.02 * torch.randn(1, 1, embed_dim))
            self.view2_encoding = nn.Parameter(0.02 * torch.randn(1, 1, embed_dim))

        # special particle
        self.out_particle = nn.Parameter(0.02 * torch.randn(1, 1, embed_dim))

    def forward(self, obs, action):
        particles = obs["achieved_goal"]
        goal_particles = obs["desired_goal"]

        if len(particles.shape) == 4:
            bs, n_views, n_particles, feature_dim = particles.shape
        else:
            bs, n_particles, feature_dim = particles.shape
            n_views = 1

        # preprocess particles and produce masks
        state_mask, goal_mask = None, None
        if self.masking:
            # prepare attention masks (based on obj_on)
            particles_obj_on = particles[..., -1].view(bs, -1)
            if self.action_particle:
                particles_obj_on = torch.cat([particles_obj_on.new_ones([bs, 1]), particles_obj_on], dim=-1)  # add special particles
            particles = particles[..., :-1]  # remove obj_on from features
            state_mask = torch.where(particles_obj_on.unsqueeze(-1) < 0, True, False).transpose(1, 2)

            if self.goal_cond:
                goal_particles_obj_on = goal_particles[..., -1].view(bs, -1)
                goal_particles = goal_particles[..., :-1]  # remove obj_on from goal features
                goal_mask = torch.where(goal_particles_obj_on.unsqueeze(-1) < 0, True, False).transpose(1, 2)

        # project particle features
        particles = self.particle_projection(particles)
        if self.multiview:
            # add view identifying encoding
            particles_view1 = particles[:, 0] + self.view1_encoding.repeat(bs, n_particles, 1)
            particles_view2 = particles[:, 1] + self.view2_encoding.repeat(bs, n_particles, 1)
            particles = torch.cat([particles_view1, particles_view2], dim=1)
        else:
            particles = particles.squeeze(1)

        # project action and add to particles
        action_particle = self.action_projection(action)
        if self.action_particle:
            x = torch.cat([action_particle.unsqueeze(1), particles], dim=1)  # [bs, n_particles + 1, embed_dim]
        else:
            x = particles  # [bs, n_particles, embed_dim]

        # forward through self-attention block1
        x = x.unsqueeze(2)  # [bs, n_particles + 1, 1, embed_dim]
        x = self.particle_self_att1(x, x_mask=state_mask)

        if self.goal_cond:
            # project goal particle features
            goal_particles = self.goal_particle_projection(goal_particles)
            if self.multiview:
                # add goal view identifying encoding
                goal_particles_view1 = goal_particles[:, 0] + self.view1_encoding.repeat(bs, n_particles, 1)
                goal_particles_view2 = goal_particles[:, 1] + self.view2_encoding.repeat(bs, n_particles, 1)
                goal_particles = torch.cat([goal_particles_view1, goal_particles_view2], dim=1)
            else:
                goal_particles = goal_particles.squeeze(1)

            # forward through cross-attention block
            c = goal_particles.unsqueeze(2)  # [bs, n_particles, 1, embed_dim]
            x = self.particle_cross_att(x, c, x_mask=goal_mask)

        # forward through self-attention block2
        x = self.particle_self_att2(x, x_mask=state_mask)

        # pool using special output particle
        out_particle = self.out_particle.repeat(bs, 1, 1)
        if self.action_particle:
            action_particle_out = x[:, 0].clone()
            x_out = torch.cat([out_particle, action_particle_out], dim=1)  # [bs, 2, embed_dim]
        else:
            x_out = out_particle
        x_out = x_out.unsqueeze(2)  # [bs, 2, 1, embed_dim]
        x_out = self.particle_pool_att(x_out, x, x_mask=state_mask)
        x_out = x_out.squeeze(2)  # [bs, 2, embed_dim]
        # final layer norm
        x_out = self.linear_out(self.ln(x_out))

        if self.action_particle:
            x_agg = torch.cat([x_out[:, 0], x_out[:, 1]], dim=-1)  # [bs, 2 * embed_dim]
        else:
            x_agg = torch.cat([x_out[:, 0], action_particle], dim=-1)  # [bs, 2 * embed_dim]

        # forward through output MLP
        output = self.output_mlp(x_agg)  # [bs, output_dim]
        return output


class EITCritic(BaseModel):
    def __init__(self, observation_space, action_space, n_critics=2, action_particle=True,
                 embed_dim=64, h_dim=256, n_head=1, dropout=0.0,
                 masking=False, goal_cond=False, **kwargs):
        super().__init__(observation_space, action_space)

        self.n_critics = n_critics
        self.q_networks = []
        for idx in range(n_critics):
            q_net = EITCriticNetwork(observation_space, action_space,
                                        embed_dim, h_dim, n_head, dropout,
                                        masking, goal_cond, action_particle)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        qvalue_outputs = []
        for i in range(self.n_critics):
            value = self.q_networks[i](obs, actions)
            qvalue_outputs.append(value)
        return tuple(qvalue_outputs)

    def q1_forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        value = self.q_networks[0](obs, actions)
        return value


"""
SMORL Policy
"""


def get_single_goal(full_goal, latent_classifier, device, check_goal_reaching=False,
                    achieved_goal=None, reward_model=None, dist_threshold=None):
    with torch.no_grad():
        full_goal = torch.from_numpy(full_goal).to(device)
        full_goal_view1 = full_goal[:, 0]
        if check_goal_reaching:
            achieved_goal = torch.from_numpy(achieved_goal).to(device)
        # obj_on condition
        obj_on_cond = full_goal_view1[..., -1]
        # latent classifier condition
        if latent_classifier is not None:
            particle_vis_features = full_goal_view1[..., 5:9]
            obj_class_cond = latent_classifier.classify(particle_vis_features)
        else:
            obj_class_cond = torch.ones_like(obj_on_cond)
        # invalid goal condition, for single goal sampling
        invalid_goal_cond = torch.logical_or(obj_on_cond < 0.9, obj_class_cond == 0)

        bs, n_views, n_particles, _ = full_goal.shape
        goal_obj_indices = torch.randint(n_particles, size=(bs,), device=device)
        active_idx = torch.arange(bs, device=device)  # indices of batches we're still actively updating goals for
        for i in range(2 * n_particles):
            if i == n_particles:
                # if we have cycled over all particles and none were chosen,
                # just choose a valid goal regardless of if it was reached or not
                check_goal_reaching = False
            # cycle to next object
            goal_obj_indices[active_idx] = (goal_obj_indices[active_idx] + 1) % n_particles
            # maybe check if goal was reached
            if check_goal_reaching:
                cur_desired_goal = full_goal_view1[active_idx, goal_obj_indices[active_idx]].unsqueeze(1).unsqueeze(2)
                obs = {
                    "achieved_goal": achieved_goal[active_idx, :1],
                    "desired_goal": cur_desired_goal,
                }
                pixel_reward = reward_model(obs, unnormalize=False)
                goal_reached = - pixel_reward.squeeze(-1) < dist_threshold
            else:
                goal_reached = torch.zeros(len(active_idx), device=device)
            # update active_idx based on goal validity
            invalid_goal = invalid_goal_cond[active_idx, goal_obj_indices[active_idx]]
            cond = torch.logical_or(invalid_goal, goal_reached)
            active_idx = active_idx[cond]
            # if active idx is empty, then all sampling is valid
            if len(active_idx) == 0:
                break

        single_goal_view1 = full_goal_view1[torch.arange(bs, device=device), goal_obj_indices].unsqueeze(1).unsqueeze(2)

        if n_views > 1:
            # find matching goal in view2
            full_goal_view2 = full_goal[:, 1]
            full_goal_view2_vis_features = full_goal_view2[..., 5:9]
            single_goal_view1_vis_features = single_goal_view1[..., 5:9].squeeze(1)
            P = batch_pairwise_dist(full_goal_view2_vis_features, single_goal_view1_vis_features, metric='l2_simple')
            min_dists, min_indices = torch.min(P, 1)
            single_goal_view2 = full_goal_view2[torch.arange(bs, device=device), min_indices.squeeze(1)].unsqueeze(1).unsqueeze(2)
            # concatenate goals in view dim
            single_goal = torch.cat([single_goal_view1, single_goal_view2], dim=1)
        else:
            single_goal = single_goal_view1

    return single_goal.cpu().numpy()


class SMORLActor(BasePolicy):
    def __init__(self, observation_space, action_space, embed_dim=32, h_dim=256, n_head=1, dropout=0.0, masking=False):
        super(SMORLActor, self).__init__(observation_space, action_space, squash_output=True)

        observation_shape = observation_space["achieved_goal"].shape
        self.multiview = observation_shape[0] > 1 and len(observation_shape) == 3
        self.masking = masking

        action_dim = get_action_dim(self.action_space)
        particle_fdim = observation_shape[-1]
        particle_dim = particle_fdim - 1 if self.masking else particle_fdim

        self.particle_projection = nn.Linear(particle_dim, embed_dim)
        self.goal_particle_projection = nn.Linear(particle_dim, embed_dim)

        self.particle_attention_gc = ParticleAttention(embed_dim, n_head, attn_pdrop=dropout, resid_pdrop=dropout,
                                                       linear_bias=False, att_type='cross')
        self.particle_attention_gu = ParticleAttention(embed_dim, n_head, attn_pdrop=dropout, resid_pdrop=dropout,
                                                       linear_bias=False, att_type='cross')

        if self.multiview:
            self.particle_attention_gc_view2 = ParticleAttention(embed_dim, n_head, attn_pdrop=dropout, resid_pdrop=dropout,
                                                                 linear_bias=False, att_type='cross')
            self.particle_attention_gu_view2 = ParticleAttention(embed_dim, n_head, attn_pdrop=dropout, resid_pdrop=dropout,
                                                                 linear_bias=False, att_type='cross')

        input_dim = 6 * embed_dim if self.multiview else 3 * embed_dim
        self.output_mlp = nn.Sequential(nn.Linear(input_dim, h_dim),
                                        nn.ReLU(True),
                                        nn.Linear(h_dim, h_dim),
                                        nn.ReLU(True),
                                        nn.Linear(h_dim, h_dim),
                                        nn.ReLU(True),
                                        nn.Linear(h_dim, action_dim))

        # special particle
        self.special_particle = nn.Parameter(0.02 * torch.randn(1, 1, embed_dim))
        if self.multiview:
            self.special_particle_view2 = nn.Parameter(0.02 * torch.randn(1, 1, embed_dim))

    def forward(self, obs):
        particles = obs["achieved_goal"]
        goal_particle = obs["desired_goal"]

        if len(particles.shape) == 4:
            bs, n_views, n_particles, feature_dim = particles.shape
        else:
            bs, n_particles, feature_dim = particles.shape
            n_views = 1

        # preprocess particles and produce mask
        mask, mask_view2 = None, None
        if self.masking:
            # prepare attention masks (with obj_on < mean)
            particles_obj_on = particles[..., -1].view(bs, -1)
            mask = torch.where(particles_obj_on.unsqueeze(-1) < 0, True, False).transpose(1, 2)
            if self.multiview:
                mask_view2 = mask[..., n_particles:]
                mask = mask[..., :n_particles]
                # in the very rare case of attention masking of all particles,
                # unmask a single particle to avoid NaN values
                full_masking = torch.all(mask_view2, dim=-1).squeeze(1)
                mask_view2[full_masking, :, 0] = False
            # remove obj_on from features
            particles = particles[..., :-1]
            goal_particle = goal_particle[..., :-1]

        # project particle features
        particles = self.particle_projection(particles)
        if self.multiview:
            particles_view2 = particles[:, 1]
            particles = particles[:, 0]
        elif len(particles.shape) == 4:
            # squeeze view dimension
            particles = particles.squeeze(1)

        # project goal particle features
        goal_particle = self.goal_particle_projection(goal_particle)
        if self.multiview:
            goal_particle_view2 = goal_particle[:, 1]
            goal_particle = goal_particle[:, 0]
        elif len(goal_particle.shape) == 4:
            # squeeze view dimension
            goal_particle = goal_particle.squeeze(1)

        special_particle = self.special_particle.repeat(bs, 1, 1)

        x_gc = goal_particle.clone().unsqueeze(2)  # [bs, 1, 1, embed_dim]
        x_gu = special_particle.unsqueeze(2)  # [bs, 1, 1, embed_dim]
        c = particles.unsqueeze(2)  # [bs, n_particles, 1, embed_dim]

        x_gc = self.particle_attention_gc(x_gc, c, mask)  # [bs, 1, T, embed_dim]
        x_gc = x_gc.squeeze(2)  # [bs, 1, embed_dim]
        x_gu = self.particle_attention_gu(x_gu, c, mask)  # [bs, 1, T, embed_dim]
        x_gu = x_gu.squeeze(2)  # [bs, 1, embed_dim]

        if self.multiview:
            special_particle_view2 = self.special_particle_view2.repeat(bs, 1, 1)

            x_gc_view2 = goal_particle_view2.clone().unsqueeze(2)  # [bs, 1, 1, embed_dim]
            x_gu_view2 = special_particle_view2.unsqueeze(2)  # [bs, 1, 1, embed_dim]
            c_view2 = particles_view2.unsqueeze(2)  # [bs, n_particles, 1, embed_dim]

            x_gc_view2 = self.particle_attention_gc_view2(x_gc_view2, c_view2, mask_view2)
            x_gc_view2 = x_gc_view2.squeeze(2)  # [bs, 1, embed_dim]
            x_gu_view2 = self.particle_attention_gu_view2(x_gu_view2, c_view2, mask_view2)
            x_gu_view2 = x_gu_view2.squeeze(2)  # [bs, 1, embed_dim]

            # concatenate output particles and goal from both views
            x_out = torch.cat([x_gc, x_gu, goal_particle, x_gc_view2, x_gu_view2, goal_particle_view2], dim=-1)  # [bs, 6*embed_dim]

        else:
            # concatenate output particles and goal
            x_out = torch.cat([x_gc, x_gu, goal_particle], dim=-1)  # [bs, 3*embed_dim]

        # forward through output MLP
        action = torch.tanh(self.output_mlp(x_out.squeeze(1)))  # [bs, action_dim]

        return action

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        # Note: the deterministic parameter is ignored in the case of TD3. Predictions are always deterministic.
        return self(observation)


class SMORLCriticNetwork(BaseModel):
    def __init__(self, observation_space, action_space, embed_dim=64, h_dim=128, n_head=1, dropout=0.0, masking=False):
        super(SMORLCriticNetwork, self).__init__(observation_space, action_space)

        observation_shape = observation_space["achieved_goal"].shape
        self.multiview = observation_shape[0] > 1 and len(observation_shape) == 3
        self.masking = masking

        action_dim = get_action_dim(self.action_space)
        particle_fdim = observation_shape[-1]
        particle_dim = particle_fdim - 1 if self.masking else particle_fdim

        self.particle_projection = nn.Linear(particle_dim, embed_dim)
        self.goal_particle_projection = nn.Linear(particle_dim, embed_dim)

        self.action_projection = nn.Sequential(nn.Linear(action_dim, h_dim),
                                               nn.ReLU(True),
                                               nn.Linear(h_dim, embed_dim))

        self.particle_attention_gc = ParticleAttention(embed_dim, n_head,
                                                       attn_pdrop=dropout, resid_pdrop=dropout,
                                                       linear_bias=False, att_type='cross')
        self.particle_attention_gu = ParticleAttention(embed_dim, n_head,
                                                       attn_pdrop=dropout, resid_pdrop=dropout,
                                                       linear_bias=False, att_type='cross')

        if self.multiview:
            self.particle_attention_gc_view2 = ParticleAttention(embed_dim, n_head,
                                                                 attn_pdrop=dropout, resid_pdrop=dropout,
                                                                 linear_bias=False, att_type='cross')
            self.particle_attention_gu_view2 = ParticleAttention(embed_dim, n_head,
                                                                 attn_pdrop=dropout, resid_pdrop=dropout,
                                                                 linear_bias=False, att_type='cross')

        input_dim = 7 * embed_dim if self.multiview else 4 * embed_dim
        self.output_mlp = nn.Sequential(nn.Linear(input_dim, h_dim),
                                        nn.ReLU(True),
                                        nn.Linear(h_dim, h_dim),
                                        nn.ReLU(True),
                                        nn.Linear(h_dim, h_dim),
                                        nn.ReLU(True),
                                        nn.Linear(h_dim, 1))

        # special particle
        self.special_particle = nn.Parameter(0.02 * torch.randn(1, 1, embed_dim))
        if self.multiview:
            self.special_particle_view2 = nn.Parameter(0.02 * torch.randn(1, 1, embed_dim))

    def forward(self, obs, action):
        particles = obs["achieved_goal"]
        goal_particle = obs["desired_goal"]

        if len(particles.shape) == 4:
            bs, n_views, n_particles, feature_dim = particles.shape
        else:
            bs, n_particles, feature_dim = particles.shape
            n_views = 1

        # preprocess particles and produce mask
        mask, mask_view2 = None, None
        if self.masking:
            # prepare attention masks (based on obj_on)
            particles_obj_on = particles[..., -1].view(bs, -1)
            mask = torch.where(particles_obj_on.unsqueeze(-1) < 0, True, False).transpose(1, 2)
            if self.multiview:
                mask_view2 = mask[..., n_particles:]
                mask = mask[..., :n_particles]
                # in the very rare case of attention masking of all particles,
                # unmask a single particle to avoid NaN values
                full_masking = torch.all(mask_view2, dim=-1).squeeze(1)
                mask_view2[full_masking, :, 0] = False
            # remove obj_on from features
            particles = particles[..., :-1]
            goal_particle = goal_particle[..., :-1]


        # project particle features
        particles = self.particle_projection(particles)
        if self.multiview:
            particles_view2 = particles[:, 1]
            particles = particles[:, 0]
        elif len(particles.shape) == 4:
            # squeeze view dimension
            particles = particles.squeeze(1)

        # project goal particle features
        goal_particle = self.goal_particle_projection(goal_particle)
        if self.multiview:
            goal_particle_view2 = goal_particle[:, 1]
            goal_particle = goal_particle[:, 0]
        elif len(goal_particle.shape) == 4:
            # squeeze view dimension
            goal_particle = goal_particle.squeeze(1)

        # project action
        action_particle = self.action_projection(action)

        # forward through attention
        special_particle = self.special_particle.repeat(bs, 1, 1)

        x_gc = goal_particle.clone().unsqueeze(2)  # [bs, n_particles + 1, 1, embed_dim]
        x_gu = special_particle.unsqueeze(2)  # [bs, n_particles + 1, 1, embed_dim]
        c = particles.unsqueeze(2)  # [bs, n_particles, 1, embed_dim]

        # forward through goal conditional attention
        x_gc = self.particle_attention_gc(x_gc, c, mask)
        x_gc = x_gc.squeeze(2)  # [bs, n_particles + 1, embed_dim]

        # forward through goal unconditional attention
        x_gu = self.particle_attention_gu(x_gu, c, mask)
        x_gu = x_gu.squeeze(2)  # [bs, n_particles + 1, embed_dim]

        if self.multiview:
            special_particle_view2 = self.special_particle_view2.repeat(bs, 1, 1)

            x_gc_view2 = goal_particle_view2.clone().unsqueeze(2)  # [bs, 1, 1, embed_dim]
            x_gu_view2 = special_particle_view2.unsqueeze(2)  # [bs, 1, 1, embed_dim]
            c_view2 = particles_view2.unsqueeze(2)  # [bs, n_particles, 1, embed_dim]

            x_gc_view2 = self.particle_attention_gc_view2(x_gc_view2, c_view2, mask_view2)
            x_gc_view2 = x_gc_view2.squeeze(2)  # [bs, 1, embed_dim]
            x_gu_view2 = self.particle_attention_gu_view2(x_gu_view2, c_view2, mask_view2)
            x_gu_view2 = x_gu_view2.squeeze(2)  # [bs, 1, embed_dim]

            # concatenate output particles and goal from both views and action
            x_out = torch.cat([x_gc, x_gu, goal_particle, x_gc_view2, x_gu_view2, goal_particle_view2, action_particle.unsqueeze(1)], dim=-1)  # [bs, 7*embed_dim]

        else:
            # concatenate output particles, goal and action
            x_out = torch.cat([x_gc, x_gu, goal_particle, action_particle.unsqueeze(1)], dim=-1) # [bs, 4*embed_dim]

        # forward through output MLP
        output = self.output_mlp(x_out.squeeze(1))  # [bs, output_dim]
        return output


class SMORLCritic(BaseModel):
    def __init__(self, observation_space, action_space, n_critics=2,
                 embed_dim=64, h_dim=256, n_head=1, dropout=0.0,
                 masking=False):
        super().__init__(observation_space, action_space)

        self.n_critics = n_critics
        self.q_networks = []
        for idx in range(n_critics):
            q_net = SMORLCriticNetwork(observation_space, action_space, embed_dim, h_dim, n_head, dropout, masking)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        qvalue_outputs = []
        for i in range(self.n_critics):
            value = self.q_networks[i](obs, actions)
            qvalue_outputs.append(value)
        return tuple(qvalue_outputs)

    def q1_forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        value = self.q_networks[0](obs, actions)
        return value


"""
MLP Policy
"""


class MLPActor(BasePolicy):
    def __init__(self, observation_space, action_space, n_hidden_layers=3, h_dim=256):
        super(MLPActor, self).__init__(observation_space, action_space, squash_output=True)

        observation_shape = observation_space["achieved_goal"].shape
        self.multiview = observation_shape[0] > 1 and len(observation_shape) == 2

        action_dim = get_action_dim(self.action_space)
        f_dim = observation_shape[-1]
        input_dim = 4 * f_dim if self.multiview else 2 * f_dim

        layers = [nn.Linear(input_dim, h_dim), nn.ReLU(True)]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(h_dim, h_dim), nn.ReLU(True)]
        layers += [nn.Linear(h_dim, action_dim)]

        self.mlp = nn.Sequential(*layers)

    def forward(self, obs):
        a_goal = obs["achieved_goal"]
        d_goal = obs["desired_goal"]

        if len(a_goal.shape) == 3:
            # concatenate views
            bs, n_views, feature_dim = a_goal.shape
            a_goal = torch.cat([a_goal[:, i] for i in range(n_views)], dim=-1)
            d_goal = torch.cat([d_goal[:, i] for i in range(n_views)], dim=-1)

        # concatenate obs and goal
        input = torch.cat([a_goal, d_goal], dim=-1)

        # forward through output MLP
        action = torch.tanh(self.mlp(input))  # [bs, action_dim]

        return action

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        # Note: the deterministic parameter is ignored in the case of TD3. Predictions are always deterministic.
        return self(observation)


class MLPCriticNetwork(BaseModel):
    def __init__(self, observation_space, action_space, n_hidden_layers=3, h_dim=256):
        super(MLPCriticNetwork, self).__init__(observation_space, action_space)

        observation_shape = observation_space["achieved_goal"].shape
        self.multiview = observation_shape[0] > 1 and len(observation_shape) == 2

        action_dim = get_action_dim(self.action_space)
        f_dim = observation_shape[-1]
        input_dim = 5 * f_dim if self.multiview else 3 * f_dim

        self.action_projection = nn.Sequential(nn.Linear(action_dim, h_dim),
                                               nn.ReLU(True),
                                               nn.Linear(h_dim, f_dim))

        layers = [nn.Linear(input_dim, h_dim), nn.ReLU(True)]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(h_dim, h_dim), nn.ReLU(True)]
        layers += [nn.Linear(h_dim, 1)]

        self.mlp = nn.Sequential(*layers)

    def forward(self, obs, action):
        a_goal = obs["achieved_goal"]
        d_goal = obs["desired_goal"]

        if len(a_goal.shape) == 3:
            # concatenate views
            bs, n_views, feature_dim = a_goal.shape
            a_goal = torch.cat([a_goal[:, i] for i in range(n_views)], dim=-1)
            d_goal = torch.cat([d_goal[:, i] for i in range(n_views)], dim=-1)

        # project action
        action = self.action_projection(action)

        # concatenate obs, goal and action
        input = torch.cat([a_goal, d_goal, action], dim=-1)

        # forward through output MLP
        value = self.mlp(input)  # [bs, 1]

        return value


class MLPCritic(BaseModel):
    def __init__(self, observation_space, action_space, n_critics=2, n_hidden_layers=2, h_dim=256):
        super().__init__(observation_space, action_space)

        self.n_critics = n_critics
        self.q_networks = []
        for idx in range(n_critics):
            q_net = MLPCriticNetwork(observation_space, action_space, n_hidden_layers, h_dim)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        qvalue_outputs = []
        for i in range(self.n_critics):
            value = self.q_networks[i](obs, actions)
            qvalue_outputs.append(value)
        return tuple(qvalue_outputs)

    def q1_forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        value = self.q_networks[0](obs, actions)
        return value