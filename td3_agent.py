"""
TD3 based agent
Author: Dan Haramati
"""

import pickle
import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from torch.nn import functional as F

from gym import spaces

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer
from stable_baselines3.common.noise import ActionNoise, NormalActionNoise, VectorizedActionNoise
from stable_baselines3.common.type_aliases import MaybeCallback, GymEnv, Schedule, TrainFrequencyUnit, RolloutReturn, \
    TrainFreq
from stable_baselines3.common.utils import polyak_update, should_collect_more_steps
from stable_baselines3.td3.policies import TD3Policy

import wandb
from moviepy.editor import ImageSequenceClip

from multi_her_replay_buffer import MultiHerReplayBuffer

from dlp2.utils.util_func import plot_keypoints_on_image

from utils import compute_gradients, compute_params, get_max_param
from utils import RMSNormalizer, get_dlp_rep, extract_slot_image

from chamfer_reward import ChamferReward, DensityAwareChamferReward
from policies import get_single_goal

"""
Agent Model
"""


class TD3HER(OffPolicyAlgorithm):
    def __init__(
            self,
            env: Union[GymEnv, str],
            policy: Union[str, Type[TD3Policy]],
            policy_kwargs: Optional[Dict[str, Any]] = None,
            policy_delay: int = 2,
            target_policy_noise: float = 0.2,
            target_noise_clip: float = 0.5,
            learning_rate: Union[float, Schedule] = 1e-3,
            batch_size: int = 100,
            tau: float = 0.005,
            gamma: float = 0.99,
            a_reg_coef: float = 1.0,
            buffer_size: int = 1_000_000,  # 1e6
            replay_buffer_class: Optional[ReplayBuffer] = None,
            replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
            chamfer_reward: bool = False,
            chamfer_reward_kwargs: Optional[Dict[str, Any]] = None,
            learning_starts: int = 100,
            train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
            gradient_steps: int = -1,
            action_noise: Optional[ActionNoise] = None,
            exploration_epsilon: Optional[float] = None,
            exploration_schedule: Optional[List] = None,
            policy_eval_freq: Optional[int] = None,
            num_eval_episodes: int = 100,
            eval_max_episode_length=50,
            smorl_meta_n_steps=25,
            model_save_freq: Optional[int] = None,
            model_save_dir: Optional[str] = None,
            seed: Optional[int] = None,
            device: Union[torch.device, str] = "auto",
            _init_setup_model: bool = True,
            wandb_log: bool = True,
            wandb_log_policy_stats: bool = True,  # setting to False saves a lot of time in training
            episode_vis_freq: Optional[int] = None,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            verbose: int = 0,
            optimize_memory_usage: bool = False,
    ):

        super(TD3HER, self).__init__(
            policy,
            env,
            TD3Policy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=spaces.Box,
            support_multi_env=True,
        )

        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        self.obs_mode = self.env.obs_mode
        self.horizon = self.env.max_episode_len
        self.num_objects = self.env.num_objects
        self.a_reg_coef = a_reg_coef

        self.stats = None
        self.policy_eval_freq = policy_eval_freq
        self.num_eval_episodes = num_eval_episodes
        self.eval_max_episode_length = eval_max_episode_length
        self.smorl_meta_n_steps = smorl_meta_n_steps
        self.episode_vis_freq = episode_vis_freq
        self.model_save_freq = model_save_freq
        self.model_save_dir = model_save_dir

        # initialize exploration parameters
        self.exp_sch = exploration_schedule
        # action noise
        if action_noise is not None:
            self.an_sigma_init = self.action_noise._sigma[0]
            self.an_sigma = self.an_sigma_init
        # random exploration
        self.epsilon_init = exploration_epsilon
        self.epsilon = self.epsilon_init

        # initialize observation normalization params
        self.rms_normalizer = RMSNormalizer(epsilon=1e-6, shape=self.env.observation_space["achieved_goal"].shape[-1])

        # reward model
        self.chamfer_reward = chamfer_reward
        self.chamfer_reward_kwargs = chamfer_reward_kwargs
        self.reward_model = None

        if _init_setup_model:
            self._setup_model()

        self.wandb_log = wandb_log
        self.wandb_log_policy_stats = wandb_log_policy_stats

    def _setup_model(self) -> None:
        """
        Added in order to use custom replay buffer class MultiHerReplayBuffer
        """
        if self.replay_buffer_class == MultiHerReplayBuffer:
            assert self.env is not None, "You must pass an environment when using `HerReplayBuffer`"

            # If using offline sampling, we need a classic replay buffer too
            if self.replay_buffer_kwargs.get("online_sampling", True):
                replay_buffer = None
            else:
                replay_buffer = DictReplayBuffer(
                    self.buffer_size,
                    self.observation_space,
                    self.action_space,
                    device=self.device,
                    optimize_memory_usage=self.optimize_memory_usage,
                )

            self.replay_buffer = MultiHerReplayBuffer(
                self.env,
                self.buffer_size,
                device=self.device,
                replay_buffer=replay_buffer,
                rms_normalizer=self.rms_normalizer,
                **self.replay_buffer_kwargs,
            )

        super(TD3HER, self)._setup_model()

        if self.chamfer_reward:
            if self.env.smorl:
                self.reward_model = ChamferReward(particle_normalizer=self.rms_normalizer,
                                                  latent_classifier=self.env.latent_classifier,
                                                  smorl=True,
                                                  **self.chamfer_reward_kwargs).to(self.device)
            else:
                self.reward_model = DensityAwareChamferReward(particle_normalizer=self.rms_normalizer,
                                                              latent_classifier=self.env.latent_classifier,
                                                              **self.chamfer_reward_kwargs).to(self.device)

        self._create_aliases()

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def learn(self, total_timesteps: int, callback: MaybeCallback = None, log_interval: Optional[int] = 4,
                          eval_env: Optional[GymEnv] = None, eval_freq: int = -1, n_eval_episodes: int = 5,
                          tb_log_name: str = "TD3", eval_log_path: Optional[str] = None,
                          reset_num_timesteps: bool = True) -> "TD3HER":

        total_timesteps, callback = self._setup_learn(total_timesteps, eval_env, callback, eval_freq, n_eval_episodes,
                                                      eval_log_path, reset_num_timesteps, tb_log_name)

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            if self.num_timesteps == 0:
                print(f"Starting warmup episodes")
            if self.num_timesteps == self.learning_starts:
                print(f"\nFinished warmup episodes")
            print(f"\n#### Episode {self._episode_num + 1} - Start ####")
            episode_start_time = time.time()

            # reset environment for new episode
            self._last_obs = self.env.reset()

            # set exploration parameters according to schedule
            if self.exp_sch is not None:
                self._update_exploration_params(self._episode_num + 1)

            # collect and store data in replay buffer
            print(f"Collecting transition data...")
            rollout_start_time = time.time()
            rollout = self.collect_rollouts(  # reset is performed at end of rollout
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )
            print(f"Collection completed in {time.time() - rollout_start_time:5.2f}s")

            # collect episode statistics
            self._get_episode_stats()

            # visualize rollout
            if self.wandb_log and self.episode_vis_freq is not None and self._episode_num % self.episode_vis_freq == 0:
                self._wandb_visualize()

            if rollout.continue_training is False:
                break

            # train agent
            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    print(f"Training agent...")
                    agent_train_start_time = time.time()
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)
                    print(f"Training completed in {time.time() - agent_train_start_time:5.2f}s")

            print(f"Elapsed time: {time.time() - episode_start_time:5.2f}s")
            print(f"####  Episode {self._episode_num} - End  ####")

            # evaluate policy
            if (self.policy_eval_freq is not None and self.num_timesteps > self.learning_starts
                    and self._episode_num % self.policy_eval_freq == 0):
                eval_stat_dict = self.evaluate_policy(self.num_eval_episodes)
                if self.wandb_log:
                    self._wandb_log_eval_stats(eval_stat_dict)
                print(f"\nTotal environment steps taken: {self._episode_num * self.horizon}")

            # save model
            if (self.model_save_freq is not None
                    and self.num_timesteps > self.learning_starts
                    and self._episode_num % self.model_save_freq == 0):
                print(f"Saving model to {self.model_save_dir}")
                self.save(self.model_save_dir)

            # wandb log and commit
            if self.wandb_log:
                wandb.log({"obs_rms_mean": np.mean(self.rms_normalizer.rms.mean)}, commit=False)
                wandb.log({"obs_rms_std": np.mean(np.sqrt(self.rms_normalizer.rms.var))}, commit=False)
                wandb.log({"exp_epsilon": self.epsilon}, commit=False)
                wandb.log({"action_noise": self.an_sigma}, commit=False)
                wandb.log({"env_timesteps": self.num_timesteps}, commit=True)

        callback.on_training_end()

        return self

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        if self.wandb_log and self.wandb_log_policy_stats:
            actor_losses, actor_gradients, actor_params, actor_max_param = [], [], [], []
            critic_losses, critic_gradients, critic_params = [], [], []

        for _ in range(gradient_steps):

            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with torch.no_grad():
                # select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # compute the next Q-values: min over all critics targets
                next_q_values = torch.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                # compute rewards
                if self.reward_model is not None:
                    rewards = self.reward_model(replay_data.next_observations)
                else:
                    rewards = replay_data.rewards  # reward from environment
                # compute target Q-values
                target_q_values = rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                # clip target Q-values to possible range
                min_total_reward = self.env.reward_range[0] / (1 - self.gamma)
                max_total_reward = self.env.reward_range[1] / (1 - self.gamma)
                target_q_values = torch.clamp(target_q_values, min_total_reward, max_total_reward)

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if self.wandb_log and self.wandb_log_policy_stats:
                critic_losses.append(critic_loss.item())
                critic_gradients.append(compute_gradients(self.critic.parameters()).item())
                critic_params.append(compute_params(self.critic.parameters()).item())

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                # switch critic to eval mode (this affects batch norm / dropout)
                self.critic.train(False)
                # maximize Q-value
                actor_action = self.actor(replay_data.observations)
                q_loss = -self.critic.q1_forward(replay_data.observations, actor_action)
                # limit action L2 to prevent policy saturation (from original HER)
                action_reg = actor_action.pow(2)
                # actor loss
                actor_loss = q_loss.mean() + self.a_reg_coef * action_reg.mean()

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                if self.wandb_log and self.wandb_log_policy_stats:
                    actor_losses.append(actor_loss.item())
                    actor_gradients.append(compute_gradients(self.actor.parameters()).item())
                    actor_params.append(compute_params(self.actor.parameters()).item())
                    actor_max_param.append(get_max_param(self.actor.parameters()).item())

                # switch critic back to train mode (this affects batch norm / dropout)
                self.critic.train(True)

        # update target networks after each training cycle
        polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
        polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

        # WANDB log
        if self.wandb_log and self.wandb_log_policy_stats:
            # actor stats
            wandb.log({"actor_loss": np.mean(actor_losses)}, commit=False)
            wandb.log({"actor_grad_norm": np.mean(actor_gradients)}, commit=False)
            wandb.log({"actor_param_norm": np.mean(actor_params)}, commit=False)
            wandb.log({"actor_max_param": np.mean(actor_max_param)}, commit=False)
            # critic stats
            wandb.log({"critic_loss": np.mean(critic_losses)}, commit=False)
            wandb.log({"critic_grad_norm": np.mean(critic_gradients)}, commit=False)
            wandb.log({"critic_param_norm": np.mean(critic_params)}, commit=False)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()

        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase: select random action
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])

        else:
            if self.epsilon is not None and np.random.rand() < self.epsilon:
                unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])

            else:
                # normalize observation
                input_obs = {
                    # "observation": self._last_obs["observation"],  # commented out to accelerate code
                    "desired_goal": self.rms_normalizer.normalize(self._last_obs["desired_goal"]),
                    "achieved_goal": self.rms_normalizer.normalize(self._last_obs["achieved_goal"])
                }
                unscaled_action, _ = self.predict(input_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action

        return action, buffer_action

    def evaluate_policy(self, num_eval_episodes=100, stat_save_path=None):
        self.policy.set_training_mode(False)
        num_eval_episodes = (num_eval_episodes // self.env.num_envs) * self.env.num_envs
        # if training on random number of cubes, evaluate only on max number of cubes
        random_obj_num = self.env.env.random_obj_num
        self.env.env.random_obj_num = False
        # prepare stats variables
        total_returns = 0
        total_latent_rep_returns = 0
        total_avg_obj_dist = 0
        total_max_obj_dist = 0
        total_goal_success_frac = 0
        total_goals_reached = 0
        if stat_save_path is not None:
            save_stat_dict = {
                "success": [],
                "success_frac": [],
                "avg_obj_dist": [],
                "max_obj_dist": [],
                "avg_return": [],
            }
        img_list = []
        goal_img = None
        ori_dist_list = []

        print(f"\nEvaluating policy on {num_eval_episodes} random goals...")
        start_time = time.time()
        for i in tqdm(range(num_eval_episodes // self.env.num_envs)):
            # prepare rollout stats variables
            ret = np.zeros(self.env.num_envs)
            latent_rep_ret = np.zeros(self.env.num_envs)
            avg_obj_dist = np.ones(self.env.num_envs)
            max_obj_dist = np.ones(self.env.num_envs)
            goal_success_frac = np.zeros(self.env.num_envs)
            goals_reached = np.zeros(self.env.num_envs)
            # perform rollout
            self._last_obs = self.env.reset()
            for t in range(self.eval_max_episode_length):
                if self.env.smorl and t % self.smorl_meta_n_steps == 0 and t > 0:
                    if self.obs_mode == 'state':
                        self.smorl_update_env_goal_state()
                    else:
                        self.smorl_update_env_goal()
                # normalize observation
                input_obs = {
                    # "observation": self._last_obs["observation"],  # commented to accelerate code
                    "desired_goal": self.rms_normalizer.normalize(self._last_obs["desired_goal"]),
                    "achieved_goal": self.rms_normalizer.normalize(self._last_obs["achieved_goal"])
                }
                # select action according to policy
                actions, _ = self.predict(input_obs)
                # perform action
                new_obs, rewards, dones, infos = self.env.step(actions)
                # gather stats
                ret += rewards
                avg_obj_dist = np.array([infos[i]["avg_obj_dist"] for i in range(len(infos))])
                max_obj_dist = np.array([infos[i]["max_obj_dist"] for i in range(len(infos))])
                goal_success_frac = np.array([infos[i]["goal_success_frac"] for i in range(len(infos))])
                if self.reward_model is not None:
                    # calculate Chamfer reward
                    with torch.no_grad():
                        obs = {"achieved_goal": self.rms_normalizer.normalize(torch.tensor(new_obs["achieved_goal"], device=self.device)),
                               "desired_goal": self.rms_normalizer.normalize(torch.tensor(new_obs["desired_goal"], device=self.device))}
                        chamfer_rewards = self.reward_model(obs)
                        latent_rep_ret += chamfer_rewards.cpu().numpy().squeeze()
                # save episode media and goals
                if i == 0:
                    if t == 0:
                        goal_img = np.moveaxis(infos[0]["goal_image"][0], 0, -1)
                    img_list.append(np.moveaxis(infos[0]["image"][0], 0, -1))
                    if t == self.eval_max_episode_length - 1:
                        if goal_success_frac[0] == 1:
                            eval_vid_success = True
                            print("Visualized eval episode was a success")
                        else:
                            eval_vid_success = False
                            print("Visualized eval episode was a failure")
                # save orientation distances
                if self.env.push_t and t == self.eval_max_episode_length - 1:
                    ori_dist_list.extend(np.array([infos[i]["ori_dist"] for i in range(len(infos))]))
                # update last_obs
                self._last_obs = new_obs

            total_returns += np.sum(ret)
            total_latent_rep_returns += np.sum(latent_rep_ret)
            total_avg_obj_dist += np.sum(avg_obj_dist)
            total_max_obj_dist += np.sum(max_obj_dist)
            total_goal_success_frac += np.sum(goal_success_frac)
            goals_reached[goal_success_frac == 1] = 1
            total_goals_reached += np.sum(goals_reached)
            if stat_save_path is not None:
                save_stat_dict["success"].extend(goals_reached)
                save_stat_dict["success_frac"].extend(goal_success_frac)
                save_stat_dict["max_obj_dist"].extend(max_obj_dist)
                save_stat_dict["avg_obj_dist"].extend(avg_obj_dist)
                save_stat_dict["avg_return"].extend(ret / self.eval_max_episode_length)

        print(f"Evaluation completed in {time.time() - start_time:5.2f}s")
        # revert 'random_obj_num' attribute to original value
        self.env.env.random_obj_num = random_obj_num

        if stat_save_path is not None:
            with open(stat_save_path, 'wb') as file:
                pickle.dump(save_stat_dict, file)
            print(f"Saved eval stats to {stat_save_path}\n")

        # compute overall stats
        mean_return = total_returns / num_eval_episodes
        mean_latent_rep_return = total_latent_rep_returns / num_eval_episodes
        mean_avg_obj_dist = total_avg_obj_dist / num_eval_episodes
        mean_max_obj_dist = total_max_obj_dist / num_eval_episodes
        mean_success_frac = total_goal_success_frac / num_eval_episodes
        succes_rate = (total_goals_reached / num_eval_episodes) * 100

        print(f"Goal success rate: {succes_rate / 100:3.3f}%")
        print(f"Goal success fraction: {mean_success_frac:3.3f}")
        print(f"Max object-goal distance: {mean_max_obj_dist:3.3f}")
        print(f"Avg. object-goal distance: {mean_avg_obj_dist:3.3f}")
        print(f"Avg. reward: {mean_return / self.eval_max_episode_length:3.3f}")

        eval_stat_dict = {
            "succes_rate": succes_rate,
            "mean_success_frac": mean_success_frac,
            "mean_avg_obj_dist": mean_avg_obj_dist,
            "mean_max_obj_dist": mean_max_obj_dist,
            "mean_return": mean_return,
            "mean_latent_rep_return": mean_latent_rep_return,
            "img_list": img_list,
            "goal_img": goal_img,
            "eval_vid_success": eval_vid_success,
            "ori_dist_array": np.concatenate(ori_dist_list) if self.env.push_t else None,
        }
        return eval_stat_dict

    def _get_episode_stats(self):
        num_episodes = int(self.train_freq[0] / self.horizon) if self.n_envs > 1 else self.train_freq[0]

        if self.replay_buffer.pos == 0:
            info = list(self.replay_buffer.info_buffer[-num_episodes:].copy())
            a_goal = self.replay_buffer._buffer["achieved_goal"][-num_episodes:].copy()
            d_goal = self.replay_buffer._buffer["desired_goal"][-num_episodes:].copy()
        elif self.replay_buffer.pos < num_episodes:
            info = list(self.replay_buffer.info_buffer[:self.replay_buffer.pos].copy())
            a_goal = self.replay_buffer._buffer["achieved_goal"][: self.replay_buffer.pos].copy()
            d_goal = self.replay_buffer._buffer["desired_goal"][: self.replay_buffer.pos].copy()
        else:
            info = list(self.replay_buffer.info_buffer[self.replay_buffer.pos - num_episodes: self.replay_buffer.pos].copy())
            a_goal = self.replay_buffer._buffer["achieved_goal"][self.replay_buffer.pos - num_episodes: self.replay_buffer.pos].copy()
            d_goal = self.replay_buffer._buffer["desired_goal"][self.replay_buffer.pos - num_episodes: self.replay_buffer.pos].copy()

        # update observation RMS
        flattened_a_goal = a_goal.reshape(-1, a_goal.shape[-1])
        self.rms_normalizer.update(flattened_a_goal)

        if self.wandb_log:
            # calculate interaction rate
            info = [list(info[episode]) for episode in range(len(info))]
            num_objects = info[0][0][0]['position'].shape[0]
            tot_num_interactions = np.zeros(num_objects + 1)
            for epi_idx in range(len(info)):
                episode_info = info[epi_idx]
                for t in range(1, len(episode_info)):
                    # keep count of how many objects moved as a result of a single action
                    prev_step_info, step_info = list(episode_info[t - 1]), list(episode_info[t])
                    prev_obj_xy = np.array([prev_step_info[env_idx]["position"] for env_idx in range(self.n_envs)])
                    obj_xy = np.array([step_info[env_idx]["position"] for env_idx in range(self.n_envs)])
                    num_objects_moved = np.zeros([obj_xy.shape[0], obj_xy.shape[1], 1])
                    pos_dif = np.sqrt(np.sum(np.square(prev_obj_xy - obj_xy), axis=-1, keepdims=True))
                    num_objects_moved[pos_dif > 1e-5] += 1
                    num_objects_moved = np.sum(num_objects_moved, axis=(1, 2))
                    num_interactions, _ = np.histogram(num_objects_moved, bins=np.arange(num_objects+2))
                    tot_num_interactions += num_interactions
            for i in range(num_objects):
                wandb.log({f"{i+1}C_interaction_rate": (tot_num_interactions[i+1] / np.sum(tot_num_interactions))}, commit=False)

    def _update_exploration_params(self, episode):
        if self.exp_sch[1] < episode <= self.exp_sch[2]:
            num_episodes = int((self.train_freq[0] / self.horizon) * self.n_envs) if self.n_envs > 1 else self.train_freq[0]
            # exploration epsilon
            epsilon_delta = ((1-self.exp_sch[0]) * self.epsilon_init) / (self.exp_sch[2] - self.exp_sch[1])
            self.epsilon = self.epsilon - num_episodes * epsilon_delta
            # action_noise
            an_sigma_delta = ((1-self.exp_sch[0]) * self.an_sigma_init) / (self.exp_sch[2] - self.exp_sch[1])
            self.an_sigma = self.an_sigma - num_episodes * an_sigma_delta
            action_len = self.action_space.shape[-1]
            self.action_noise = NormalActionNoise(mean=np.zeros(action_len), sigma=self.an_sigma * np.ones(action_len))

    def extract_dlp_image(self, images):
        orig_image_shape = images.shape
        if len(orig_image_shape) == 3:
            images = np.expand_dims(images, axis=0)
        normalized_images = images.astype('float32') / 255
        normalized_images = torch.tensor(normalized_images, device=self.device)

        with torch.no_grad():
            encoded_output = self.env.latent_rep_model.encode_all(normalized_images, deterministic=True)
            dlp_features = get_dlp_rep(encoded_output)
            normalized_dlp_features = torch.tensor(self.rms_normalizer.normalize(dlp_features.cpu().numpy()), device=self.device)
            if self.actor.masking:
                pixel_xy = encoded_output['z']
                obj_on = normalized_dlp_features[:, :, -1]
                mask = torch.where(obj_on < 0, False, True)
                pixel_xy = [pixel_xy[i][mask[i]] for i in range(len(pixel_xy))]
            else:
                pixel_xy = encoded_output['z']

        dlp_images = []
        for kp_xy, image in zip(pixel_xy, normalized_images):
            dlp_images.append(plot_keypoints_on_image(kp_xy, image, radius=2, thickness=1, kp_range=(-1, 1), plot_numbers=False))

        if len(dlp_images) == 1:
            dlp_images = dlp_images[0]

        return dlp_images

    def extract_dlp_visuals(self, images):
        orig_image_shape = images.shape
        if len(orig_image_shape) == 3:
            images = np.expand_dims(images, axis=0)
        normalized_images = images.astype('float32') / 255
        normalized_images = torch.tensor(normalized_images, device=self.device)

        with torch.no_grad():
            model_output = self.env.latent_rep_model(normalized_images, deterministic=True)
            # image with particles
            pixel_xy = model_output['z']
            dlp_images = []
            for kp_xy, image in zip(pixel_xy, normalized_images):
                dlp_images.append(plot_keypoints_on_image(kp_xy, image, radius=2, thickness=1, kp_range=(-1, 1), plot_numbers=False))
            if len(dlp_images) == 1:
                dlp_images = dlp_images[0]
            # decoded foreground image
            dec_objects = model_output['dec_objects'].squeeze().cpu().numpy()
            dec_objects = np.moveaxis(dec_objects, 0, -1)
            # decoded object glimpses
            dec_object_glimpses = model_output['dec_objects_original'].squeeze()
            _, dec_object_glimpses = torch.split(dec_object_glimpses, [1, 3], dim=1)
            dec_object_glimpses = torch.cat([dec_object_glimpses[i] for i in range(len(dec_object_glimpses))], dim=1)
            dec_object_glimpses = dec_object_glimpses.cpu().numpy()
            dec_object_glimpses = np.moveaxis(dec_object_glimpses, 0, -1)
            # obj_on normalized values
            dlp_features = get_dlp_rep(model_output)
            dlp_features = self.rms_normalizer.normalize(dlp_features.cpu().numpy())
            obj_on = dlp_features[0, :, -1]

        return dlp_images, dec_objects, dec_object_glimpses, obj_on

    def smorl_update_env_goal(self):
        single_goal = get_single_goal(self.env.full_goal, self.env.latent_classifier, self.device,
                                      check_goal_reaching=True, achieved_goal=self._last_obs["achieved_goal"],
                                      reward_model=self.reward_model, dist_threshold=0.3)
        # update goal
        self.env.goal = single_goal
        # update last obs goal
        self._last_obs["desired_goal"] = self.env.goal

    def smorl_update_env_goal_state(self):
        achieved_goal = self._last_obs["achieved_goal"][:, 1:]
        desired_goal = self.env.full_goal[:, 1:]
        goal_obj_index = self.env.goal_obj_index - 1  # disregard arm in goal choosing

        active_idx = np.arange(self.env.num_envs)  # indices of envs we're still actively updating goals for
        for i in range(self.num_objects):
            # cycle to next object
            goal_obj_index[active_idx] = (goal_obj_index[active_idx] + 1) % self.num_objects
            # check if updated goals are not yet reached
            achieved_subgoal = achieved_goal[active_idx, goal_obj_index[active_idx]]
            desired_subgoal = desired_goal[active_idx, goal_obj_index[active_idx]]
            dist = np.linalg.norm(achieved_subgoal - desired_subgoal, axis=-1)
            subgoals_reached = dist < self.env.dist_threshold
            # update active_idx based on goal reaching
            active_idx = active_idx[subgoals_reached]
            num_active_idx = len(active_idx)
            # If active idx is empty, then all sampling is valid :D
            if num_active_idx == 0:
                break

        # update goal
        goal_obj_index += 1  # count arm as index 0 for env
        self.env.goal = np.expand_dims(self.env.full_goal[np.arange(self.env.num_envs), goal_obj_index], -2)
        self.env.goal_obj_index = goal_obj_index
        # update last obs goal
        self._last_obs["desired_goal"] = self.env.goal

    def _excluded_save_params(self) -> List[str]:
        return super(TD3HER, self)._excluded_save_params() + ["actor", "critic", "actor_target", "critic_target", "reward_model"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        return state_dicts, []

    def _wandb_visualize(self):
        # get latest rollout from buffer
        rollout_info = list(self.replay_buffer.info_buffer[self.replay_buffer.pos - 1].copy())
        # compute average object distance
        avg_obj_dist = np.mean([np.linalg.norm(rollout_info[t][0]["position"] - rollout_info[t][0]["goal_pos"], axis=-1)
                                for t in range(len(rollout_info))])
        # get goal image
        n_views = self.env.n_views
        image_views = ["Frontview", "Sideview"]
        rollout_goal = rollout_info[0][0]["goal_image"]
        for i in range(n_views):
            if self.obs_mode == "dlp":
                goal_img = self.extract_dlp_image(rollout_goal[i])
            else:
                goal_img = np.moveaxis(rollout_goal[i], 0, -1)
            wandb.log({f"Goal Image - {image_views[i]}": wandb.Image(goal_img)}, commit=False)
        # create episode video
        for i in range(n_views):
            vid_save_dir = f'./results/{image_views[i]}_episode_video.gif'
            if self.obs_mode == "dlp":
                img_array = np.zeros([len(rollout_info), *rollout_info[0][0]["image"][i].shape])
                for t in range(len(rollout_info)):
                    img_array[t] = rollout_info[t][0]["image"][i]
                img_list = self.extract_dlp_image(img_array)
            else:
                img_list = []
                for t in range(len(rollout_info)):
                    img_list.append(np.moveaxis(rollout_info[t][0]["image"][i], 0, -1))
            clip = ImageSequenceClip(img_list, fps=15)
            clip.write_gif(vid_save_dir, fps=15)
            vid = wandb.Video(data_or_path=vid_save_dir, caption=f"Avg Dist = {avg_obj_dist:2.4f}", fps=15)
            wandb.log({f"Episode Video - {image_views[i]}": vid}, commit=False)
        # visualize slots
        if self.obs_mode == "slot":
            img_array = np.zeros([8, *rollout_info[0][0]["image"][0].shape])
            img_array[0] = rollout_info[0][0]["goal_image"][0]
            for i in range(7):
                img_array[i + 1] = rollout_info[5 * i][0]["image"][0]
            images = extract_slot_image(img_array, self.env.latent_rep_model, self.device)
            wandb.log({f"Slot Image": wandb.Image(images)}, commit=False)

    def _wandb_log_eval_stats(self, eval_stat_dict):
        # log stats
        wandb.log({"eval_goal_achievement_%": eval_stat_dict["succes_rate"]}, commit=False)
        wandb.log({"mean_success_frac": eval_stat_dict["mean_success_frac"]}, commit=False)
        wandb.log({"mean_avg_obj_dist": eval_stat_dict["mean_avg_obj_dist"]}, commit=False)
        wandb.log({"mean_max_obj_dist": eval_stat_dict["mean_max_obj_dist"]}, commit=False)
        wandb.log({"eval_mean_reward": eval_stat_dict["mean_return"]}, commit=False)
        if self.reward_model is not None:
            wandb.log({"eval_mean_dlp_reward": eval_stat_dict["mean_latent_rep_return"]}, commit=False)
        # log goal image
        wandb.log({f"Eval Goal Image": wandb.Image(eval_stat_dict["goal_img"])}, commit=False)
        # log episode video
        vid_save_dir = f'./results/eval_episode_video.gif'
        clip = ImageSequenceClip(eval_stat_dict["img_list"], fps=15)
        clip.write_gif(vid_save_dir, fps=15)
        vid_caption = "Success" if eval_stat_dict["eval_vid_success"] else "Failure"
        vid = wandb.Video(data_or_path=vid_save_dir, caption=vid_caption, fps=15)
        wandb.log({f"Eval Episode Video": vid}, commit=False)
        # log orientation distance distribution plot
        if self.env.push_t:
            hist_fig = plt.figure(1, figsize=(5, 5), clear=True)
            plt.hist(eval_stat_dict["ori_dist_array"], bins=np.linspace(0, np.pi, num=50), edgecolor='black')
            wandb.log({f"Distribution of Orientation Distance from Goal": wandb.Image(hist_fig)}, commit=False)