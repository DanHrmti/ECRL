from collections import OrderedDict

import time
import os
import yaml
from pathlib import Path
from copy import deepcopy

import isaacgym

import numpy as np
import torch

import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip

from gym import spaces
from gym.core import GoalEnv

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.env_checker import check_env

from isaac_panda_push_env import IsaacPandaPush

from policies import get_single_goal
from utils import load_pretrained_rep_model, load_latent_classifier, get_dlp_rep, extract_dlp_image, check_config


class SB3VecEnvAdapter(VecEnv):

    def __init__(self, num_envs: int, observation_space: spaces.Space, action_space: spaces.Space):
        super().__init__(num_envs, observation_space, action_space)

    def step_async(self, actions):
        pass

    def step_wait(self):
        pass

    def get_attr(self, attr_name, indices=None):
        pass

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        pass

    def seed(self, seed):
        pass

    def env_is_wrapped(self):
        pass

    def render(self):
        pass


class IsaacPandaPushGoalSB3Wrapper(GoalEnv, SB3VecEnvAdapter):
    def __init__(self, env, obs_mode, n_views, latent_rep_model, latent_classifier, reward_cfg, smorl=False, **kwargs):

        self.env = env
        self.device = self.env.device

        super().__init__(self.env.num_envs, self.env.observation_space, self.env.action_space)

        # Gym specific attributes
        self.name = "PandaPush"
        self.spec = None
        self.metadata = None

        # observation related attributes
        self.obs_mode = obs_mode
        self.n_views = n_views
        self.smorl = smorl
        if self.smorl:
            self.env.max_episode_length = self.env.cfg["env"]["episodeLength"][0]  # smorl trains on single object goals
            assert self.obs_mode in ['state', 'dlp']

        # reward related attributes
        self.reward_scale = reward_cfg.get("reward_scale", 1.0)
        self.dist_threshold = reward_cfg.get("dist_threshold", np.sqrt(2) * self.env.cube_size)
        self.ori_threshold = reward_cfg.get("ori_threshold", 0.3)
        self.only_ori_reward = reward_cfg.get("only_ori_reward", False)
        self.reward_range = [-1 * self.reward_scale, 0 * self.reward_scale]

        # other attributes
        self.horizon = self.env.max_episode_length

        # representation model
        self.latent_rep_model = latent_rep_model.to(self.device) if latent_rep_model is not None else None
        # latent classifier model
        self.latent_classifier = latent_classifier.to(self.device) if latent_classifier is not None else None

        # set up observation space
        obs_dict = self.env.reset()

        if self.obs_mode == 'state':
            obs = self._get_state_obs(obs_dict)
            obs_shape = obs.shape[1:]
            obs_low = -np.inf
            obs_high = np.inf
            obs_dtype = np.float32

            goal_shape = obs_shape
            goal_low = obs_low
            goal_high = obs_high
            goal_dtype = np.float32

            a_goal_shape = goal_shape
            d_goal_shape = [1, goal_shape[-1]] if self.smorl else goal_shape

        elif self.obs_mode == 'state_unstruct':
            obs = self._get_state_unstruct_obs(obs_dict)
            obs_shape = obs.shape[1:]
            obs_low = -np.inf
            obs_high = np.inf
            obs_dtype = np.float32

            goal_shape = obs_shape
            goal_low = obs_low
            goal_high = obs_high
            goal_dtype = np.float32

            a_goal_shape = goal_shape
            d_goal_shape = goal_shape

        elif self.obs_mode in ['dlp', 'vae', 'slot']:
            obs = self._get_state_obs(obs_dict)
            obs_shape = obs.shape[1:]
            obs_low = -np.inf
            obs_high = np.inf
            obs_dtype = np.float32

            goal = self._get_latent_obs(obs_dict)
            goal_shape = goal.shape[1:]
            goal_low = -np.inf
            goal_high = np.inf
            goal_dtype = np.float32

            a_goal_shape = goal_shape
            d_goal_shape = [goal_shape[0], 1, goal_shape[-1]] if self.smorl else goal_shape

        else:  # obs_mode == "raw"
            obs = self._get_image_obs(obs_dict)
            obs_shape = obs.shape[1:]
            obs_low = 0
            obs_high = 255
            obs_dtype = np.uint8

            goal_shape = obs_shape
            goal_low = 0
            goal_high = 255
            goal_dtype = np.uint8

            a_goal_shape = goal_shape
            d_goal_shape = goal_shape

        self.observation_space = spaces.Dict({
            # "observation": spaces.Box(low=obs_low, high=obs_high, shape=obs_shape, dtype=obs_dtype),  # commented out to accelerate code
            "desired_goal": spaces.Box(low=goal_low, high=goal_high, shape=d_goal_shape, dtype=goal_dtype),
            "achieved_goal": spaces.Box(low=goal_low, high=goal_high, shape=a_goal_shape, dtype=goal_dtype),
        })

        # set up goal
        self.goal = None
        self.goal_pos = {}
        self.goal_image = None
        if self.smorl:
            self.goal_obj_index = None
            self.full_goal = None

        # set up action space
        low, high = self.env.act_space.low, self.env.act_space.high
        low, high = low[:3], high[:3]  # for allowing vertical movements and closed gripper only
        self.action_space = spaces.Box(low=low, high=high)

    def reset(self):
        """
        Extends env reset method to return Goal Environment observation instead of normal OrderedDict.

        Returns:
            dict: GoalEnv observation after reset occurs
        """
        goal_obs_dict = self.get_random_goal()  # resets env

        obs_dict = self.env.reset()

        # extract observation and achieved goal
        if self.obs_mode == 'state':
            observation = self._get_state_obs(obs_dict)
            achieved_goal = observation
            self.goal = self._get_state_obs(goal_obs_dict)
            if self.smorl:
                self.full_goal = self.goal
                rand_obj = np.random.randint(1, self.num_objects+1, self.num_envs)
                rand_single_goal = self.full_goal[np.arange(self.num_envs), rand_obj]
                self.goal = np.expand_dims(rand_single_goal, -2)
                self.goal_obj_index = rand_obj  # for goal info

        elif self.obs_mode == 'state_unstruct':
            observation = self._get_state_unstruct_obs(obs_dict)
            achieved_goal = observation
            self.goal = self._get_state_unstruct_obs(goal_obs_dict)

        elif self.obs_mode in ['dlp', 'vae', 'slot']:
            observation = self._get_state_obs(obs_dict)
            achieved_goal = self._get_latent_obs(obs_dict)  # [n_views, *(latent_dims)]
            self.goal = self._get_latent_obs(goal_obs_dict)  # [n_views, *(latent_dims)]
            if self.smorl:
                self.full_goal = self.goal
                self.goal = get_single_goal(self.full_goal, self.latent_classifier, self.device, check_goal_reaching=False)

        else:  # obs_mode == 'raw'
            observation = self._get_image_obs(obs_dict)  # [n_views, 3, h, w]
            achieved_goal = observation
            self.goal = self._get_image_obs(goal_obs_dict)  # [n_views, 3, h, w]

        # set goal info
        goal_observation = self._get_state_obs(goal_obs_dict)
        self.goal_pos = goal_observation[:, 1:, :-(self.num_objects+1)]
        if self.push_t:
            self.goal_pos = self.goal_pos[:, 0:1]
        self.goal_image = self._get_image_obs(goal_obs_dict)

        # create GoalEnv observation
        obs = {
            # "observation": observation,  # commented out to accelerate code
            "desired_goal": self.goal,
            "achieved_goal": achieved_goal
        }

        return obs

    def step(self, action):
        """
        Extends env step() function call to:
            - return goal environment observation instead of normal observation
            - compute reward based on goal and current state

        Args:
            action (torch.tensor): action to take in environment

        Returns:
            4-tuple:
                - observations based on obs_mode
                - reward from the environment
                - whether the current episode is completed or not
                - misc information
        """
        # modify action to fit env action space and allow vertical movements with closed gripper only
        action_xyz = torch.tensor(action, device=self.device, dtype=torch.float32)
        action_rest = torch.tensor([0, 0, 0, -1], device=self.device).unsqueeze(0).expand(self.num_envs, -1)
        action = torch.cat([action_xyz, action_rest], dim=-1)

        # take policy step
        obs_dict, _, episode_done, info = self.env.step(action)

        # extract observation
        if self.obs_mode == 'state':
            observation = self._get_state_obs(obs_dict)
            achieved_goal = observation

        elif self.obs_mode == 'state_unstruct':
            observation = self._get_state_unstruct_obs(obs_dict)
            achieved_goal = observation

        elif self.obs_mode in ['dlp', 'vae', 'slot']:
            observation = self._get_state_obs(obs_dict)
            achieved_goal = self._get_latent_obs(obs_dict)

        else:  # obs_mode == 'raw'
            observation = self._get_image_obs(obs_dict)
            achieved_goal = observation

        # create GoalEnv observation
        obs = {
            # "observation": observation,  # commented out to accelerate code
            "desired_goal": self.goal,
            "achieved_goal": achieved_goal
        }

        # save info
        vec_info = {
            "position": self._get_state_obs(obs_dict)[:, 1:, :-(self.num_objects+1)],
            "image": self._get_image_obs(obs_dict),
            "goal_pos": self.goal_pos,
            "goal_image": self.goal_image,
        }
        if self.random_obj_num:
            vec_info["cur_num_obj"] = self.cur_num_objects * np.ones(self.num_envs, dtype=int)
        if self.smorl and self.obs_mode == 'state':
            vec_info["goal_obj_index"] = self.goal_obj_index

        # get reward
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], vec_info)

        # add goal reaching info
        goal_frac_reached, avg_obj_dist, max_obj_dist, ori_dist = self.check_success(obs["achieved_goal"], obs["desired_goal"], vec_info)
        vec_info["goal_success_frac"] = goal_frac_reached
        vec_info["avg_obj_dist"] = avg_obj_dist
        vec_info["max_obj_dist"] = max_obj_dist
        if self.push_t:
            vec_info["ori_dist"] = ori_dist

        # set done flag
        done = episode_done.cpu().numpy()  # shouldn't get done signal even if reached goal

        # add info for HerReplayBuffer use (ignoring done due to episode termination)
        vec_info["TimeLimit.truncated"] = episode_done.cpu().numpy()

        # convert info to tuple of dicts for SB3 compatibility
        info = tuple([{key: vec_info[key][i] for key in vec_info} for i in range(self.num_envs)])

        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info={}):
        """
        Reward function for the goal conditioned task: negative distance from goal averaged over objects

        Args:
            achieved_goal: current state representation
            desired_goal:  goal state representation
            info: contains additional information

        Returns:
            goal conditioned reward
        """
        if type(info) == dict:
            a_goal, d_goal = info['position'].copy(), info['goal_pos'].copy()
            if self.random_obj_num:
                a_goal, d_goal = a_goal[:, :self.cur_num_objects], d_goal[:, :self.cur_num_objects]
                cur_num_obj = self.cur_num_objects

            if self.smorl and self.obs_mode == "state":
                goal_obj_index = info["goal_obj_index"] - 1
                a_goal = np.expand_dims(a_goal[np.arange(len(goal_obj_index)), goal_obj_index], -2)
                d_goal = np.expand_dims(d_goal[np.arange(len(goal_obj_index)), goal_obj_index], -2)

        else:  # numpy array of dicts from HER replay buffer
            a_goal = np.array([info[i]['position'] for i in range(len(info))])
            d_goal = np.array([info[i]['goal_pos'] for i in range(len(info))])
            if self.random_obj_num:
                bs = a_goal.shape[0]
                cur_num_obj = np.array([info[i]['cur_num_obj'] for i in range(len(info))])
                mask_cond = np.expand_dims(cur_num_obj, -1).repeat(self.num_objects, axis=-1)
                obj_idx = np.tile(np.arange(self.num_objects), bs).reshape(bs, -1)
                obj_mask = obj_idx >= mask_cond
                a_goal[obj_mask] = 0
                d_goal[obj_mask] = 0

            if self.smorl and self.obs_mode == "state":
                goal_obj_index = np.array([info[i]['goal_obj_index'] for i in range(len(info))]) - 1
                a_goal = np.expand_dims(a_goal[np.arange(len(goal_obj_index)), goal_obj_index], -2)
                d_goal = np.expand_dims(d_goal[np.arange(len(goal_obj_index)), goal_obj_index], -2)

        if self.push_t:
            # normalize orientation by pi
            a_goal = a_goal[..., 2:] / np.pi
            d_goal = d_goal[..., 2:] / np.pi
            # calculate per object orientation distance (minimum of both directions)
            dist = np.linalg.norm(a_goal - d_goal, ord=2, axis=-1)
            dist = np.minimum(dist, np.linalg.norm(a_goal - (d_goal + 2), ord=2, axis=-1))
            dist = np.minimum(dist, np.linalg.norm((a_goal + 2) - d_goal, ord=2, axis=-1))
            # calculate reward
            reward = -np.mean(dist, axis=-1)

        else:
            # normalize xy positions by table scale
            table_diag_len = (np.linalg.norm([(self.table_dims[0]) / 2, (self.table_dims[1]) / 2]))
            a_goal[..., :2] /= table_diag_len
            d_goal[..., :2] /= table_diag_len
            # calculate per object distance
            dist = np.linalg.norm(a_goal - d_goal, axis=-1)
            # calculate reward
            if self.random_obj_num:
                reward = -np.sum(dist, axis=-1) / cur_num_obj
            else:
                reward = -np.mean(dist, axis=-1)

        # rescale reward
        reward = reward * self.reward_scale

        return reward

    def check_success(self, achieved_goal, desired_goal, info={}):
        """
        Checks goal reaching success
        Args:
            achieved_goal: current state representation
            desired_goal:  goal state representation
            info: contains additional information

        Returns:
            fraction of goals reached
        """
        a_goal, d_goal = info['position'], info['goal_pos']
        if self.random_obj_num:
            a_goal, d_goal = a_goal[:, :self.cur_num_objects], d_goal[:, :self.cur_num_objects]

        if self.push_t:
            ori_dist = np.abs(a_goal[..., -1] - d_goal[..., -1])
            ori_dist = np.minimum(ori_dist, np.abs((a_goal[..., -1] + 2 * np.pi) - d_goal[..., -1]))
            ori_dist = np.minimum(ori_dist, np.abs((a_goal[..., -1] - 2 * np.pi) - d_goal[..., -1]))
            dist = ori_dist
            obj_goal_reached = ori_dist < self.ori_threshold

        else:
            ori_dist = None
            dist = np.linalg.norm(a_goal[..., :2] - d_goal[..., :2], axis=-1)
            obj_goal_reached = dist < self.dist_threshold

        goal_frac_reached = np.mean(obj_goal_reached, axis=-1)
        avg_obj_dist = np.mean(dist, axis=-1)
        max_obj_dist = np.max(dist, axis=-1)
        return goal_frac_reached, avg_obj_dist, max_obj_dist, ori_dist

    def get_random_goal(self):
        # pre reset
        self.env.goal_reset = True
        # reset
        self.env.reset()
        # post reset
        self.env.goal_reset = False
        # move arm one step back
        action = torch.tensor([-1, 0, 0, 0, 0, 0, -1], device=self.device).unsqueeze(0).expand(self.env.num_envs, -1)
        self.env.step(action)
        # get goal
        goal_obs_dict, _, _, _ = self.env.step(action)
        goal_obs_dict = deepcopy(goal_obs_dict)
        return goal_obs_dict

    def _get_state_obs(self, obs_dict):
        """
           Gets simulation state from environment, reshapes to add an entity dimension
           and concatenates 1-hot features to each entity (eef + objects)

           Args:
               obs_dict (OrderedDict): ordered dictionary of observations

           Returns:
               np.ndarray: [num_envs, num_entities, state_dim + 1-hot_identifier]
       """
        num_envs = self.num_envs
        num_entities = self.num_objects + 1

        obs = obs_dict["obs"].reshape(num_envs, num_entities, -1)
        one_hot_id = torch.eye(num_entities, device=self.device).unsqueeze(0).expand(num_envs, -1, -1)

        if self.push_t:
            obs = torch.cat([obs[..., :2], obs[..., 5:], one_hot_id], dim=-1)
        else:
            obs = torch.cat([obs[..., :2], one_hot_id], dim=-1)

        return obs.cpu().numpy().squeeze()

    def _get_state_unstruct_obs(self, obs_dict):
        """
           Gets simulation state from environment, concatenates states from all entities

           Args:
               obs_dict (OrderedDict): ordered dictionary of observations

           Returns:
               np.ndarray: [num_envs, num_entities * state_dim]
       """
        obs = obs_dict["obs"][..., :2].reshape(self.num_envs, -1)
        return obs.cpu().numpy().squeeze()

    def _get_image_obs(self, obs_dict):
        """
        Gets multiview image observations

        Args:
            obs_dict (OrderedDict): ordered dictionary of observations

        Returns:
            np.array: [num_envs, num_views, channels, height, width]
        """
        obs = obs_dict["media"][:, :self.n_views]
        return obs.cpu().numpy()

    def _get_latent_obs(self, obs_dict):
        """
        Gets multiview latent representations

        Args:
            obs_dict (OrderedDict): ordered dictionary of observations

        Returns:
            np.array: [num_envs, num_views, num_entities, feature_dim]
        """
        image_obs = obs_dict["media"][:, :self.n_views]
        obs = self._image_to_latent_rep(image_obs)
        return obs.cpu().numpy()

    def _image_to_latent_rep(self, image_obs):
        orig_obs_shape = image_obs.shape
        if len(orig_obs_shape) == 4:  # no batch dim
            image_obs = image_obs.unsqueeze(0)

        if self.obs_mode == 'dlp':
            latent_obs = [self._extract_dlp_features(image_obs[:, i]) for i in range(self.n_views)]
        elif self.obs_mode == 'vae':
            latent_obs = [self._extract_vae_features(image_obs[:, i]) for i in range(self.n_views)]
        elif self.obs_mode == 'slot':
            latent_obs = [self._extract_slot_features(image_obs[:, i]) for i in range(self.n_views)]
        else:
            raise NotImplementedError

        latent_obs = torch.cat([view.unsqueeze(1) for view in latent_obs], dim=1)
        if len(orig_obs_shape) == 4:  # no batch dim
            latent_obs = latent_obs.squeeze(0)
        return latent_obs

    def _extract_dlp_features(self, image):
        normalized_image = image.to(torch.float32) / 255

        with torch.no_grad():
            encoded_output = self.latent_rep_model.encode_all(normalized_image, deterministic=True)
            dlp_features = get_dlp_rep(encoded_output)

        return dlp_features

    def _extract_vae_features(self, image):
        normalized_image = image.to(torch.float32) / 255

        with torch.no_grad():
            normalized_image = self.latent_rep_model.preprocess_rgb(normalized_image)
            vae_features = self.latent_rep_model.get_latent_rep(normalized_image, deterministic=True)

        return vae_features

    def _extract_slot_features(self, image):
        normalized_image = image.to(torch.float32) / 255

        with torch.no_grad():
            slots = self.latent_rep_model.predict(normalized_image, do_transforms=True, return_slots=True)
            slots = slots.squeeze()

        return slots

    def _get_ori_aware_goal(self, goal):
        pos, ori = goal[..., :2], goal[..., 2:]
        pos_list = []
        # for i in range(self.num_objects):
        pos_list.append(pos)
        pos_list.append(np.concatenate([pos[..., 0:1] + 4 * 0.03 * np.cos(ori),
                                        pos[..., 1:] + 4 * 0.03 * np.sin(ori)], axis=-1))
        pos_list.append(np.concatenate([pos[..., 0:1] + 4 * 0.03 * np.cos(ori + np.pi/2),
                                        pos[..., 1:] + 4 * 0.03 * np.sin(ori + np.pi/2)], axis=-1))

        ori_aware_goal = np.concatenate(pos_list, axis=-2)
        return ori_aware_goal

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        if method_name == "compute_reward":
            return self.compute_reward(*method_args)
        else:
            raise NotImplementedError(f"Method {method_name} is not implemented in this env")

    @property
    def num_objects(self) -> int:
        """Get the (maximum) number of objects in the environment."""
        return self.env.num_objects

    @property
    def cur_num_objects(self) -> int:
        """Get the current number of objects in the environment."""
        return self.env.cur_num_objects

    @property
    def num_colors(self) -> int:
        """Get the (maximum) number of objects in the environment."""
        return self.env.num_colors

    @property
    def max_episode_len(self) -> int:
        """Get the number of objects in the environment."""
        return self.env.max_episode_length

    @property
    def table_dims(self) -> list:
        """Table dimensions"""
        return self.env.table_dims

    @property
    def random_obj_num(self) -> bool:
        """Boolean 'random number of cubes' env indicator"""
        return self.env.random_obj_num

    @property
    def adjacent_goals(self) -> bool:
        """Boolean 'close cube goals' task indicator"""
        return self.env.adjacent_goals

    @property
    def small_table(self) -> bool:
        """Boolean 'close cube goals' task indicator"""
        return self.env.table_dims[0] < 0.5

    @property
    def ordered_push(self) -> bool:
        """Boolean 'stack push' task indicator"""
        return self.env.ordered_push

    @property
    def push_t(self) -> bool:
        """Boolean 'push_t' task indicator"""
        return self.env.push_t


if __name__ == '__main__':

    """
    This script can be used to debug the 'IsaacPandaPush' environment and it's wrapper.
    To visualize the environment simulation, set:
    - plot_images=True for single image visualization per timestep of a single env
    - create_episode_gif=True for episode video visualization in the .gif format of a single env
    - vis_dlp=True to visualize dlp particle locations on the above visualizations
    - debug_returns=True to visualize two consecutive states with a change in reward and the corresponding rewards
    """

    plot_images = True
    create_episode_gif = False
    vis_dlp = False
    debug_returns = False

    # load config files
    config = yaml.safe_load(Path('config/n_cubes/Config.yaml').read_text())
    isaac_env_cfg = yaml.safe_load(Path('config/n_cubes/IsaacPandaPushConfig.yaml').read_text())

    check_config(config, isaac_env_cfg)

    cuda_device = config['cudaDevice']

    # output directories
    results_dir = './results'
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
        print(f"Created directory {results_dir}")

    if vis_dlp:
        assert config["Model"]["obsMode"] == 'dlp', "vis_dlp can be set only with DLP observation mode"

    #################################
    #         Representation        #
    #################################

    latent_rep_model = load_pretrained_rep_model(dir_path=config['Model']['latentRepPath'], model_type=config['Model']['obsMode'])
    latent_classifier = load_latent_classifier(config, num_objects=isaac_env_cfg["env"]["numObjects"])

    #################################
    #          Environment          #
    #################################

    # create environments
    envs = IsaacPandaPush(
        cfg=isaac_env_cfg,
        rl_device=f"cuda:{config['cudaDevice']}",
        sim_device=f"cuda:{config['cudaDevice']}",
        graphics_device_id=config['cudaDevice'],
        headless=True,
        virtual_screen_capture=False,
        force_render=False,
    )

    # wrap enviroments for GoalEnv and SB3 compatibility
    env = IsaacPandaPushGoalSB3Wrapper(
        env=envs,
        obs_mode=config['Model']['obsMode'],
        n_views=config['Model']['numViews'],
        latent_rep_model=latent_rep_model,
        latent_classifier=latent_classifier,
        reward_cfg=config['Reward']['GT'],
        smorl=(config['Model']['method'] == 'SMORL'),
    )

    if config['envCheck']:
        check_env(env, warn=True, skip_render_check=True)  # verify SB3 compatibility

    print(f"Finished setting up environment")

    # perform rollouts with random policy
    for _ in range(100):
        start_time = time.time()
        obs = env.reset()
        if create_episode_gif:
            img_list_fview = []
            img_list_sview = []

        for i in range(envs.max_episode_length):
            # choose random action
            action = -2 * torch.rand((env.num_envs, 3), device=f"cuda:{cuda_device}") + 1  # uniform [-1, 1] in all dimensions
            # perform action
            if i > 0 and debug_returns:
                prev_obs, prev_reward, prev_done, prev_info = obs, reward, done, info
            obs, reward, done, info = env.step(action)

            # plot goal media
            if (plot_images or create_episode_gif) and i == 0:
                frontview_image = info[0]["goal_image"][0]
                sideview_image = info[0]["goal_image"][1]
                if vis_dlp:
                    frontview_image = extract_dlp_image(frontview_image, latent_rep_model, f"cuda:{cuda_device}")
                    sideview_image = extract_dlp_image(sideview_image, latent_rep_model, f"cuda:{cuda_device}")
                else:
                    frontview_image = np.moveaxis(frontview_image, 0, -1)
                    sideview_image = np.moveaxis(sideview_image, 0, -1)

                plt.imshow(frontview_image)
                plt.axis('off')
                plt.show()

                plt.imshow(sideview_image)
                plt.axis('off')
                plt.show()

            if i > 0 and debug_returns:
                for j in range(env.num_envs):
                    if np.abs(reward[j] - prev_reward[j]) > 1e-5:
                        print(f"prev_reward = {prev_reward[j]}, reward = {reward[j]}")
                        plt.imshow(np.moveaxis(prev_info[j]["image"][0], 0, -1))
                        plt.axis('off')
                        plt.show()
                        plt.imshow(np.moveaxis(info[j]["image"][0], 0, -1))
                        plt.axis('off')
                        plt.show()
                        plt.imshow(np.moveaxis(info[j]["goal_image"][0], 0, -1))
                        plt.axis('off')
                        plt.show()
                        print("\n")

            if plot_images or create_episode_gif:
                frontview_image = info[0]["image"][0]
                sideview_image = info[0]["image"][1]
                if vis_dlp:
                    frontview_image = extract_dlp_image(frontview_image, latent_rep_model, f"cuda:{cuda_device}")
                    sideview_image = extract_dlp_image(sideview_image, latent_rep_model, f"cuda:{cuda_device}")
                else:
                    frontview_image = np.moveaxis(frontview_image, 0, -1)
                    sideview_image = np.moveaxis(sideview_image, 0, -1)

                # plot current state
                if plot_images:
                    plt.imshow(frontview_image)
                    plt.axis('off')
                    plt.show()

                    plt.imshow(sideview_image)
                    plt.axis('off')
                    plt.show()

                print("\n")

                # save current state for episode video
                if create_episode_gif:
                    img_list_fview.append(frontview_image)
                    img_list_sview.append(sideview_image)

        print(f"Episode completed in {time.time() - start_time:5.2f}s")

        if create_episode_gif:
            clip = ImageSequenceClip(img_list_fview, fps=15)
            clip.write_gif(f'./results/episode_video_fview.gif', fps=15)
            clip = ImageSequenceClip(img_list_sview, fps=15)
            clip.write_gif(f'./results/episode_video_sview.gif', fps=15)
