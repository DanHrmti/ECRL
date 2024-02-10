# Copyright (c) 2021-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Modified by Dan Haramati from https://github.com/NVIDIA-Omniverse/IsaacGymEnvs
"""

import os
import time
import yaml
from pathlib import Path

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

import numpy as np
import torch

from moviepy.editor import ImageSequenceClip
import matplotlib.pyplot as plt

from isaac_vec_env import IsaacVecEnv

from panda_controller.osc import OSCController


def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat


def quat2axisangle(quat):
    """
    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.
    Args:
        quat (tensor): (..., 4) tensor where the final dim is (x,y,z,w) quaternion
    Returns:
        tensor: (..., 3) axis-angle exponential coordinates
    """
    # reshape quaternion
    quat_shape = quat.shape[:-1]      # ignore last dim
    quat = quat.reshape(-1, 4)
    # clip quaternion
    quat[:, 3] = torch.clamp(quat[:, 3], -1., 1.)
    # Calculate denominator
    den = torch.sqrt(1. - quat[:,3] * quat[:,3])
    # Create return array
    ret = torch.zeros_like(quat)[:, :3]
    idx = torch.nonzero(den).reshape(-1)
    ret[idx, :] = (quat[idx, :3] * 2. * torch.acos(quat[idx, 3]).unsqueeze(-1)) / den[idx].unsqueeze(-1)

    # Reshape and return output
    ret = ret.reshape(list(quat_shape) + [3, ])
    return ret


class IsaacPandaPush(IsaacVecEnv):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        if type(self.cfg["env"]["episodeLength"]) is list:
            self.max_episode_length = self.cfg["env"]["episodeLength"][self.cfg["env"]["numObjects"] - 1]
        else:
            self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.adjacent_goals = self.cfg["env"].get("AdjacentGoals", False)
        self.ordered_push = self.cfg["env"].get("OrderedPush", False)
        self.push_t = self.cfg["env"].get("PushT", False)

        self.random_obj_colors = self.cfg["env"].get("RandColor", False)
        self.random_obj_num = self.cfg["env"].get("RandNumObj", False)
        self.random_obj_num_dist = self.cfg["env"].get("RandNumObjDist", None)
        self.sort_push = self.cfg["env"].get("SortPush", False)

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.franka_position_noise = self.cfg["env"]["frankaPositionNoise"]
        self.franka_rotation_noise = self.cfg["env"]["frankaRotationNoise"]
        self.franka_dof_noise = self.cfg["env"]["frankaDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # Create dicts to pass to reward function
        self.reward_settings = {}

        # Controller type
        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {"osc", "joint_tor"},\
            "Invalid control type specified. Must be one of: {osc, joint_tor}"

        # dimensions
        # obs include: eef_pos (3) + cube_pos (num_cubes * 3)
        self.cfg["env"]["numObservations"] = 3 + 3 * self.cfg["env"]["numObjects"] if self.control_type == "osc" else 26
        # actions include: delta EEF if OSC (6) or joint torques (7) + bool gripper (1)
        self.cfg["env"]["numActions"] = 7 if self.control_type == "osc" else 8

        # important quantities
        self.num_objects = self.cfg["env"]["numObjects"]
        self.cur_num_objects = self.num_objects
        num_other_actors = 6 if self.ordered_push else 3
        self.num_actors = num_other_actors + self.num_objects
        self.num_colors = self.cfg["env"].get("numColors", self.num_objects)
        self.cube_size = self.cfg["env"]["cubeSize"]

        # Values to be filled in at runtime
        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        self.num_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed
        self._init_obj_state_dict = {f"cube{i+1}": None for i in range(self.num_objects)}  # Initial state of cubes for the current env
        self._obj_state_dict = {f"cube{i+1}": None for i in range(self.num_objects)}  # Current state of cubes for the current env
        self._obj_id_dict = {f"cube{i+1}": None for i in range(self.num_objects)}  # Actor ID corresponding to each cube for a given env

        # Tensor placeholders
        self._root_state = None  # State of root body        (n_envs, 13)
        self._dof_state = None   # State of all joints       (n_envs, n_dof)
        self._q = None           # Joint positions           (n_envs, n_dof)
        self._qd = None          # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._contact_forces = None     # Contact forces in sim
        self._eef_state = None  # end effector state (at grasping point)
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._arm_control = None  # Tensor buffer for controlling arm
        self._gripper_control = None  # Tensor buffer for controlling gripper
        self._pos_control = None            # Position actions
        self._effort_control = None         # Torque actions
        self._franka_effort_limits = None        # Actuator effort limits for franka
        self._global_indices = None         # Unique indices corresponding to all envs in flattened array

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device,
                         graphics_device_id=graphics_device_id, headless=headless,
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # Franka defaults
        self.franka_default_dof_pos = to_torch(
            [0, 0.1963, 0, -2.6180, 0, 2.9416, 0.7854, 0.0, 0.0], device=self.device
        )

        # Define controller
        table_x_half = self.cfg["env"]["tableDims"][0] / 2
        table_y_half = self.cfg["env"]["tableDims"][1] / 2
        table_height = self.cfg["env"]["tablePos"][2] + self.cfg["env"]["tableDims"][2] / 2
        cube_size = self.cube_size
        pos_limits = torch.tensor(
            [[-(table_x_half + (cube_size / 2)), -(table_y_half + (cube_size / 2)), table_height + 0.02],
             [(table_x_half + (cube_size / 2)), (table_y_half + (cube_size / 2)), table_height + cube_size]],
            device=self.device
        )

        self.controller = OSCController(
            input_min=-1,
            input_max=1,
            output_min=[-0.125, -0.125, -0.125, -0.5, -0.5, -0.5],
            output_max=[0.125, 0.125, 0.125, 0.5, 0.5, 0.5],
            control_min=-self._franka_effort_limits[:7],
            control_max=self._franka_effort_limits[:7],
            control_noise=0,
            control_dim=7,
            device=self.device,
            kp=150.0,
            kp_limits=(10.0, 300.),
            kp_null=10.0,
            kp_null_limits=(0.0, 50.0),
            pos_limits=pos_limits,
            rest_qpos=self.franka_default_dof_pos[:7],
        )

        self.control_timestep = 1.0 / self.cfg["env"]["controlFrequency"]

        # Reset all environments
        self.goal_reset = False
        self.franka_default_eef_pos = torch.ones_like(self._eef_state)
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.gym.simulate(self.sim)

        # Refresh tensors
        self._refresh()

        # Get initial eef xy position for cube sampling
        self.franka_default_eef_pos = self._eef_state.clone()

        # Reset controller
        self.controller.reset(self.get_control_dict())

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        self.gym.set_light_parameters(self.sim, 0, gymapi.Vec3(0.4, 0.4, 0.4), gymapi.Vec3(0.3, 0.3, 0.3), gymapi.Vec3(0.5, 0.5, 0.55))
        self.gym.set_light_parameters(self.sim, 1, gymapi.Vec3(0.4, 0.4, 0.4), gymapi.Vec3(0.3, 0.3, 0.3), gymapi.Vec3(0.5, -0.5, 0.55))
        self.gym.set_light_parameters(self.sim, 2, gymapi.Vec3(0.4, 0.4, 0.4), gymapi.Vec3(0.3, 0.3, 0.3), gymapi.Vec3(-0.5, 0.5, 0.55))
        self.gym.set_light_parameters(self.sim, 3, gymapi.Vec3(0.4, 0.4, 0.4), gymapi.Vec3(0.3, 0.3, 0.3), gymapi.Vec3(-0.5, -0.5, 0.55))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
        franka_asset_file = "urdf/franka_description/robots/franka_panda_gripper.urdf"
        t_block_asset_file = "urdf/t_block.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)

        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        franka_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0, 0, 5000., 5000.], dtype=torch.float, device=self.device)
        franka_dof_damping = to_torch([0, 0, 0, 0, 0, 0, 0, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        # Create table asset
        self.table_dims = self.cfg["env"]["tableDims"]
        table_pos = self.cfg["env"]["tablePos"]
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *self.table_dims, table_opts)

        # Create table stand asset
        table_stand_dims = [0.2, 0.2, 0.05]
        table_stand_pos = [-(self.table_dims[0] / 2 + table_stand_dims[0] / 2 + 0.16), 0.0, 0.92]
        # table_stand_pos = [-(self.table_dims[0] / 2 + table_stand_dims[0] / 2 + 0.16), 0.0, 0.91]
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, *table_stand_dims, table_stand_opts)

        # Create object assets
        self.object_colors = [
            gymapi.Vec3(0.6, 0.1, 0.0),  # Red
            gymapi.Vec3(0.0, 0.6, 0.1),  # Green
            gymapi.Vec3(0.0, 0.1, 0.8),  # Blue
            gymapi.Vec3(0.7, 0.7, 0.0),  # Yellow
            gymapi.Vec3(0.5, 0.0, 0.5),  # Purple
            gymapi.Vec3(0.0, 0.9, 0.9),  # Cyan
            gymapi.Vec3(1.0, 0.4, 0.7),  # Pink
            gymapi.Vec3(0.55, 0.27, 0.07),  # Brown
            gymapi.Vec3(1.0, 0.55, 0.0),  # Orange
        ]

        asset_options = gymapi.AssetOptions()

        if self.push_t:
            object_assets = [self.gym.load_asset(self.sim, asset_root, t_block_asset_file, asset_options)
                             for _ in range(self.num_objects)]
        else:
            object_assets = [self.gym.create_box(self.sim, *([self.cube_size] * 3), asset_options)
                             for _ in range(self.num_objects)]

        if self.ordered_push:
            # Create corridor assets
            sidewall_dims = [0.12, 0.08, 0.025]
            backwall_dims = [0.01, 2 * sidewall_dims[1] + (self.cube_size + 0.02), 0.025]
            sidewall_x_pos = table_pos[0] + (self.table_dims[0] / 2) - (sidewall_dims[0] / 2) - backwall_dims[0]
            sidewall_y_pos = table_pos[1] + (sidewall_dims[1] / 2) + (self.cube_size + 0.02) / 2
            backwall_x_pos = table_pos[0] + (self.table_dims[0] / 2) - (backwall_dims[0] / 2)
            backwall_y_pos = table_pos[1]
            z_pos = table_pos[2] + (self.table_dims[2] / 2) + (sidewall_dims[2] / 2)
            corridor_asset_opts = gymapi.AssetOptions()
            corridor_asset_opts.fix_base_link = True
            rightwall_asset = self.gym.create_box(self.sim, *sidewall_dims, corridor_asset_opts)
            leftwall_asset = self.gym.create_box(self.sim, *sidewall_dims, corridor_asset_opts)
            backwall_asset = self.gym.create_box(self.sim, *backwall_dims, corridor_asset_opts)

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)

        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        self._franka_effort_limits = []
        for i in range(self.num_franka_dofs):
            franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS if i > 6 else gymapi.DOF_MODE_EFFORT
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
                franka_dof_props['damping'][i] = franka_dof_damping[i]
            else:
                franka_dof_props['stiffness'][i] = 7000.0
                franka_dof_props['damping'][i] = 50.0

            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])
            self._franka_effort_limits.append(franka_dof_props['effort'][i])

        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self._franka_effort_limits = to_torch(self._franka_effort_limits, device=self.device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[[7, 8]] = 0.1
        franka_dof_props['effort'][7] = 200
        franka_dof_props['effort'][8] = 200

        # Define start pose for franka
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(table_stand_pos[0] + 0.05, table_stand_pos[1], table_stand_pos[2] + self.table_dims[2] / 2)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(table_pos) + np.array([0, 0, self.table_dims[2] / 2])

        # Define start pose for table stand
        table_stand_start_pose = gymapi.Transform()
        table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
        table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for cubes (doesn't really matter since they get overridden during reset())
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        object_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        max_agg_bodies = num_franka_bodies + 2 + self.num_objects  # 1 for table, table stand, cubes
        max_agg_shapes = num_franka_shapes + 2 + self.num_objects  # 1 for table, table stand, cubes

        if self.ordered_push:
            # Define start pose for walls
            # right
            right_wall_start_pose = gymapi.Transform()
            right_wall_start_pose.p = gymapi.Vec3(sidewall_x_pos, sidewall_y_pos, z_pos)
            right_wall_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            # left
            left_wall_start_pose = gymapi.Transform()
            left_wall_start_pose.p = gymapi.Vec3(sidewall_x_pos, -sidewall_y_pos, z_pos)
            left_wall_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            # back
            back_wall_start_pose = gymapi.Transform()
            back_wall_start_pose.p = gymapi.Vec3(backwall_x_pos, backwall_y_pos, z_pos)
            back_wall_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            # consider in max bodies/shapes
            max_agg_bodies += 3  # 3 walls
            max_agg_shapes += 3  # 3 walls

        self.frankas = []
        self.envs = []
        self.fview_cam_tensors = []
        self.sview_cam_tensors = []

        # Create environments
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: franka should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create franka
            # Potentially randomize start pose
            if self.franka_position_noise > 0:
                rand_xy = self.franka_position_noise * (-1. + np.random.rand(2) * 2.0)
                franka_start_pose.p = gymapi.Vec3(-0.45 + rand_xy[0], 0.0 + rand_xy[1], 1.0 + self.table_dims[2] / 2)
            if self.franka_rotation_noise > 0:
                rand_rot = torch.zeros(1, 3)
                rand_rot[:, -1] = self.franka_rotation_noise * (-1. + np.random.rand() * 2.0)
                new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
                franka_start_pose.r = gymapi.Quat(*new_quat)
            franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            self.frankas.append(franka_actor)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create table
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
            # self.gym.set_rigid_body_color(env_ptr, table_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.9, 0.9, 0.9))

            # Create robot table stand
            table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand", i, 1, 0)

            if self.ordered_push:
                # create corridor walls
                rightwall_actor = self.gym.create_actor(env_ptr, rightwall_asset, right_wall_start_pose, "rightwall", i, 1, 0)
                leftwall_actor = self.gym.create_actor(env_ptr, leftwall_asset, left_wall_start_pose, "leftwall", i, 1, 0)
                backwall_actor = self.gym.create_actor(env_ptr, backwall_asset, back_wall_start_pose, "backwall", i, 1, 0)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create objects and set colors
            for j, key in enumerate(self._obj_id_dict):
                self._obj_id_dict[key] = self.gym.create_actor(env_ptr, object_assets[j], object_start_pose, key, i, 0, 0)
                self.gym.set_rigid_body_color(env_ptr, self._obj_id_dict[key], 0, gymapi.MESH_VISUAL, self.object_colors[j % self.num_colors])

            if self.enable_camera_sensors:
                # Create frontview camera
                cam_props = gymapi.CameraProperties()
                cam_props.width = self.cfg["env"]["cameraRes"]
                cam_props.height = self.cfg["env"]["cameraRes"]
                cam_props.supersampling_horizontal = self.cfg["env"]["cameraSupersampleRatio"]
                cam_props.supersampling_vertical = self.cfg["env"]["cameraSupersampleRatio"]
                cam_props.horizontal_fov = 35
                cam_props.enable_tensors = True
                cam_handle = self.gym.create_camera_sensor(env_ptr, cam_props)
                self.gym.set_camera_location(cam_handle, env_ptr, gymapi.Vec3(0.8, 0, 1.80),
                                                                  gymapi.Vec3(0.12, 0, self._table_surface_pos[2]))
                # Obtain camera tensor
                cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_handle, gymapi.IMAGE_COLOR)
                torch_cam_tensor = gymtorch.wrap_tensor(cam_tensor)
                torch_cam_tensor = torch_cam_tensor.permute((2, 0, 1))
                self.fview_cam_tensors.append(torch_cam_tensor.view(1, *torch_cam_tensor.shape))

                # Create sideview camera
                cam_props = gymapi.CameraProperties()
                cam_props.width = self.cfg["env"]["cameraRes"]
                cam_props.height = self.cfg["env"]["cameraRes"]
                cam_props.supersampling_horizontal = self.cfg["env"]["cameraSupersampleRatio"]
                cam_props.supersampling_vertical = self.cfg["env"]["cameraSupersampleRatio"]
                cam_props.horizontal_fov = 35
                cam_props.enable_tensors = True
                cam_handle = self.gym.create_camera_sensor(env_ptr, cam_props)
                self.gym.set_camera_location(cam_handle, env_ptr, gymapi.Vec3(0, -0.8, 1.65),
                                                                  gymapi.Vec3(0, -0.12, self._table_surface_pos[2]))
                # Obtain camera tensor
                cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_handle, gymapi.IMAGE_COLOR)
                torch_cam_tensor = gymtorch.wrap_tensor(cam_tensor)
                torch_cam_tensor = torch_cam_tensor.permute((2, 0, 1))
                self.sview_cam_tensors.append(torch_cam_tensor.view(1, *torch_cam_tensor.shape))

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)

        # Setup init state buffer
        for key in self._init_obj_state_dict:
            self._init_obj_state_dict[key] = torch.zeros(self.num_envs, 13, device=self.device)

        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        franka_handle = 0
        self.handles = {
            # Franka
            "hand": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_hand"),
            "leftfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_leftfinger_tip"),
            "rightfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_rightfinger_tip"),
            "grip_site": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_grip_site"),
        }
        # Cube handles
        # for key in self._obj_id_dict:
        #     self.handles[f"{key}_body_handle"] = self.gym.find_actor_rigid_body_handle(self.envs[0],
        #                                                                                self._obj_id_dict[key], "box")

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._eef_state = self._rigid_body_state[:, self.handles["grip_site"], :]
        self._eef_lf_state = self._rigid_body_state[:, self.handles["leftfinger_tip"], :]
        self._eef_rf_state = self._rigid_body_state[:, self.handles["rightfinger_tip"], :]
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        hand_joint_index = self.gym.get_actor_joint_dict(env_ptr, franka_handle)['panda_hand_joint']
        self._j_eef = jacobian[:, hand_joint_index, :, :7]
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :7, :7]
        for key in self._obj_state_dict:
            self._obj_state_dict[key] = self._root_state[:, self._obj_id_dict[key], :]

        # Initialize states
        self.states.update({
            f"{key}_size": torch.ones_like(self._eef_state[:, 0]) * self.cube_size for key in self._obj_state_dict
        })

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize control
        self._arm_control = self._effort_control[:, :7]
        self._gripper_control = self._pos_control[:, 7:9]

        # Initialize indices
        self._global_indices = torch.arange(self.num_envs * self.num_actors, dtype=torch.int32,
                                            device=self.device).view(self.num_envs, -1)

    def _update_states(self):
        # Franka
        self.states.update({
            "q": self._q[:, :],
            "q_gripper": self._q[:, -2:],
            "eef_pos": self._eef_state[:, :3],
            "eef_ori": quat2axisangle(self._eef_state[:, 3:7]),
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
            "eef_lf_pos": self._eef_lf_state[:, :3],
            "eef_rf_pos": self._eef_rf_state[:, :3],
        })
        # Cubes
        for key in self._obj_state_dict:
            self.states.update({
                f"{key}_pos": self._obj_state_dict[key][:, :3],
                f"{key}_ori": quat2axisangle(self._obj_state_dict[key][:, 3:7]),
                f"{key}_quat": self._obj_state_dict[key][:, 3:7],
            })

        if self.enable_camera_sensors:
            # Camera Images
            self.states.update({
                "fview_image": torch.cat(self.fview_cam_tensors)[:, :3].unsqueeze(1),
                "sview_image": torch.cat(self.sview_cam_tensors)[:, :3].unsqueeze(1),
            })

    def _refresh(self, update_states=False):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        if update_states:
            # render camera media
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            # Refresh states
            self._update_states()
            self.gym.end_access_image_tensors(self.sim)

    def compute_reward(self):
        pass

    def compute_observations(self):
        self._refresh(update_states=True)
        if self.push_t:
            obs = ["eef_pos", "eef_ori"]
            for key in self._obj_state_dict:
                obs = obs + [f"{key}_pos", f"{key}_ori"]
        else:
            obs = ["eef_pos"]
            for key in self._obj_state_dict:
                obs = obs + [f"{key}_pos"]
        images = ["fview_image", "sview_image"]
        self.obs_buf = torch.cat([self.states[ob].unsqueeze(1) for ob in obs], dim=1)
        if self.enable_camera_sensors:
            self.image_buf = torch.cat([self.states[img] for img in images], dim=1)

    def reset(self):
        """
        Resets all the environments.
        Returns:
            Observation dictionary
        """
        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        if self.goal_reset and self.random_obj_colors:
            # Set random cube colors
            rand_indices = torch.randperm(self.num_colors)[:self.num_objects]
            for i in range(self.num_envs):
                for j, key in zip(rand_indices, self._obj_id_dict):
                    self.gym.set_rigid_body_color(self.envs[i], self._obj_id_dict[key], 0, gymapi.MESH_VISUAL, self.object_colors[j])

        # Step simulation
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        # Step graphics
        self.gym.step_graphics(self.sim)
        # Refresh tensors and compute observations
        self.compute_observations()
        # reset controller
        self.controller.reset(self.get_control_dict())
        # populate obs_dict
        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        self.obs_dict["media"] = self.image_buf.to(self.rl_device)
        if self.num_states > 0:
            self.obs_dict["states"] = torch.clamp(self.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        return self.obs_dict

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # Reset cubes
        self._reset_init_cube_states(env_ids=env_ids, check_valid=True)

        # Write these new init states to the sim states
        for key in self._obj_state_dict:
            self._obj_state_dict[key][env_ids] = self._init_obj_state_dict[key][env_ids]

        # Reset agent
        reset_noise = torch.rand((len(env_ids), 9), device=self.device)
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) + self.franka_dof_noise * 2.0 * (reset_noise - 0.5),
            self.franka_dof_lower_limits.unsqueeze(0), self.franka_dof_upper_limits)

        # Overwrite gripper init pos (no noise since these are always position controlled)
        pos[:, -2:] = self.franka_default_dof_pos[-2:]

        # Reset the internal obs accordingly
        self._q[env_ids, :] = pos
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        self._pos_control[env_ids, :] = pos
        self._effort_control[env_ids, :] = torch.zeros_like(pos)

        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._pos_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._effort_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        # Update cube states
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, -self.num_objects:].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def _reset_init_cube_states(self, env_ids, check_valid=True):
        if self.goal_reset and self.adjacent_goals:
            self._reset_init_cube_state(cube="cube1", env_ids=env_ids, check_valid=check_valid)
            offset = (self.cube_size + 0.005) * torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1]], device=self.device)
            for i, key in enumerate(self._init_obj_state_dict):
                self._init_obj_state_dict[key][env_ids] = self._init_obj_state_dict["cube1"][env_ids].clone()
                self._init_obj_state_dict[key][env_ids, :2] += offset[i]

        elif self.goal_reset and self.ordered_push:
            num_resets = len(env_ids)
            sampled_cube_state = torch.zeros(num_resets, 13, device=self.device)
            sampled_cube_state[:, 2] = self._table_surface_pos[2] + self.cube_size / 2  # fixed height
            sampled_cube_state[:, 6] = 1.0  # no rotation (quat w = 1)
            # x_positions = 0.5 * (self.table_dims[0] - 0.02 - self.cube_size * torch.range(1, 2 * (self.num_objects + 1), 2, device=self.device))
            x_positions = 0.5 * (self.table_dims[0] - 0.02 - self.cube_size * torch.tensor([1, 6], device=self.device))
            rand_indices = torch.randint(self.num_objects, (num_resets,))
            for i, key in enumerate(self._init_obj_state_dict):
                sampled_cube_state[:, 0] = x_positions[(rand_indices + i) % self.num_objects]
                self._init_obj_state_dict[key][env_ids] = sampled_cube_state.clone()

        elif self.goal_reset and self.sort_push:
            num_resets = len(env_ids)
            sampled_cube_state = torch.zeros(num_resets, 13, device=self.device)
            sampled_cube_state[:, 2] = self._table_surface_pos[2] + self.cube_size / 2  # fixed height
            sampled_cube_state[:, 6] = 1.0  # no rotation (quat w = 1)
            # hard coded goal positions
            pos_x = 0.5 * self.table_dims[0] - 4.5 * self.cube_size
            pos_y = 0.5 * self.table_dims[1] - 4.5 * self.cube_size
            positions = torch.tensor([[pos_x, pos_y], [-pos_x, -pos_y], [pos_x, -pos_y], [-pos_x, pos_y]], device=self.device)
            rand_indices = torch.randint(self.num_colors, (num_resets,))
            for i, key in enumerate(self._init_obj_state_dict):
                if i < self.num_colors:
                    sampled_cube_state[:, :2] = positions[(rand_indices + i) % self.num_colors]
                    # sampled_cube_state[:, :2] = positions[i]
                    self._init_obj_state_dict[key][env_ids] = sampled_cube_state.clone()
                else:
                    self._init_obj_state_dict[key] = torch.zeros(len(env_ids), 13, device=self.device)

        elif self.goal_reset and self.push_t:
            for i, key in enumerate(self._init_obj_state_dict):
                if i < 1:
                    self._reset_init_cube_state(cube=key, env_ids=env_ids, check_valid=check_valid)
                else:
                    self._init_obj_state_dict[key] = torch.zeros(len(env_ids), 13, device=self.device)

        elif self.random_obj_num:
            if self.goal_reset:
                if self.random_obj_num_dist is not None:
                    self.cur_num_objects = np.random.choice(self.num_objects, p=self.random_obj_num_dist) + 1
                else:
                    self.cur_num_objects = np.random.randint(self.num_objects) + 1
            for i, key in enumerate(self._init_obj_state_dict):
                if i < self.cur_num_objects:
                    self._reset_init_cube_state(cube=key, env_ids=env_ids, check_valid=check_valid)
                else:
                    self._init_obj_state_dict[key] = torch.zeros(len(env_ids), 13, device=self.device)

        else:
            for key in self._init_obj_state_dict:
                self._reset_init_cube_state(cube=key, env_ids=env_ids, check_valid=check_valid)

    def _reset_init_cube_state(self, cube, env_ids, check_valid=True):
        """
        Simple method to sample @cube's position based on self.startPositionNoise and self.startRotationNoise, and
        automatically reset the pose internally. Populates the appropriate self._init_obj_state_dict entry

        If @check_valid is True, then this will also make sure that the sampled position is not in contact with the
        other cubes.

        Args:
            cube(str): Which cube to sample location for.
            env_ids (tensor or None): Specific environments to reset cube for
            check_valid (bool): Whether to make sure sampled position is collision-free with the other cubes.
        """
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_cube_state = torch.zeros(num_resets, 13, device=self.device)

        # Get correct references depending on which one was selected
        cube_heights = self.states[f"{cube}_size"]
        this_cube_state_all = self._init_obj_state_dict[cube]
        other_objs_state_list = []
        other_objs_state_list.append(self.franka_default_eef_pos[env_ids, :].clone().unsqueeze(-2))
        for key in self._init_obj_state_dict:
            if key == cube:
                break
            cube_state = self._init_obj_state_dict[key][env_ids, :].clone()
            other_objs_state_list.append(cube_state.unsqueeze(-2))

        # Minimum cube distance for guarenteed collision-free sampling is the sum of each cube's effective radius,
        # assuming all cubes are the same size
        if self.push_t:
            min_dists = 6 * self.states[f"{cube}_size"][env_ids].unsqueeze(-1) * np.sqrt(2) / 2.0
        else:
            min_dists = 2 * self.states[f"{cube}_size"][env_ids].unsqueeze(-1) * np.sqrt(2) / 2.0

        # Sampling references and boundaries
        max_cube_x_state = torch.tensor((self.table_dims[0] / 2) - 0.1, device=self.device, dtype=torch.float32)
        max_cube_y_state = torch.tensor((self.table_dims[1] / 2) - 0.1, device=self.device, dtype=torch.float32)
        if self.ordered_push:
            max_cube_x_state = torch.tensor((self.table_dims[0] / 2) - (0.12 + 0.5 * self.cube_size), device=self.device, dtype=torch.float32)
            max_cube_y_state = torch.tensor((self.table_dims[1] / 2) - 0.15, device=self.device, dtype=torch.float32)
        min_cube_x_state = -max_cube_x_state
        min_cube_y_state = -max_cube_y_state

        # Set z value, which is fixed height
        sampled_cube_state[:, 2] = self._table_surface_pos[2] + cube_heights[env_ids] / 2

        # Initialize rotation, which is no rotation (quat w = 1)
        sampled_cube_state[:, 6] = 1.0

        # If we're verifying valid sampling, we need to check and re-sample if any are not collision-free
        # We use a simple heuristic of checking based on cubes' radius to determine if a collision would occur
        if check_valid and len(other_objs_state_list) > 0:
            other_cube_states = torch.cat(other_objs_state_list, dim=-2)
            success = False
            # Indexes corresponding to envs we're still actively sampling for
            active_idx = torch.arange(num_resets, device=self.device)
            num_active_idx = len(active_idx)
            for i in range(10000):
                # Sample x values
                sampled_cube_state[active_idx, 0] = (min_cube_x_state - max_cube_x_state) * torch.rand_like(sampled_cube_state[active_idx, 0]) + max_cube_x_state
                # Sample y values
                sampled_cube_state[active_idx, 1] = (min_cube_y_state - max_cube_y_state) * torch.rand_like(sampled_cube_state[active_idx, 1]) + max_cube_y_state
                # Check if sampled values are valid
                sampled_cube_state_copy = torch.unsqueeze(sampled_cube_state, -2).expand(-1, other_cube_states.shape[-2], -1)
                cube_dist = torch.linalg.norm(sampled_cube_state_copy[..., :2] - other_cube_states[..., :2], dim=-1)
                active_idx = torch.nonzero(torch.any(cube_dist < min_dists, dim=-1), as_tuple=True)[0]
                num_active_idx = len(active_idx)
                # If active idx is empty, then all sampling is valid :D
                if num_active_idx == 0:
                    success = True
                    break
            # Make sure we succeeded at sampling
            assert success, "Sampling cube locations was unsuccessful! ):"
        else:
            # We just directly sample
            # Sample x values
            sampled_cube_state[:, 0] = (min_cube_x_state - max_cube_x_state) * torch.rand_like(sampled_cube_state[:, 0]) + max_cube_x_state
            # Sample y values
            sampled_cube_state[:, 1] = (min_cube_y_state - max_cube_y_state) * torch.rand_like(sampled_cube_state[:, 1]) + max_cube_y_state

        # Sample rotation value
        if self.start_rotation_noise > 0:
            aa_rot = torch.zeros(num_resets, 3, device=self.device)
            aa_rot[:, 2] = 2.0 * self.start_rotation_noise * (torch.rand(num_resets, device=self.device) - 0.5)
            sampled_cube_state[:, 3:7] = quat_mul(axisangle2quat(aa_rot), sampled_cube_state[:, 3:7])
        if self.push_t:
            rand_rot = torch.zeros([num_resets, 3], device=self.device)
            rand_rot[:, -1] = np.pi * (-1. + torch.rand(num_resets) * 2.0)
            sampled_cube_state[:, 3:7] = axisangle2quat(rand_rot)

        # Lastly, set these sampled values as the new init state
        this_cube_state_all[env_ids, :] = sampled_cube_state

    def step(self, actions):
        """
        Perform a policy step, taking multiple simulation steps based on specified control frequency

        Args:
            actions: actions delta to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """

        # randomize actions
        if self.dr_randomizations.get('actions', None):
            actions = self.dr_randomizations['actions']['noise_lambda'](actions)

        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        # set OSC goal and apply gripper command
        self.pre_physics_step(action_tensor)

        # step physics
        for i in range(int(self.control_timestep / self.sim_timestep)):
            self._refresh()
            # get torques
            u_arm = self.controller.compute_control(self.get_control_dict())
            self._arm_control[:, :] = u_arm
            # deploy arm actions
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))
            # render if using viewer
            if self.force_render:
                self.render()
            # step simulation physics
            self.gym.simulate(self.sim)

        if self.viewer == None:
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 1. Timeout == 1 makes sense only if the reset buffer is 1.
        self.timeout_buf = (self.progress_buf > self.max_episode_length - 1) & (self.reset_buf != 0)

        # randomize observations
        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        self.obs_dict["media"] = self.image_buf.to(self.rl_device)
        if self.num_states > 0:
            self.obs_dict["states"] = torch.clamp(self.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        # Split arm and gripper command
        u_arm, u_gripper = self.actions[:, :-1], self.actions[:, -1]

        # Set goal for OSC arm control
        self.controller.update_goal(self.get_control_dict(), u_arm)

        # Control gripper
        u_fingers = torch.zeros_like(self._gripper_control)
        u_fingers[:, 0] = torch.where(u_gripper >= 0.0, self.franka_dof_upper_limits[-2].item(),
                                      self.franka_dof_lower_limits[-2].item())
        u_fingers[:, 1] = torch.where(u_gripper >= 0.0, self.franka_dof_upper_limits[-1].item(),
                                      self.franka_dof_lower_limits[-1].item())
        # Write gripper command to appropriate tensor buffer
        self._gripper_control[:, :] = u_fingers

        # Deploy gripper action
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))

    def post_physics_step(self):
        self.progress_buf += 1
        self.reset_buf[:] = torch.where((self.progress_buf > self.max_episode_length - 1),
                                        torch.ones_like(self.reset_buf[:]), self.reset_buf[:])

        # uncomment if episodes can terminate early so only some envs need to be reset
        # env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # if len(env_ids) > 0:
        #     self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # Grab relevant states to visualize
            eef_pos = self.states["eef_pos"]
            eef_rot = self.states["eef_quat"]
            pos_list = [eef_pos] + [self.states[f"cube{i+1}_pos"] for i in range(self.num_objects)]
            rot_list = [eef_rot] + [self.states[f"cube{i+1}_quat"] for i in range(self.num_objects)]

            # Plot visualizations
            for i in range(self.num_envs):
                for pos, rot in zip(pos_list, rot_list):
                    px = (pos[i] + quat_apply(rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                    py = (pos[i] + quat_apply(rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                    pz = (pos[i] + quat_apply(rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                    p0 = pos[i].cpu().numpy()
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

    def get_control_dict(self):
        control_dict = {
            "eef_state": self._eef_state,
            "q": self._q[:, :7],
            "qd": self._qd[:, :7],
            "mm": self._mm,
            "j_eef": self._j_eef
        }
        return control_dict


if __name__ == '__main__':

    """
    This script can be used to debug the 'IsaacPandaPush' environment.
    To visualize the environment simulation, set:
    - visualize_isaacgym_envs=True for IsaacGym GUI visualization of all parallel envs
    - plot_images=True for single image visualization per timestep of a single env
    - create_episode_gif=True for episode video visualization of a single env
    """

    visualize_isaacgym_envs = True
    plot_images = False
    create_episode_gif = False

    # config
    cfg = yaml.safe_load(Path('config/n_cubes/IsaacPandaPushConfig.yaml').read_text())
    cuda_device = 0

    envs = IsaacPandaPush(
        cfg=cfg,
	    rl_device=f"cuda:{cuda_device}",
		sim_device=f"cuda:{cuda_device}",
		graphics_device_id=cuda_device,
		headless=not visualize_isaacgym_envs,
		virtual_screen_capture=False,
		force_render=True,
	)

    print("Finished setting up simulation env")

    # perform rollouts with random policy
    for _ in range(100):
        start_time = time.time()
        obs = envs.reset()
        if create_episode_gif:
            img_list_fview = [obs["media"][0, 0].permute((1, 2, 0)).cpu().numpy()]
            img_list_sview = [obs["media"][0, 1].permute((1, 2, 0)).cpu().numpy()]
        if plot_images:
            plt.imshow(obs["media"][0, 0].permute((1, 2, 0)).cpu().numpy())
            plt.axis('off')
            plt.show()
            plt.imshow(obs["media"][0, 1].permute((1, 2, 0)).cpu().numpy())
            plt.axis('off')
            plt.show()
        for i in range(envs.max_episode_length):
            # choose random action
            action_xyz = -2 * torch.rand((envs.num_envs, 3), device="cuda:0") + 1  # uniform [-1, 1]
            # modify action to fit env action space and allow vertical movements with closed gripper only
            action_rest = torch.tensor([0, 0, 0, -1], device="cuda:0").unsqueeze(0).expand(envs.num_envs, -1)
            action = torch.cat([action_xyz, action_rest], dim=-1)
            # perform action
            obs, reward, done, info = envs.step(action)
            # visualize
            if create_episode_gif:
                img_list_fview.append(obs["media"][0, 0].permute((1, 2, 0)).cpu().numpy())
                img_list_sview.append(obs["media"][0, 1].permute((1, 2, 0)).cpu().numpy())
                # img_list_oview.append(obs["env_images"][0, 0].permute((1, 2, 0)).cpu().numpy())
            if plot_images:
                plt.imshow(obs["media"][0, 0].permute((1, 2, 0)).cpu().numpy())
                plt.axis('off')
                plt.show()
                plt.imshow(obs["media"][0, 1].permute((1, 2, 0)).cpu().numpy())
                plt.axis('off')
                plt.show()

        print(f"Episode completed in {time.time() - start_time:5.2f}s")

        if create_episode_gif:
            clip = ImageSequenceClip(img_list_fview, fps=10)
            clip.write_gif(f'./results/episode_video_fview.gif', fps=10)
            clip = ImageSequenceClip(img_list_sview, fps=10)
            clip.write_gif(f'./results/episode_video_sview.gif', fps=10)

        print("\n")
