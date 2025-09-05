# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
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
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from torch.nn.modules.loss import F

class Yun1WRoughCfg( LeggedRobotCfg ):
    
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_one_step_observations = 57
        num_observations = num_one_step_observations * 6
        num_one_step_privileged_obs = 57 + 3 + 3 + 187 # additional: base_lin_vel, external_forces, scan_dots
        num_privileged_obs = num_one_step_privileged_obs * 1 # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 16
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'  # "heightfield" # none, plane, heightfield or trimesh
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.35] # x,y,z [m] - Reduced height for stability
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'fl_hip_Joint': 0.0,   # [rad]
            'rl_hip_Joint': 0.0,   # [rad]
            'fr_hip_Joint': 0.0,  # [rad]
            'rr_hip_Joint': 0.0,   # [rad]

            'fl_thigh_Joint': 0.6,     # [rad] - 确保腿部正确伸展
            'rl_thigh_Joint': -0.6,     # [rad]
            'fr_thigh_Joint': 0.6,     # [rad]
            'rr_thigh_Joint': -0.6,     # [rad]

            'fl_calf_Joint': -1.2,   # [rad] - 确保foot接触地面
            'rl_calf_Joint': 1.2,    # [rad]
            'fr_calf_Joint': -1.2,  # [rad]
            'rr_calf_Joint': 1.2,    # [rad]

            'fl_wheel_Joint': 0.0,   # [rad]
            'rl_wheel_Joint': 0.0,   # [rad]
            'fr_wheel_Joint': 0.0,  # [rad]
            'rr_wheel_Joint': 0.0,    # [rad]
        }

        start_joint_angles = { # = target angles [rad] when stand still
            'fl_hip_Joint': 0.0,   # [rad]
            'rl_hip_Joint': 0.0,   # [rad]
            'fr_hip_Joint': 0.0,  # [rad]
            'rr_hip_Joint': 0.0,   # [rad]

            'fl_thigh_Joint': 0.6,     # [rad] - 确保腿部正确伸展
            'rl_thigh_Joint': -0.6,     # [rad]
            'fr_thigh_Joint': 0.6,     # [rad]
            'rr_thigh_Joint': -0.6,     # [rad]

            'fl_calf_Joint': -1.2,   # [rad] - 确保foot接触地面
            'rl_calf_Joint': 1.2,    # [rad]
            'fr_calf_Joint': -1.2,  # [rad]
            'rr_calf_Joint': 1.2,    # [rad]

            'fl_wheel_Joint': 0.0,   # [rad]
            'rl_wheel_Joint': 0.0,   # [rad]
            'fr_wheel_Joint': 0.0,  # [rad]
            'rr_wheel_Joint': 0.0,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip_Joint': 15.,'thigh_Joint': 15.,'calf_Joint': 15.,"wheel_Joint":0}  # [N*m/rad]
        damping = {'hip_Joint': 0.3,'thigh_Joint': 0.3,'calf_Joint': 0.3,"wheel_Joint":0.12}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25

        vel_scale = 10 # 轮子的速度缩放超参数
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        # wheel_speed = 1
        

    class commands( LeggedRobotCfg.commands ):
            curriculum = True
            max_curriculum = 1.5
            num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
            resampling_time = 10. # time before command are changed[s]
            heading_command = False # if true: compute ang vel command from heading error
            class ranges( LeggedRobotCfg.commands.ranges):
                lin_vel_x = [-0.0, 0.0]  # min max [m/s] x轴方向线速度
                lin_vel_y = [-0.0, 0.0]  # min max [m/s] y轴方向线速度
                ang_vel_yaw = [-1, 1]  # min max [rad/s] 角速度
                heading = [-3.14, 3.14]  # 航向 实际上没有使用这个维度

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/Yun1W2/urdf/yun1w.urdf'
        name = "yun1w"
        foot_name = "wheel"
        wheel_name =["wheel"]
        penalize_contacts_on = ["thigh_Link", "calf_Link", "base_link"]  # Fixed link names
        terminate_after_contacts_on = ["base_link"]
        privileged_contacts_on = ["base_link", "thigh_Link", "calf_Link"]  # 特权学习：训练时提供接触信息
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
  
    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            # general
            termination = -0.8
            # velocity-tracking
            tracking_lin_vel = 3.0
            tracking_ang_vel = 1.5
            # root
            lin_vel_z = -2
            ang_vel_xy = -0.05
            orientation = -2
            base_height = -10
            # joint
            torques = -0.0002
            dof_vel = -1e-7
            dof_acc = -1e-7
            joint_power = -2e-5
            dof_pos_limits = -0.9
            stand_still = -0.5
            hip_pos = -0.2  # 髋关节hip（0,3,6,9）位置与默认位置的偏差 惩罚
            thigh_pose = -0.1
            calf_pose = -0.1
            # action
            action_rate = -0.01
            hip_action_l2 = -0.1
            smoothness = -0.01
            # contact
            collision = -0.1
            stumble = -0.1
            no_wheel_spin_in_air = -0.001
            #yaw_feet_air_time = 1.0
            no_feet_air_time = -1.2
            has_contact = 0.5

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 1.
        base_height_target = 0.27
        max_contact_force = 100.

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0

class Yun1WRoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        num_steps_per_env = 24 # per iteration
        max_iterations = 50000 # number of policy updates
        runner_class_name = 'HIMOnPolicyRunner'
        run_name = ''
        experiment_name = 'rough_yun1w'
