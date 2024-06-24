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
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Pai2Cfg(LeggedRobotCfg):
    """
    Configuration class for the XBotL humanoid robot.
    """

    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 15
        c_frame_stack = 3
        num_single_obs = 47
        num_observations = int(frame_stack * (num_single_obs))
        single_num_privileged_obs = 73
        num_privileged_obs = int(c_frame_stack * (single_num_privileged_obs))
        num_actions = 12
        num_envs = 4096
        episode_length_s = 24  # episode length in seconds
        use_ref_actions = False

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/pai_12dof/urdf/pai_12dof.urdf"

        name = "Pai2"
        foot_name = "ankle_roll"
        knee_name = "calf"
        thigh_name = "thigh"

        terminate_after_contacts_on = ["base_link"]
        penalize_contacts_on = ["base_link"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False
        stiffness = 0.8
        damping = 0.5

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        # mesh_type = 'trimesh'
        curriculum = False
        # rough terrain only:
        measure_heights = False
        static_friction = 0.8
        dynamic_friction = 0.6
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.0

    class noise:
        add_noise = True
        noise_level = 0.6  # scales other values

        class noise_scales:
            dof_pos = 0.2 # 0.05
            dof_vel = 0.5
            ang_vel = 0.1
            lin_vel = 0.05
            quat = 0.03
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):

        model_path = 'robots/pai_12dof/urdf/pai_12dof.urdf'
        pos_action = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        pos = [0.0, 0.0, 0.410]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "left_hip_yaw_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_pitch_joint": 0.0,
            "left_knee_joint": 0.0,
            "left_ankle_pitch_joint": 0.0,
            "left_ankle_roll_joint": 0.0,

            "right_hip_yaw_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_pitch_joint": 0.0,
            "right_knee_joint": 0.0,
            "right_ankle_pitch_joint": 0.0,
            "right_ankle_roll_joint": 0.0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters: 高警建议 kp 15~30  kd 0.01~2
        # kps = np.array([40, 40, 80, 80, 15, 15, 40, 40, 80, 80, 15, 15], dtype=np.double)
        # kds = np.array([0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125], dtype=np.double)
        stiffness = {
            "hip_yaw_joint": 26.042,#40.0,
            "hip_roll_joint": 26.042,#80.0,
            "hip_pitch_joint": 45.573,#15.0,
            "knee_joint": 45.573,#40.0,
            "ankle_pitch_joint": 15.0,#80.0,
            "ankle_roll_joint": 15.0,#15.0,
        }
        damping = {
            "hip_yaw_joint":0.195,# 0.125, 
            "hip_roll_joint":0.195,# 0.125, 
            "hip_pitch_joint":0.195,#0.125, 
            "knee_joint":0.195,# 0.125, 
            "ankle_pitch_joint":0.195,# 0.125, 
            "ankle_roll_joint":0.195,# 0.125,
        }

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 100hz
        # decimation = 20  # 100hz
        decimation_rate = 1

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 500 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 20
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand:
        randomize_friction = True
        friction_range = [0.1, 2.0]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.2
        max_push_ang_vel = 0.4
        dynamic_randomization = 0.02

    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.0  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-0.3, 0.6]  # min max [m/s]
            lin_vel_y = [-0.3, 0.3]  # min max [m/s]
            ang_vel_yaw = [-0.3, 0.3]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards:
        base_height_target = 0.35
        min_dist = 0.15
        max_dist = 0.2
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.17 # 0.17  # rad
        target_feet_height = 0.06  # m
        cycle_time = 0.64  # sec
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        # tracking reward = exp(error*sigma)
        tracking_sigma = 5
        max_contact_force = 80  # forces above this value are penalized

        class scales:
            # reference motion tracking
            joint_pos = 1.5  # github:1.6, paper(Joint Position Tracking):1.5
            feet_clearance = 1.0 # github:1.0, paper:-
            feet_contact_number = 1.0 # github:1.2, paper(Contact Pattern):1.0
            # gait
            feet_air_time = 1.0 # github:1.0
            foot_slip = -0.05 # github:-0.05
            feet_distance = 0.16  # github:0.2
            knee_distance = 0.16  # github:0.2
            # contact
            feet_contact_forces = -0.01 # github:-0.01, paper(Large contact):-0.01
            # vel tracking
            tracking_lin_vel = 1.2 # github:1.2, paper(Lin. velocity tracking):1.2
            tracking_ang_vel = 1.0 # github:1.1, paper(Ang. velocity tracking):1.0
            vel_mismatch_exp = 0.5  # lin_z; ang x,y github:0.5, paper(Velocity mismatch)0.5
            low_speed = 0.5 # github:0.2
            track_vel_hard = 0.5 # github:0.5
            # base pos
            default_joint_pos = 0.2 # github:0.5, paper(Default Joint):0.2
            orientation = 1.0 # github:1.0, paper(Orientation tracking):1.0
            base_height = 0.5 # github:0.2, paper(Base height tracking):0.5
            base_acc = 0.2 # github:0.2
            # energy
            action_smoothness = -0.01 # github:-0.002, paper(Action Smoothness):-0.01
            torques = -1e-4 # github:-1e-5, paper(Energy Cost):-0.0001
            dof_vel = -5e-3 # github:-5e-4
            dof_acc = -1e-6 # github:-1e-7
            collision = -1.0 # github::-1.0

            target_pos_progress = 1.0

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            quat = 1.0
            height_measurements = 5.0

        clip_observations = 18.0
        clip_actions = 3.0


class Pai2CfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = "OnPolicyRunner"  # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9 # github:0.9, paper(GAE discount factor):0.95
        num_mini_batches = 4

    class runner:
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 60  # per iteration github:60, paper:24
        # max_iterations = 10001  # number of policy updates
        max_iterations = 10000  # number of policy updates  zhe_v2

        # logging
        save_interval = 100  # check for potential saves every this many iterations
        experiment_name = "Pai2_ppo"
        run_name = ""
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
