import time

from isaacgym import gymtorch
import torch
from pycore.base import Core

from pycore.logger import Logger as CoreLogger
from pycore.utils.file_utils import FileUtils

from humanoid.envs.pai.pai_config import PaiCfg
from humanoid.utils.convert_utils import kp_kd_array
from humanoid.utils.time_utils import get_timestamp
from humanoid.sim.simulation import Simulation
from humanoid.utils import get_args, export_policy_as_jit, task_registry, Logger, convert_utils
import numpy as np
import pandas as pd
import statistics

cfg = PaiCfg()
cfg.default_dof_drive_mode = 3
num_dofs = 12
gym_step = 0


# ************************** computer target_q  **************************
def computer_pd_tar(actions):
    actions = torch.tensor(actions)
    # 1 clip
    clip_actions = cfg.normalization.clip_actions
    actions = torch.clip(actions, -clip_actions, clip_actions)
    # 2 computer
    for i in range(num_dofs):
        actions[i] = actions[i] * cfg.control.action_scale
    actions = actions.unsqueeze(0)
    return actions


def get_target_q_queue(device):
    action1 = [
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0
    ]
    action2 = [
        0, 0, -1,
        -3, -1, 0,
        0, 0, 1,
        3, 1, 0
    ]

    target_q_1 = computer_pd_tar(actions=action1).to(device)
    target_q_2 = computer_pd_tar(actions=action2).to(device)
    _queue = [target_q_1, target_q_2]
    return _queue


dt = cfg.sim.dt * cfg.control.decimation


def _get_phase():
    cycle_time = cfg.rewards.cycle_time
    phase = gym_step * dt / cycle_time
    return phase


def compute_ref_state(dof_pos):
    phase = _get_phase()
    phase = torch.tensor(phase)
    sin_pos = torch.sin(2 * torch.pi * phase)
    sin_pos_l = sin_pos.clone()
    sin_pos_r = sin_pos.clone()
    ref_dof_pos = torch.zeros_like(dof_pos)
    scale_1 = cfg.rewards.target_joint_pos_scale * 2
    scale_2 = 2 * scale_1
    # left foot stance phase set to default joint pos
    sin_pos_l[sin_pos_l > 0] = 0
    ref_dof_pos[:, 2] = sin_pos_l * scale_1  # left_hip_pitch_joint
    ref_dof_pos[:, 3] = sin_pos_l * scale_2  # left_knee_joint
    ref_dof_pos[:, 4] = sin_pos_l * scale_1  # left_ankle_pitch_joint
    # right foot stance phase set to default joint pos
    sin_pos_r[sin_pos_r < 0] = 0
    ref_dof_pos[:, 8] = sin_pos_r * scale_1  # right_hip_pitch_joint
    ref_dof_pos[:, 9] = sin_pos_r * scale_2  # right_knee_joint
    ref_dof_pos[:, 10] = sin_pos_r * scale_1  # right_knee_joint
    # Double support phase
    ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0

    return ref_dof_pos


def compute_torques(p_gains,d_gains, actions, dof_pos, dof_vel, torque_limits):
    # pd controller

    # dof_pos[0][3] = dof_pos[0][3] * 1.2
    # dof_pos[0][3+6] = dof_pos[0][3+6] * 1.2
    # print(f"curr dof_pos: {dof_pos}")
    torques = p_gains * (actions - dof_pos) - d_gains * dof_vel
    torques = torch.clip(torques, -torque_limits, torque_limits)
    # print(f"torques: {torques}")
    return torques


def play_tau():
    global gym_step
    # **************************  实例化模拟器 **************************
    simulation = Simulation(cfg=cfg)
    gym = simulation.gym
    sim = simulation.sim
    robot_asset = simulation.load_pai_asset(file_name="pai_12dof")
    device = simulation.device

    p_gains, d_gains = kp_kd_array(cfg)
    p_gains = np.array(p_gains).astype("float32")
    d_gains = np.array(d_gains).astype("float32")
    p_gains = torch.from_numpy(p_gains).to(device)
    d_gains = torch.from_numpy(d_gains).to(device)

    print(f"p_gains: {p_gains}")
    print(f"d_gains: {d_gains}")
    logger = Logger(dt)
    # **************************  actor asset特征  **************************
    simulation.set_asset_props(robot_asset=robot_asset)

    # ************************** env actor  **************************
    # 实例化env
    env_handle = simulation.creat_env(num=1)
    # 实例化actor
    actor_handle = simulation.create_actor(env_handle=env_handle)
    dof_props_asset = gym.get_asset_dof_properties(robot_asset)
    torque_limits = torch.zeros(num_dofs, dtype=torch.float, device=device, requires_grad=False)
    for i in range(num_dofs):
        dof_props_asset["stiffness"][i] = cfg.asset.stiffness  # stiffness_var[i]
        dof_props_asset["damping"][i] = cfg.asset.damping  # damping_var[i]
        torque_limits[i] = dof_props_asset["effort"][i].item() * cfg.safety.torque_limit
    #  设置actor的关节特性
    gym.set_actor_dof_properties(env_handle, actor_handle, dof_props_asset)
    print(f"torque_limits： {torque_limits}")

    dof_state_tensor = gym.acquire_dof_state_tensor(sim)  # 实时监听仿真器的state
    net_contact_forces = gym.acquire_net_contact_force_tensor(sim) #仿真器监听力量
    contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(1, -1, 3)  # shape: num_envs, num_bodies, xyz axis
    dof_state = gymtorch.wrap_tensor(dof_state_tensor)
    torques = torch.zeros(1, 12, dtype=torch.float, device=device, requires_grad=False)
    dof_pos = dof_state.view(1, 12, 2)[..., 0]
    dof_vel = dof_state.view(1, 12, 2)[..., 1]
    # must call prepare_sim to initialize the internal data structures used by the tensor API:
    gym.prepare_sim(sim)

    # ************************** render  **************************
    viewer = simulation.create_views(env_handle)
    action_queue = get_target_q_queue(device)
    target_q = action_queue[0]

    last_switch_time = get_timestamp()
    switch_freq = cfg.control.decimation  # ms
    save_queue = []
    # while not gym.query_viewer_has_closed(viewer):

    action_cost_array = []
    computer_cost_array = []
    sim_cost_array = []
    feet_indices = torch.tensor([6, 12]).to(device)
    while gym_step < 640:
        # # 监听键盘事件
        # for evt in gym.query_viewer_action_events(viewer):
        #     pass
        simulation.render()

        timestamp_ns_s = time.time_ns()

        target_q = compute_ref_state(dof_pos)
        target_q_save = target_q.detach().cpu().numpy()[0]
        save_queue.append(target_q_save * 4)
        # CoreLogger.instance().info(f"gym_step: {gym_step} 切换发送目标:  {target_q}")
        gym_step += 1

        for i in range(int(cfg.control.decimation / 2)):
            # ******************** 计算扭矩 ***********************
            t_ns_comp_s = time.time_ns()
            torques = compute_torques(p_gains, d_gains, target_q, dof_pos, dof_vel, torque_limits).view(torques.shape)
            torques_tensor = gymtorch.unwrap_tensor(torques)
            gym.set_dof_actuation_force_tensor(sim, torques_tensor)
            t_ns_cimp_e = time.time_ns()
            computer_cost_array.append((t_ns_cimp_e - t_ns_comp_s)/ 1_000_000)

            # ******************** 执行仿真 ***********************
            t_ns_sim_s = time.time_ns()
            gym.simulate(sim)  # 执行一次仿真子步骤 dt 1ms
            t_ns_sim_e = time.time_ns()
            sim_cost_array.append((t_ns_sim_e - t_ns_sim_s) / 1_000_000)

            # 刷新关节状态张量，更新仿真状态。
            gym.refresh_dof_state_tensor(sim)
            gym.fetch_results(sim, True)
            # if i % 2 == 0:
            #     simulation.render()

        timestamp_ns_e = time.time_ns()
        action_cost_array.append((timestamp_ns_e- timestamp_ns_s)/1_000_000)

        gym.refresh_net_contact_force_tensor(sim)
        logger.log_states(
            {
                'hip_pitch_target': target_q[0,2].item(),
                'hip_pitch': dof_pos[0, 2].item(),
                'knee_target': target_q[0, 3].item(),
                'knee': dof_pos[0, 3].item(),
                'dof_vel': dof_vel[0, 3].item(),
                'dof_torque': torques[0, 3].item(),
                # 'command_x': env.commands[robot_index, 0].item(),
                # 'command_y': env.commands[robot_index, 1].item(),
                # 'command_yaw': env.commands[robot_index, 2].item(),
                # 'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                # 'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                # 'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                # 'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                'contact_forces_z': contact_forces[0, feet_indices, 2].cpu().numpy()
            }
        )

    print(f"******** 测试统计数据 **************")
    print(f"sim dt: {cfg.sim.dt}")
    print(f"env dt: {dt}")
    print(f"decimation: {cfg.control.decimation/2}")
    print(f"target_joint_pos_scale: {cfg.rewards.target_joint_pos_scale}")
    print(f"action平均时间：  {statistics.mean(action_cost_array[1:])}")
    print(f"扭矩平均计算时间： {statistics.mean(computer_cost_array[1:])}")
    print(f"仿真平均消耗时间： {statistics.mean(sim_cost_array[1:])}")
    print(f"kp： {p_gains}")
    print(f"kd： {p_gains}")
    print(f"**********************************")




    save_queue = np.array(save_queue)
    df = pd.DataFrame(save_queue)
    # 将DataFrame写入CSV文件
    csv_file = f'{FileUtils.get_project_path()}/test/csv/action_tau_phase.csv'
    df.to_csv(csv_file, index=False)  # index=False 表示不保存索引列
    print(f"save {csv_file} success!")
    simulation.destroy()

    logger.print_rewards()
    logger.plot_states()


if __name__ == '__main__':
    play_tau()



