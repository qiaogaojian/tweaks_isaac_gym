from isaacgym import gymtorch
import torch

from pycore.logger import Logger

from humanoid.envs.pai.pai_config import PaiCfg
from humanoid.utils.time_utils import get_timestamp
from humanoid.sim.simulation import Simulation
import numpy as np
import pandas as pd

cfg = PaiCfg()
cfg.default_dof_drive_mode = 1
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
    scale_1 = cfg.rewards.target_joint_pos_scale
    scale_2 = 2 * scale_1
    # left foot stance phase set to default joint pos
    sin_pos_l[sin_pos_l > 0] = 0
    ref_dof_pos[:, 2] = sin_pos_l * scale_1  # left_hip_pitch_joint
    ref_dof_pos[:, 3] = sin_pos_l * scale_2  # left_knee_joint
    ref_dof_pos[:, 4] = sin_pos_l * scale_1 # left_ankle_pitch_joint
    # right foot stance phase set to default joint pos
    sin_pos_r[sin_pos_r < 0] = 0
    ref_dof_pos[:, 8] = sin_pos_r * scale_1 # right_hip_pitch_joint
    ref_dof_pos[:, 9] = sin_pos_r * scale_2 # right_knee_joint
    ref_dof_pos[:, 10] = sin_pos_r * scale_1 # right_knee_joint
    # Double support phase
    ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0

    return ref_dof_pos


def play_pos():
    global gym_step
    # **************************  实例化模拟器 **************************
    simulation = Simulation(cfg=cfg)
    gym = simulation.gym
    sim = simulation.sim
    robot_asset = simulation.load_pai_asset(file_name="pai_12dof")
    device = simulation.device

    # **************************  actor asset特征  **************************
    simulation.set_asset_props(robot_asset=robot_asset)

    # ************************** env actor  **************************
    # 实例化env
    env_handle = simulation.creat_env(num=1)
    # 实例化actor
    actor_handle = simulation.create_actor(env_handle=env_handle)
    dof_props_asset = gym.get_asset_dof_properties(robot_asset)
    for i in range(num_dofs):
        dof_props_asset["stiffness"][i] = 150   # stiffness_var[i]
        dof_props_asset["damping"][i] = 5  # damping_var[i]
    #  设置actor的关节阻尼特性
    gym.set_actor_dof_properties(env_handle, actor_handle, dof_props_asset)

    dof_state_tensor = gym.acquire_dof_state_tensor(sim)  # 实时监听仿真器的state
    dof_state = gymtorch.wrap_tensor(dof_state_tensor)
    dof_pos = dof_state.view(1, 12, 2)[..., 0]
    # must call prepare_sim to initialize the internal data structures used by the tensor API:
    gym.prepare_sim(sim)

    # ************************** render  **************************
    viewer = simulation.create_views(env_handle)
    action_queue = get_target_q_queue(device)
    target_q = action_queue[0]

    last_switch_time = get_timestamp()
    switch_freq = cfg.control.decimation  # ms
    save_queue = []
    #while not gym.query_viewer_has_closed(viewer):
    while gym_step < 640*24:
        # # 监听键盘事件
        # for evt in gym.query_viewer_action_events(viewer):
        #     pass
        simulation.render()

        _now = get_timestamp()
        if _now - last_switch_time > switch_freq:  # ms
            # 设定目标
            # target_q = action_queue[gym_step % 2]
            target_q = compute_ref_state(dof_pos)
            save_queue.append(target_q.detach().cpu().numpy()[0] * 4)
            Logger.instance().info(f"gym_step: {gym_step} 切换发送目标:  {target_q}")
            gym_step += 1
            last_switch_time = _now

        pd_tar_tensor = gymtorch.unwrap_tensor(target_q)
        gym.set_dof_position_target_tensor(sim, pd_tar_tensor)
        gym.simulate(sim)  # 执行一次仿真子步骤

    save_queue = np.array(save_queue)
    df = pd.DataFrame(save_queue)
    # 将DataFrame写入CSV文件
    csv_file = f'action_pos_phase.csv'
    df.to_csv(csv_file, index=False)  # index=False 表示不保存索引列
    print(f"save {csv_file} success!")
    simulation.destroy()


if __name__ == '__main__':
    play_pos()
