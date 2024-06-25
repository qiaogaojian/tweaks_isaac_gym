from pycore.utils.file_utils import FileUtils
from humanoid.envs import *
from humanoid.utils import get_args, task_registry
from humanoid.utils.matplot_logger import MatplotLogger

import time
import torch
import numpy as np
import pandas as pd

from pycore.base import Core
from pycore.logger import Logger
from pycore.utils.tools import Tools


def get_timestamp():
    timestamp_ns = time.time_ns()
    # 转换为毫秒
    timestamp_ms = timestamp_ns / 1_000_000
    return timestamp_ms


def read(args):
    # ********************* step1 init env *********************
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env_cfg.domain_rand.push_robots = False  # hack 屏蔽对机器人的推力
    device = env.device
    print("************ config: ************")
    print(f"env.dt: {env.dt}")
    matplot_logger = MatplotLogger(dt=env.dt, png_name="gym", pai_config=env_cfg)

    # ********************* action queue *********************
    # 使用pandas读取CSV文件
    csv_file = args.csv_file
    df = pd.read_csv(csv_file)
    # 将DataFrame转换为numpy数组
    np_array = df.values

    action_queue = []
    for row in np_array:
        ten = torch.Tensor(row).to(device)
        action = ten.unsqueeze(0)
        action_queue.append(action)
    print(f"load csv: {csv_file} success!")

    # ********************* render *********************
    if FIX_COMMAND:
        env.commands[:, 0] = 0.0  # 1.0
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 0.0
        env.commands[:, 3] = 0.0

    # action = torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).to(device)
    # action = action.unsqueeze(0)

    # idle_step = 1
    # while idle_step < 320:
    #     env.play(action.detach())

    #     idle_step += 1
    #     matplot_logger.log_states({"target": action[0].cpu().numpy() * env.cfg.control.action_scale, "gym_dof_pos": env.dof_pos[0].cpu().numpy(), "gym_dof_vel": env.dof_vel[0].cpu().numpy(), "gym_torques": env.torques[0].cpu().numpy(), "gym_command": env.commands[0].cpu().numpy(), "gym_base_line_vel": env.base_lin_vel[0].cpu().numpy(), "gym_base_ang_vel": env.base_ang_vel[0].cpu().numpy(), "gym_contact_forces_z": env.contact_forces[0, env.feet_indices, 2].cpu().numpy()})

    gym_step = 0
    while gym_step < 540:
        action = action_queue[gym_step]
        env.play(action.detach())

        gym_step += 1
        matplot_logger.log_states({"target": action[0].cpu().numpy() * env.cfg.control.action_scale, "gym_dof_pos": env.dof_pos[0].cpu().numpy(), "gym_dof_vel": env.dof_vel[0].cpu().numpy(), "gym_torques": env.torques[0].cpu().numpy(), "gym_command": env.commands[0].cpu().numpy(), "gym_base_line_vel": env.base_lin_vel[0].cpu().numpy(), "gym_base_ang_vel": env.base_ang_vel[0].cpu().numpy(), "gym_contact_forces_z": env.contact_forces[0, env.feet_indices, 2].cpu().numpy()})

    # 生成图表
    matplot_logger.plot_states()


if __name__ == "__main__":
    EXPORT_POLICY = True
    RENDER = True
    FIX_COMMAND = True
    args = get_args()
    args.is_play = True
    args.headless = False
    args.num_envs = 1
    args.csv_file = FileUtils.get_project_path("resources/csv/action_Jun12_20-01-14_pos_v13.csv")

    core = Core()
    core.init(env="dev")
    Logger.instance().info("********************************* Test *********************************")

    read(args)
