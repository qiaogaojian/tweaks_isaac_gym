import time

from pycore.logger import Logger
from humanoid.envs import *
from humanoid.utils import get_args, task_registry
import torch
from tqdm import tqdm
import csv
import numpy as np
import pandas as pd

# target_q = action_queue[0]
switch_freq = 1000  # ms
gym_step = 0


def get_timestamp():
    timestamp_ns = time.time_ns()
    # 转换为毫秒
    timestamp_ms = timestamp_ns / 1_000_000
    return timestamp_ms


def play(args):
    # ********************* step1 init env *********************
    _, train_cfg = task_registry.get_cfgs(name=args.task)

    args.num_envs = 1
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    device = env.device
    print(f"env: {type(env_cfg)}")
    obs = env.get_observations()

    # ********************* load policy *********************
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # ********************* render *********************
    action_queue = []
    for i in tqdm(range(540)):
        actions = policy(obs.detach())  # * 0.
        if FIX_COMMAND:
            env.commands[:, 0] = 0.5    # 1.0
            env.commands[:, 1] = 0.
            env.commands[:, 2] = 0.
            env.commands[:, 3] = 0.

        obs, critic_obs, rews, dones, infos = env.step(actions.detach())
        action_queue.append(actions.detach().cpu().numpy()[0])

    last_switch_time = get_timestamp()
    # ********************* save action queue *********************
    action_queue = np.array(action_queue)
    df = pd.DataFrame(action_queue)
    path_array = args.load_model.split('/')
    # 将DataFrame写入CSV文件
    csv_file = f'csv/action_{path_array[-2]}.csv'
    df.to_csv(csv_file, index=False)  # index=False 表示不保存索引列
    print(f"save {csv_file} success!")


if __name__ == '__main__':
    EXPORT_POLICY = True
    RENDER = True
    FIX_COMMAND = True
    args = get_args()
    args.load_model = "/home/mega/python_projects/livelybot_rl_control/logs/Pai_ppo/Jun12_20-01-14_pos_v13/model_6900.pt"
    args.run_name = "pos_v13"
    args.is_play = True

    play(args)

