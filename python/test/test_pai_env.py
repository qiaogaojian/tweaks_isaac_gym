import time

from pycore.logger import Logger

from humanoid.envs import *
from humanoid.utils import get_args, task_registry
import torch

# ********************* step1 init env *********************
from humanoid.utils.time_utils import get_timestamp

args = get_args()
args.num_envs = 1
env, env_cfg = task_registry.make_env(name=args.task, args=args)
device = env.device
print(f"env: {type(env_cfg)}")

# ********************* action queue *********************
action1 = torch.Tensor([
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0
]).to(device)
actions1 = action1.unsqueeze(0)

action2 = torch.Tensor([
    0, 0, -1,
    -3, -1, 0,
    0, 0, 1,
    3, 1, 0
]).to(device)
action2 = action2.unsqueeze(0)
action_queue = [action1, action2]

# ********************* render *********************
target_q = action_queue[0]
switch_freq = 1000  # ms
gym_step = 0

last_switch_time = get_timestamp()
while True:
    # v1
    _now = get_timestamp()
    if _now - last_switch_time > switch_freq:  # 1s
        target_q = action_queue[gym_step % 2] * 1
        Logger.instance().info(f"切换发送目标 {target_q}")
        gym_step += 1
        last_switch_time = _now

    # v2
    # target_q = action_queue[gym_step % 2] * 1
    # Logger.instance().info(f"切换发送目标 {target_q}")
    # gym_step += 1

    env.step(target_q)
