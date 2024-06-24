from humanoid.envs import *
from humanoid.utils import get_args, task_registry
from humanoid.utils.time_utils import get_timestamp

import torch
from pycore.logger import Logger

# ********************* config *********************
args = get_args()
args.num_envs = 1

# *********************  env *********************
env, env_cfg = task_registry.make_env(name=args.task, args=args)
device = env.device
print(f"env: {type(env_cfg)}")

# ********************* action *********************
action1 = torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).to(device)
actions1 = action1.unsqueeze(0)

action2 = torch.Tensor([0, 0, -1, -2, -1, 0, 0, 0, 1, 2, 1, 0]).to(device)
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
    if _now - last_switch_time > switch_freq:  # 1000 ms
        target_q = action_queue[gym_step % 2] * 1
        Logger.instance().info(f"切换发送目标 {target_q}")

        last_switch_time = _now
        gym_step += 1

    env.step(target_q)
