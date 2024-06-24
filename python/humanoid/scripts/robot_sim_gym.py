import rospy
from std_msgs.msg import Float32MultiArray

from humanoid.envs import *
from humanoid.utils import get_args, task_registry
import torch
import time
import queue

def read(args):
    # ********************* step1 init env *********************
    args.num_envs = 1
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    device = env.device
    print(f"env: {type(env_cfg)}")

    init_status = env_cfg.init_state.pos_action

    for num in range(0, 100):
        render(env, device, init_status)
    # ********************* action queue *********************
    rospy.loginfo("Action Sim init Success")
    while True:
        ac_data = q_action.get()
        # print(f"ac_data {ac_data}")
        print(f"q_action size {q_action.qsize()}")
        if ac_data:
            render(env, device, ac_data.data)
            # ten = torch.Tensor(ac_data.data).to(device)
            # action = ten.unsqueeze(0)

            # if FIX_COMMAND:
            #         env.commands[:, 0] = 0.5    # 1.0
            #         env.commands[:, 1] = 0.
            #         env.commands[:, 2] = 0.
            #         env.commands[:, 3] = 0.

            # obs, critic_obs, rews, dones, infos = env.step(action.detach())

def render(env, device, action):
    ten = torch.Tensor(action).to(device)
    action = ten.unsqueeze(0)

    if FIX_COMMAND:
            env.commands[:, 0] = 0.5    # 1.0
            env.commands[:, 1] = 0.
            env.commands[:, 2] = 0.
            env.commands[:, 3] = 0.

    obs, critic_obs, rews, dones, infos = env.step(action.detach())

q_action=queue.Queue(1000)

def callback(data):
    global q_action
    q_action.put(data)


def listener():
    print("++++++++++++++++rospy listener+++++++++++++++++++++++++")

    rospy.init_node('mujoco_robot', anonymous=True)
    rospy.Subscriber('/model/reference/results', Float32MultiArray, callback)

    print("Subscriber /model/reference/results")
    print("++++++++++++++++rospy listener+++++++++++++++++++++++++")

if __name__ == '__main__':
    print("++++++++++++++++__main__+++++++++++++++++++++++++")
    try:
        listener()
    except rospy.ROSInterruptException:
        pass

    EXPORT_POLICY = True
    RENDER = True
    FIX_COMMAND = True
    args = get_args()
    args.is_play = True
    read(args)