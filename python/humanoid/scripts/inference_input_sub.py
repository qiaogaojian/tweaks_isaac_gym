#!/usr/bin/env python

from os import access
import rospy
from humanoid.envs import *
from humanoid.utils import get_args, CSVWriter, task_registry, Logger
from std_msgs.msg import Float32MultiArray
from livelybot_msg.msg import Dim2Array # type: ignore
from collections import deque
import torch
import pandas as pd
import numpy as np
import queue

def inference(obs_array, cfg, policy, csv_input: CSVWriter, csv_action: CSVWriter):

    i = 0
    action = np.zeros((cfg.env.num_actions), dtype=np.double)
    policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
    for obs_ele in obs_array.data:
        obs_np = np.array(obs_ele.data, dtype=np.float32)
        # print(f"obs_np i {i}")
        # print(obs_np)
        obs = np.clip(obs_np, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)
    # for i in range(cfg.env.frame_stack):
        policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = obs
        i += 1
    # print("policy_input")
    # print(policy_input)
    # csv_input.write_row(policy_input[0])
    action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
    action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
    # csv_action.write_row(action)
    return policy_input[0], action
    
def read(args):
    # ********************* step1 init env *********************
    args.num_envs = 1
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    device = env.device
    logger_robot = Logger(env.dt)
    logger_sim = Logger(env.dt)
    print(f"env: {type(env_cfg)}")

    model_path = '/home/mega/Downloads/ycw.pt'

    policy = torch.jit.load(model_path)

    sim_action_arr = []
    robot_action_arr = []
    model_input_arr = []
    target_obs_arr = []
    compare_len = 516

    init_status = env_cfg.init_state.pos_action

    csv_input = CSVWriter("model_input");
    csv_action = CSVWriter("model_action");

    for num in range(0, 100):
        render(env, device, init_status)

    # ********************* action queue *********************
    rospy.loginfo("Action Sim init Success")
    while True:
        obs = ros_input.get()
        # print(f"ac_data {ac_data}")
        # print(f"ros_input {ros_input.qsize()}")
        if obs and len(sim_action_arr) < compare_len:
            model_input, action = inference(obs, env_cfg, policy, csv_input, csv_action)
            
            target_obs = render(env, device, action).clone().cpu().numpy()[0]
            target_obs_arr.append(target_obs)
            # csv_input.write_row(target_obs)
            sim_action_arr.append(action)
            model_input_arr.append(model_input)
        # print(f"len(sim_action_arr) {len(sim_action_arr)}")

        robot_action = ros_result.get()
        # print(f"robot_action {robot_action.qsize()}")
        if robot_action and len(robot_action_arr) < compare_len:
            robot_action_arr.append(robot_action.data)

        if len(sim_action_arr) == compare_len and len(robot_action_arr) == compare_len:
            # print("sim_action_arr")
            # print(sim_action_arr)
            # print("robot_action_arr")
            # print(robot_action_arr)
            # print(f"model inference times {compare_len}")
            calculate_similarity(sim_action_arr, robot_action_arr)
            calculate_average_error(sim_action_arr, robot_action_arr, compare_len)
            calculate_pos_action_average_error("Real", model_input_arr)
            calculate_pos_action_average_error("Sim", target_obs_arr)
            # inference_draw_log(env_cfg, logger, sim_action_arr, robot_action_arr)
            robot_action_draw_log(env_cfg, logger_robot, model_input_arr)
            robot_action_draw_log(env_cfg, logger_sim, target_obs_arr)
            break

def robot_action_draw_log(env_cfg, logger : Logger, model_input_arr): 
    for model_input in model_input_arr:
        logger.log_states(
        {
            'hip_pitch_target': model_input[29 + 2] * env_cfg.control.action_scale,
            'hip_pitch': model_input[5 + 2],
            'knee_target': model_input[29 + 3] * env_cfg.control.action_scale,
            'knee': model_input[5 + 3],
            'dof_vel': model_input[17 + 2],
            # 'dof_torque': env.torques[robot_index, 1].item(),
            'command_x': model_input[2],
            'command_y': model_input[3],
            'command_yaw': model_input[4],
            # 'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
            # 'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
            # 'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
            # 'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
            # 'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
        })
    logger.print_rewards()
    logger.plot_states()

def inference_draw_log(env_cfg, logger : Logger, sim_action_arr, robot_action_arr):
    logger_len = len(sim_action_arr)
    for i in range (0, logger_len):
        logger.log_states(
        {
            'hip_pitch_target': sim_action_arr[i][2] * env_cfg.control.action_scale,
            'hip_pitch': robot_action_arr[i][2],
            'knee_target': sim_action_arr[i][3] * env_cfg.control.action_scale,
            'knee': robot_action_arr[i][3],
            # 'dof_vel': env.dof_vel[robot_index, 1].item(),
            # 'dof_torque': env.torques[robot_index, 1].item(),
            # 'command_x': env.commands[robot_index, 0].item(),
            # 'command_y': env.commands[robot_index, 1].item(),
            # 'command_yaw': env.commands[robot_index, 2].item(),
            # 'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
            # 'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
            # 'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
            # 'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
            # 'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
        }
    )
    logger.print_rewards()
    logger.plot_states()

def calculate_pos_action_average_error(name, model_input_arr):
    """
    Compare position and action.
    """
    pa_average_error_arr = []
    pa_pos = []
    pa_action = []
    ii = 0
    for model_input in model_input_arr:
        ii += 1
        if ii < 16:
            continue
        e = 0
        p = 0
        a = 0
        for i in range (0 , 12):
            p += abs(model_input[i + 5])
            pa_pos.append(p)
            a += abs(model_input[i + 29] * 0.25)
            pa_action.append(a)
            e += abs(model_input[i + 5] - (model_input[i + 29] * 0.25))
        pa_average_error_arr.append(e)
    pa_average_error_total = sum(pa_average_error_arr) / len(pa_average_error_arr)
    pa_per_average_error_total = sum(pa_average_error_arr) / len(pa_average_error_arr) / 12
    pa_pos_percent_total = round(sum(pa_average_error_arr) * 100 / sum(pa_pos), 2)
    pa_action_percent_total = round(sum(pa_average_error_arr) * 100 / sum(pa_action), 2)
    print(f"{name} average difference between position and action {pa_average_error_total}")
    print(f"{name} average difference for each motor in position and action {pa_per_average_error_total}")   
    print(f"{name} percent of average error in pos {pa_pos_percent_total}%")   
    print(f"{name} percent of average error in action {pa_action_percent_total}%")   

def calculate_similarity(sim_action_arr, robot_action_arr, threshold=0.0000001):
    """
    Compare corresponding elements in two lists and check if the absolute difference
    between them is less than the threshold.
    """
    percent_equal = []
    for robot_action, sim_action in zip(robot_action_arr, sim_action_arr):
        percent = compare_lists(robot_action, sim_action)
        percent_equal.append(percent)
    percent_total = sum(percent_equal) / len(percent_equal)
    print(f"model inference similarity percent {percent_total}%, threshold {threshold}")

def calculate_average_error(sim_action_arr, robot_action_arr, compare_len):
    """
    计算两个数组之间的平均误差。
    """
    if len(robot_action_arr) != len(sim_action_arr):
        raise ValueError("Arrays must have the same length")
    average_error_arr = []
    for robot_action, sim_action in zip(robot_action_arr, sim_action_arr):
        average_error = average_error_list(robot_action, sim_action)
        average_error_arr.append(average_error)
    average_error_total = sum(average_error_arr) / len(average_error_arr)
    print(f"model inference average error {average_error_total}")
    return average_error_total

def average_error_list(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Lists must be of the same length")

    average_error = 0
    total_count = len(list1)

    for elem1, elem2 in zip(list1, list2):
        average_error += abs(elem1 - elem2)

    return (average_error / total_count) if total_count > 0 else 0    

def compare_lists(list1, list2, threshold=0.001):
    """
    Compare corresponding elements in two lists and check if the absolute difference
    between them is less than the threshold.
    """
    if len(list1) != len(list2):
        raise ValueError("Lists must be of the same length")

    equal_count = 0
    total_count = len(list1)

    for elem1, elem2 in zip(list1, list2):
        if abs(elem1 - elem2) < threshold:
            equal_count += 1

    return (equal_count / total_count) * 100 if total_count > 0 else 0    

def render(env, device, action):
    ten = torch.Tensor(action).to(device)
    action = ten.unsqueeze(0)

    if FIX_COMMAND:
            env.commands[:, 0] = 0.5    # 1.0
            env.commands[:, 1] = 0.
            env.commands[:, 2] = 0.
            env.commands[:, 3] = 0.

    target_obs, critic_obs, rews, dones, infos = env.step(action.detach())
    return target_obs

ros_input = queue.Queue(1000)
ros_result = queue.Queue(1000)

def callback(data):
    global ros_input
    ros_input.put(data)

def callback_r(data):
    global ros_result
    ros_result.put(data)

def listener():
    print("++++++++++++++++rospy listener+++++++++++++++++++++++++")

    rospy.init_node('gym_robot', anonymous=True)
    rospy.Subscriber('/model/reference/input', Dim2Array, callback)
    rospy.Subscriber('/model/reference/results', Float32MultiArray, callback_r)

    print("Subscriber /model/reference/input")
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
    args.headless = False
    read(args)