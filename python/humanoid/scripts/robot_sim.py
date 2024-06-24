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

import rospy
import math
import numpy as np
import mujoco, mujoco_viewer
from std_msgs.msg import Float32MultiArray
from collections import deque
from scipy.spatial.transform import Rotation as R
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import PaiCfg
import os
import argparse
import time
import queue

from humanoid.utils.helpers import get_args

class Sim2RobotCfg(PaiCfg):

    class sim_config:
        # mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/pai_12dof/mjcf/pai_12dof.xml'
        mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/pai_12dof_v2/mjcf/pai_12dof.xml'
        sim_duration = 60.0
        dt = 0.005
        decimation = 10

    class robot_config:
        kps = np.array([30, 30, 90, 90, 15, 15, 30, 30, 90, 90, 15, 15], dtype=np.double)
        kds = np.array([0.5, 0.5, 1.5, 1.5, 0.2, 0.2, 0.5, 0.5, 1.5, 1.5, 0.2, 0.2], dtype=np.double)
        tau_limit = 10. * np.ones(12, dtype=np.double)

class cmd:
    vx = 0.2
    vy = 0.0
    dyaw = 0.0

args = get_args()
count_lowlevel = 0
print("++++++++++++++++Sim2RobotCfg+++++++++++++++++++++++++")
cfg = Sim2RobotCfg()
print(f"mujoco_model_path: {cfg.sim_config.mujoco_model_path}")
print("++++++++++++++++Sim2RobotCfg+++++++++++++++++++++++++")

print("++++++++++++++++MjModel+++++++++++++++++++++++++")
model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
model.opt.timestep = cfg.sim_config.dt
mj_data = mujoco.MjData(model)
print(f"data: {mj_data}")
mujoco.mj_step(model, mj_data)
viewer = mujoco_viewer.MujocoViewer(model, mj_data)
# mujoco.mj_step(model, mj_data)
viewer.render()
target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
action = np.zeros((cfg.env.num_actions), dtype=np.double)

mj_data.ctrl = np.array(list([0.03424922,-0.11412111,0.21214561,0.78471703,0.95770369,2.10723316,-0.01714044,-0.16668367,-0.66534658,-1.69914467,-0.45487402,3.43415007]))

mujoco.mj_step(model, mj_data)
print("++++++++++++++++mujoco+++++++++++++++++++++++++")
viewer.render()

hist_obs = deque()
for _ in range(cfg.env.frame_stack):
    hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))

print("++++++++++++++++MjModel+++++++++++++++++++++++++")

test_aciton = np.array(list(cfg.init_state.pos_action))

def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z]) 

def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    # print("p:", (target_q - q) * kp )
    # print("d", (target_dq - dq) * kd)
    return (target_q - q) * kp + (target_dq - dq) * kd

q_action=queue.Queue(1000)

def callback(data):
    global q_action
    q_action.put(data)
    # print("++++++++++++++++rospy listener data+++++++++++++++++++++++++")
    # print(f"/model/reference/results {data.data}")
    # print("++++++++++++++++rospy listener data+++++++++++++++++++++++++")

def mjc_exec(ac_data):
     # Update Mujoco model based on received message
    

    global count_lowlevel, action, mj_data, target_q, test_aciton

    # Obtain an observation
    q, dq, quat, v, omega, gvec = get_obs(mj_data)
    q = q[-cfg.env.num_actions:]
    dq = dq[-cfg.env.num_actions:]

    for i in range(6):
        tmpq = q[i]
        q[i] = q[i+6]
        q[i+6] = tmpq

        tmpdq = dq[i]
        dq[i] = dq[i+6]
        dq[i+6] = tmpdq
    # print(f"===========================cfg.env.num_single_obs {cfg.env.num_single_obs}")
    # 1000hz -> 100hz
    if count_lowlevel % 10 == 0:
        obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
        eu_ang = quaternion_to_euler_array(quat)
        eu_ang[eu_ang > math.pi] -= 2 * math.pi

        obs[0, 0] = math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / 0.64)
        obs[0, 1] = math.cos(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / 0.64)
        obs[0, 2] = cmd.vx * cfg.normalization.obs_scales.lin_vel
        obs[0, 3] = cmd.vy * cfg.normalization.obs_scales.lin_vel
        obs[0, 4] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
        obs[0, 5:17] = q * cfg.normalization.obs_scales.dof_pos
        obs[0, 17:29] = dq * cfg.normalization.obs_scales.dof_vel
        obs[0, 29:41] = action
        obs[0, 41:44] = omega
        obs[0, 44:47] = eu_ang

        obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)
        hist_obs.append(obs)
        hist_obs.popleft()

        # TODO add _model_sub std_msgs::Float32MultiArray
        if ac_data:
            action = np.array(list(ac_data.data))
        else: 
            action = test_aciton
        # action = np.array(list([0,0,0,0,0,0,0,0,0,0,0,0]))
        # print("++++++++++++++++action+++++++++++++++++++++++++")
        # print(f"action {action}")
        # print("++++++++++++++++action+++++++++++++++++++++++++")

        target_q = action * cfg.control.action_scale

    target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
    
    # Generate PD control
    tau = pd_control(target_q, q, cfg.robot_config.kps,
                    target_dq, dq, cfg.robot_config.kds)  # Calc torques
    tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
    for i in range(6):
        tmptau = tau[i]
        tau[i] = tau[i+6]
        tau[i+6] = tmptau
    mj_data.ctrl = tau
    print("++++++++++++++++tau+++++++++++++++++++++++++")
    print(f"tau {tau}")
    print("++++++++++++++++tau+++++++++++++++++++++++++")

    try:
        mujoco.mj_step(model, mj_data)
        # print("++++++++++++++++mujoco+++++++++++++++++++++++++")
        viewer.render()
    except BaseException as e:
        print(e);
    count_lowlevel += 1
    # time.sleep(11111)

def listener():
    print("++++++++++++++++rospy listener+++++++++++++++++++++++++")

    rospy.init_node('mujoco_robot', anonymous=True)
    rospy.Subscriber('/model/reference/results', Float32MultiArray, callback)

    print("Subscriber /model/reference/results")
    print("++++++++++++++++rospy listener+++++++++++++++++++++++++")

    

if __name__ == '__main__':
    print("++++++++++++++++__main__+++++++++++++++++++++++++")
    for num in range(0, 10001):
        mjc_exec(False)
    time.sleep(100000)
    try:
        listener()
    except rospy.ROSInterruptException:
        pass

    while True:
        ac_data = q_action.get()
        # print(f"ac_data {ac_data}")
        print(f"q_action size {q_action.qsize()}")
        if ac_data:
            mjc_exec(ac_data)
    # rospy.spin()
    
