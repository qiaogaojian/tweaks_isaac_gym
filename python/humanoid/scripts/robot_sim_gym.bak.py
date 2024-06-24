import os

from isaacgym import gymapi, gymtorch
import torch
from isaacgym.torch_utils import to_torch
import rospy
from std_msgs.msg import Float32MultiArray

from humanoid.envs import PaiCfg
from humanoid.sim import Simulation
import numpy as np
from humanoid import LEGGED_GYM_ROOT_DIR
import time
import queue

cfg = PaiCfg()


# 加载pai asset
def load_pai_asset(s: Simulation):
    gym = s.gym
    sim = s.sim
    # actor 资源
    asset_root = LEGGED_GYM_ROOT_DIR + "/resources"
    asset_file = cfg.init_state.model_path
    # asset_file = "robots/pai_12dof_v2/urdf/pai_12dof.urdf"
    # actor属性
    asset_options = gymapi.AssetOptions()
    asset_options.default_dof_drive_mode = 1  # cfg.asset.default_dof_drive_mode 1pos模式
    asset_options.collapse_fixed_joints = cfg.asset.collapse_fixed_joints
    asset_options.replace_cylinder_with_capsule = cfg.asset.replace_cylinder_with_capsule
    asset_options.flip_visual_attachments = cfg.asset.flip_visual_attachments
    asset_options.fix_base_link = cfg.asset.fix_base_link
    asset_options.density = cfg.asset.density
    asset_options.angular_damping = cfg.asset.angular_damping
    asset_options.linear_damping = cfg.asset.linear_damping
    asset_options.max_angular_velocity = cfg.asset.max_angular_velocity
    asset_options.max_linear_velocity = cfg.asset.max_linear_velocity
    asset_options.armature = cfg.asset.armature
    asset_options.thickness = cfg.asset.thickness
    asset_options.disable_gravity = cfg.asset.disable_gravity
    

    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    return asset

def kp_kd_array(cfg):
    stiffness_var = []
    damping_var = []


    for val in cfg.control.stiffness.values():
        stiffness_var.append(val)
    stiffness_var = np.array(stiffness_var)
    stiffness_var = np.tile(stiffness_var, 2).astype(float)

    for val in cfg.control.damping.values():
        damping_var.append(val)
    damping_var = np.array(damping_var)
    damping_var = np.tile(damping_var, 2).astype(float)

    return stiffness_var, damping_var

# **************************  实例化模拟器 **************************
simulation = Simulation(dt=1 / 60, cfg=cfg)
gym = simulation.gym
sim = simulation.sim
robot_asset = load_pai_asset(s=simulation)

# 检查是否有可用的CUDA设备
print("cuda", torch.cuda.is_available())
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(device)

# **************************  actor asset特征  **************************
rigid_shape_props_asset = gym.get_asset_rigid_shape_properties(robot_asset)
for i in range(len(rigid_shape_props_asset)):
    rigid_shape_props_asset[i].friction = 0.5  # 刚体摩擦力系数0.1~2
gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props_asset)
dof_names = gym.get_asset_dof_names(robot_asset)
print(f"dof_names: {dof_names}")
num_dof = gym.get_asset_dof_count(robot_asset)
num_dofs = len(dof_names)

# 2 init base line
base_init_state_list = cfg.init_state.pos + cfg.init_state.rot + cfg.init_state.lin_vel + cfg.init_state.ang_vel
base_init_state = to_torch(base_init_state_list, device=device, requires_grad=False)
start_pose = gymapi.Transform()
# start_pose.p = gymapi.Vec3(0.0, 0.0, 0.38)
start_pose.p = gymapi.Vec3(*base_init_state[:3])
print(f">\nstart_pose: {base_init_state[:3]}")

# ************************** env actor  **************************
# 3 实例化env
env_handle = simulation.creat_env(num=1)
# 实例化actor
actor_handle = simulation.gym.create_actor(env_handle, robot_asset, start_pose, "pai", 0, cfg.asset.self_collisions, 0)
dof_props_asset = gym.get_asset_dof_properties(robot_asset)
stiffness_var, damping_var = kp_kd_array(cfg)
print(f"stiffness_var: {stiffness_var}")
print(f"damping_var: {damping_var}")
for i in range(12):
    dof_props_asset["stiffness"][i] = 0.8
    dof_props_asset["damping"][i] = 0.5
# actor 特性
gym.set_actor_dof_properties(env_handle, actor_handle, dof_props_asset)

# TODO 环境特征
# must call prepare_sim to initialize the internal data structures used by the tensor API:
gym.prepare_sim(sim)

# ************************** init  pos  **************************
# 4 初始化actor pos
dof_names = gym.get_asset_dof_names(robot_asset)
num_dof = gym.get_asset_dof_count(robot_asset)
num_dofs = len(dof_names)
default_dof_pos = torch.zeros(num_dof, dtype=torch.float, device=device, requires_grad=False)
for i in range(num_dofs):
    name = dof_names[i]
    default_dof_pos[i] = cfg.init_state.default_joint_angles[name]
print(f">\ndefault_dof_pos: {default_dof_pos}")

def computer_pd_tar(actions):
    # 1 clip
    clip_actions = cfg.normalization.clip_actions
    actions = torch.clip(actions, -clip_actions, clip_actions)
    # 2 computer
    for i in range(num_dofs):
       # actions[i] = (actions[i] * cfg.control.action_scale + default_dof_pos[i]) * cfg.init_state.motor_direction[i]
        actions[i] = actions[i] * cfg.control.action_scale + default_dof_pos[i]

    # actions[3] = actions[3] - actions[2]
    # actions[9] = actions[9] - actions[8]
    return actions

# print(f"motor_direction: {cfg.init_state.motor_direction}")

def action_status(action_status):
    # start_tar = torch.zeros(num_dof, dtype=torch.float, device=device)
    # start_tar = np.array(start_tar)
    action_s = torch.tensor(action_status)
    # print(f"start_action: {action_s}")
    action_s = computer_pd_tar(actions=action_s)
    # print(f"target_q: {action_s} ")
    action_s = action_s.unsqueeze(0)
    return action_s

# ************************** render  **************************
#  相机看向actor
viewer = simulation.create_views(env_handle)

# 5 渲染
def render(r_action):
    # 监听键盘事件
    for evt in gym.query_viewer_action_events(viewer):
        pass

    # 设定目标
    # step the physics
    exe_action = action_status(r_action)
    print(f'exe_action {exe_action}')
    pd_tar_tensor = gymtorch.unwrap_tensor(exe_action)
    
    gym.set_dof_position_target_tensor(sim, pd_tar_tensor)

    gym.simulate(sim)  # 执行一次仿真子步骤
    if device == 'cpu':
        gym.fetch_results(sim, True)
    # 刷新关节状态张量，更新仿真状态。
    gym.refresh_dof_state_tensor(sim)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

q_action=queue.Queue(1000)

def callback(data):
    global q_action
    q_action.put(data)
    # print("++++++++++++++++rospy listener data+++++++++++++++++++++++++")
    # print(f"/model/reference/results {data.data}")
    # print("++++++++++++++++rospy listener data+++++++++++++++++++++++++")

def listener():
    print("++++++++++++++++rospy listener+++++++++++++++++++++++++")

    rospy.init_node('mujoco_robot', anonymous=True)
    rospy.Subscriber('/model/reference/results', Float32MultiArray, callback)

    print("Subscriber /model/reference/results")
    print("++++++++++++++++rospy listener+++++++++++++++++++++++++")


# same to unity update function
# while not gym.query_viewer_has_closed(viewer):
#    render(init_status) 
def init_render(n):
    for num in range(0, n):
        render(init_status)
        # print(num)
        # print(f">\nstart_pose: {base_init_state[:3]}")

# init_status = [
# 0, 0, -1,
# -3, -1, 0,
# 0,0,1,
# 3, 1, 0
# ]

init_status = cfg.init_state.pos_action

# simulation.destroy()
if __name__ == '__main__':
    print("++++++++++++++++__main__+++++++++++++++++++++++++")
    # 5 渲染
    # same to unity update function
    init_render(100)
    
    try:
        listener()
    except rospy.ROSInterruptException:
        pass
    
    # while True:
    #     render(init_status)

    while True:
        ac_data = q_action.get()
        # print(f"ac_data {ac_data}")
        print(f"q_action size {q_action.qsize()}")
        if ac_data:
            render(ac_data.data)
    simulation.destroy()