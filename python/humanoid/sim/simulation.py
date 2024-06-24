import torch
from isaacgym import gymapi
from pycore.utils.file_utils import FileUtils

from humanoid.envs.pai.pai_config import PaiCfg
from humanoid.utils import get_args, class_to_dict
from isaacgym.torch_utils import to_torch
from humanoid.utils.helpers import parse_sim_params


class Simulation:

    def __init__(self, cfg: PaiCfg):
        self.cfg = cfg
        self.gym, self.sim = self.create_sim(cfg)
        # 检查是否有可用的CUDA设备
        print("cuda", torch.cuda.is_available())
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def create_sim(self, cfg: PaiCfg):
        # gym instance
        gym = gymapi.acquire_gym()
        args = get_args()
        sim_params = {"sim": class_to_dict(cfg.sim)}
        self.sim_params = parse_sim_params(args, sim_params)
        self.sim_params.substeps = self.cfg.sim.substeps
        # self.sim_params.use_gpu_pipeline = False # 暂时禁用gpu tensor

        print(f"sim_params.dt: {self.sim_params.dt}")

        # sim instance
        sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, gymapi.SIM_PHYSX, self.sim_params)

        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction    # cfg.terrain.static_friction # 地板的静态摩擦系数。取值范围为 0 到 1。
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction  # cfg.terrain.dynamic_friction # 地板的动态摩擦系数。取值范围为 0 到 1。
        plane_params.restitution = cfg.terrain.restitution
        gym.add_ground(sim, plane_params)

        return gym, sim

    def creat_env(self, num = 1):
        # 创建env句柄
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        env_handle = self.gym.create_env(self.sim, env_lower, env_upper, num)
        return env_handle

    def destroy(self):
        print("Done")
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def config_viewer(self):
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_ESCAPE, "QUIT")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_F, "follow")

    def create_views(self, env_handle):
        # create viewer
        cam_props = gymapi.CameraProperties()
        self.viewer = self.gym.create_viewer(self.sim, cam_props)
        if self.viewer is None:
            print("*** Failed to create viewer")
            quit()

        position = self.cfg.viewer.pos
        lookat = self.cfg.viewer.lookat
        cam_pos = gymapi.Vec3(2, 2, 2)
        cam_target = gymapi.Vec3(0, 0, 1)
        # cam_target = env_handle
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return self.viewer

    # 加载pai asset
    def load_pai_asset(self, file_name):
        gym = self.gym
        sim = self.sim
        # actor 资源
        asset_root = FileUtils.get_project_root() + "/resources"
        asset_file = f"robots/pai_12dof/urdf/{file_name}.urdf"
        # actor属性
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode # 1pos模式
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        self.robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
        return self.robot_asset

    def render(self):
        self.gym.fetch_results(self.sim, True)

        # update the viewer
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)

        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        self.gym.sync_frame_time(self.sim)

    def set_asset_props(self, robot_asset):
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        for i in range(len(rigid_shape_props_asset)):
            rigid_shape_props_asset[i].friction = 0.5  # 刚体摩擦力系数0.1~2
        self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props_asset)
        # num_dof = gym.get_asset_dof_count(robot_asset)

    def create_actor(self, env_handle):
        # 2 init base line
        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        # start_pose.p = gymapi.Vec3(0.0, 0.0, 0.38)
        start_pose.p = gymapi.Vec3(*base_init_state[:3])
        print(f">\nstart_pose: {base_init_state[:3]}")
        return self.gym.create_actor(env_handle, self.robot_asset, start_pose, "pai", 0, self.cfg.asset.self_collisions, 0)