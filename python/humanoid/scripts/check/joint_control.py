import os
import math
import numpy as np
from isaacgym import gymapi, gymutil
from humanoid.envs import *
from humanoid.utils import get_args, task_registry
from pycore.base import Core
from pycore.logger import Logger
from pycore.utils.tools import Tools

from isaacgym.torch_utils import *
from humanoid.utils.helpers import class_to_dict, parse_sim_params


def init_config():
    args = get_args()
    args.show_axis = True

    env_cfg = PaiCfg()
    train_cfg = PaiCfgPPO()

    # parse sim params (convert to dict first)
    sim_params = {"sim": class_to_dict(env_cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)

    # configure sim
    # sim_params = gymapi.SimParams()
    sim_params.dt = dt = 1.0 / 60.0
    sim_params.use_gpu_pipeline = False
    if args.use_gpu_pipeline:
        print("WARNING: Forcing CPU pipeline.")

    return env_cfg, train_cfg, sim_params, args


def create_sim():
    gym = gymapi.acquire_gym()
    sim_device_type, sim_device_id = gymutil.parse_device_str(device)
    graphics_device_id = sim_device_id
    sim = gym.create_sim(sim_device_id, graphics_device_id, args.physics_engine, sim_params)

    _create_ground_plane(gym, sim)
    robot_asset = _create_envs(gym, sim)

    gym.prepare_sim(sim)

    # enable_viewer_sync = True
    # if running with a viewer, set up keyboard shortcuts and camera
    # subscribe to keyboard shortcuts
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    # gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ESCAPE, "QUIT")
    # gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_V, "toggle_viewer_sync")

    # camera_properties = gymapi.CameraProperties()
    # camera_properties.width = 720
    # camera_properties.height = 480
    # camera_handle = gym.create_camera_sensor(envs[0], camera_properties)

    return gym, sim, viewer, robot_asset


def _create_ground_plane(gym, sim):
    """Adds a ground plane to the simulation, sets friction and restitution based on the cfg."""
    # add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    plane_params.static_friction = cfg.terrain.static_friction
    plane_params.dynamic_friction = cfg.terrain.dynamic_friction
    plane_params.restitution = cfg.terrain.restitution
    gym.add_ground(sim, plane_params)


def _create_envs(gym, sim):
    """Creates environments:
    1. loads the robot URDF/MJCF asset,
    2. For each environment
    2.1 creates the environment,
    2.2 calls DOF and Rigid shape properties callbacks,
    2.3 create actor with these properties and add them to the env
    3. Store indices of different bodies of the robot
    """
    num_envs = cfg.env.num_envs

    asset_path = cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    asset_root = os.path.dirname(asset_path)
    asset_file = os.path.basename(asset_path)

    asset_options = gymapi.AssetOptions()
    asset_options.default_dof_drive_mode = cfg.asset.default_dof_drive_mode
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
    asset_options.disable_gravity = True

    robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    num_dof = gym.get_asset_dof_count(robot_asset)
    num_bodies = gym.get_asset_rigid_body_count(robot_asset)
    dof_props_asset = gym.get_asset_dof_properties(robot_asset)
    rigid_shape_props_asset = gym.get_asset_rigid_shape_properties(robot_asset)

    # # save body names from the asset
    # body_names = gym.get_asset_rigid_body_names(robot_asset)
    # dof_names = gym.get_asset_dof_names(robot_asset)
    # num_bodies = len(body_names)
    # num_dofs = len(dof_names)
    # feet_names = [s for s in body_names if cfg.asset.foot_name in s]
    # knee_names = [s for s in body_names if cfg.asset.knee_name in s]
    # penalized_contact_names = []
    # for name in cfg.asset.penalize_contacts_on:
    #     penalized_contact_names.extend([s for s in body_names if name in s])
    # termination_contact_names = []
    # for name in cfg.asset.terminate_after_contacts_on:
    #     termination_contact_names.extend([s for s in body_names if name in s])

    # base_init_state_list = cfg.init_state.pos + cfg.init_state.rot + cfg.init_state.lin_vel + cfg.init_state.ang_vel

    # base_init_state = to_torch(base_init_state_list, device=device, requires_grad=False)
    # start_pose = gymapi.Transform()
    # start_pose.p = gymapi.Vec3(*base_init_state[:3])

    # _get_env_origins()
    # env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
    # env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
    # actor_handles = []
    # envs = []
    # env_frictions = torch.zeros(num_envs, 1, dtype=torch.float32, device=device)

    # body_mass = torch.zeros(num_envs, 1, dtype=torch.float32, device=device, requires_grad=False)

    # for i in range(num_envs):
    #     # create env instance
    #     env_handle = gym.create_env(sim, env_lower, env_upper, int(np.sqrt(num_envs)))
    #     pos = env_origins[i].clone()
    #     pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=device).squeeze(1)
    #     start_pose.p = gymapi.Vec3(*pos)

    #     rigid_shape_props = _process_rigid_shape_props(rigid_shape_props_asset, i, num_envs)
    #     gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
    #     actor_handle = gym.create_actor(env_handle, robot_asset, start_pose, cfg.asset.name, i, cfg.asset.self_collisions, 0)
    #     dof_props = _process_dof_props(dof_props_asset, i, num_dof)
    #     gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
    #     body_props = gym.get_actor_rigid_body_properties(env_handle, actor_handle)
    #     body_props = _process_rigid_body_props(body_props, i)
    #     gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
    #     envs.append(env_handle)
    #     actor_handles.append(actor_handle)

    # feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=device, requires_grad=False)
    # for i in range(len(feet_names)):
    #     feet_indices[i] = gym.find_actor_rigid_body_handle(envs[0], actor_handles[0], feet_names[i])
    # knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=device, requires_grad=False)
    # for i in range(len(knee_names)):
    #     knee_indices[i] = gym.find_actor_rigid_body_handle(envs[0], actor_handles[0], knee_names[i])

    # penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=device, requires_grad=False)
    # for i in range(len(penalized_contact_names)):
    #     penalised_contact_indices[i] = gym.find_actor_rigid_body_handle(envs[0], actor_handles[0], penalized_contact_names[i])

    # termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=device, requires_grad=False)
    # for i in range(len(termination_contact_names)):
    #     termination_contact_indices[i] = gym.find_actor_rigid_body_handle(envs[0], actor_handles[0], termination_contact_names[i])

    return robot_asset


def _get_env_origins():
    """Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
    Otherwise create a grid.
    """
    custom_origins = False
    num_envs = cfg.env.num_envs
    env_origins = torch.zeros(num_envs, 3, device=device, requires_grad=False)
    # create a grid of robots
    num_cols = np.floor(np.sqrt(num_envs))
    num_rows = np.ceil(num_envs / num_cols)
    xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
    spacing = cfg.env.env_spacing
    env_origins[:, 0] = spacing * xx.flatten()[:num_envs]
    env_origins[:, 1] = spacing * yy.flatten()[:num_envs]
    env_origins[:, 2] = 0.0

    return env_origins


# ------------- Callbacks --------------
def _process_rigid_shape_props(props, env_id, num_envs):
    """Callback allowing to store/change/randomize the rigid shape properties of each environment.
        Called During environment creation.
        Base behavior: randomizes the friction of each environment

    Args:
        props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
        env_id (int): Environment id

    Returns:
        [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
    """
    if cfg.domain_rand.randomize_friction:
        if env_id == 0:
            # prepare friction randomization
            friction_range = cfg.domain_rand.friction_range
            num_buckets = 256
            bucket_ids = torch.randint(0, num_buckets, (num_envs, 1))
            friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1), device="cpu")
            friction_coeffs = friction_buckets[bucket_ids]

        for s in range(len(props)):
            props[s].friction = friction_coeffs[env_id]
    return props


def _process_dof_props(props, env_id, num_dof):
    """Callback allowing to store/change/randomize the DOF properties of each environment.
        Called During environment creation.
        Base behavior: stores position, velocity and torques limits defined in the URDF

    Args:
        props (numpy.array): Properties of each DOF of the asset
        env_id (int): Environment id

    Returns:
        [numpy.array]: Modified DOF properties
    """
    if env_id == 0:
        dof_pos_limits = torch.zeros(num_dof, 2, dtype=torch.float, device=device, requires_grad=False)
        dof_vel_limits = torch.zeros(num_dof, dtype=torch.float, device=device, requires_grad=False)
        torque_limits = torch.zeros(num_dof, dtype=torch.float, device=device, requires_grad=False)
        for i in range(len(props)):
            dof_pos_limits[i, 0] = props["lower"][i].item() * cfg.safety.pos_limit
            dof_pos_limits[i, 1] = props["upper"][i].item() * cfg.safety.pos_limit
            dof_vel_limits[i] = props["velocity"][i].item() * cfg.safety.vel_limit
            torque_limits[i] = props["effort"][i].item() * cfg.safety.torque_limit
    return props


def _process_rigid_body_props(props, env_id):
    # randomize base mass
    if cfg.domain_rand.randomize_base_mass:
        rng = cfg.domain_rand.added_mass_range
        props[0].mass += np.random.uniform(rng[0], rng[1])

    return props


def sim_controll():
    dt = sim_params.dt
    asset = robot_asset

    # get array of DOF names
    dof_names = gym.get_asset_dof_names(asset)

    # get array of DOF properties
    dof_props = gym.get_asset_dof_properties(asset)

    # create an array of DOF states that will be used to update the actors
    num_dofs = gym.get_asset_dof_count(asset)
    dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)

    # get list of DOF types
    dof_types = [gym.get_asset_dof_type(asset, i) for i in range(num_dofs)]

    # get the position slice of the DOF state array
    dof_positions = dof_states["pos"]

    # get the limit-related slices of the DOF properties array
    stiffnesses = dof_props["stiffness"]
    dampings = dof_props["damping"]
    armatures = dof_props["armature"]
    has_limits = dof_props["hasLimits"]
    lower_limits = dof_props["lower"]
    upper_limits = dof_props["upper"]

    # initialize default positions, limits, and speeds (make sure they are in reasonable ranges)
    defaults = np.zeros(num_dofs)
    speeds = np.zeros(num_dofs)
    for i in range(num_dofs):
        if has_limits[i]:
            if dof_types[i] == gymapi.DOF_ROTATION:
                lower_limits[i] = clamp(lower_limits[i], -math.pi, math.pi)
                upper_limits[i] = clamp(upper_limits[i], -math.pi, math.pi)
            # make sure our default position is in range
            if lower_limits[i] > 0.0:
                defaults[i] = lower_limits[i]
            elif upper_limits[i] < 0.0:
                defaults[i] = upper_limits[i]
        else:
            # set reasonable animation limits for unlimited joints
            if dof_types[i] == gymapi.DOF_ROTATION:
                # unlimited revolute joint
                lower_limits[i] = -math.pi
                upper_limits[i] = math.pi
            elif dof_types[i] == gymapi.DOF_TRANSLATION:
                # unlimited prismatic joint
                lower_limits[i] = -1.0
                upper_limits[i] = 1.0
        # set DOF position to default
        dof_positions[i] = defaults[i]
        # set speed depending on DOF type and range of motion
        if dof_types[i] == gymapi.DOF_ROTATION:
            speeds[i] = args.speed_scale * clamp(2 * (upper_limits[i] - lower_limits[i]), 0.25 * math.pi, 3.0 * math.pi)
        else:
            speeds[i] = args.speed_scale * clamp(2 * (upper_limits[i] - lower_limits[i]), 0.1, 7.0)

    # Print DOF properties
    for i in range(num_dofs):
        print("DOF %d" % i)
        print("  Name:     '%s'" % dof_names[i])
        print("  Type:     %s" % gym.get_dof_type_string(dof_types[i]))
        print("  Stiffness:  %r" % stiffnesses[i])
        print("  Damping:  %r" % dampings[i])
        print("  Armature:  %r" % armatures[i])
        print("  Limited?  %r" % has_limits[i])
        if has_limits[i]:
            print("    Lower   %f" % lower_limits[i])
            print("    Upper   %f" % upper_limits[i])

    # set up the env grid
    num_envs = 1
    num_per_row = 1
    spacing = 1
    env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)

    # position the camera
    cam_pos = gymapi.Vec3(1.0, 1.0, 1.0)
    cam_target = gymapi.Vec3(0.0, 0.0, 1)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # cache useful handles
    envs = []
    actor_handles = []

    print("Creating %d environments" % num_envs)
    for i in range(num_envs):
        # create env
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        # add actor
        pose = gymapi.Transform()
        # pose.p = gymapi.Vec3(2.0, 2.32, 2.0)
        pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        actor_handle = gym.create_actor(env, asset, pose, "actor", i, 1)
        actor_handles.append(actor_handle)

        # set default DOF positions
        gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)

    # joint animation states
    ANIM_SEEK_LOWER = 1
    ANIM_SEEK_UPPER = 2
    ANIM_SEEK_DEFAULT = 3
    ANIM_FINISHED = 4

    # initialize animation state
    anim_state = ANIM_SEEK_LOWER
    current_dof = 0
    print("Animating DOF %d ('%s')" % (current_dof, dof_names[current_dof]))

    while not gym.query_viewer_has_closed(viewer):

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        speed = speeds[current_dof]

        # animate the dofs
        if anim_state == ANIM_SEEK_LOWER:
            dof_positions[current_dof] -= speed * dt
            if dof_positions[current_dof] <= lower_limits[current_dof]:
                dof_positions[current_dof] = lower_limits[current_dof]
                anim_state = ANIM_SEEK_UPPER
        elif anim_state == ANIM_SEEK_UPPER:
            dof_positions[current_dof] += speed * dt
            if dof_positions[current_dof] >= upper_limits[current_dof]:
                dof_positions[current_dof] = upper_limits[current_dof]
                anim_state = ANIM_SEEK_DEFAULT
        if anim_state == ANIM_SEEK_DEFAULT:
            dof_positions[current_dof] -= speed * dt
            if dof_positions[current_dof] <= defaults[current_dof]:
                dof_positions[current_dof] = defaults[current_dof]
                anim_state = ANIM_FINISHED
        elif anim_state == ANIM_FINISHED:
            dof_positions[current_dof] = defaults[current_dof]
            current_dof = (current_dof + 1) % num_dofs
            anim_state = ANIM_SEEK_LOWER
            print("Animating DOF %d ('%s')" % (current_dof, dof_names[current_dof]))

        if args.show_axis:
            gym.clear_lines(viewer)

        # clone actor state in all of the environments
        for i in range(num_envs):
            gym.set_actor_dof_states(envs[i], actor_handles[i], dof_states, gymapi.STATE_POS)

            if args.show_axis:
                # get the DOF frame (origin and axis)
                dof_handle = gym.get_actor_dof_handle(envs[i], actor_handles[i], current_dof)
                frame = gym.get_dof_frame(envs[i], dof_handle)

                # draw a line from DOF origin along the DOF axis
                p1 = frame.origin
                p2 = frame.origin + frame.axis * 0.7
                color = gymapi.Vec3(1.0, 0.0, 0.0)
                gymutil.draw_line(p1, p2, color, gym, viewer, envs[i])

        # update the viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        gym.sync_frame_time(sim)

    print("Done")

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)


if __name__ == "__main__":
    core = Core()
    core.init("dev")
    Logger.instance().info("********************************* Joint Controll *********************************")

    device = "cuda:0"

    cfg, train_cfg, sim_params, args = init_config()

    env_origins = _get_env_origins()
    gym, sim, viewer, robot_asset = create_sim()

    sim_controll()
