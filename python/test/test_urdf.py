import os
import urdf_parser_py.urdf

def parse_urdf(urdf_file_path):
    with open(urdf_file_path, 'r') as urdf_file:
        urdf_data = urdf_file.read()
    robot = urdf_parser_py.urdf.URDF.from_xml_string(urdf_data)
    return robot

def get_link_extents(robot):
    min_z = float('inf')
    max_z = float('-inf')
    
    for link in robot.links:
        if link.visuals:
            for visual in link.visuals:
                print(visual)
                if hasattr(visual.geometry, 'box') and visual.geometry.box:
                    size = visual.geometry.box.size
                    min_z = min(min_z, visual.origin.position[2] - size[2]/2)
                    max_z = max(max_z, visual.origin.position[2] + size[2]/2)
                elif hasattr(visual.geometry, 'cylinder') and visual.geometry.cylinder:
                    height = visual.geometry.cylinder.length
                    min_z = min(min_z, visual.origin.position[2] - height/2)
                    max_z = max(max_z, visual.origin.position[2] + height/2)
                elif hasattr(visual.geometry, 'sphere') and visual.geometry.sphere:
                    radius = visual.geometry.sphere.radius
                    min_z = min(min_z, visual.origin.position[2] - radius)
                    max_z = max(max_z, visual.origin.position[2] + radius)
                elif hasattr(visual.geometry, 'mesh') and visual.geometry.mesh:
                    # Assuming the mesh origin gives a representative height
                    # This is a simplification; actual mesh processing would be more complex
                    min_z = min(min_z, visual.origin.position[2])
                    max_z = max(max_z, visual.origin.position[2])
                else:
                    # print(f"Unknown geometry type in link {link.name}")
                    pass
    return min_z, max_z

def calculate_robot_height(urdf_file_path):
    robot = parse_urdf(urdf_file_path)
    min_z, max_z = get_link_extents(robot)
    height = max_z - min_z
    return height

urdf_file_path = '/home/mega/ocs2_ws/src/livelybot_rl_control/resources/robots/pai_12dof/urdf/pai_12dof.urdf'
robot_height = calculate_robot_height(urdf_file_path)
print(f'The height of the robot is: {robot_height}')
