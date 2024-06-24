import numpy as np
import stl
from humanoid import LEGGED_GYM_ROOT_DIR

file_path = LEGGED_GYM_ROOT_DIR + '/resources/robots/pai_12dof/meshes/base_link.STL'
# 给定质量
mass = 2.4140796

# 加载 STL 文件
mesh = stl.mesh.Mesh.from_file(file_path)

# 计算质心
volume, center_of_mass, inertia = mesh.get_mass_properties()
inertia /= volume  # The default inertia is multiplied by the volume, so we undo that.
inertia *= mass

# 计算质心
centroid = np.mean(mesh.vectors.reshape(-1, 3), axis=0)
print("质心:")
print(centroid)

# 初始化惯量张量
inertia_tensor = np.zeros((3, 3))

# 遍历每个三角形面
for i in range(len(mesh.vectors)):
    vertices = mesh.vectors[i]
    # 计算三角形的面积和相应的质量
    triangle_area = 0.5 * np.linalg.norm(np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0]))
    triangle_mass = (triangle_area / mesh.areas.sum()) * mass
    
    # 三角形的质心
    triangle_centroid = np.mean(vertices, axis=0)
    
    # 计算每个三角形的惯量张量（平行轴定理应用之前）
    triangle_inertia = np.zeros((3, 3))
    for j in range(3):
        for k in range(3):
            if j == k:
                triangle_inertia[j, k] = (triangle_mass / 6) * (
                    (vertices[0][(j + 1) % 3] ** 2 + vertices[0][(j + 2) % 3] ** 2 +
                     vertices[1][(j + 1) % 3] ** 2 + vertices[1][(j + 2) % 3] ** 2 +
                     vertices[2][(j + 1) % 3] ** 2 + vertices[2][(j + 2) % 3] ** 2))
            else:
                triangle_inertia[j, k] = (triangle_mass / 12) * (
                    vertices[0][j] * vertices[0][k] + vertices[1][j] * vertices[1][k] +
                    vertices[2][j] * vertices[2][k] + vertices[0][j] * vertices[1][k] +
                    vertices[1][j] * vertices[0][k] + vertices[1][j] * vertices[2][k] +
                    vertices[2][j] * vertices[1][k] + vertices[2][j] * vertices[0][k] +
                    vertices[0][j] * vertices[2][k] + vertices[2][j] * vertices[0][k])
    
    # 平行轴定理将三角形的惯量张量转移到全局坐标系原点
    distance_to_centroid = triangle_centroid - centroid
    inertia_tensor += triangle_inertia + triangle_mass * (
        np.dot(distance_to_centroid, distance_to_centroid) * np.eye(3) - np.outer(distance_to_centroid, distance_to_centroid)
    )

# print("惯量张量相对于原点:")
# print(inertia_tensor)

# 打印各个元素
ixx = inertia_tensor[0, 0]
ixy = inertia_tensor[0, 1]
ixz = inertia_tensor[0, 2]
iyy = inertia_tensor[1, 1]
iyz = inertia_tensor[1, 2]
izz = inertia_tensor[2, 2]

print("惯量张量:")
print(f'ixx="{ixx}"')
print(f'ixy="{ixy}"')
print(f'ixz="{ixz}"')
print(f'iyy="{iyy}"')
print(f'iyz="{iyz}"')
print(f'izz="{izz}"')