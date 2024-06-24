# http://wiki.ros.org/urdf/Tutorials/Adding%20Physical%20and%20Collision%20Properties%20to%20a%20URDF%20Model
# *ixx* *ixy* *ixz*
# ixy   *iyy* *iyz*
# ixz    iyz  *izz*

import numpy as np

# 给定参数
mass = 2.4140796

width = 0.08
height = 0.08
depth = 0.08

# 质心位置
cx, cy, cz = {0.036389, -1.95E-05, -0.0235966}

# 计算惯性矩
Ixx = (1 / 12) * mass * (height ** 2 + depth ** 2)
Iyy = (1 / 12) * mass * (width ** 2 + height ** 2)
Izz = (1 / 12) * mass * (width ** 2 + depth ** 2)

Ixy = 0
Ixz = 0
Iyz = 0

inertia_tensor = {
    'ixx': Ixx,
    'iyy': Iyy,
    'izz': Izz,
    'ixy': Ixy,
    'ixz': Ixz,
    'iyz': Iyz
}

# 打印结果
for key, value in inertia_tensor.items():
    print(f"{key}: {value:.8f}")
