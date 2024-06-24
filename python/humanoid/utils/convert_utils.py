import numpy as np


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
