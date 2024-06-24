from turtle import pu
import torch

decimation = 7
print(f"decimation: {decimation}")
sim_dt = 0.002
print(f"sim_dt: {sim_dt}")

self_dt = decimation * sim_dt
print(f"self_dt: {self_dt}")

max_episode_length_s = 20
max_episode_length = int(max_episode_length_s / self_dt)
print(f"max_episode_length: {max_episode_length}")

push_interval_s = 4
push_interval = int(push_interval_s / self_dt)
print(f"push_interval: {push_interval}")

resampling_time = 8
resampling_time_1 = int(resampling_time / self_dt)
print(f"resampling_time_1: {resampling_time_1}")

dof_distance = 1
dof_vel = 1 / self_dt
print(f"dof_vel: {dof_vel}")

feet_air_time = 10
feet_air_time += self_dt
print(f"feet_air_time: {feet_air_time}")

cycle_time = 0.64
episode_length_buf = 1
phase = episode_length_buf * self_dt / cycle_time
print(f"phase: {phase}")

##########################################################################
# decimation: 10
# sim_dt: 0.01
# self_dt: 0.1
# max_episode_length: 200
# push_interval: 40
# resampling_time_1: 80
# dof_vel: 10.0
# feet_air_time: 10.1
# phase: 0.15625
##########################################################################
# decimation: 4
# sim_dt: 0.005
# self_dt: 0.02
# max_episode_length: 1000
# push_interval: 200
# resampling_time_1: 400
# dof_vel: 50.0
# feet_air_time: 10.02
# phase: 0.03125
##########################################################################
# decimation: 7
# sim_dt: 0.002
# self_dt: 0.014
# max_episode_length: 1428
# push_interval: 285
# resampling_time_1: 571
# dof_vel: 71.42857142857143
# feet_air_time: 10.014
# phase: 0.021875