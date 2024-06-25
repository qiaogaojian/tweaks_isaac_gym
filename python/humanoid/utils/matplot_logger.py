import datetime

import matplotlib
import matplotlib.pyplot as plt
from pycore.utils.file_utils import FileUtils

from humanoid.envs import PaiCfg

matplotlib.use('Agg')
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value
from PIL import Image
from datetime import datetime

class MatplotLogger:

    def __init__(self, dt, png_name="matplot", pai_config:PaiCfg=None):
        """
        绘制target和测量图表，成功日志报告
        """
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None
        self.png_name = png_name
        self.platforms = [
            {"key":"gym",   "color": "C0", "draw": False, "command_color":"C1"},
            {"key": "real", "color": "C2", "draw": False, "command_color":"C3"},
        ]
        self.pai_config = pai_config

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        self.plot_process = Process(target=self._plot)
        self.plot_process.start()

    def spawn_sub_pos(self, a, time, title, joint_id):
        log = self.state_log
        target = np.array(log["target"])
        # print(f"target shape: {target.shape}")
        a.plot(time, target[:, joint_id], label='target', color='C1') # 黄色

        for item in self.platforms:
            key = item["key"]
            if not item["draw"]: continue
            dof_pos = np.array(log[f"{key}_dof_pos"])
            a.plot(time, dof_pos[:, joint_id], label=key, color=item["color"])
        a.set(xlabel='time [s]', ylabel='Position [rad]', title=title)
        a.legend()
        return a
    
    def spawn_sub_vel(self, a, time, title, joint_id):
        log = self.state_log
        for item in self.platforms:
            key = item["key"]
            if not item["draw"]: continue
            dof_vel = np.array(log[f"{key}_dof_vel"])
            a.plot(time, dof_vel[:, joint_id], label=key)
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title=title)
        a.legend()
        return a
    
    def spawn_sub_tau(self, a, time, title, joint_id):
        log = self.state_log
        for item in self.platforms:
            key = item["key"]
            if not item["draw"]: continue
            torques = np.array(log[f"{key}_torques"])
            a.plot(time, torques[:, joint_id], label=key, color=item["color"])
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title=title)
        a.legend()
        return a

    def spawn_sub_tau_vel(self, a, time, title, joint_id):
        log = self.state_log
        for item in self.platforms:
            key = item["key"]
            if not item["draw"]: continue
            torques = np.array(log[f"{key}_torques"])
            dof_vel = np.array(log[f"{key}_dof_vel"])
            a.plot(dof_vel[:, joint_id], torques[:, joint_id], 'x', label='gym', color=item["color"])
        a.set(xlabel='Joint vel [rad/s]', ylabel='Joint Torque [Nm]', title=f'{title} Torque/velocity curves')
        a.legend()
        return a

    def spawn_sub_base_vel(self,a, time, title, id):
        log = self.state_log
        for item in self.platforms:
            key = item["key"]
            if not item["draw"]: continue
            command = np.array(log[f"{key}_command"])
            a.plot(time, command[:, id], label=f'{key}_command', color=item["command_color"])
            if id == 2: # yaw旋转
                base_ang_vel = np.array(log[f"{key}_base_ang_vel"])
                a.plot(time, base_ang_vel[:, id], label=key, color=item["color"])
                a.set(xlabel='time [s]', ylabel='base ang vel [rad/s]', title=title)
            else:
                base_line_vel = np.array(log[f"{key}_base_line_vel"])
                a.plot(time, base_line_vel[:, id], label=key, color=item["color"])
                a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title=title)

        a.legend()
        return a

    def create_group_pos(self, time, axs):
        a = axs[0, 0]
        self.spawn_sub_pos(a, time=time, title='L Yaw',joint_id=0)
        a = axs[0, 1]
        self.spawn_sub_pos(a, time=time, title='L Roll',joint_id=1)
        # Hip_Pitch
        a = axs[0, 2]
        self.spawn_sub_pos(a, time=time, title='L Pitch',joint_id=2)
        # knee 膝盖
        a = axs[0, 3]
        self.spawn_sub_pos(a, time=time, title='L Knee', joint_id=3)
        # Ankle_Pitch
        a = axs[0, 4]
        self.spawn_sub_pos(a, time=time, title='L Ankle_Pitch', joint_id=4)
        # Ankle_Roll
        a = axs[0, 5]
        self.spawn_sub_pos(a, time=time, title='L Ankle_Roll',  joint_id=5)

    def create_group_torque(self, time, axs):
        a = axs[1, 0]
        self.spawn_sub_tau(a, time=time, title='L Yaw', joint_id=0)
        a = axs[1, 1]
        self.spawn_sub_tau(a, time=time, title='L Roll', joint_id=1)
        # Hip_Pitch
        a = axs[1, 2]
        self.spawn_sub_tau(a, time=time, title='L Pitch', joint_id=2)
        # knee 膝盖
        a = axs[1, 3]
        self.spawn_sub_tau(a, time=time, title='L Knee', joint_id=3)
        # Ankle_Pitch
        a = axs[1, 4]
        self.spawn_sub_tau(a, time=time, title='L Ankle_Pitch', joint_id=4)
        # Ankle_Roll
        a = axs[1, 5]
        self.spawn_sub_tau(a, time=time, title='L Ankle_Roll', joint_id=5)

    def create_group_tau_vel(self, time, axs):
        a = axs[2, 0]
        self.spawn_sub_tau_vel(a, time=time, title='L Yaw', joint_id=0)
        a = axs[2, 1]
        self.spawn_sub_tau_vel(a, time=time, title='L Roll', joint_id=1)
        # Hip_Pitch
        a = axs[2, 2]
        self.spawn_sub_tau_vel(a, time=time, title='L Pitch', joint_id=2)
        # knee 膝盖
        a = axs[2, 3]
        self.spawn_sub_tau_vel(a, time=time, title='L Knee', joint_id=3)
        # Ankle_Pitch
        a = axs[2, 4]
        self.spawn_sub_tau_vel(a, time=time, title='L Ankle_Pitch', joint_id=4)
        # Ankle_Roll
        a = axs[2, 5]
        self.spawn_sub_tau_vel(a, time=time, title='L Ankle_Roll', joint_id=5)

    def check_draw(self):
        for item in  self.platforms:
            platform = item["key"]
            if f"{platform}_dof_pos" in  self.state_log:
                item["draw"] = True
        print(f"{self.platforms}")


    def _plot(self):
        self.check_draw()
        nb_rows = 4
        nb_cols = 6
        # 设置画布大小为3440 1440
        fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(34.4, 20))
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break
        log = self.state_log

        # ************************  第一行 pos **************************
        self.create_group_pos(time, axs)

        # ************************  第二行 torque **************************
        self.create_group_torque(time, axs)

        # ************************  第三行 torque/vel curves **************************
        self.create_group_tau_vel(time, axs)

        #  *************************  第四行 base line  **********************
        # plot contact forces
        a = axs[3, 0]
        if log["gym_contact_forces_z"]:
            forces = np.array(log["gym_contact_forces_z"])
            print(f"forces shape: {forces.shape}")
            for i in range(forces.shape[1]):
                a.plot(time, forces[:,i], label=f'force {i}')
        a.set(xlabel='time [s]', ylabel='Forces z [N]', title='Vertical Contact forces')
        a.legend()
        # plot base vel x
        a = axs[3, 1]
        self.spawn_sub_base_vel(a, time, title="Base velocity x", id=0)
        # plot base vel y
        a = axs[3, 2]
        self.spawn_sub_base_vel(a, time, title="Base velocity y", id=1)
        # plot base vel yaw
        a = axs[3, 3]
        self.spawn_sub_base_vel(a, time, title="Base velocity yaw", id=2)

        # plot base vel z
        a = axs[3, 4]
        if log["gym_base_line_vel"]:
            gym_base_line_vel = np.array(log["gym_base_line_vel"])
            a.plot(time, gym_base_line_vel[:,2], label='gym')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity z')
        a.legend()
    
        # pitch velocity
        a = axs[3, 5]
        self.spawn_sub_vel(a, time, title="Pitch Velocity", joint_id=2)
        # plt.show()

        time_str = datetime.now().strftime('%m%d_%H-%M-%S')
        png_file = FileUtils.get_project_path() + f"videos/Pai_ppo/benchmark_{self.png_name}_{time_str}.png"
        log_file = FileUtils.get_project_path() + f"videos/Pai_ppo/benchmark_{self.png_name}_{time_str}.log"
        self.print_rate_log(log_file)
        # 保存图像到文件
        plt.savefig(png_file)
        print(f"save benchmark png: {png_file}")
        # 使用pillow库读取并显示图像
        image = Image.open(png_file)
        image.show()

    def print_rate_log(self, log_file):
        for item in self.platforms:
            key = item["key"]
            if not item["draw"]: continue
            with open(log_file, 'w', encoding='utf-8') as f:
                self._write_log_line(f, f"***** {key} *****")
                # 配置信息
                if self.pai_config:
                    self._write_log_line(f, f"kp: {self.pai_config.control.stiffness}")
                    self._write_log_line(f, f"kd: {self.pai_config.control.damping}")
                    self._write_log_line(f, f"actor stiffness: {self.pai_config.asset.stiffness}")
                    self._write_log_line(f, f"actor damping:   {self.pai_config.asset.damping}")
                    self._write_log_line(f, f"height: {self.pai_config.init_state.pos}")
                # 各个姿态的达成百分比
                target = np.array(self.state_log["target"]).astype("float32")
                dof_pos = np.array(self.state_log[f"{key}_dof_pos"]).astype("float32")
                # # 计算 b 除以 a 的结果
                # result = dof_pos / target
                # result = np.where(result == np.inf, 1, result)
                # print(result)
                # # 计算列方向上的平均值
                # column_means = np.mean(result, axis=0)
                # print(f"column_means: {column_means}")
                for i in range(12):
                    target_max = np.max(target[:,i])
                    dof_max = np.max(dof_pos[:,i])
                    max_rate = 0
                    if target_max > 0:
                        max_rate = dof_max * 100 / target_max
                    target_min = np.min(target[:,i])
                    dof_min = np.min(dof_pos[:,i])
                    min_rate = 0
                    if target_min < 0 :
                        min_rate = dof_min*100 / target_min
                    self._write_log_line(f, f"关节pos：{i}, target_max: {target_max:.5f}， dof_max: {dof_max:.5f},max_rate:{max_rate:.2f}%, "
                                            f"target_min: {target_min:.5f}， dof_min: {dof_min:.5f} minrate:{min_rate:.2f}%")


    def _write_log_line(self, f, str):
        print(str)
        f.write(f"{str}\n")


    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")

    def __del__(self):
        pass
        # if self.plot_process is not None:
        #     self.plot_process.kill()