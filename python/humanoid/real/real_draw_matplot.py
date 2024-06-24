from humanoid.envs import *
import argparse
import pandas as pd
from pycore.utils.file_utils import FileUtils

from humanoid.utils.matplot_logger import MatplotLogger


def load_benchmark_target(matplot_logger):
    target_csv = FileUtils.get_project_path() + "/test/csv/action_benchmark.csv"
    df = pd.read_csv(target_csv)
    np_array = df.values
    for row in np_array:
        matplot_logger.log_states(
            {
                "target": row * 0.25,
            }
        )
    print(f"load action_benchmark success: {target_csv}")


def draw_with_csv(args, matplot_logger):
    real_csv = args.real_csv
    df = pd.read_csv(real_csv)
    print(f"load real_csv success: {real_csv}")
    np_array = df.values
    # shape(641x 47)
    for i in range(640):
        row = np_array[i]
        matplot_logger.log_states(
            {
                "real_dof_pos": row[5:17],
                "real_dof_vel": row[17:29],
                "real_torques": [0,0,0,0,0,0,0,0,0,0,0,0], # 默认0
                'real_command': row[2:5],
                'real_base_ang_vel': row[41:44],
                'real_base_line_vel': [0,0,0], # base的线速度默认0
                'real_contact_forces_z': [0,0] # 脚步压力默认0
            }
        )
    # 生成图表
    matplot_logger.plot_states()

if __name__ == '__main__':
    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser(description='A simple command-line tool')
    # 添加命令行参数
    parser.add_argument('--real_csv', type=str,default=None, help='Your name')

    # 解析命令行参数
    args = parser.parse_args()

    matplot_logger = MatplotLogger(dt=0.01, png_name="real", pai_config=None)
    load_benchmark_target(matplot_logger)

    if args.real_csv is None:
        args.real_csv = FileUtils.get_project_path() + "/test/csv/q20ms.csv"
    draw_with_csv(args, matplot_logger)
