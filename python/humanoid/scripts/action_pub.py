#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32MultiArray
import pandas as pd
import numpy as np

def sim_action():

    csv_path = '/home/mega/ocs2_ws/src/livelybot_rl_control/test/action_queue.csv'

    rospy.init_node('sim_action_pub', anonymous=True)
    pub = rospy.Publisher('/model/reference/results', Float32MultiArray, queue_size=10)
    rate = rospy.Rate(100)  # 100 Hz

    # 读取 CSV 文件
    df = pd.read_csv(csv_path, header=None)

    # 将 DataFrame 转换为 NumPy 数组
    data_array = df.values

    # 如果需要将数据转换为浮点数类型，可以使用 astype
    data_array = data_array.astype(float)

    # 转换为 Python 列表
    data_list = data_array.tolist()
    
    for row in data_list:
        # print(row)
# for num in range(0, 1000):
        # 创建并填充 Float32MultiArray 消息
        msg = Float32MultiArray()
        # msg.data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 这里可以填充你需要的数据
        msg.data = row

        # 发布消息
        rospy.loginfo("Publishing: %s", msg.data)
        pub.publish(msg)

        # 睡眠以保持发布频率
        rate.sleep()

if __name__ == '__main__':
    try:
        sim_action()
    except rospy.ROSInterruptException:
        pass
