import os
from datetime import datetime

from pycore.utils.file_utils import FileUtils
import cv2

def create_bench_video():
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    time_str = datetime.now().strftime('%m%d_%H-%M-%S')
    dir = FileUtils.get_project_path() + f"/videos/Pai_ppo/benchmark_gym_{time_str}.mp4"
    video = cv2.VideoWriter(dir, fourcc, 50.0, (1920, 1080))
    return video

def release_video(video):
    video.release()
