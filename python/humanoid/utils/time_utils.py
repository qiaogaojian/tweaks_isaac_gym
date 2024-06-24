import time


def get_timestamp():
    timestamp_ns = time.time_ns()
    # 转换为毫秒
    timestamp_ms = timestamp_ns / 1_000_000
    return timestamp_ms
