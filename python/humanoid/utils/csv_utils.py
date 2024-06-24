import csv
import os
import time

class CSVWriter:
    def __init__(self, base_file_name):
        """
        初始化CSVWriter对象
        :param base_file_name: CSV文件的基础名称（不包含扩展名）
        """
        # 获取当前时间并格式化为字符串
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        # 生成包含时间戳的文件名
        self.file_path = f"{base_file_name}_{timestamp}.csv"
        self._file_exists = os.path.isfile(self.file_path)
    
    def _generate_header(self, data_length):
        """
        根据数据长度生成表头
        :param data_length: 数据的长度
        :return: 生成的表头（列表形式）
        """
        return [f'Column{i+1}' for i in range(data_length)]
    
    def _write_header(self, header):
        """
        写入表头
        :param header: 表头（列表形式）
        """
        with open(self.file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
    
    def write_row(self, data):
        """
        写入一行数据
        :param data: 要写入的数据（列表形式）
        """
        if not self._file_exists:
            # 如果文件不存在，根据数据的长度生成表头并写入
            header = self._generate_header(len(data))
            self._write_header(header)
            self._file_exists = True  # 更新文件存在的状态
        
        with open(self.file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)

def example():
    # 示例数据
    data1 = [1, 2, 3]
    data2 = [4, 5, 6]
    data3 = [7, 8, 9]
    
    # CSV文件基础名称
    base_file_name = 'output'
    
    # 创建CSVWriter对象
    csv_writer = CSVWriter(base_file_name)
    
    # 将数据增量写入CSV文件
    csv_writer.write_row(data1)
    csv_writer.write_row(data2)
    csv_writer.write_row(data3)
    
    print(f"数据已成功写入到 {csv_writer.file_path}")

if __name__ == "__main__":
    example()
