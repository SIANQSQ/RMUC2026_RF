import numpy as np
from gnuradio import gr
import os

class ota_frame_generator(gr.sync_block):
    """
    自定义空口帧生成块
    生成固定结构：[8字节Access Code][4字节双份Header][15字节Payload]
    """
    def __init__(self, Interface_KEY = "123456"):
        self.log("Start Here")
        """
        参数:
            access_code_choice: 选择Access Code类型，"信息波"或"干扰波"
            payload_data: 15字节的Payload数据（bytes类型），默认全0x00
        """
        gr.sync_block.__init__(
            self,
            name="Interference wave Generator",  
            in_sig=None,  
            out_sig=[np.uint8]  
        )

        # Access Code
        # 干扰波 Access Code: 0x16E8D377151C712D（大端序）
        self.access_code_jam = bytes([
            0x16, 0xE8, 0xD3, 0x77,
            0x15, 0x1C, 0x71, 0x2D
        ])

        # Header 
        # 固定 0x00 0x0F 0x00 0x0F
        self.header = bytes([0x00, 0x0F, 0x00, 0x0F])

        # # Payload
        # if payload_data is None:
        #     # 默认Payload：15字节全0x00（干扰数据）
        #     self.payload = bytes([0x00] * 15)
        # else:
        #     # 确保Payload是15字节
        #     if len(payload_data) != 15:
        #         raise ValueError("Payload必须是15字节！")
        #     self.payload = payload_data

        # -------------------------- 4. 组装完整帧 --------------------------
        # self.frame = self.access_code + self.header + self.payload
        # self.frame_len = len(self.frame)  # 固定27字节

        # # 用于循环输出的索引
        # self.current_idx = 0

    def generate_9byte_random_data(self):
        random_fill = os.urandom(9)
        return random_fill
    
    def log(self,str):
        print("[Python Block Debug]:"+str)
        
    def work(self, input_items, output_items):
        out = output_items[0]
        out_len = len(out)

        # 循环将帧数据填充到输出缓冲区
        bytes_written = 0
        print(self.generate_9byte_random_data)
        # while bytes_written < out_len:
        #     # 计算本次可写入的字节数
        #     remaining_in_frame = self.frame_len - self.current_idx
        #     remaining_in_out = out_len - bytes_written
        #     write_len = min(remaining_in_frame, remaining_in_out)

        #     # 复制数据到输出
        #     out[bytes_written:bytes_written+write_len] = np.frombuffer(
        #         self.frame[self.current_idx:self.current_idx+write_len],
        #         dtype=np.uint8
        #     )

        #     # 更新索引
        #     self.current_idx += write_len
        #     bytes_written += write_len

        #     # 如果一帧写完，重置索引，循环发送
        #     if self.current_idx >= self.frame_len:
        #         self.current_idx = 0

        # return bytes_written
