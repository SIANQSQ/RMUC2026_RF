#!/usr/bin/env python3
import numpy as np
from gnuradio import gr
import random
import string


# 干扰波数据生成块 (密钥+随机填充)
class gfsk_data_source(gr.sync_block):
    def __init__(self):
        gr.sync_block.__init__(self, name="GFSK Data Source", in_sig=None, out_sig=[np.uint8])
        
        # 帧结构参数
        self.ACCESS_CODE = bytes([0x16, 0xE8, 0xD3, 0x77, 0x15, 0x1C, 0x71, 0x2D])
        self.LENGTH_CHECK = bytes([0x00, 0x0F, 0x00, 0x0F])
        self.SOF = 0xA5
        self.CMD_ID = bytes([0x0A, 0x06])
        self.DATALENGTH = 6
        self.TARGET_BYTES_PER_SEC = 1350                                                                      # 干扰输出:1350 Bytes/s
        self.PACKET_FREQ_HZ = 10                                                                              # 数据包体推送速率: 10Hz
        self.CYCLE_TIME = 1.0 / self.PACKET_FREQ_HZ
        self.BYTES_PER_CYCLE = self.TARGET_BYTES_PER_SEC // self.PACKET_FREQ_HZ
        
        # 计算帧长度
        self.data_frame_len = 5 + len(self.CMD_ID) + self.DATALENGTH + 2                                      # 数据帧 5 + 6 + 2 + 2 = 15
        self.air_frame_len = len(self.ACCESS_CODE) + len(self.LENGTH_CHECK) + self.data_frame_len             # 空口帧 8 + 4 + 15 = 27
        self.pad_len = self.BYTES_PER_CYCLE - self.air_frame_len                                              # 填充长度 135 - 27 = 108
        self.seq = 0                                                                                          # 包序号 0-255循环

        self._packet_buffer = bytearray()
        print(f"Packet length (bytes): {self.air_frame_len + self.pad_len}")

    def crc8_atm(self, data: bytes) -> int:
        poly = 0x07
        crc = 0x00
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x80:
                    crc = ((crc << 1) ^ poly) & 0xFF
                else:
                    crc = (crc << 1) & 0xFF
        return crc

    def crc16_ccitt(self, data: bytes) -> int:
        poly = 0x1021
        crc = 0xFFFF
        for byte in data:
            crc ^= (byte << 8)
            for _ in range(8):
                if crc & 0x8000:
                    crc = ((crc << 1) ^ poly) & 0xFFFF
                else:
                    crc = (crc << 1) & 0xFFFF
        return crc

    def generate_ascii_bytes(self, length: int) -> bytes:
        chars = string.printable[:95]
        return ''.join(random.choices(chars, k=length)).encode('ascii')

    def build_structured_frame(self, seq: int) -> bytes:
        data = b"rmnb66"
        datalength_bytes = bytes([(self.DATALENGTH >> 8) & 0xFF, self.DATALENGTH & 0xFF])
        header_prefix = bytes([self.SOF]) + datalength_bytes + bytes([seq])
        crc8 = self.crc8_atm(header_prefix)
        frame_header = header_prefix + bytes([crc8])
        crc_input = frame_header + self.CMD_ID + data
        crc16_val = self.crc16_ccitt(crc_input)
        crc16_bytes = bytes([(crc16_val >> 8) & 0xFF, crc16_val & 0xFF])
        return crc_input + crc16_bytes

    def next_packet(self):  #生成完整数据包
        structured = self.build_structured_frame(self.seq)
        packet = self.ACCESS_CODE + self.LENGTH_CHECK + structured
        padding = self.generate_ascii_bytes(self.pad_len) if self.pad_len > 0 else b''
        self.seq = (self.seq + 1) % 256
        return packet + padding

    def work(self, input_items, output_items):
        out = output_items[0]
        max_output = len(out)
        bytes_written = 0

        # 缓存为空时，生成新数据包
        if len(self._packet_buffer) == 0:
            self._packet_buffer = bytearray(self.next_packet())

        send_len = min(len(self._packet_buffer), max_output)
        if send_len > 0:
            # 复制缓存数据到输出
            out[:send_len] = np.frombuffer(self._packet_buffer[:send_len], dtype=np.uint8)
            # 移除已发送的数据
            del self._packet_buffer[:send_len]
            bytes_written = send_len

        return bytes_written

