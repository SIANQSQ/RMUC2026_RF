#!/usr/bin/env python3
import sys
import random
import string
import time

# ---------- 空口帧配置 ----------
# 空口帧_帧头：Access Code (8 Bytes) + Length (2 Bytes) + Length (2 Bytes)
ACCESS_CODE   = bytes([0x16, 0xE8, 0xD3, 0x77, 0x15, 0x1C, 0x71, 0x2D])    # 同步头 0x16E8D377151C712D
LENGTH_CHECK   = bytes([0x00, 0x0F, 0x00, 0x0F])                           # 空口帧长度校验 000F，000F  接收时校验二者是否一致，不一致则丢包
# ---------- Payload配置 ----------
# 空口帧_Payload([干扰密钥 6 Bytes] + 填充随机字节)(总推送速率 1350 Bytes/s, 密钥信息流 10 Hz)

# 干扰密钥数据包
SOF           = 0xA5                             # 帧起始
CMD_ID        = bytes([0x0A, 0x06])              # 命令字
DATALENGTH    = 6

# 速率控制
TARGET_BYTES_PER_SEC = 1350
PACKET_FREQ_HZ       = 10
CYCLE_TIME           = 1.0 / PACKET_FREQ_HZ       # 0.1 s
BYTES_PER_CYCLE      = TARGET_BYTES_PER_SEC // PACKET_FREQ_HZ  # 135

# ---------- CRC 实现 ----------
def crc8_atm(data: bytes) -> int:
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

def crc16_ccitt(data: bytes) -> int:
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

# ---------- 数据生成 ----------
def generate_ascii_bytes(length: int) -> bytes:
    """生成 length 字节随机可打印 ASCII (0x20-0x7E)"""
    chars = string.printable[:95]
    return ''.join(random.choices(chars, k=length)).encode('ascii')

def build_structured_frame(seq: int) -> bytes:
    # 数据区（保证6字节，不足补随机）
    data = generate_ascii_bytes(6)

    # 帧头前4字节：SOF + datalength + seq
    datalength_bytes = bytes([(DATALENGTH >> 8) & 0xFF, DATALENGTH & 0xFF])
    header_prefix = bytes([SOF]) + datalength_bytes + bytes([seq])

    # CRC8 保护 header 前4字节
    crc8 = crc8_atm(header_prefix)
    frame_header = header_prefix + bytes([crc8])

    # CRC16 整包校验 (header + cmd_id + data)
    crc_input = frame_header + CMD_ID + data
    crc16_val = crc16_ccitt(crc_input)
    crc16_bytes = bytes([(crc16_val >> 8) & 0xFF, crc16_val & 0xFF])

    return crc_input + crc16_bytes

def format_hex(data: bytes) -> str:
    """将字节串转换为 '0xHH 0xHH ...' 格式的字符串"""
    return ' '.join(f'0x{byte:02X}' for byte in data)

# ---------- 主循环 ----------
def main():
    # 计算帧长和填充量
    structured_len = 5 + len(CMD_ID) + DATALENGTH + 2  # 15
    frame_len = len(ACCESS_CODE) + len(LENGTH_CHECK) + structured_len
    pad_len = BYTES_PER_CYCLE - frame_len
    if pad_len < 0:
        raise ValueError(f"帧过长! frame_len={frame_len} > 每周期字节数{BYTES_PER_CYCLE}")

    seq = 0
    print(f"ACCESS_CODE: {ACCESS_CODE.hex().upper()}", file=sys.stderr)
    print(f"帧长: {frame_len} B, 填充: {pad_len} B, 周期: {CYCLE_TIME} s", file=sys.stderr)

    try:
        while True:
            t_start = time.perf_counter()

            # 1. 构造数据包
            structured = build_structured_frame(seq)
            packet = ACCESS_CODE + LENGTH_CHECK + structured
            padding = generate_ascii_bytes(pad_len) if pad_len > 0 else b''

            # 2. 打印十六进制格式（每个包一行）
            hex_line = format_hex(packet)
            if padding:
                hex_line += ' ' + format_hex(padding)
            print(hex_line)
            sys.stdout.flush()

            # 3. 等待至周期结束
            seq = (seq + 1) % 256
            elapsed = time.perf_counter() - t_start
            sleep_time = CYCLE_TIME - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        # out = sys.stdout.buffer  # 使用二进制输出通道
        # while True:
        #     t_start = time.perf_counter()

        #     # 1. 发送一个完整的数据包
        #     structured = build_structured_frame(seq)
        #     packet = ACCESS_CODE + LENGTH_CHECK + structured
        #     out.write(packet)

        #     # 2. 发送填充字节 (随机 ASCII)
        #     if pad_len > 0:
        #         padding = generate_ascii_bytes(pad_len)
        #         out.write(padding)

        #     out.flush()

        #     # 3. 等待至本周期结束，保证恒定时钟
        #     seq = (seq + 1) % 256
        #     elapsed = time.perf_counter() - t_start
        #     sleep_time = CYCLE_TIME - elapsed
        #     if sleep_time > 0:
        #         time.sleep(sleep_time)

    except AttributeError:
        print("错误：请在管道或重定向下运行本脚本。", file=sys.stderr)
        print("示例：python script.py | xxd  或 > stream.bin", file=sys.stderr)
    except KeyboardInterrupt:
        print("\n输出已停止。", file=sys.stderr)
    except BrokenPipeError:
        pass

if __name__ == "__main__":
    main()