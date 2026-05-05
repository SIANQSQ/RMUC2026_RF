#!/usr/bin/env python3
import time
import os
import numpy as np
from gnuradio import gr


class gfsk_data_source(gr.sync_block):
    """
    GFSK Data Source

    逻辑说明：

    1. 原始 Payload 数据流速率为 1350 byte/s。
    2. Payload 按 15 字节为一组切片。
    3. 每个 15 字节 Payload 被封装为空口包：

        ACCESS_CODE + 00 0F 00 0F + 15-byte Payload

    4. Payload 数据流中包含：
        - 0x0A06 结构化数据帧，10Hz，每帧 15 字节
        - 其余字节为随机填充

    5. 因为每个 15-byte Payload 都会增加 12-byte 空口头，
       所以封装后的实际输出速率为：

        Payload速率: 1350 byte/s
        空口输出速率: 1350 / 15 * 27 = 2430 byte/s
    """

    def __init__(self, enable_internal_timing=True):
        gr.sync_block.__init__(
            self,
            name="GFSK Data Source",
            in_sig=None,
            out_sig=[np.uint8]
        )

        # =========================
        # 空口包参数
        # =========================
        self.ACCESS_CODE = bytes([
            0x16, 0xE8, 0xD3, 0x77,
            0x15, 0x1C, 0x71, 0x2D
        ])

        # Header / Length Check: 00 0F 00 0F
        self.HEADER = bytes([0x00, 0x0F, 0x00, 0x0F])

        # 每个 Payload 分片长度
        self.PAYLOAD_SLICE_LEN = 15

        # 单个空口包长度：8 + 4 + 15 = 27 bytes
        self.AIR_PACKET_LEN = (
            len(self.ACCESS_CODE)
            + len(self.HEADER)
            + self.PAYLOAD_SLICE_LEN
        )

        # =========================
        # Payload 数据流参数
        # =========================
        # 原始 Payload 数据流速率：1350 byte/s
        self.TARGET_PAYLOAD_BYTES_PER_SEC = 1350

        # 0x0A06 结构化数据帧发送频率：10Hz
        self.STRUCTURED_FRAME_FREQ_HZ = 10

        # 每个 10Hz 周期时间：0.1s
        self.CYCLE_TIME = 1.0 / self.STRUCTURED_FRAME_FREQ_HZ

        # 每个周期内的 Payload 字节数：1350 / 10 = 135 bytes
        self.PAYLOAD_BYTES_PER_CYCLE = (
            self.TARGET_PAYLOAD_BYTES_PER_SEC
            // self.STRUCTURED_FRAME_FREQ_HZ
        )

        # 每个周期内的 Payload 分片数量：135 / 15 = 9
        self.SLICES_PER_CYCLE = (
            self.PAYLOAD_BYTES_PER_CYCLE
            // self.PAYLOAD_SLICE_LEN
        )

        # 每个周期内随机填充字节数：135 - 15 = 120 bytes
        self.RANDOM_BYTES_PER_CYCLE = (
            self.PAYLOAD_BYTES_PER_CYCLE
            - self.PAYLOAD_SLICE_LEN
        )

        # 每个周期输出的空口字节数：9 * 27 = 243 bytes
        self.AIR_BYTES_PER_CYCLE = (
            self.SLICES_PER_CYCLE
            * self.AIR_PACKET_LEN
        )

        # 封装后的实际空口输出速率：243 * 10 = 2430 byte/s
        self.TARGET_AIR_BYTES_PER_SEC = (
            self.AIR_BYTES_PER_CYCLE
            * self.STRUCTURED_FRAME_FREQ_HZ
        )

        # =========================
        # 0x0A06 结构化数据帧参数
        # =========================
        self.SOF = 0xA5
        self.CMD_ID = bytes([0x0A, 0x06])
        self.DATALENGTH = 6
        self.DATA = b"rmnb66"

        # 数据帧结构：
        # SOF          1 byte
        # DATALENGTH   2 bytes
        # SEQ          1 byte
        # CRC8         1 byte
        # CMD_ID       2 bytes
        # DATA         6 bytes
        # CRC16        2 bytes
        #
        # 总长度：1 + 2 + 1 + 1 + 2 + 6 + 2 = 15 bytes
        self.STRUCTURED_FRAME_LEN = (
            1 + 2 + 1 + 1 + len(self.CMD_ID) + self.DATALENGTH + 2
        )

        if self.STRUCTURED_FRAME_LEN != self.PAYLOAD_SLICE_LEN:
            raise ValueError(
                f"Structured frame length must be {self.PAYLOAD_SLICE_LEN} bytes, "
                f"but got {self.STRUCTURED_FRAME_LEN}"
            )

        if self.PAYLOAD_BYTES_PER_CYCLE % self.PAYLOAD_SLICE_LEN != 0:
            raise ValueError(
                "PAYLOAD_BYTES_PER_CYCLE must be divisible by PAYLOAD_SLICE_LEN"
            )

        if self.RANDOM_BYTES_PER_CYCLE < 0:
            raise ValueError(
                "Payload bytes per cycle is smaller than one structured frame"
            )

        # 包序号
        self.seq = 0

        # 输出缓存
        self._packet_buffer = bytearray()

        # 是否在 block 内部进行 10Hz 节拍控制
        self.enable_internal_timing = enable_internal_timing
        self._next_cycle_time = time.monotonic()

        print("========== GFSK Data Source Config ==========")
        print(f"ACCESS_CODE length            : {len(self.ACCESS_CODE)} bytes")
        print(f"HEADER length                 : {len(self.HEADER)} bytes")
        print(f"Payload slice length          : {self.PAYLOAD_SLICE_LEN} bytes")
        print(f"Air packet length             : {self.AIR_PACKET_LEN} bytes")
        print(f"Payload rate                  : {self.TARGET_PAYLOAD_BYTES_PER_SEC} byte/s")
        print(f"Structured 0x0A06 frame rate  : {self.STRUCTURED_FRAME_FREQ_HZ} Hz")
        print(f"Payload bytes per cycle       : {self.PAYLOAD_BYTES_PER_CYCLE} bytes")
        print(f"Payload slices per cycle      : {self.SLICES_PER_CYCLE}")
        print(f"Random bytes per cycle        : {self.RANDOM_BYTES_PER_CYCLE} bytes")
        print(f"Air bytes per cycle           : {self.AIR_BYTES_PER_CYCLE} bytes")
        print(f"Actual air output rate        : {self.TARGET_AIR_BYTES_PER_SEC} byte/s")
        print("=============================================")

    # =========================
    # CRC8 ATM
    # =========================
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

    # =========================
    # CRC16 CCITT
    # =========================
    def crc16_ccitt(self, data: bytes) -> int:
        poly = 0x1021
        crc = 0xFFFF

        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = ((crc << 1) ^ poly) & 0xFFFF
                else:
                    crc = (crc << 1) & 0xFFFF

        return crc

    # =========================
    # 生成 15 字节 0x0A06 结构化数据帧
    # =========================
    def build_structured_frame(self, seq: int) -> bytes:
        """
        生成一个 15 字节的 0x0A06 Payload 数据帧。

        结构：
            SOF          1 byte
            DATALENGTH   2 bytes
            SEQ          1 byte
            CRC8         1 byte
            CMD_ID       2 bytes
            DATA         6 bytes
            CRC16        2 bytes
        """

        data = self.DATA

        if len(data) != self.DATALENGTH:
            raise ValueError(
                f"DATA length must be {self.DATALENGTH}, got {len(data)}"
            )

        # DATALENGTH: big-endian
        datalength_bytes = bytes([
            (self.DATALENGTH >> 8) & 0xFF,
            self.DATALENGTH & 0xFF
        ])

        # Header prefix: SOF + DATALENGTH + SEQ
        header_prefix = bytes([self.SOF]) + datalength_bytes + bytes([seq])

        # CRC8 over SOF + DATALENGTH + SEQ
        crc8 = self.crc8_atm(header_prefix)

        # Frame header: SOF + DATALENGTH + SEQ + CRC8
        frame_header = header_prefix + bytes([crc8])

        # CRC16 input: frame_header + CMD_ID + DATA
        crc16_input = frame_header + self.CMD_ID + data

        crc16_val = self.crc16_ccitt(crc16_input)

        crc16_bytes = bytes([
            (crc16_val >> 8) & 0xFF,
            crc16_val & 0xFF
        ])

        structured_frame = crc16_input + crc16_bytes

        if len(structured_frame) != self.PAYLOAD_SLICE_LEN:
            raise ValueError(
                f"Structured frame must be {self.PAYLOAD_SLICE_LEN} bytes, "
                f"got {len(structured_frame)}"
            )

        return structured_frame

    # =========================
    # 生成随机填充 Payload
    # =========================
    def generate_random_bytes(self, length: int) -> bytes:
        """
        生成随机填充字节。
        如果你希望随机填充限制为可打印 ASCII，可在这里改成 ASCII 随机。
        当前实现为 0x00~0xFF 全范围随机字节。
        """
        return os.urandom(length)

    # =========================
    # 封装一个 15 字节 Payload 为空口包
    # =========================
    def build_air_packet(self, payload: bytes) -> bytes:
        """
        空口包格式：

            ACCESS_CODE + 00 0F 00 0F + 15-byte Payload
        """

        if len(payload) != self.PAYLOAD_SLICE_LEN:
            raise ValueError(
                f"Payload length must be {self.PAYLOAD_SLICE_LEN} bytes, "
                f"got {len(payload)}"
            )

        return self.ACCESS_CODE + self.HEADER + payload

    # =========================
    # 生成一个 0.1 秒周期内的所有空口包
    # =========================
    def build_cycle_air_packets(self) -> bytes:
        """
        每个 10Hz 周期生成：

            Payload 数据流 135 bytes

        其中：
            15 bytes  : 一个 0x0A06 结构化数据帧
            120 bytes : 随机填充

        然后按 15 bytes 切片：

            135 / 15 = 9 个 Payload

        每个 Payload 封装为：

            ACCESS_CODE + 00 0F 00 0F + 15-byte Payload

        因此每周期输出：

            9 * 27 = 243 bytes
        """

        # 1. 生成一个 15 字节 0x0A06 结构化 Payload
        structured_frame = self.build_structured_frame(self.seq)

        # 2. 生成其余随机 Payload
        random_padding = self.generate_random_bytes(self.RANDOM_BYTES_PER_CYCLE)

        # 3. 构造本周期内的 Payload 数据流
        payload_stream = structured_frame + random_padding

        if len(payload_stream) != self.PAYLOAD_BYTES_PER_CYCLE:
            raise ValueError(
                f"Payload stream length must be {self.PAYLOAD_BYTES_PER_CYCLE}, "
                f"got {len(payload_stream)}"
            )

        # 4. 按 15 字节切片并逐片封装为空口包
        air_stream = bytearray()

        for offset in range(0, len(payload_stream), self.PAYLOAD_SLICE_LEN):
            payload_slice = payload_stream[
                offset: offset + self.PAYLOAD_SLICE_LEN
            ]

            air_packet = self.build_air_packet(payload_slice)
            air_stream += air_packet

        if len(air_stream) != self.AIR_BYTES_PER_CYCLE:
            raise ValueError(
                f"Air stream length must be {self.AIR_BYTES_PER_CYCLE}, "
                f"got {len(air_stream)}"
            )

        # 5. 更新序号
        self.seq = (self.seq + 1) & 0xFF

        return bytes(air_stream)

    # =========================
    # 内部节拍控制
    # =========================
    def wait_for_next_cycle(self):
        """
        让每个周期按照 10Hz 产生。

        注意：
        GNU Radio 中更推荐用 Throttle 或硬件 Sink 控制速率。
        这里提供内部 sleep，是为了让该 Source Block 单独运行时也能接近目标节拍。
        """

        if not self.enable_internal_timing:
            return

        now = time.monotonic()

        if now < self._next_cycle_time:
            time.sleep(self._next_cycle_time - now)

        # 设定下一个周期时间
        self._next_cycle_time += self.CYCLE_TIME

        # 如果系统调度严重滞后，避免一直追赶导致突发输出
        now_after_sleep = time.monotonic()
        if self._next_cycle_time < now_after_sleep - self.CYCLE_TIME:
            self._next_cycle_time = now_after_sleep + self.CYCLE_TIME

    # =========================
    # GNU Radio work
    # =========================
    def work(self, input_items, output_items):
        out = output_items[0]
        max_output = len(out)

        if max_output == 0:
            return 0

        # 如果缓存为空，按 10Hz 生成一个周期的空口包
        if len(self._packet_buffer) == 0:
            self.wait_for_next_cycle()
            self._packet_buffer = bytearray(self.build_cycle_air_packets())

        # 从缓存中拷贝到 GNU Radio 输出缓冲区
        send_len = min(len(self._packet_buffer), max_output)

        out[:send_len] = np.frombuffer(
            self._packet_buffer[:send_len],
            dtype=np.uint8
        )

        del self._packet_buffer[:send_len]

        return send_len
