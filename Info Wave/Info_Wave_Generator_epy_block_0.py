#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
from gnuradio import gr


class gfsk_data_source(gr.sync_block):
    """
    GFSK Data Source

    新版业务数据流：

    每 0.1 秒，即 10Hz，生成 135 字节有效业务数据：

        0A01: 9 + 24 = 33 bytes
        0A02: 9 + 12 = 21 bytes
        0A03: 9 + 10 = 19 bytes
        0A04: 9 + 8  = 17 bytes
        0A05: 9 + 36 = 45 bytes

        总计: 33 + 21 + 19 + 17 + 45 = 135 bytes

    每个业务帧格式：

        SOF          1 byte
        DATALENGTH   2 bytes
        SEQ          1 byte
        CRC8         1 byte
        CMD_ID       2 bytes
        DATA         N bytes
        CRC16        2 bytes

    其中固定开销：

        1 + 2 + 1 + 1 + 2 + 2 = 9 bytes

    空口封装格式：

        ACCESS_CODE + 00 0F 00 0F + 15-byte payload

    每 0.1 秒：

        135 bytes payload stream
        → 切成 9 个 15-byte payload
        → 9 个空口包
        → 9 * 27 = 243 bytes

    因此：

        payload rate = 135 * 10 = 1350 byte/s
        air rate     = 243 * 10 = 2430 byte/s
    """

    def __init__(self, enable_internal_timing=True, seq_per_frame=False):
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

        self.PAYLOAD_SLICE_LEN = 15

        self.AIR_PACKET_LEN = (
            len(self.ACCESS_CODE)
            + len(self.HEADER)
            + self.PAYLOAD_SLICE_LEN
        )

        # =========================
        # 业务流速率参数
        # =========================
        self.STRUCTURED_FRAME_FREQ_HZ = 10
        self.CYCLE_TIME = 1.0 / self.STRUCTURED_FRAME_FREQ_HZ

        # 每周期业务流 135 byte
        self.PAYLOAD_BYTES_PER_CYCLE = 135

        # 每周期 9 个 15-byte payload
        self.SLICES_PER_CYCLE = (
            self.PAYLOAD_BYTES_PER_CYCLE
            // self.PAYLOAD_SLICE_LEN
        )

        # 每周期空口字节数：9 * 27 = 243
        self.AIR_BYTES_PER_CYCLE = (
            self.SLICES_PER_CYCLE
            * self.AIR_PACKET_LEN
        )

        # 业务速率：135 * 10 = 1350 byte/s
        self.TARGET_PAYLOAD_BYTES_PER_SEC = (
            self.PAYLOAD_BYTES_PER_CYCLE
            * self.STRUCTURED_FRAME_FREQ_HZ
        )

        # 空口速率：243 * 10 = 2430 byte/s
        self.TARGET_AIR_BYTES_PER_SEC = (
            self.AIR_BYTES_PER_CYCLE
            * self.STRUCTURED_FRAME_FREQ_HZ
        )

        if self.PAYLOAD_BYTES_PER_CYCLE % self.PAYLOAD_SLICE_LEN != 0:
            raise ValueError(
                "PAYLOAD_BYTES_PER_CYCLE must be divisible by PAYLOAD_SLICE_LEN"
            )

        # =========================
        # 业务帧参数
        # =========================
        self.SOF = 0xA5

        # 每个业务帧固定开销：
        # SOF 1 + DATALENGTH 2 + SEQ 1 + CRC8 1 + CMD_ID 2 + CRC16 2 = 9
        self.APP_FRAME_OVERHEAD = 9

        self.CMD_0A01 = bytes([0x0A, 0x01])
        self.CMD_0A02 = bytes([0x0A, 0x02])
        self.CMD_0A03 = bytes([0x0A, 0x03])
        self.CMD_0A04 = bytes([0x0A, 0x04])
        self.CMD_0A05 = bytes([0x0A, 0x05])

        self.LEN_0A01 = 24
        self.LEN_0A02 = 12
        self.LEN_0A03 = 10
        self.LEN_0A04 = 8
        self.LEN_0A05 = 36

        self.EXPECTED_APP_BYTES_PER_CYCLE = (
            self.APP_FRAME_OVERHEAD + self.LEN_0A01
            + self.APP_FRAME_OVERHEAD + self.LEN_0A02
            + self.APP_FRAME_OVERHEAD + self.LEN_0A03
            + self.APP_FRAME_OVERHEAD + self.LEN_0A04
            + self.APP_FRAME_OVERHEAD + self.LEN_0A05
        )

        if self.EXPECTED_APP_BYTES_PER_CYCLE != self.PAYLOAD_BYTES_PER_CYCLE:
            raise ValueError(
                f"App bytes per cycle must be {self.PAYLOAD_BYTES_PER_CYCLE}, "
                f"but got {self.EXPECTED_APP_BYTES_PER_CYCLE}"
            )

        # seq_per_frame = False:
        #   同一个 10Hz 周期内 0A01~0A05 使用同一个 SEQ
        #
        # seq_per_frame = True:
        #   每个业务帧单独递增 SEQ
        self.seq_per_frame = bool(seq_per_frame)

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
        print(f"0A01 data length              : {self.LEN_0A01} bytes")
        print(f"0A02 data length              : {self.LEN_0A02} bytes")
        print(f"0A03 data length              : {self.LEN_0A03} bytes")
        print(f"0A04 data length              : {self.LEN_0A04} bytes")
        print(f"0A05 data length              : {self.LEN_0A05} bytes")
        print(f"App bytes per cycle           : {self.PAYLOAD_BYTES_PER_CYCLE} bytes")
        print(f"Payload slices per cycle      : {self.SLICES_PER_CYCLE}")
        print(f"Air bytes per cycle           : {self.AIR_BYTES_PER_CYCLE} bytes")
        print(f"Payload rate                  : {self.TARGET_PAYLOAD_BYTES_PER_SEC} byte/s")
        print(f"Actual air output rate        : {self.TARGET_AIR_BYTES_PER_SEC} byte/s")
        print(f"Frame rate                    : {self.STRUCTURED_FRAME_FREQ_HZ} Hz")
        print(f"seq_per_frame                 : {self.seq_per_frame}")
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
    # 固定测试数据生成函数
    # 后面你要改真实数据，就改这些函数
    # =========================
    def get_data_0a01(self) -> bytes:
        """
        0A01 数据区，24 字节。
        当前用固定字节 0x11 填充，便于检验。
        """
        return bytes([0x11] * self.LEN_0A01)

    def get_data_0a02(self) -> bytes:
        """
        0A02 数据区，12 字节。
        当前用固定字节 0x22 填充，便于检验。
        """
        return bytes([0x22] * self.LEN_0A02)

    def get_data_0a03(self) -> bytes:
        """
        0A03 数据区，10 字节。
        当前用固定字节 0x33 填充，便于检验。
        """
        return bytes([0x33] * self.LEN_0A03)

    def get_data_0a04(self) -> bytes:
        """
        0A04 数据区，8 字节。
        当前用固定字节 0x44 填充，便于检验。
        """
        return bytes([0x44] * self.LEN_0A04)

    def get_data_0a05(self) -> bytes:
        """
        0A05 数据区，36 字节。
        当前用固定字节 0x55 填充，便于检验。
        """
        return bytes([0x55] * self.LEN_0A05)

    # =========================
    # 构造一个业务帧
    # =========================
    def build_app_frame(self, cmd_id: bytes, data: bytes, seq: int) -> bytes:
        """
        业务帧格式：

            SOF          1 byte
            DATALENGTH   2 bytes, big-endian
            SEQ          1 byte
            CRC8         1 byte
            CMD_ID       2 bytes
            DATA         N bytes
            CRC16        2 bytes

        总长度：

            9 + len(DATA)
        """

        if len(cmd_id) != 2:
            raise ValueError("cmd_id must be 2 bytes")

        data_len = len(data)

        datalength_bytes = bytes([
            (data_len >> 8) & 0xFF,
            data_len & 0xFF
        ])

        # CRC8 输入：SOF + DATALENGTH + SEQ
        header_prefix = (
            bytes([self.SOF])
            + datalength_bytes
            + bytes([seq & 0xFF])
        )

        crc8_val = self.crc8_atm(header_prefix)

        frame_without_crc16 = (
            header_prefix
            + bytes([crc8_val])
            + cmd_id
            + data
        )

        crc16_val = self.crc16_ccitt(frame_without_crc16)

        crc16_bytes = bytes([
            (crc16_val >> 8) & 0xFF,
            crc16_val & 0xFF
        ])

        frame = frame_without_crc16 + crc16_bytes

        expected_len = self.APP_FRAME_OVERHEAD + data_len

        if len(frame) != expected_len:
            raise ValueError(
                f"App frame length error, expected {expected_len}, got {len(frame)}"
            )

        return frame

    # =========================
    # 生成一个 0.1 秒周期内的 135 字节有效业务流
    # =========================
    def build_cycle_payload_stream(self) -> bytes:
        """
        每个周期生成 5 个有效业务帧：

            0A01 + 24 bytes data
            0A02 + 12 bytes data
            0A03 + 10 bytes data
            0A04 + 8  bytes data
            0A05 + 36 bytes data

        拼接后总长度刚好 135 字节。
        """

        cycle_seq = self.seq & 0xFF

        frames = bytearray()

        if self.seq_per_frame:
            # 每个业务帧单独递增 seq
            frames += self.build_app_frame(
                self.CMD_0A01,
                self.get_data_0a01(),
                self.seq
            )
            self.seq = (self.seq + 1) & 0xFF

            frames += self.build_app_frame(
                self.CMD_0A02,
                self.get_data_0a02(),
                self.seq
            )
            self.seq = (self.seq + 1) & 0xFF

            frames += self.build_app_frame(
                self.CMD_0A03,
                self.get_data_0a03(),
                self.seq
            )
            self.seq = (self.seq + 1) & 0xFF

            frames += self.build_app_frame(
                self.CMD_0A04,
                self.get_data_0a04(),
                self.seq
            )
            self.seq = (self.seq + 1) & 0xFF

            frames += self.build_app_frame(
                self.CMD_0A05,
                self.get_data_0a05(),
                self.seq
            )
            self.seq = (self.seq + 1) & 0xFF

        else:
            # 同一个 10Hz 周期内 5 个业务帧共用同一个 seq
            frames += self.build_app_frame(
                self.CMD_0A01,
                self.get_data_0a01(),
                cycle_seq
            )

            frames += self.build_app_frame(
                self.CMD_0A02,
                self.get_data_0a02(),
                cycle_seq
            )

            frames += self.build_app_frame(
                self.CMD_0A03,
                self.get_data_0a03(),
                cycle_seq
            )

            frames += self.build_app_frame(
                self.CMD_0A04,
                self.get_data_0a04(),
                cycle_seq
            )

            frames += self.build_app_frame(
                self.CMD_0A05,
                self.get_data_0a05(),
                cycle_seq
            )

            self.seq = (self.seq + 1) & 0xFF

        payload_stream = bytes(frames)

        if len(payload_stream) != self.PAYLOAD_BYTES_PER_CYCLE:
            raise ValueError(
                f"Payload stream length must be {self.PAYLOAD_BYTES_PER_CYCLE}, "
                f"got {len(payload_stream)}"
            )

        return payload_stream

    # =========================
    # 封装一个 15 字节 payload 为空口包
    # =========================
    def build_air_packet(self, payload: bytes) -> bytes:
        """
        空口包格式：

            ACCESS_CODE + 00 0F 00 0F + 15-byte payload
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
        每个 10Hz 周期：

            1. 生成 135 字节有效业务流
            2. 按 15 字节切成 9 片
            3. 每片封装为 27 字节空口包
            4. 总输出 9 * 27 = 243 字节
        """

        payload_stream = self.build_cycle_payload_stream()

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

        return bytes(air_stream)

    # =========================
    # 内部节拍控制
    # =========================
    def wait_for_next_cycle(self):
        """
        让每个周期按照 10Hz 产生。

        GNU Radio 中更推荐用 Throttle 或硬件 Sink 控制速率。
        这里保留内部 sleep，方便单独运行 Source Block 时也能接近目标节拍。
        """

        if not self.enable_internal_timing:
            return

        now = time.monotonic()

        if now < self._next_cycle_time:
            time.sleep(self._next_cycle_time - now)

        self._next_cycle_time += self.CYCLE_TIME

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

        # 缓存为空时，生成一个 10Hz 周期的空口数据
        if len(self._packet_buffer) == 0:
            self.wait_for_next_cycle()
            self._packet_buffer = bytearray(self.build_cycle_air_packets())

        send_len = min(len(self._packet_buffer), max_output)

        out[:send_len] = np.frombuffer(
            self._packet_buffer[:send_len],
            dtype=np.uint8
        )

        del self._packet_buffer[:send_len]

        return send_len