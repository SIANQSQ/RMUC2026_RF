#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from gnuradio import gr
import pmt


class gfsk_key_decoder(gr.basic_block):
    """
    GFSK Key Decoder

    输入：
        GFSK Demod 输出的 unpacked bit stream。
        每个 uint8 只使用 LSB，值为 0 或 1。

    空口包格式：
        ACCESS_CODE + 00 0F 00 0F + 15-byte payload

    业务帧格式：
        SOF             1 byte   0xA5
        DATALENGTH      2 bytes  0x0006
        SEQ             1 byte
        CRC8            1 byte   CRC8(SOF + DATALENGTH + SEQ)
        CMD_ID          2 bytes  0x0A 0x06
        DATA            6 bytes
        CRC16           2 bytes  CRC16(前 13 字节)

    输出：
        每解析出一个合法 0x0A06 业务帧，输出其中 6 字节 DATA。
        例如 b"rmnb66"。
    """

    def __init__(
        self,
        max_access_errors=0,
        debug=True,
        invert_bits=False
    ):
        gr.basic_block.__init__(
            self,
            name="GFSK Key Decoder",
            in_sig=[np.uint8],
            out_sig=[np.uint8],
        )

        # =========================
        # 空口包参数
        # =========================
        self.ACCESS_CODE = bytes([
            0x16, 0xE8, 0xD3, 0x77,
            0x15, 0x1C, 0x71, 0x2D
        ])

        # 新发送端中的 header: 00 0F 00 0F
        self.AIR_HEADER_LEN = 4
        self.AIR_PAYLOAD_LEN = 15

        self.max_access_errors = int(max_access_errors)
        self.debug = bool(debug)
        self.invert_bits = bool(invert_bits)

        self.access_bits = self.bytes_to_bits(self.ACCESS_CODE)

        # =========================
        # 业务帧参数
        # =========================
        self.APP_FRAME_LEN = 15

        self.SOF = 0xA5
        self.CMD_ID = bytes([0x0A, 0x06])
        self.KEY_LEN = 6

        # =========================
        # 缓存
        # =========================
        # bitbuf：GFSK demod 后的 bit 流缓存，用于查找 ACCESS_CODE
        self.bitbuf = bytearray()

        # app_buf：从多个空口 payload 拼接得到的连续业务流
        self.app_buf = bytearray()

        # frame_queue：已经解析出来但还没输出的 6 字节 DATA
        # 元素格式: (seq, key_data)
        self.frame_queue = []

        # =========================
        # tag
        # =========================
        self.packet_len_key = pmt.intern("packet_len")
        self.seq_key = pmt.intern("seq")

        # =========================
        # 统计
        # =========================
        self.air_packet_count = 0
        self.valid_frame_count = 0
        self.false_sof_count = 0
        self.bad_header_count = 0

        if self.debug:
            print("========== GFSK Key Decoder Config ==========")
            print("ACCESS_CODE        :", self.ACCESS_CODE.hex(" "))
            print("AIR HEADER         : 00 0f 00 0f")
            print("AIR PAYLOAD LEN    :", self.AIR_PAYLOAD_LEN)
            print("APP FRAME LEN      :", self.APP_FRAME_LEN)
            print("CMD_ID             :", self.CMD_ID.hex(" "))
            print("DATA LEN           :", self.KEY_LEN)
            print("max_access_errors  :", self.max_access_errors)
            print("invert_bits        :", self.invert_bits)
            print("=============================================")

    # =========================
    # byte -> bit, MSB first
    # =========================
    def bytes_to_bits(self, data: bytes):
        bits = []
        for b in data:
            for i in range(7, -1, -1):
                bits.append((b >> i) & 0x01)
        return bits

    # =========================
    # bit -> byte, MSB first
    # =========================
    def bits_to_bytes(self, bits):
        n = len(bits) // 8
        out = bytearray(n)

        for i in range(n):
            v = 0
            for b in bits[i * 8:(i + 1) * 8]:
                v = (v << 1) | (b & 0x01)
            out[i] = v & 0xFF

        return bytes(out)

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
    # 在 bitbuf 中查找 access code
    # =========================
    def find_access_code(self):
        """
        在 bitbuf 中搜索 ACCESS_CODE。
        返回匹配起点；找不到返回 -1。
        支持 max_access_errors 个 bit 错误。
        """

        access_len = len(self.access_bits)
        limit = len(self.bitbuf) - access_len + 1

        if limit <= 0:
            return -1

        max_err = self.max_access_errors
        access = self.access_bits

        for i in range(limit):
            err = 0

            for j in range(access_len):
                if self.bitbuf[i + j] != access[j]:
                    err += 1

                    if err > max_err:
                        break

            if err <= max_err:
                return i

        return -1

    # =========================
    # 解析 15 字节业务帧
    # =========================
    def parse_app_frame(self, frame: bytes):
        """
        解析一个候选 15 字节业务帧。

        成功：
            return seq, key_data

        失败：
            return None
        """

        if len(frame) != self.APP_FRAME_LEN:
            return None

        # SOF
        if frame[0] != self.SOF:
            return None

        # DATALENGTH
        data_len = (frame[1] << 8) | frame[2]
        if data_len != self.KEY_LEN:
            return None

        seq = frame[3]
        rx_crc8 = frame[4]

        # CRC8 over SOF + DATALENGTH + SEQ
        calc_crc8 = self.crc8_atm(frame[0:4])
        if rx_crc8 != calc_crc8:
            return None

        # CMD_ID
        cmd_id = frame[5:7]
        if cmd_id != self.CMD_ID:
            return None

        # DATA
        data_start = 7
        data_end = data_start + data_len
        key_data = frame[data_start:data_end]

        # CRC16
        rx_crc16 = (frame[13] << 8) | frame[14]
        calc_crc16 = self.crc16_ccitt(frame[0:13])

        if rx_crc16 != calc_crc16:
            return None

        return seq, key_data

    # =========================
    # 业务流解析：从 app_buf 中搜索完整 0x0A06 帧
    # =========================
    def parse_app_buf(self):
        """
        app_buf 是多个 15 字节 payload 拼接后的连续业务流。

        这里不假设业务帧一定正好落在一个 payload 内。
        即使业务帧跨两个 payload，也可以解析。
        """

        while True:
            if len(self.app_buf) < self.APP_FRAME_LEN:
                return

            # 先找 SOF = 0xA5
            pos = self.app_buf.find(bytes([self.SOF]))

            if pos < 0:
                # 没找到 SOF，丢掉大部分随机数据。
                # 保留少量尾部，防止边界情况。
                keep = self.APP_FRAME_LEN - 1
                if len(self.app_buf) > keep:
                    del self.app_buf[:-keep]
                return

            # 丢掉 SOF 前面的随机数据
            if pos > 0:
                del self.app_buf[:pos]

            # 等待完整 15 字节
            if len(self.app_buf) < self.APP_FRAME_LEN:
                return

            candidate = bytes(self.app_buf[:self.APP_FRAME_LEN])

            parsed = self.parse_app_frame(candidate)

            if parsed is not None:
                seq, key_data = parsed
                self.valid_frame_count += 1

                # 放入输出队列，等 general_work 输出
                self.frame_queue.append((seq, key_data))

                if self.debug:
                    try:
                        ascii_text = key_data.decode("ascii")
                    except Exception:
                        ascii_text = repr(key_data)

                    print(
                        "[GFSK RX] valid_frame=%d seq=%03d key_hex=%s key_ascii=%s"
                        % (
                            self.valid_frame_count,
                            seq,
                            key_data.hex(" "),
                            ascii_text
                        )
                    )

                # 删除这个完整业务帧
                del self.app_buf[:self.APP_FRAME_LEN]

            else:
                # 找到了 0xA5，但后续 CRC/CMD/LEN 不对，说明是假帧头。
                # 只删除这个 0xA5，继续向后找下一个 0xA5。
                self.false_sof_count += 1

                if self.debug:
                    print(
                        "[GFSK RX] false SOF, drop one byte, candidate:",
                        candidate.hex(" ")
                    )

                del self.app_buf[0]

    # =========================
    # 空口包解析：从 bitbuf 中取出 payload
    # =========================
    def parse_air_packets(self):
        """
        从 bitbuf 中解析：

            ACCESS_CODE + 00 0F 00 0F + 15-byte payload

        每解析出一个 payload，就追加到 app_buf。
        """

        access_bits_len = len(self.access_bits)
        header_bits_len = self.AIR_HEADER_LEN * 8

        min_header_bits = access_bits_len + header_bits_len

        while True:
            pos = self.find_access_code()

            if pos < 0:
                # 保留最后 63 bit，防止 ACCESS_CODE 跨 work 边界
                keep = access_bits_len - 1
                if len(self.bitbuf) > keep:
                    del self.bitbuf[:-keep]
                return

            # 丢弃 access code 前面的杂散 bit
            if pos > 0:
                del self.bitbuf[:pos]

            # 等待 access code + 4 字节 header
            if len(self.bitbuf) < min_header_bits:
                return

            header_bits = self.bitbuf[access_bits_len:min_header_bits]
            header = self.bits_to_bytes(header_bits)

            # header 格式：00 0F 00 0F
            len1 = (header[0] << 8) | header[1]
            len2 = (header[2] << 8) | header[3]

            if (
                len1 != len2
                or len1 != self.AIR_PAYLOAD_LEN
            ):
                self.bad_header_count += 1

                if self.debug:
                    print(
                        "[GFSK RX] bad air header:",
                        header.hex(" "),
                        "len1 =", len1,
                        "len2 =", len2
                    )

                # header 不对，说明是假同步，右移 1 bit 继续搜索
                del self.bitbuf[0]
                continue

            payload_len_bytes = len1
            total_bits = (
                access_bits_len
                + header_bits_len
                + payload_len_bytes * 8
            )

            # 等待完整空口包
            if len(self.bitbuf) < total_bits:
                return

            payload_bits = self.bitbuf[min_header_bits:total_bits]
            payload = self.bits_to_bytes(payload_bits)

            self.air_packet_count += 1

            if self.debug:
                print(
                    "[GFSK RX] air_packet=%d payload=%s"
                    % (
                        self.air_packet_count,
                        payload.hex(" ")
                    )
                )

            # 关键点：
            # 这里不直接 parse_app_frame(payload)。
            # 而是把 payload 拼接进连续业务流 app_buf。
            self.app_buf.extend(payload)

            # 删除整个已解析的空口包
            del self.bitbuf[:total_bits]

            # 尝试从连续业务流中解析 0x0A06 业务帧
            self.parse_app_buf()

    # =========================
    # GNU Radio forecast
    # =========================
    def forecast(self, noutput_items, ninput_items_required):
        # 本 block 是变速率 block，输入 bit 多吃少吃都可以
        ninput_items_required[0] = 1

    # =========================
    # GNU Radio general_work
    # =========================
    def general_work(self, input_items, output_items):
        inp = input_items[0]
        out = output_items[0]
        
        # 1. 接收 GFSK Demod 输出 bit
        if len(inp) > 0:
            bits = (inp & 0x01).astype(np.uint8).tolist()

            if self.invert_bits:
                bits = [b ^ 0x01 for b in bits]

            self.bitbuf.extend(bits)

            # 告诉 GNU Radio：这些输入已经消费
            self.consume(0, len(inp))
        else:
            self.consume(0, 0)

        # 2. 从 bitbuf 中解析空口包，并把 payload 拼接到 app_buf
        self.parse_air_packets()

        # 3. 输出已经解析出的业务 DATA
        produced = 0

        while (
            len(self.frame_queue) > 0
            and produced + self.KEY_LEN <= len(out)
        ):
            seq, key_data = self.frame_queue.pop(0)

            abs_off = self.nitems_written(0) + produced

            self.add_item_tag(
                0,
                abs_off,
                self.packet_len_key,
                pmt.from_long(len(key_data)),
            )

            self.add_item_tag(
                0,
                abs_off,
                self.seq_key,
                pmt.from_long(seq),
            )

            out[produced:produced + len(key_data)] = np.frombuffer(
                key_data,
                dtype=np.uint8
            )

            produced += len(key_data)

        return produced