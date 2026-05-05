#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from gnuradio import gr
import pmt


class gfsk_key_decoder(gr.basic_block):
    """
    GFSK Receiver for 0A01~0A05 effective data stream

    输入：
        GFSK Demod 输出的 unpacked bit stream。
        每个 uint8 只使用 LSB，值为 0 或 1。

    空口包格式：
        ACCESS_CODE + 00 0F 00 0F + 15-byte payload

    业务层帧格式：
        SOF          1 byte   0xA5
        DATALENGTH   2 bytes  big-endian
        SEQ          1 byte
        CRC8         1 byte   CRC8(SOF + DATALENGTH + SEQ)
        CMD_ID       2 bytes
        DATA         N bytes
        CRC16        2 bytes  CRC16(前面所有字段，不含 CRC16)

    支持的 CMD_ID：
        0A01: DATA 24 bytes, full frame 33 bytes
        0A02: DATA 12 bytes, full frame 21 bytes
        0A03: DATA 10 bytes, full frame 19 bytes
        0A04: DATA 8  bytes, full frame 17 bytes
        0A05: DATA 36 bytes, full frame 45 bytes

    每 10Hz 周期：
        33 + 21 + 19 + 17 + 45 = 135 bytes

    输出：
        默认 output_full_frame=True：
            输出完整业务帧，即 A5 ... CRC16。
            每周期输出 135 bytes。

        如果 output_full_frame=False：
            只输出 DATA 数据区。
            每周期输出 24 + 12 + 10 + 8 + 36 = 90 bytes。
    """

    def __init__(
        self,
        max_access_errors=0,
        debug=True,
        invert_bits=False,
        output_full_frame=True,
        debug_air_packet=False
    ):
        gr.basic_block.__init__(
            self,
            name="GFSK 0A01-0A05 Decoder",
            in_sig=[np.uint8],
            out_sig=[np.uint8],
        )

        # =========================
        # 空口层参数
        # =========================
        self.ACCESS_CODE = bytes([
            0x16, 0xE8, 0xD3, 0x77,
            0x15, 0x1C, 0x71, 0x2D
        ])

        self.AIR_HEADER_LEN = 4
        self.AIR_PAYLOAD_LEN = 15

        self.access_bits = self.bytes_to_bits(self.ACCESS_CODE)

        self.max_access_errors = int(max_access_errors)
        self.debug = bool(debug)
        self.invert_bits = bool(invert_bits)
        self.output_full_frame = bool(output_full_frame)
        self.debug_air_packet = bool(debug_air_packet)

        # =========================
        # 业务层参数
        # =========================
        self.SOF = 0xA5

        self.CMD_0A01 = bytes([0x0A, 0x01])
        self.CMD_0A02 = bytes([0x0A, 0x02])
        self.CMD_0A03 = bytes([0x0A, 0x03])
        self.CMD_0A04 = bytes([0x0A, 0x04])
        self.CMD_0A05 = bytes([0x0A, 0x05])

        self.expected_data_len = {
            self.CMD_0A01: 24,
            self.CMD_0A02: 12,
            self.CMD_0A03: 10,
            self.CMD_0A04: 8,
            self.CMD_0A05: 36,
        }

        self.APP_FRAME_OVERHEAD = 9
        self.MIN_APP_HEADER_LEN = 7

        # =========================
        # 缓存
        # =========================
        # GFSK demod 后的 bit 流缓存
        self.bitbuf = bytearray()

        # 多个空口 payload 拼接后的连续业务流
        self.app_buf = bytearray()

        # 已解析但还未输出的帧队列
        # 元素格式：
        #   {
        #       "cmd_id": bytes,
        #       "seq": int,
        #       "data": bytes,
        #       "frame": bytes,
        #       "output": bytes
        #   }
        self.frame_queue = []

        # =========================
        # GNU Radio tags
        # =========================
        self.packet_len_key = pmt.intern("packet_len")
        self.seq_key = pmt.intern("seq")
        self.cmd_id_key = pmt.intern("cmd_id")
        self.data_len_key = pmt.intern("data_len")
        self.frame_len_key = pmt.intern("frame_len")

        # =========================
        # 统计
        # =========================
        self.air_packet_count = 0
        self.valid_frame_count = 0
        self.bad_header_count = 0
        self.false_sof_count = 0
        self.bad_crc8_count = 0
        self.bad_crc16_count = 0
        self.bad_cmd_count = 0
        self.bad_len_count = 0

        if self.debug:
            print("========== GFSK 0A01-0A05 Decoder Config ==========")
            print("ACCESS_CODE           :", self.ACCESS_CODE.hex(" "))
            print("AIR HEADER            : 00 0f 00 0f")
            print("AIR PAYLOAD LEN       :", self.AIR_PAYLOAD_LEN)
            print("Supported frames:")
            print("  0A01 DATA LEN        : 24, FRAME LEN 33")
            print("  0A02 DATA LEN        : 12, FRAME LEN 21")
            print("  0A03 DATA LEN        : 10, FRAME LEN 19")
            print("  0A04 DATA LEN        : 8,  FRAME LEN 17")
            print("  0A05 DATA LEN        : 36, FRAME LEN 45")
            print("Cycle payload bytes    : 135")
            print("max_access_errors     :", self.max_access_errors)
            print("invert_bits           :", self.invert_bits)
            print("output_full_frame     :", self.output_full_frame)
            print("debug_air_packet      :", self.debug_air_packet)
            print("====================================================")

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
    # 查找 ACCESS_CODE
    # =========================
    def find_access_code(self):
        access_len = len(self.access_bits)
        limit = len(self.bitbuf) - access_len + 1

        if limit <= 0:
            return -1

        access = self.access_bits
        max_err = self.max_access_errors

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
    # CMD_ID 显示
    # =========================
    def cmd_to_int(self, cmd_id: bytes) -> int:
        return (cmd_id[0] << 8) | cmd_id[1]

    def cmd_to_text(self, cmd_id: bytes) -> str:
        return "%02X%02X" % (cmd_id[0], cmd_id[1])

    # =========================
    # 解析一个业务帧候选
    # =========================
    def parse_app_frame(self, frame: bytes):
        """
        输入一个完整候选业务帧。

        成功返回：
            seq, cmd_id, data

        失败返回：
            None
        """

        if len(frame) < self.MIN_APP_HEADER_LEN + 2:
            return None

        if frame[0] != self.SOF:
            return None

        data_len = (frame[1] << 8) | frame[2]
        seq = frame[3]
        rx_crc8 = frame[4]
        cmd_id = frame[5:7]

        # CMD 检查
        if cmd_id not in self.expected_data_len:
            self.bad_cmd_count += 1
            return None

        expected_len = self.expected_data_len[cmd_id]

        if data_len != expected_len:
            self.bad_len_count += 1
            return None

        full_len = self.APP_FRAME_OVERHEAD + data_len

        if len(frame) != full_len:
            return None

        # CRC8: SOF + DATALENGTH + SEQ
        calc_crc8 = self.crc8_atm(frame[0:4])
        if rx_crc8 != calc_crc8:
            self.bad_crc8_count += 1
            return None

        # DATA
        data_start = 7
        data_end = data_start + data_len
        data = frame[data_start:data_end]

        # CRC16: 除最后 2 字节外的所有字段
        rx_crc16 = (frame[-2] << 8) | frame[-1]
        calc_crc16 = self.crc16_ccitt(frame[:-2])

        if rx_crc16 != calc_crc16:
            self.bad_crc16_count += 1
            return None

        return seq, cmd_id, data

    # =========================
    # 从 app_buf 中解析可变长度业务帧
    # =========================
    def parse_app_buf(self):
        """
        app_buf 是由多个 15-byte 空口 payload 拼接出来的连续业务流。

        本函数支持业务帧跨 15-byte payload 边界。
        """

        while True:
            # 至少要有 A5 + LEN2 + SEQ + CRC8 + CMD2
            if len(self.app_buf) < self.MIN_APP_HEADER_LEN:
                return

            # 找 SOF = 0xA5
            pos = self.app_buf.find(bytes([self.SOF]))

            if pos < 0:
                # 没有 SOF，丢掉无效数据。
                # SOF 是单字节，不需要保留很多。
                if len(self.app_buf) > 0:
                    self.app_buf.clear()
                return

            # 丢掉 SOF 前面的数据
            if pos > 0:
                del self.app_buf[:pos]

            # 等待最小帧头
            if len(self.app_buf) < self.MIN_APP_HEADER_LEN:
                return

            # 读取长度和 CMD_ID
            data_len = (self.app_buf[1] << 8) | self.app_buf[2]
            cmd_id = bytes(self.app_buf[5:7])

            # 长度基本合法性检查
            # 目前最大 DATA 为 36，给一点余量，超过 512 基本就是假帧头。
            if data_len <= 0 or data_len > 512:
                self.false_sof_count += 1

                if self.debug:
                    print(
                        "[GFSK RX] false SOF: invalid data_len =",
                        data_len
                    )

                del self.app_buf[0]
                continue

            # 如果 CMD_ID 已经不是支持的类型，说明这个 A5 很可能是随机或误码。
            if cmd_id not in self.expected_data_len:
                self.bad_cmd_count += 1

                if self.debug:
                    print(
                        "[GFSK RX] unsupported CMD_ID after SOF:",
                        cmd_id.hex(" ")
                    )

                del self.app_buf[0]
                continue

            expected_data_len = self.expected_data_len[cmd_id]

            if data_len != expected_data_len:
                self.bad_len_count += 1

                if self.debug:
                    print(
                        "[GFSK RX] length mismatch cmd=%s rx_len=%d expected=%d"
                        % (
                            self.cmd_to_text(cmd_id),
                            data_len,
                            expected_data_len
                        )
                    )

                del self.app_buf[0]
                continue

            full_frame_len = self.APP_FRAME_OVERHEAD + data_len

            # 等待完整业务帧
            if len(self.app_buf) < full_frame_len:
                return

            candidate = bytes(self.app_buf[:full_frame_len])

            parsed = self.parse_app_frame(candidate)

            if parsed is None:
                # 找到了 A5，但 CRC 或字段不对。
                # 删除这个 A5，继续搜索下一个 SOF。
                self.false_sof_count += 1

                if self.debug:
                    print(
                        "[GFSK RX] invalid app frame candidate:",
                        candidate.hex(" ")
                    )

                del self.app_buf[0]
                continue

            seq, cmd_id, data = parsed
            self.valid_frame_count += 1

            if self.output_full_frame:
                output_bytes = candidate
            else:
                output_bytes = data

            self.frame_queue.append({
                "seq": seq,
                "cmd_id": cmd_id,
                "data": data,
                "frame": candidate,
                "output": output_bytes,
            })

            if self.debug:
                data_preview = data.hex(" ")
                if len(data_preview) > 96:
                    data_preview = data_preview[:96] + " ..."

                print(
                    "[GFSK RX] valid_frame=%d cmd=%s seq=%03d data_len=%d frame_len=%d data=%s"
                    % (
                        self.valid_frame_count,
                        self.cmd_to_text(cmd_id),
                        seq,
                        len(data),
                        len(candidate),
                        data_preview
                    )
                )

            # 删除完整业务帧
            del self.app_buf[:full_frame_len]

    # =========================
    # 解析空口包
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
                # 保留最后 ACCESS_CODE_LEN - 1 bit，防止包头跨 work 边界
                keep = access_bits_len - 1

                if len(self.bitbuf) > keep:
                    del self.bitbuf[:-keep]

                return

            # 丢掉 access code 前面的杂散 bit
            if pos > 0:
                del self.bitbuf[:pos]

            # 等待 access code + 4 字节 header
            if len(self.bitbuf) < min_header_bits:
                return

            header_bits = self.bitbuf[access_bits_len:min_header_bits]
            header = self.bits_to_bytes(header_bits)

            len1 = (header[0] << 8) | header[1]
            len2 = (header[2] << 8) | header[3]

            # header 应该是 00 0F 00 0F
            if len1 != len2 or len1 != self.AIR_PAYLOAD_LEN:
                self.bad_header_count += 1

                if self.debug:
                    print(
                        "[GFSK RX] bad air header:",
                        header.hex(" "),
                        "len1 =", len1,
                        "len2 =", len2
                    )

                # 假同步，右移 1 bit 继续找 ACCESS_CODE
                del self.bitbuf[0]
                continue

            total_bits = (
                access_bits_len
                + header_bits_len
                + self.AIR_PAYLOAD_LEN * 8
            )

            # 等待完整空口包
            if len(self.bitbuf) < total_bits:
                return

            payload_bits = self.bitbuf[min_header_bits:total_bits]
            payload = self.bits_to_bytes(payload_bits)

            self.air_packet_count += 1

            if self.debug_air_packet:
                print(
                    "[GFSK RX] air_packet=%d payload=%s"
                    % (
                        self.air_packet_count,
                        payload.hex(" ")
                    )
                )

            # 关键：
            # 这里只恢复空口 payload，不直接按业务帧解析。
            # 因为 0A01~0A05 都可能跨 15-byte payload 边界。
            self.app_buf.extend(payload)

            # 删除整个空口包
            del self.bitbuf[:total_bits]

            # 解析业务流
            self.parse_app_buf()

    # =========================
    # GNU Radio forecast
    # =========================
    def forecast(self, noutput_items, ninput_items_required):
        ninput_items_required[0] = 1

    # =========================
    # GNU Radio general_work
    # =========================
    def general_work(self, input_items, output_items):
        inp = input_items[0]
        out = output_items[0]

        # 1. 收 GFSK Demod 输出 bit
        if len(inp) > 0:
            bits = (inp & 0x01).astype(np.uint8).tolist()

            if self.invert_bits:
                bits = [b ^ 0x01 for b in bits]

            self.bitbuf.extend(bits)

            self.consume(0, len(inp))
        else:
            self.consume(0, 0)

        # 2. 空口层解析，再进入业务层解析
        self.parse_air_packets()

        # 3. 输出已解析的业务帧
        produced = 0

        while len(self.frame_queue) > 0:
            item = self.frame_queue[0]
            output_bytes = item["output"]

            # 如果当前 GNU Radio 输出缓冲不够放一整个帧，就留到下一次。
            if produced + len(output_bytes) > len(out):
                break

            seq = item["seq"]
            cmd_id = item["cmd_id"]
            data = item["data"]
            frame = item["frame"]

            start_off = produced
            end_off = produced + len(output_bytes)

            out[start_off:end_off] = np.frombuffer(
                output_bytes,
                dtype=np.uint8
            )

            abs_off = self.nitems_written(0) + produced

            # 添加 tags，方便后面 File Sink / 自定义 block 判断帧边界
            self.add_item_tag(
                0,
                abs_off,
                self.packet_len_key,
                pmt.from_long(len(output_bytes)),
            )

            self.add_item_tag(
                0,
                abs_off,
                self.seq_key,
                pmt.from_long(seq),
            )

            self.add_item_tag(
                0,
                abs_off,
                self.cmd_id_key,
                pmt.from_long(self.cmd_to_int(cmd_id)),
            )

            self.add_item_tag(
                0,
                abs_off,
                self.data_len_key,
                pmt.from_long(len(data)),
            )

            self.add_item_tag(
                0,
                abs_off,
                self.frame_len_key,
                pmt.from_long(len(frame)),
            )

            produced += len(output_bytes)

            # 删除已经输出的帧
            self.frame_queue.pop(0)

        return produced