#!/usr/bin/env python3
import numpy as np
from gnuradio import gr
import pmt


class gfsk_key_decoder(gr.basic_block):
    """
    输入：GFSK Demod 输出的 unpacked bit stream，每个 uint8 只使用 LSB，值为 0 或 1。
    输出：每帧解出的 6 字节密钥数据，例如 b"123456"。
    """

    def __init__(self, max_access_errors=0, debug=True):
        gr.basic_block.__init__(
            self,
            name="GFSK Key Decoder",
            in_sig=[np.uint8],
            out_sig=[np.uint8],
        )

        self.ACCESS_CODE = bytes([0x16, 0xE8, 0xD3, 0x77, 0x15, 0x1C, 0x71, 0x2D])
        self.EXPECTED_LEN = 15
        self.SOF = 0xA5
        self.CMD_ID = bytes([0x0A, 0x06])
        self.KEY_LEN = 6

        self.max_access_errors = int(max_access_errors)
        self.debug = bool(debug)

        self.access_bits = self.bytes_to_bits(self.ACCESS_CODE)
        self.bitbuf = bytearray()

        self.packet_len_key = pmt.intern("packet_len")
        self.seq_key = pmt.intern("seq")

    def bytes_to_bits(self, data: bytes):
        bits = []
        for b in data:
            for i in range(7, -1, -1):
                bits.append((b >> i) & 1)
        return bits

    def bits_to_bytes(self, bits):
        n = len(bits) // 8
        out = bytearray(n)
        for i in range(n):
            v = 0
            for b in bits[i * 8:(i + 1) * 8]:
                v = (v << 1) | (b & 1)
            out[i] = v
        return bytes(out)

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

    def find_access_code(self):
        """
        在 bitbuf 中搜索 access code。
        返回匹配起点；找不到返回 -1。
        """
        n = len(self.access_bits)
        limit = len(self.bitbuf) - n + 1
        if limit <= 0:
            return -1

        access = self.access_bits
        max_err = self.max_access_errors

        for i in range(limit):
            err = 0
            for j in range(n):
                if self.bitbuf[i + j] != access[j]:
                    err += 1
                    if err > max_err:
                        break
            if err <= max_err:
                return i

        return -1

    def parse_payload(self, payload: bytes):
        """
        payload 是 access code + 4字节 LENGTH_CHECK 之后的 15 字节结构帧：

        SOF             1 byte
        DATALENGTH      2 bytes
        SEQ             1 byte
        CRC8            1 byte
        CMD_ID          2 bytes
        DATA            6 bytes
        CRC16           2 bytes
        """
        if len(payload) != self.EXPECTED_LEN:
            return None

        if payload[0] != self.SOF:
            return None

        data_len = (payload[1] << 8) | payload[2]
        seq = payload[3]
        rx_crc8 = payload[4]

        header_prefix = payload[0:4]
        calc_crc8 = self.crc8_atm(header_prefix)
        if rx_crc8 != calc_crc8:
            return None

        cmd = payload[5:7]
        if cmd != self.CMD_ID:
            return None

        if data_len != self.KEY_LEN:
            return None

        data_start = 7
        data_end = data_start + data_len
        key_data = payload[data_start:data_end]

        rx_crc16 = (payload[-2] << 8) | payload[-1]
        calc_crc16 = self.crc16_ccitt(payload[:-2])
        if rx_crc16 != calc_crc16:
            return None

        return seq, key_data

    def general_work(self, input_items, output_items):
        inp = input_items[0]
        out = output_items[0]

        # GFSK Demod 输出每个 uint8 的 LSB 是 bit。
        if len(inp) > 0:
            self.bitbuf.extend((inp & 0x01).astype(np.uint8).tolist())

        produced = 0

        access_bits_len = len(self.access_bits)
        header_bits_len = 32
        min_header_bits = access_bits_len + header_bits_len

        while produced + self.KEY_LEN <= len(out):
            pos = self.find_access_code()

            if pos < 0:
                # 保留最后 63 bit，防止 access code 跨 work 调用边界。
                keep = access_bits_len - 1
                if len(self.bitbuf) > keep:
                    del self.bitbuf[:-keep]
                break

            # 丢弃 access code 前面的杂散 bit。
            if pos > 0:
                del self.bitbuf[:pos]

            # 等待 access code + 32 bit 长度头。
            if len(self.bitbuf) < min_header_bits:
                break

            header_bits = self.bitbuf[access_bits_len:min_header_bits]
            header = self.bits_to_bytes(header_bits)

            # header 格式：00 0F 00 0F，即 16-bit length 重复两次。
            len1 = (header[0] << 8) | header[1]
            len2 = (header[2] << 8) | header[3]

            if len1 != len2 or len1 <= 0 or len1 > 512:
                # 假同步，右移 1 bit 继续搜索。
                del self.bitbuf[0]
                continue

            payload_len_bytes = len1
            total_bits = access_bits_len + header_bits_len + payload_len_bytes * 8

            if len(self.bitbuf) < total_bits:
                break

            payload_bits = self.bitbuf[min_header_bits:total_bits]
            payload = self.bits_to_bytes(payload_bits)

            parsed = self.parse_payload(payload)

            if parsed is not None:
                seq, key_data = parsed

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
                    key_data, dtype=np.uint8
                )
                produced += len(key_data)

                if self.debug:
                    try:
                        ascii_text = key_data.decode("ascii")
                    except Exception:
                        ascii_text = repr(key_data)
                    print(
                        "[GFSK RX] seq=%03d key_hex=%s key_ascii=%s"
                        % (seq, key_data.hex(" "), ascii_text)
                    )

                # 丢弃整个已解析帧。
                del self.bitbuf[:total_bits]
            else:
                # CRC/SOF/CMD 错误，说明是假同步或 bit 错误，右移 1 bit 重搜。
                del self.bitbuf[0]

        self.consume(0, len(inp))
        return produced