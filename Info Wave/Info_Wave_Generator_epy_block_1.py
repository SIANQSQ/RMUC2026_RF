#!/usr/bin/env python3
import numpy as np
from gnuradio import gr
import pmt


class gfsk_key_decoder(gr.basic_block):
    """
    输入：
        GFSK Demod 输出的 unpacked bit stream，每个 uint8 只使用 LSB，值为 0 或 1。

    空口帧格式固定为 27 字节：

        ACCESS_CODE      8 bytes
        LENGTH_CHECK     4 bytes，固定 00 0F 00 0F
        DATA_SLICE       15 bytes

    每个空口帧只有最后 15 字节是有效数据切片。

    一整套有效数据为 135 字节：

        135 / 15 = 9

    所以接收端需要连续接收 9 个空口帧，提取其中的 15 字节 DATA_SLICE，
    拼接成 135 字节完整数据，然后再解析内部的五个子包。

    135 字节内部包含五个子数据包：

        CMD_ID 0x0A01, DATA 长度 24
        CMD_ID 0x0A02, DATA 长度 12
        CMD_ID 0x0A03, DATA 长度 10
        CMD_ID 0x0A04, DATA 长度 8
        CMD_ID 0x0A05, DATA 长度 36

    每个子数据包格式：

        SOF             1 byte
        DATALENGTH      2 bytes
        SEQ             1 byte
        CRC8            1 byte
        CMD_ID          2 bytes
        DATA            N bytes
        CRC16           2 bytes

    输出：
        将五个子包的 DATA 按顺序输出：

            DATA_0A01 + DATA_0A02 + DATA_0A03 + DATA_0A04 + DATA_0A05

        总输出长度：

            24 + 12 + 10 + 8 + 36 = 90 bytes
    """

    def __init__(self, max_access_errors=0, debug=True):
        gr.basic_block.__init__(
            self,
            name="GFSK 15-Byte Slice Packet Decoder",
            in_sig=[np.uint8],
            out_sig=[np.uint8],
        )

        ##################################################
        # 空口帧参数
        ##################################################

        self.ACCESS_CODE = bytes([
            0x16, 0xE8, 0xD3, 0x77,
            0x15, 0x1C, 0x71, 0x2D
        ])

        # ACCESS_CODE 后面的长度校验固定为 00 0F 00 0F
        self.SLICE_LEN = 15

        # 8 字节 ACCESS_CODE + 4 字节 LENGTH_CHECK + 15 字节 DATA_SLICE
        self.AIR_FRAME_LEN = 8 + 4 + self.SLICE_LEN

        # 一整套数据是 135 字节，也就是 9 个 15 字节切片
        self.FULL_PAYLOAD_LEN = 135
        self.SLICE_COUNT = self.FULL_PAYLOAD_LEN // self.SLICE_LEN

        ##################################################
        # 135 字节内部子包参数
        ##################################################

        self.SOF = 0xA5

        # 每个子包固定开销：
        # SOF 1 + DATALENGTH 2 + SEQ 1 + CRC8 1 + CMD_ID 2 + CRC16 2 = 9
        self.PACKET_OVERHEAD = 9

        # 五个子包，顺序固定
        self.PACKET_SPECS = [
            (0x0A01, 24),
            (0x0A02, 12),
            (0x0A03, 10),
            (0x0A04, 8),
            (0x0A05, 36),
        ]

        # 五个子包 DATA 总长度
        self.TOTAL_DATA_LEN = sum(data_len for _, data_len in self.PACKET_SPECS)

        calc_payload_len = sum(
            data_len + self.PACKET_OVERHEAD
            for _, data_len in self.PACKET_SPECS
        )

        if calc_payload_len != self.FULL_PAYLOAD_LEN:
            raise ValueError(
                "135-byte payload length mismatch: calc=%d expected=%d"
                % (calc_payload_len, self.FULL_PAYLOAD_LEN)
            )

        self.max_access_errors = int(max_access_errors)
        self.debug = bool(debug)

        self.access_bits = self.bytes_to_bits(self.ACCESS_CODE)

        # bit 级缓存，用来寻找空口帧
        self.bitbuf = bytearray()

        # 15 字节切片拼接缓存
        self.slice_buf = bytearray()

        ##################################################
        # 输出 tag
        ##################################################

        self.packet_len_key = pmt.intern("packet_len")
        self.seq_key = pmt.intern("seq")
        self.cmd_key = pmt.intern("cmd_id")
        self.frame_index_key = pmt.intern("frame_index")

        # 尽量让调度器一次给足 90 字节输出空间
        try:
            self.set_min_noutput_items(self.TOTAL_DATA_LEN)
        except Exception:
            pass

    ##################################################
    # bit/byte 转换
    ##################################################

    def bytes_to_bits(self, data: bytes):
        """
        MSB first:
            0xA5 -> 1 0 1 0 0 1 0 1
        """
        bits = []
        for b in data:
            for i in range(7, -1, -1):
                bits.append((b >> i) & 1)
        return bits

    def bits_to_bytes(self, bits):
        """
        MSB first bits 转 bytes。
        """
        n = len(bits) // 8
        out = bytearray(n)

        for i in range(n):
            v = 0
            for b in bits[i * 8:(i + 1) * 8]:
                v = (v << 1) | (b & 1)
            out[i] = v

        return bytes(out)

    ##################################################
    # CRC
    ##################################################

    def crc8_atm(self, data: bytes) -> int:
        """
        CRC-8/ATM
        poly = 0x07
        init = 0x00
        """
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
        """
        CRC-16/CCITT-FALSE
        poly = 0x1021
        init = 0xFFFF
        """
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

    ##################################################
    # ACCESS_CODE 搜索
    ##################################################

    def find_access_code(self):
        """
        在 bitbuf 中搜索 ACCESS_CODE。

        返回：
            找到：起始 bit 位置
            找不到：-1
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

    ##################################################
    # 解析 135 字节内部子包
    ##################################################

    def parse_one_packet(self, packet: bytes, expected_cmd: int, expected_data_len: int):
        """
        解析一个子数据包。

        子包格式：

            SOF             1 byte
            DATALENGTH      2 bytes
            SEQ             1 byte
            CRC8            1 byte
            CMD_ID          2 bytes
            DATA            N bytes
            CRC16           2 bytes
        """

        expected_packet_len = expected_data_len + self.PACKET_OVERHEAD

        if len(packet) != expected_packet_len:
            return None

        ##################################################
        # SOF
        ##################################################

        if packet[0] != self.SOF:
            return None

        ##################################################
        # DATALENGTH
        ##################################################

        data_len = (packet[1] << 8) | packet[2]

        if data_len != expected_data_len:
            return None

        ##################################################
        # SEQ
        ##################################################

        seq = packet[3]

        ##################################################
        # CRC8
        #
        # 和原始代码保持一致：
        # CRC8 计算范围为 SOF + DATALENGTH + SEQ，也就是 packet[0:4]
        ##################################################

        rx_crc8 = packet[4]
        calc_crc8 = self.crc8_atm(packet[0:4])

        if rx_crc8 != calc_crc8:
            return None

        ##################################################
        # CMD_ID
        ##################################################

        cmd = (packet[5] << 8) | packet[6]

        if cmd != expected_cmd:
            return None

        ##################################################
        # DATA
        ##################################################

        data_start = 7
        data_end = data_start + data_len
        data = packet[data_start:data_end]

        ##################################################
        # CRC16
        #
        # 和原始代码保持一致：
        # CRC16 计算范围为整个子包去掉最后两个 CRC16 字节
        ##################################################

        rx_crc16 = (packet[-2] << 8) | packet[-1]
        calc_crc16 = self.crc16_ccitt(packet[:-2])

        if rx_crc16 != calc_crc16:
            return None

        return {
            "cmd": cmd,
            "seq": seq,
            "data": data,
        }

    def parse_full_payload(self, payload: bytes):
        """
        解析拼接出来的 135 字节完整有效数据。

        135 字节内部顺序固定为：

            0x0A01, DATA 24 bytes
            0x0A02, DATA 12 bytes
            0x0A03, DATA 10 bytes
            0x0A04, DATA 8 bytes
            0x0A05, DATA 36 bytes
        """

        if len(payload) != self.FULL_PAYLOAD_LEN:
            return None

        packets = []
        offset = 0

        for index, (expected_cmd, expected_data_len) in enumerate(self.PACKET_SPECS):
            packet_len = expected_data_len + self.PACKET_OVERHEAD

            if offset + packet_len > len(payload):
                return None

            packet = payload[offset:offset + packet_len]

            parsed = self.parse_one_packet(
                packet,
                expected_cmd=expected_cmd,
                expected_data_len=expected_data_len,
            )

            if parsed is None:
                return None

            parsed["index"] = index
            packets.append(parsed)

            offset += packet_len

        if offset != len(payload):
            return None

        return packets

    ##################################################
    # 输出五个子包 DATA
    ##################################################

    def output_packets(self, packets, out, produced):
        """
        将五个子包的 DATA 输出到 GNU Radio 流中。
        """

        for pkt in packets:
            cmd = pkt["cmd"]
            seq = pkt["seq"]
            data = pkt["data"]
            index = pkt["index"]

            abs_off = self.nitems_written(0) + produced

            self.add_item_tag(
                0,
                abs_off,
                self.packet_len_key,
                pmt.from_long(len(data)),
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
                self.cmd_key,
                pmt.from_long(cmd),
            )

            self.add_item_tag(
                0,
                abs_off,
                self.frame_index_key,
                pmt.from_long(index),
            )

            out[produced:produced + len(data)] = np.frombuffer(
                data,
                dtype=np.uint8,
            )

            produced += len(data)

            if self.debug:
                try:
                    ascii_text = data.decode("ascii")
                except Exception:
                    ascii_text = repr(data)

                print(
                    "[GFSK RX] index=%d seq=%03d cmd=0x%04X len=%d data_hex=%s data_ascii=%s"
                    % (
                        index,
                        seq,
                        cmd,
                        len(data),
                        data.hex(" "),
                        ascii_text,
                    )
                )

        return produced

    ##################################################
    # GNU Radio work
    ##################################################

    def general_work(self, input_items, output_items):
        inp = input_items[0]
        out = output_items[0]

        ##################################################
        # GFSK Demod 输出每个 uint8 的 LSB 是 bit
        ##################################################

        if len(inp) > 0:
            self.bitbuf.extend((inp & 0x01).astype(np.uint8).tolist())

        produced = 0

        access_bits_len = len(self.access_bits)

        # ACCESS_CODE 后面 4 字节 LENGTH_CHECK
        length_check_bits_len = 32

        # ACCESS_CODE + LENGTH_CHECK
        min_header_bits = access_bits_len + length_check_bits_len

        # 一个完整空口帧：
        # ACCESS_CODE 8 字节 + LENGTH_CHECK 4 字节 + DATA_SLICE 15 字节
        air_frame_bits_len = self.AIR_FRAME_LEN * 8

        # DATA_SLICE 在空口帧中的 bit 位置
        slice_start_bits = min_header_bits
        slice_end_bits = slice_start_bits + self.SLICE_LEN * 8

        ##################################################
        # 每成功解析 135 字节，输出 90 字节
        ##################################################

        while produced + self.TOTAL_DATA_LEN <= len(out):

            ##################################################
            # 如果已经收集到至少 135 字节切片数据，则尝试解析
            ##################################################

            if len(self.slice_buf) >= self.FULL_PAYLOAD_LEN:
                full_payload = bytes(self.slice_buf[:self.FULL_PAYLOAD_LEN])

                parsed_packets = self.parse_full_payload(full_payload)

                if parsed_packets is not None:
                    if self.debug:
                        print(
                            "[GFSK RX] full 135-byte payload parsed successfully"
                        )

                    produced = self.output_packets(
                        parsed_packets,
                        out,
                        produced,
                    )

                    # 丢弃已经解析的 135 字节
                    del self.slice_buf[:self.FULL_PAYLOAD_LEN]

                    continue

                else:
                    ##################################################
                    # 如果解析失败，说明当前 135 字节的切片边界可能不对，
                    # 或者中间有误码/丢帧。
                    #
                    # 因为每个切片是 15 字节，所以这里丢弃最早的一个
                    # 15 字节切片，再继续等待后续切片重新尝试。
                    ##################################################

                    if self.debug:
                        print(
                            "[GFSK RX] 135-byte payload parse failed, drop one 15-byte slice and resync"
                        )

                    del self.slice_buf[:self.SLICE_LEN]
                    continue

            ##################################################
            # slice_buf 不足 135 字节时，继续从 bitbuf 中提取空口帧
            ##################################################

            pos = self.find_access_code()

            if pos < 0:
                ##################################################
                # 未找到 ACCESS_CODE。
                # 保留最后 access_bits_len - 1 bit，
                # 防止 ACCESS_CODE 跨 work 调用边界。
                ##################################################

                keep = access_bits_len - 1

                if len(self.bitbuf) > keep:
                    del self.bitbuf[:-keep]

                break

            ##################################################
            # 丢弃 ACCESS_CODE 前面的杂散 bit
            ##################################################

            if pos > 0:
                del self.bitbuf[:pos]

            ##################################################
            # 等待 ACCESS_CODE + LENGTH_CHECK
            ##################################################

            if len(self.bitbuf) < min_header_bits:
                break

            ##################################################
            # 检查 LENGTH_CHECK
            #
            # 格式固定：
            #   00 0F 00 0F
            #
            # 即 len1 = 15, len2 = 15
            ##################################################

            length_bits = self.bitbuf[access_bits_len:min_header_bits]
            length_bytes = self.bits_to_bytes(length_bits)

            len1 = (length_bytes[0] << 8) | length_bytes[1]
            len2 = (length_bytes[2] << 8) | length_bytes[3]

            if len1 != len2 or len1 != self.SLICE_LEN:
                ##################################################
                # 假同步，右移 1 bit 继续搜索 ACCESS_CODE
                ##################################################

                if self.debug:
                    print(
                        "[GFSK RX] length check error: len1=%d len2=%d expected=%d"
                        % (len1, len2, self.SLICE_LEN)
                    )

                del self.bitbuf[0]
                continue

            ##################################################
            # 等待完整 27 字节空口帧
            ##################################################

            if len(self.bitbuf) < air_frame_bits_len:
                break

            ##################################################
            # 提取后 15 字节有效切片
            ##################################################

            slice_bits = self.bitbuf[slice_start_bits:slice_end_bits]
            data_slice = self.bits_to_bytes(slice_bits)

            if len(data_slice) != self.SLICE_LEN:
                # 理论上不会发生
                del self.bitbuf[0]
                continue

            ##################################################
            # 将 15 字节切片追加到拼接缓存
            ##################################################

            self.slice_buf.extend(data_slice)

            if self.debug:
                print(
                    "[GFSK RX] got one 15-byte slice, collected=%d/%d bytes, slice_hex=%s"
                    % (
                        len(self.slice_buf),
                        self.FULL_PAYLOAD_LEN,
                        data_slice.hex(" "),
                    )
                )

            ##################################################
            # 丢弃已经处理的完整空口帧
            ##################################################

            del self.bitbuf[:air_frame_bits_len]

        ##################################################
        # 消费本次输入
        ##################################################

        self.consume(0, len(inp))
        return produced
