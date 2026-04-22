import numpy as np
import serial
from gnuradio import gr

class blk(gr.sync_block):
    def __init__(self, port='/dev/ttyUSB0', baudrate=9600):
        gr.sync_block.__init__(self, name='REFEREE Output', in_sig=[], out_sig=[np.uint8])
        self.ser = serial.Serial(port, baudrate, timeout=1)

    def work(self, input_items, output_items):
        if self.ser.in_waiting > 0:
            data = self.ser.read(self.ser.in_waiting)
            data_array = np.frombuffer(data, dtype=np.uint8)
            n_items = min(len(data_array), len(output_items[0]))
            output_items[0][:n_items] = data_array[:n_items]
            return n_items
        return 0
