# core/serial_comm.py
import serial
import time

class Arduino:
    def __init__(self, port="/dev/ttyUSB0", baud=115200):
        self.ser = serial.Serial(port, baud, timeout=0.1)
        time.sleep(2)

    def send_servo(self, servo_id, angle):
        cmd = f"S {servo_id} {angle}\n"
        self.ser.write(cmd.encode())
