import serial
import time
PACKET_SIZE = 35
HEADER = b'\xAA\x01'

ser = serial.Serial("COM4", 115200, timeout=1)

def read_one_packet(ser):
    ser.reset_input_buffer()

    # 不斷往前找 header
    while True:
        b = ser.read(1)
        if not b:
            return None  # timeout

        if b == HEADER[:1]:   # 先遇到 0xAB
            b2 = ser.read(1)
            if not b2:
                return None

            if b2 == HEADER[1:]:   # 接著是 0xAA
                # 已經抓到 header 的前 2 bytes
                rest = ser.read(PACKET_SIZE - 2)
                if len(rest) != PACKET_SIZE - 2:
                    return None
                return HEADER + rest

def read_serial_packet(ser):
    p = ser.read(PACKET_SIZE)
    if not p:
        return None
    return p

packet_received = 0;
while packet_received < 17:
    packet = read_one_packet(ser)
    if packet:
        print(packet.hex(" "))

        with open("press_img.txt", "a", encoding="utf-8") as f:
            f.write(str(packet_received) + ": " + packet.hex(" ") + '\n')
        packet_received += 1
        time.sleep(3)