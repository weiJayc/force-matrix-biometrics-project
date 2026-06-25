import serial
import time

PACKET_SIZE = 35
TIMESET = 5 #5s
ser = serial.Serial('COM4', 115200, timeout=1)

count = 0
timeCnt = 0
start = time.time()

while timeCnt < TIMESET:
    data = ser.read(PACKET_SIZE)
        
    if data[0] == 0xaa and data[1] == 0x01:
        count += 1

    if time.time() - start >= 1:
        timeCnt += 1
        print(f"{count} packets/s")
        count = 0
        start = time.time()