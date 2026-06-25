import serial
import time
from datetime import datetime

PACKET_SIZE = 35
TIMESET = 5 #5s
ser = serial.Serial('COM4', 115200, timeout=1)

count = 0
timeCnt = 0
start = time.time()
dataArr = []
while timeCnt < TIMESET:
    data = ser.read(PACKET_SIZE)

    

    dataArr.append(data)

    if data[0] == 0xaa and data[1] == 0x01:
        count += 1

    if time.time() - start >= 1:
        timeCnt += 1
        print(f"{count} packets/s")
        count = 0
        start = time.time()
        
filename = datetime.now().strftime("pressData_%Y%m%d_%H%M%S.txt")

with open(filename, "w", encoding="utf-8") as f:
    for data in dataArr:
        f.write(data.hex(" ") + "\n")

print(f"Saved to {filename}")