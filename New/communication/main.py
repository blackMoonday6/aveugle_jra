import serial
import  time
serialcom=serial.Serial('/dev/ttyACM0',9600)
serialcom.timeout=1
while True:
	i=input("put position value").strip()
	if i=='done':
		print('programmefinnie')
		break


	serialcom.write(i.encode())
	iinput = serialcom.readline().decode('ascii')
	serialcom.write(i.encode())
	print(iinput)

serialcom.close()


