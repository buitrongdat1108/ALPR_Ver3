import socket

clientSocket = socket.socket()
clientSocket.connect(('127.0.0.1', 8888))
receivedPlateDir = "Plate_examples/2233785415201474700_2_2_04_20210609174216.jpg"
clientSocket.send(receivedPlateDir.encode())
plateNum = clientSocket.recv(1024)
print("Bien so xe nhan duoc qua Socket: " + plateNum.decode())
timeReq = "Time Request"
clientSocket.send(timeReq.encode())
recvTimeReq = clientSocket.recv(1024)
print("--- Detected in %s seconds ---" % recvTimeReq.decode())


