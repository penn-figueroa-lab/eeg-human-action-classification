import socket
import struct


class EEGUDP:
    def __init__(self):
        UDP_IP = "192.168.200.240"
        UDP_PORT = 50000

        self.sock = socket.socket(socket.AF_INET,  # Internet
                                  socket.SOCK_DGRAM)  # UDP
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((UDP_IP, UDP_PORT))

    def reading(self, decode=True):
        data, addr = self.sock.recvfrom(1024 * 10)  # buffer size is 1024 bytes
        # 32 -> 960, 40 -> 1200
        if decode:
            return {'eeg': struct.unpack('>BBBBIHHQQ' + 'B' * 960, data)}
        else:
            return data
