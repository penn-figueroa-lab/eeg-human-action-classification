#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32MultiArray, MultiArrayDimension
import socket
import struct
import numpy as np

int24_coeff = np.array([256 ** 2, 256, 1]).reshape((1, 3))


class EEGUDP:
    def __init__(self, n_electodes=32, packet_size=1,
                 device='tesla', channel_type='dc', freq=250,
                 udp_ip="192.168.200.240", udp_port=50000):
        self.n_electrodes = n_electodes
        self.packet_size = packet_size
        self.device = device
        self.channel_type = channel_type

        if device == 'exg' and channel_type == 'ac':
            self.scaling_factor = 1
        elif device == 'exg' and channel_type == 'dc':
            self.scaling_factor = 100
        elif device == 'tesla' and channel_type == 'ac':
            self.scaling_factor = 20
        elif device == 'tesla' and channel_type == 'dc':
            self.scaling_factor = 100

        self.sock = socket.socket(socket.AF_INET,  # Internet
                                  socket.SOCK_DGRAM)  # UDP
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((udp_ip, udp_port))

        rospy.init_node('eeg_publish')
        self.pub = rospy.Publisher('eeg_publisher', Int32MultiArray, queue_size=10)
        self.rate = rospy.Rate(freq)  # frequency of packets not sampling
        self.msg = Int32MultiArray()

        self.dim1 = MultiArrayDimension()
        self.dim1.label = 'packet size'
        self.dim1.size = self.packet_size
        self.dim1.stride = self.packet_size * self.n_electrodes

        self.dim2 = MultiArrayDimension()
        self.dim2.label = 'number of electrodes'
        self.dim2.size = self.n_electrodes
        self.dim2.stride = self.n_electrodes

        self.msg.layout.dim = [self.dim1, self.dim2]

    def run(self):
        while not rospy.is_shutdown():
            data, addr = self.sock.recvfrom(1024 * 10)  # buffer size is 1024 bytes
            data = struct.unpack('>BBBBIHHQQ' + 'B' * self.n_electrodes * self.packet_size * 3, data)[9:]

            data = np.array(data).reshape((self.n_electrodes * self.packet_size, 3)) * int24_coeff
            data = np.sum(data, axis=1)
            data = np.where(data > 2 ** 24 // 2, data - 2 ** 24, data) * self.scaling_factor

            # subscriber has to do the following transformation
            # data = data.reshape((self.packet_size, self.n_electrodes)).T

            self.msg.data = data
            self.pub.publish(self.msg)
            self.rate.sleep()


eegudp = EEGUDP()
eegudp.run()
