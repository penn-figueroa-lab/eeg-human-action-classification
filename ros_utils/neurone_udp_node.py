#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32MultiArray, MultiArrayDimension

import socket
import struct

import numpy as np

# Coefficients required for converting int24 to int32
# Bittium NeurOne Tesla EEG system uses 24 bit system
# Check the device manual for further details
int24_coeff = np.array([256 ** 2, 256, 1]).reshape((1, 3))


class EEGPublisher:
    def __init__(self, n_electrodes: int = 32, packet_size: int = 1,
                 device: str = 'tesla', channel_type: str = 'dc', freq: float = 250.,
                 udp_ip: str = "192.168.200.240", udp_port: int = 50000):

        # Intialization
        self.n_electrodes = n_electrodes
        self.packet_size = packet_size
        self.device = device
        self.channel_type = channel_type

        # Scaling factor based on the device manual
        if device == 'exg' and channel_type == 'ac':
            self.scaling_factor = 1
        elif device == 'exg' and channel_type == 'dc':
            self.scaling_factor = 100
        elif device == 'tesla' and channel_type == 'ac':
            self.scaling_factor = 20
        elif device == 'tesla' and channel_type == 'dc':
            self.scaling_factor = 100

        # UDP socket initialization
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((udp_ip, udp_port))

        # EEG node initialization
        rospy.init_node('eeg_publish')
        self.pub = rospy.Publisher('eeg_node', Int32MultiArray, queue_size=10)
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
            # Receive UDP data and decode it
            data, addr = self.sock.recvfrom(1024 * 10)  # buffer size is 1024 bytes
            data = struct.unpack('>BBBBIHHQQ' + 'B' * self.n_electrodes * self.packet_size * 3, data)[9:]

            # Reshaping the data as a NumPy array
            data = np.array(data).reshape((self.n_electrodes * self.packet_size, 3)) * int24_coeff
            data = np.sum(data, axis=1)
            data = np.where(data > 2 ** 24 // 2, data - 2 ** 24, data) * self.scaling_factor

            # Publish the data
            self.msg.data = data
            self.pub.publish(self.msg)
            self.rate.sleep()
        print("EEG publisher terminated")


class EEGSubscriber:
    def __init__(self, n_electrodes: int = 32, packet_size: int = 1):
        # Initialization
        self.n_electrodes = n_electrodes
        self.packet_size = packet_size

        # EEG subscriber initialization
        self.sub = rospy.Subscriber("/eeg_node",
                                    Int32MultiArray, self.callback, queue_size=1,
                                    buff_size=100)
        self.data = [0] * self.n_electrodes * self.packet_size
        self.label = []
        for i in range(self.packet_size):
            for j in range(self.n_electrodes):
                self.label += ["Electrode " + str(j + 1) + " (" + str(i + 1) + ")"]

    def callback(self, reading):
        self.data = list(reading.data)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_electrodes', type=int, required=True, default=32)
    parser.add_argument('--packet_size', type=int, required=True, default=1)
    parser.add_argument('--device', type=str, required=True, default="tesla")
    parser.add_argument('--channel_type', type=str, required=True, default="ac")
    parser.add_argument('--udp_ip', type=str, required=True, default="192.168.200.240")
    parser.add_argument('--udp_port', type=int, required=True, default=50000)
    args = parser.parse_args()

    eeg_pub = EEGPublisher(n_electrodes=args['n_electrodes'], packet_size=args['packet_size'],
                           device=args['device'], channel_type=args['channel_type'],
                           udp_ip=args['udp_ip'], udp_port=args['udp_port'])
    eeg_sub = EEGSubscriber(n_electrodes=args['n_electrodes'], packet_size=args['packet_size'])
    eeg_pub.run()
