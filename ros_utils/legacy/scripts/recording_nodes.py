#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32MultiArray, Int32MultiArray, String
from geometry_msgs.msg import PoseArray

class LABELPUB:
    def __init__(self):
        self.pub = rospy.Publisher("label_publisher", String, queue_size=10)


class OPTITRACKSUB:
    def __init__(self):
        self.listener = rospy.Subscriber("/natnet_ros/fullbody/pose", PoseArray, self.callback, queue_size=1,
                                         buff_size=100)
        self.data = [0] * 16 * 3
        # self.label = ["right eye vertical", "left eye vertical", "right eye horizontal", "left eye horizontal",
        #               "right eye blink", "left eye blink"]
        self.label = []
        self.xyz = ['x', 'y', 'z']
        for i in range(51):
            for j in range(3):
                self.label += ["body tracker " + str(i + 1) + " (" + self.xyz[j] + ")"]

    def callback(self, reading):
        data = [[i.position.x, i.position.y, i.position.z] for i in reading.poses]
        self.data = sum(data, [])


class WEBCAMSUB:
    def __init__(self):
        self.listener = rospy.Subscriber("/eye_publisher", Float32MultiArray, self.callback, queue_size=1,
                                         buff_size=100)
        self.data = [0] * 16 * 3
        # self.label = ["right eye vertical", "left eye vertical", "right eye horizontal", "left eye horizontal",
        #               "right eye blink", "left eye blink"]
        self.label = []
        self.xyz = ['x', 'y', 'z']
        for i in range(16):
            for j in range(3):
                self.label += ["eye tracker " + str(i + 1) + " (" + self.xyz[j] + ")"]

    def callback(self, reading):
        self.data = list(reading.data)


class EEGSUB:
    def __init__(self):
        self.listener = rospy.Subscriber("/eeg_publisher", Int32MultiArray, self.callback, queue_size=1,
                                         buff_size=100)
        self.data = [0] * 32 * 1
        self.label = []
        for i in range(1):
            for j in range(32):
                self.label += ["electrode " + str(j + 1) + " (" + str(i + 1) + ")"]

    def callback(self, reading):
        self.data = list(reading.data)
