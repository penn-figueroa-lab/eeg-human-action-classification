#!/usr/bin/env python
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np

class WEBCAMROS:
    def __init__(self):
        rospy.init_node('listener')
        self.listener = rospy.Subscriber("/facelandmarks", PointCloud2, self.callback, queue_size=1, buff_size=1000)
        self.xyz = np.zeros((16, 3))
    
    def callback(self, ros_point_cloud):
        xyz = np.empty((0, 3))
        gen = pc2.read_points(ros_point_cloud, skip_nans=True)
        int_data = list(gen)

        for x in int_data:
            xyz = np.append(xyz,[[x[0],x[1],x[2]]], axis = 0)
        self.xyz = xyz
        
    def reading(self):
        return {'webcam': self.xyz.tolist()}

  
