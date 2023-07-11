#!/usr/bin/env python
import rospy
import tf2_ros


def tf2_list(tf2):
    return [tf2.transform.translation.x,
            tf2.transform.translation.y,
            tf2.transform.translation.z,
            tf2.transform.rotation.x,
            tf2.transform.rotation.y,
            tf2.transform.rotation.z,
            tf2.transform.rotation.w]


class OPTITRACKROS:
    def __init__(self):
        rospy.init_node('tf2_listener')
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.rate = rospy.Rate(100.0)
        self.tf_list = ['shoulder_l', 'elbow_l', 'wrist_l',
                        'shoulder_r', 'elbow_r', 'wrist_r']

    def reading(self):
        readings = {}
        try:
            for tf in self.tf_list:
                readings[tf] = tf2_list(self.tfBuffer.lookup_transform('world', tf, rospy.Time(0)))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return
        return readings

