#!/usr/bin/env python
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import facemarkers
import numpy as np
import cv2
import time

import rospy
from std_msgs.msg import Float32MultiArray


def eye_tracking(detection_result):
    face_landmarks_list = detection_result.face_landmarks
    # T = detection_result.facial_transformation_matrixes

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]
        points = np.array(
            [[face_landmarks[i].x, face_landmarks[i].y, face_landmarks[i].z] for i in facemarkers.FACEMESH_EYES],
            dtype="float32")
        return points.flatten()

        # r_vertical_v = (points[10] + points[8]) / 2 - points[11]
        # r_vertical_c = np.mean(points[12:16, :], axis=0) - points[11]
        # r_vertical = r_vertical_v @ r_vertical_c / np.linalg.norm(r_vertical_v)
        #
        # l_vertical_v = (points[2] + points[0]) / 2 - points[3]
        # l_vertical_c = np.mean(points[4:8, :], axis=0) - points[3]
        # l_vertical = l_vertical_v @ l_vertical_c / np.linalg.norm(l_vertical_v)
        #
        # r_horizon_v = (points[10] + points[8]) / 2 - points[10]
        # r_horizon_c = np.mean(points[12:16, :], axis=0) - points[10]
        # r_horizon = r_horizon_v @ r_horizon_c / np.linalg.norm(r_horizon_v)
        #
        # l_horizon_v = (points[2] + points[0]) / 2 - points[2]
        # l_horizon_c = np.mean(points[4:8, :], axis=0) - points[2]
        # l_horizon = l_horizon_v @ l_horizon_c / np.linalg.norm(l_horizon_v)
        #
        # r_close = np.linalg.norm(points[11] - points[9])
        # l_close = np.linalg.norm(points[3] - points[1])
        # print(r_horizon * T[idx][2, 3])
        # return np.array([r_vertical, l_vertical, r_horizon, l_horizon, r_close, l_close])


VisionRunningMode = mp.tasks.vision.RunningMode
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult

tracking_data = Float32MultiArray()


def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global tracking_data
    eye_data = eye_tracking(result)
    if eye_data is not None:
        tracking_data.data = eye_data


base_options = python.BaseOptions(model_asset_path='/home/choi/catkin_ws/src/eeg_data_recording/scripts/face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=False,
                                       output_facial_transformation_matrixes=True,
                                       running_mode=VisionRunningMode.LIVE_STREAM,
                                       result_callback=print_result,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

vid = cv2.VideoCapture(0)
prev_frame_time = 0
new_frame_time = 0
try:
    with FaceLandmarker.create_from_options(options) as landmarker:
        rospy.init_node('eye_publish')
        pub = rospy.Publisher('eye_publisher', Float32MultiArray, queue_size=10)
        rate = rospy.Rate(30)  # frequency of packets not sampling

        flag = 0
        while not rospy.is_shutdown():
            ret, color_image = vid.read()
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_image)

            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time

            landmarker.detect_async(image, int(time.time() * 1000))
            # if flag < 30 * 5 or flag > 30 * 15:
            pub.publish(tracking_data)
            # flag += 1
            rate.sleep()
finally:
    vid.release()
