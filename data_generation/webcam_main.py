#!/usr/bin/env python
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import facemarkers
import numpy as np
import cv2
import time

import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointField, PointCloud2
from std_msgs.msg import Header

points = np.zeros((16, 3), dtype="float32")


def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])
        global points
        points = np.array(
            [[face_landmarks[i].x, face_landmarks[i].y, face_landmarks[i].z] for i in markers.FACEMESH_EYES], dtype="float32")

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=markers.FACEMESH_LEFT_EYE,
            landmark_drawing_spec=None)

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=markers.FACEMESH_RIGHT_EYE,
            landmark_drawing_spec=None)

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=markers.FACEMESH_LEFT_IRIS,
            landmark_drawing_spec=None)

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=markers.FACEMESH_RIGHT_IRIS,
            landmark_drawing_spec=None)

        # solutions.drawing_utils.draw_landmarks(
        #     image=annotated_image,
        #     landmark_list=face_landmarks_proto,
        #     connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=mp.solutions.drawing_styles
        #     .get_default_face_mesh_tesselation_style())
        # solutions.drawing_utils.draw_landmarks(
        #     image=annotated_image,
        #     landmark_list=face_landmarks_proto,
        #     connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=mp.solutions.drawing_styles
        #     .get_default_face_mesh_contours_style())
        # solutions.drawing_utils.draw_landmarks(
        #     image=annotated_image,
        #     landmark_list=face_landmarks_proto,
        #     connections=mp.solutions.face_mesh.FACEMESH_IRISES,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=mp.solutions.drawing_styles
        #     .get_default_face_mesh_iris_connections_style())

    return annotated_image


VisionRunningMode = mp.tasks.vision.RunningMode
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult

annotated_image = np.zeros([480, 640])


def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global annotated_image
    annotated_image = draw_landmarks_on_image(output_image.numpy_view(), result)


base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
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
        rospy.init_node('pc2_publisher')
        pub = rospy.Publisher('facelandmarks', PointCloud2, queue_size=100)


        def publishPC2():
            global points
            fields = [PointField('x', 0, PointField.FLOAT32, 1),
                      PointField('y', 4, PointField.FLOAT32, 1),
                      PointField('z', 8, PointField.FLOAT32, 1)]

            header = Header()
            header.frame_id = "map"
            header.stamp = rospy.Time.now()
            pc2 = point_cloud2.create_cloud(header, fields, points)
            pub.publish(pc2)


        while not rospy.is_shutdown():
            # while True:

            ret, color_image = vid.read()

            # STEP 3: Load the input image.
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_image)

            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time

            landmarker.detect_async(image, int(time.time() * 1000))
            publishPC2()

            cv2.imshow("image", annotated_image)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

finally:

    # Stop streaming
    vid.release()
    cv2.destroyAllWindows()
