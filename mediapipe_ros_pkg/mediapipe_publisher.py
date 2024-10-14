from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from cv_bridge import CvBridge
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pyrealsense2 import intrinsics, rs2_deproject_pixel_to_point
from rclpy.node import Node
from sensor_msgs.msg import Image
from sklearn.decomposition import PCA


class MediaPipePublisher(Node):
    def __init__(self):
        super().__init__("mediapipe_gesture_publisher")
        self.gesture_image_publisher = self.create_publisher(
            Image, "/mediapipe_gesture", 10
        )

        # TODO: Fix this
        mediapipe_model_path = Path(
            "/home/ws/src/mediapipe_ros_pkg/models/gesture_recognizer.task"
        )

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.base_options = python.BaseOptions(mediapipe_model_path)
        self.options = vision.GestureRecognizerOptions(
            base_options=self.base_options, num_hands=2
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(self.options)

        self.dtype = np.float32

        self.bridge = CvBridge()

        self.exposed_landmarks = [
            self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
            self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_DIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
        ]

    def forward(self, rgbd_msg):
        rgb_image_msg = rgbd_msg.rgb
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=self.bridge.imgmsg_to_cv2(rgb_image_msg, "rgb8"),
        )

        recognition_result = self.recognizer.recognize(mp_image)

        annotated_image = mp_image.numpy_view().copy()

        for hand_landmarks in recognition_result.hand_landmarks:
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    )
                    for landmark in hand_landmarks
                ]
            )

            self.mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style(),
            )

            self.pointing_vector_estimation(
                annotated_image,
                self.bridge.imgmsg_to_cv2(rgbd_msg.depth, "passthrough"),
                rgbd_msg.rgb_camera_info,
                rgbd_msg.depth_camera_info,
                hand_landmarks_proto,
            )

        self.gesture_image_publisher.publish(
            self.bridge.cv2_to_imgmsg(annotated_image, "rgb8")
        )

    def pointing_vector_estimation(
        self,
        rgb_image,
        depth_image,
        rgb_camera_info,
        depth_camera_info,
        hand_landmarks_proto,
    ):
        image_points = np.array(
            [
                (
                    hand_landmarks_proto.landmark[landmark_idx].x
                    * (rgb_image.shape[1] - 1),
                    hand_landmarks_proto.landmark[landmark_idx].y
                    * (rgb_image.shape[0] - 1),
                )
                for landmark_idx in self.exposed_landmarks
                if 0 <= hand_landmarks_proto.landmark[landmark_idx].x <= 1
                and 0 <= hand_landmarks_proto.landmark[landmark_idx].y <= 1
            ],
            dtype=np.uint32,
        )

        object_points = self.estimate_3d_coordinates(
            image_points, depth_image, depth_camera_info
        )

        if len(object_points) >= 4:  # Because pnp method needs at least 4 points
            direction_vector, mean = self.pca(object_points)

            # 2. Printing line by using PNP method(origin is INDEX_FINGER_TIP) and one
            self.visualize_line(
                rgb_image,
                rgb_camera_info,
                image_points,
                object_points,
                (direction_vector, mean),
            )

    def estimate_3d_coordinates(self, image_points, depth_image, depth_camera_info):
        hand_landmarks_3d = []
        _intrinsics = intrinsics()
        _intrinsics.width = depth_camera_info.width
        _intrinsics.height = depth_camera_info.height
        _intrinsics.fx = depth_camera_info.k[0]
        _intrinsics.fy = depth_camera_info.k[4]
        _intrinsics.ppx = depth_camera_info.k[2]
        _intrinsics.ppy = depth_camera_info.k[5]

        for landmark_2d in image_points:
            pixel = [int(landmark_2d[0]), int(landmark_2d[1])]

            # Crop around the pixel and get the depth value using the nearest 12.5% of the depth value
            depth_value = depth_image[pixel[1], pixel[0]]
            if depth_value > 0:
                crop_size = max(
                    1, int(1000 / depth_value)
                )  # Inverse proportion with depth value
            else:
                crop_size = 5  # Default crop size if depth value is invalid
            half_crop_size = crop_size // 2
            cropped_depth = depth_image[
                max(0, pixel[1] - half_crop_size) : min(
                    depth_image.shape[0] - 1, pixel[1] + half_crop_size + 1
                ),
                max(0, pixel[0] - half_crop_size) : min(
                    depth_image.shape[1] - 1, pixel[0] + half_crop_size + 1
                ),
            ]
            valid_depths = cropped_depth[
                cropped_depth > 0
            ]  # Filter out invalid depth values

            if valid_depths.size > 0:
                depth = np.percentile(
                    valid_depths, 12.0
                )  # Use the nearest 12.0% depth value
            else:
                continue

            point_3d = rs2_deproject_pixel_to_point(_intrinsics, pixel, depth * 0.001)
            hand_landmarks_3d.append(point_3d)

        return np.array(hand_landmarks_3d, dtype=self.dtype)

    def pca(self, data):
        pca = PCA(n_components=1)
        pca.fit(data)

        return pca.components_[0], pca.mean_

    def visualize_line(
        self, rgb_image, rgb_camera_info, image_points, object_points, line_params
    ):
        k = np.array(rgb_camera_info.k).reshape(3, 3)
        d = np.array(rgb_camera_info.d)

        # Solve PnP
        retval, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points.astype(self.dtype),
            k,
            d,
            flags=cv2.SOLVEPNP_P3P,
        )

        if retval:
            direction_vector, mean = line_params

            # 0.3m away from the mean point
            start_object_point = mean + direction_vector * 0.3
            end_object_point = mean - direction_vector * 0.3

            # set start point self.mp_hands.HandLandmark.INDEX_FINGER_TIP
            start_point, _ = cv2.projectPoints(
                start_object_point,
                rvec,
                tvec,
                k,
                d,
            )

            end_point, _ = cv2.projectPoints(end_object_point, rvec, tvec, k, d)

            self.write_poining_vector(rgb_image, start_point[0][0], end_point[0][0])

    def write_poining_vector(self, image, start_point, end_point):
        cv2.line(
            image,
            (int(start_point[0]), int(start_point[1])),
            (int(end_point[0]), int(end_point[1])),
            (0, 0, 0),
            3,
        )
