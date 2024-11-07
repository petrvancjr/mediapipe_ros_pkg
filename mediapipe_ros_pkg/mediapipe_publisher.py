import warnings
from copy import deepcopy
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from cv_bridge import CvBridge
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
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

        # TODO: Fix hard code
        mediapipe_model_path = Path(
            "/home/student/tellshow_ros2_ws/src/mediapipe_ros_pkg/models/gesture_recognizer.task"
            # "/home/ws/src/mediapipe_ros_pkg/models/gesture_recognizer.task"
        )

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.base_options = python.BaseOptions(mediapipe_model_path)
        self.options = vision.GestureRecognizerOptions(
            base_options=self.base_options, num_hands=1
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(self.options)

        self.dtype = np.float32

        self.bridge = CvBridge()

        self.exposed_landmarks = list(range(21))

        self.kalman_filter_config()
        self.msg_timestamp = None

    def kalman_filter_config(self):
        dim_x = 6
        dim_z = 3

        kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)

        # Measurement function
        kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
        )

        # Measurement noise covariance
        kf.R = np.eye(dim_z) * 0.05

        self.kf_dict = {
            landmark_idx: deepcopy(kf) for landmark_idx in self.exposed_landmarks
        }

        self.kalman_filter_init()

    def kalman_filter_init(self):
        for key, kf in self.kf_dict.items():
            # Initial state covariance
            kf.P = np.eye(kf.dim_x) * 1.0

            # Initial state
            kf.x = np.zeros(kf.dim_x)

    def kalman_filter_update(self, object_points_dict, dt):
        for key, kf in self.kf_dict.items():
            # State transition matrix
            kf.F = np.array(
                [
                    [1, 0, 0, dt, 0, 0],
                    [0, 1, 0, 0, dt, 0],
                    [0, 0, 1, 0, 0, dt],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1],
                ],
                dtype=self.dtype,
            )

            # Process noise covariance
            kf.Q = Q_discrete_white_noise(
                dim=kf.dim_x / kf.dim_z, dt=dt, block_size=kf.dim_z, var=1.0
            )

            kf.predict()
            z = object_points_dict[key]
            if z is not None:
                y = z - kf.H @ kf.x
                S = kf.H @ kf.P @ kf.H.T + kf.R
                d_M2 = y.T @ np.linalg.inv(S) @ y
                # Mahalanobis distance threshold
                if d_M2 < 9.21:
                    kf.update(z)
                else:
                    print(f"Mahalanobis distance threshold: {d_M2}")

    def get_state(self):
        object_points_dict = {}
        for key, kf in self.kf_dict.items():
            object_points_dict[key] = kf.H @ kf.x
        return object_points_dict

    def forward(self, rgbd_msg):
        rgb_image_msg = rgbd_msg.rgb
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=self.bridge.imgmsg_to_cv2(rgb_image_msg, "rgb8"),
        )

        # TODO : https://stackoverflow.com/questions/78841248/userwarning-symboldatabase-getprototype-is-deprecated-please-use-message-fac
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
                float(rgbd_msg.header.stamp.sec + rgbd_msg.header.stamp.nanosec * 1e-9),
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
        msg_timestamp,
        hand_landmarks_proto,
    ):
        image_points_dict = {}
        for landmark_idx in self.exposed_landmarks:
            if (
                0 <= hand_landmarks_proto.landmark[landmark_idx].x <= 1
                and 0 <= hand_landmarks_proto.landmark[landmark_idx].y <= 1
            ):
                image_points_dict[landmark_idx] = (
                    hand_landmarks_proto.landmark[landmark_idx].x
                    * (rgb_image.shape[1] - 1),
                    hand_landmarks_proto.landmark[landmark_idx].y
                    * (rgb_image.shape[0] - 1),
                )
            else:
                image_points_dict[landmark_idx] = None

        # Time stamp update
        if self.msg_timestamp is None or msg_timestamp - self.msg_timestamp > 1.0:
            self.kalman_filter_init()
            dt = 0.0
        else:
            dt = msg_timestamp - self.msg_timestamp
        self.msg_timestamp = msg_timestamp

        # Estimate 3D coordinates
        object_points_dict = self.estimate_3d_coordinates(
            image_points_dict, depth_image, depth_camera_info
        )

        self.visualize_line(
            rgb_image,
            rgb_camera_info,
            image_points_dict,
            object_points_dict,
            (0, 0, 0),
        )

        # Apply Kalman filter
        self.kalman_filter_update(object_points_dict, dt)
        object_points_dict = self.get_state()
        self.visualize_line(
            rgb_image,
            rgb_camera_info,
            image_points_dict,
            object_points_dict,
            (0, 0, 255),
        )

    def estimate_3d_coordinates(self, image_points, depth_image, depth_camera_info):
        hand_landmarks_3d = {}
        _intrinsics = intrinsics()
        _intrinsics.width = depth_camera_info.width
        _intrinsics.height = depth_camera_info.height
        _intrinsics.fx = depth_camera_info.k[0]
        _intrinsics.fy = depth_camera_info.k[4]
        _intrinsics.ppx = depth_camera_info.k[2]
        _intrinsics.ppy = depth_camera_info.k[5]

        for key, value in image_points.items():
            if value is None:
                hand_landmarks_3d[key] = None
                continue
            else:
                pixel = [int(value[0]), int(value[1])]

                depth_value = self.get_state()[key][2]
                if depth_value > 0:
                    # Inverse proportion with depth value
                    # TODO: Fix hard code
                    crop_size = max(1, int(30 / depth_value))
                else:
                    crop_size = 1

                half_crop_size = crop_size // 2
                cropped_depth = depth_image[
                    max(0, pixel[1] - half_crop_size) : min(
                        depth_image.shape[0] - 1, pixel[1] + half_crop_size + 1
                    ),
                    max(0, pixel[0] - half_crop_size) : min(
                        depth_image.shape[1] - 1, pixel[0] + half_crop_size + 1
                    ),
                ]

                # Filter out invalid depth values
                valid_depths = cropped_depth[cropped_depth > 0]

                # Crop around the pixel and get the depth value using the nearest 12.5% of the depth value
                # TODO: Fix hard code
                if valid_depths.size > 0:
                    depth = np.percentile(valid_depths, 12.5)
                    point_3d = rs2_deproject_pixel_to_point(
                        _intrinsics, pixel, depth * 0.001
                    )
                    hand_landmarks_3d[key] = point_3d
                else:
                    hand_landmarks_3d[key] = None

        return hand_landmarks_3d

    def pca(self, data):
        pca = PCA(n_components=1)
        pca.fit(data)

        return pca.components_[0], pca.mean_

    def visualize_line(
        self,
        rgb_image,
        rgb_camera_info,
        image_points_dict,
        object_points_dict,
        color,
    ):
        image_points = np.array(
            [
                image_points
                for image_points in image_points_dict.values()
                if image_points is not None
            ],
            dtype=np.uint32,
        )
        object_points = np.array(
            [
                object_points
                for object_points in object_points_dict.values()
                if object_points is not None
            ],
            dtype=self.dtype,
        )

        k = np.array(rgb_camera_info.k).reshape(3, 3)
        d = np.array(rgb_camera_info.d)

        # Solve PnP
        try:
            retval, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points.astype(self.dtype),
                k,
                d,
                flags=cv2.SOLVEPNP_P3P,
            )
        except cv2.error:
            retval = False

        if retval:
            direction_vector, mean = self.pca(object_points)

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

            self.write_poining_vector(
                rgb_image, start_point[0][0], end_point[0][0], color
            )

    def write_poining_vector(self, image, start_point, end_point, color):
        cv2.line(
            image,
            (int(start_point[0]), int(start_point[1])),
            (int(end_point[0]), int(end_point[1])),
            color,
            3,
        )
