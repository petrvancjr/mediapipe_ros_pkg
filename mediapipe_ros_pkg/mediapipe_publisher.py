from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from cv_bridge import CvBridge
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from rclpy.node import Node
from sensor_msgs.msg import Image


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
        self.options = vision.GestureRecognizerOptions(base_options=self.base_options)
        self.recognizer = vision.GestureRecognizer.create_from_options(self.options)

        self.dtype = np.float32

        self.bridge = CvBridge()

    def forward(self, image_msg):
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=self.bridge.imgmsg_to_cv2(image_msg, "rgb8"),
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
                annotated_image, None, hand_landmarks_proto, method="pnp"
            )

        # TODO: Send as Topic
        # try:
        #     top_gesture = recognition_result.gestures[0][0]
        #     self.ax.set_title(
        #         f"{top_gesture.category_name} ({top_gesture.score:.2f})",
        #         fontsize=16,
        #         color="black",
        #         fontdict={"verticalalignment": "center"},
        #         pad=16,
        #     )
        # except:
        #     pass

        self.gesture_image_publisher.publish(
            self.bridge.cv2_to_imgmsg(annotated_image, "rgb8")
        )

    # To estimate direction by using pnp method or depth
    # When use pnp method, please use index finger tip and wrist landmarks
    def pointing_vector_estimation(
        self, rgb_image, depth_image, hand_landmarks_proto, method="pnp"
    ):
        (
            index_finger_mcp,
            index_finger_pip,
            index_finger_dip,
            index_finger_tip,
        ) = [
            hand_landmarks_proto.landmark[landmark_idx]
            for landmark_idx in [
                self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
                self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
                self.mp_hands.HandLandmark.INDEX_FINGER_DIP,
                self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            ]
        ]

        # Define 2D image points of the index finger mcp, pip, dip, and tip
        image_points = np.array(
            [
                [
                    index_finger_mcp.x * rgb_image.shape[1],
                    index_finger_mcp.y * rgb_image.shape[0],
                ],
                [
                    index_finger_pip.x * rgb_image.shape[1],
                    index_finger_pip.y * rgb_image.shape[0],
                ],
                [
                    index_finger_dip.x * rgb_image.shape[1],
                    index_finger_dip.y * rgb_image.shape[0],
                ],
                [
                    index_finger_tip.x * rgb_image.shape[1],
                    index_finger_tip.y * rgb_image.shape[0],
                ],
            ],
            dtype=self.dtype,
        )

        # Use PnP method for direction estimation
        if method == "pnp":
            index_finger_object_points = np.array(
                [
                    [0.0, 0.0, -90.0],
                    [0.0, 0.0, -70.0],
                    [0.0, 0.0, -45.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=self.dtype,
            )

            # Camera internals
            focal_length = rgb_image.shape[1]
            center = (rgb_image.shape[1] / 2, rgb_image.shape[0] / 2)
            camera_matrix = np.array(
                [
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1],
                ],
                dtype=self.dtype,
            )

            dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

            # Solve PnP
            _, rvec, tvec = cv2.solvePnP(
                index_finger_object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
            )

            start_point, _ = cv2.projectPoints(
                np.array([0.0, 0.0, 0.0], dtype=self.dtype),
                rvec,
                tvec,
                camera_matrix,
                dist_coeffs,
            )

            end_point, _ = cv2.projectPoints(
                np.array([0.0, 0.0, -300.0], dtype=self.dtype),
                rvec,
                tvec,
                camera_matrix,
                dist_coeffs,
            )

        # TODO
        elif method == "depth":
            # Use depth image for direction estimation
            pass  # Implement depth-based direction estimation if needed

        self.write_poining_vector(rgb_image, start_point, end_point)

    def write_poining_vector(self, image, start_point, end_point):
        cv2.line(
            image,
            (int(start_point[0][0][0]), int(start_point[0][0][1])),
            (int(end_point[0][0][0]), int(end_point[0][0][1])),
            (0, 0, 0),
            3,
        )
