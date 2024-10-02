import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from rclpy.node import Node
from sensor_msgs.msg import Image


class MediaPipePublisher(Node):
    def __init__(self, model_path):
        super().__init__("mediapipe_gesture_publisher")
        self.publiser = self.create_publisher(Image, "/mediapipe_gesture", 10)

        self.model_path = model_path

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.base_options = python.BaseOptions(model_asset_path=self.model_path)
        self.options = vision.GestureRecognizerOptions(base_options=self.base_options)
        self.recognizer = vision.GestureRecognizer.create_from_options(self.options)

    def forward(self, image_msg):
        image = self._ros_image_to_cv2(image_msg)

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        recognition_result = self.recognizer.recognize(image)

        hand_landmarks = recognition_result.hand_landmarks

        image = image.numpy_view().copy()

        for hand_landmarks in hand_landmarks:
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
                image,
                hand_landmarks_proto,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style(),
            )

        # TODO: Fix this
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

        self.publiser.publish(self._cv2_to_ros_image(image))

    def _ros_image_to_cv2(self, image_msg):
        if image_msg.encoding == "rgb8":
            img = np.frombuffer(image_msg.data, np.uint8).reshape(
                image_msg.height, image_msg.width, 3
            )
        elif image_msg.encoding == "bgr8":
            img = np.frombuffer(image_msg.data, np.uint8).reshape(
                image_msg.height, image_msg.width, 3
            )
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(
                "Unsupported image encoding: {}".format(image_msg.encoding)
            )

        return img

    def _cv2_to_ros_image(self, cv2_image):
        image_msg = Image()
        image_msg.height = cv2_image.shape[0]
        image_msg.width = cv2_image.shape[1]
        image_msg.encoding = "rgb8"
        image_msg.is_bigendian = 0
        image_msg.data = cv2_image.tobytes()
        image_msg.step = len(image_msg.data) // image_msg.height
        return image_msg
