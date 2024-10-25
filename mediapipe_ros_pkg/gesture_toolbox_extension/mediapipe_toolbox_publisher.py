import mediapipe as mp
from mediapipe_ros_pkg.gesture_toolbox_extension.extended_frame_lib import FrameAdder
from mediapipe_ros_pkg.mediapipe_publisher import MediaPipePublisher

from gesture_msgs.msg import Frame

class MediaPipePublisherExtended(MediaPipePublisher):
    def __init__(self):
        super(MediaPipePublisherExtended, self).__init__()

        self.frame_adder = FrameAdder()
        self.frame_publisher = self.create_publisher(Frame, '/hand_frame', 5)

    def forward(self, rgbd_msg):
        rgb_image_msg = rgbd_msg.rgb
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=self.bridge.imgmsg_to_cv2(rgb_image_msg, "rgb8"),
        )

        recognition_result = self.recognizer.recognize(mp_image)

        # annotated_image = mp_image.numpy_view().copy()

        # 
        # for hand_landmarks in recognition_result.hand_landmarks:
        #     hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        #     hand_landmarks_proto.landmark.extend(
        #         [
        #             landmark_pb2.NormalizedLandmark(
        #                 x=landmark.x, y=landmark.y, z=landmark.z
        #             )
        #             for landmark in hand_landmarks
        #         ]
        #     )

        #     self.mp_drawing.draw_landmarks(
        #         annotated_image,
        #         hand_landmarks_proto,
        #         self.mp_hands.HAND_CONNECTIONS,
        #         self.mp_drawing_styles.get_default_hand_landmarks_style(),
        #         self.mp_drawing_styles.get_default_hand_connections_style(),
        #     )

        #     self.pointing_vector_estimation(
        #         annotated_image,
        #         self.bridge.imgmsg_to_cv2(rgbd_msg.depth, "passthrough"),
        #         rgbd_msg.rgb_camera_info,
        #         rgbd_msg.depth_camera_info,
        #         hand_landmarks_proto,
        #     )

        # self.gesture_image_publisher.publish(
        #     self.bridge.cv2_to_imgmsg(annotated_image, "rgb8")
        # )

        frame = self.frame_adder.add_frame(recognition_result.hand_landmarks)
        self.frame_publisher.publish(frame.to_ros())