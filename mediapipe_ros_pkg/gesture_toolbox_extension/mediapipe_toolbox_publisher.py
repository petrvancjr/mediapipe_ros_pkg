import mediapipe as mp
from mediapipe_ros_pkg.gesture_toolbox_extension.extended_frame_lib import FrameAdder
from mediapipe_ros_pkg.mediapipe_publisher import MediaPipePublisher
from mediapipe.framework.formats import landmark_pb2
from gesture_msgs.msg import Frame
from scipy.interpolate import interp1d
import numpy as np

class V():
    def __init__(self, x,y,z):
        self.x = x
        self.y = y
        self.z = z

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

        annotated_image = mp_image.numpy_view().copy()

        recognition_result_hand_landmarks_3d = []
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
            
            # Prepare format to estimate 3d coordinates
            image_points_dict = {}
            for landmark_idx in self.exposed_landmarks:
                if (
                    0 <= hand_landmarks_proto.landmark[landmark_idx].x <= 1
                    and 0 <= hand_landmarks_proto.landmark[landmark_idx].y <= 1
                ):
                    image_points_dict[landmark_idx] = (
                        hand_landmarks_proto.landmark[landmark_idx].x
                        * (annotated_image.shape[1] - 1),
                        hand_landmarks_proto.landmark[landmark_idx].y
                        * (annotated_image.shape[0] - 1),
                        hand_landmarks_proto.landmark[landmark_idx].z,
                    )
                else:
                    image_points_dict[landmark_idx] = None

            # Get landmark z-axis coordinates
            object_points_dict_z = self.estimate_3d_coordinates(
                image_points_dict,
                self.bridge.imgmsg_to_cv2(rgbd_msg.depth, "passthrough"),
                rgbd_msg.depth_camera_info,
            )

            # Use original landmarks and use estimated z-axis coordinates
            image_points_list = []
            zvalues = []
            for i in range(21):
                if object_points_dict_z[i] is not None:
                    image_points_list.append(V(hand_landmarks[i].x, hand_landmarks[i].y, object_points_dict_z[i][2]))
                    zvalues.append(object_points_dict_z[i][2])
                else:
                    image_points_list.append(V(hand_landmarks[i].x, hand_landmarks[i].y, None))
            
            zvaluemin = min(zvalues)

            for i in range(21):
                if image_points_list[i].z is None:
                    image_points_list[i].z = zvaluemin
                if image_points_list[i].z > zvaluemin * 1.5:
                    image_points_list[i].z = zvaluemin

            # Initialize lists for indices and z values where data is not None
            # indices = []
            # z_values = []

            # # Collect indices and available z values
            # for i in range(21):
            #     if object_points_dict_z[i] is not None:
            #         indices.append(i)
            #         z_values.append(object_points_dict_z[i][2])

            # # Use linear interpolation for missing z values
            # if len(indices) > 1:  # Ensure there are enough points for interpolation
            #     interp_function = interp1d(indices, z_values, kind='linear', fill_value='extrapolate')
            # else:
            #     continue
            #     # raise ValueError("Not enough data points to interpolate missing values.")

            # # Fill in image_points_list with interpolated z values where needed
            # image_points_list = []
            # for i in range(21):
            #     if object_points_dict_z[i] is not None:
            #         z_value = object_points_dict_z[i][2]
            #     else:
            #         # Estimate missing value
            #         z_value = interp_function(i)
            #     print("asdasd", hand_landmarks[i].x, hand_landmarks[i].y, z_value)
            #     image_points_list.append(V(hand_landmarks[i].x, hand_landmarks[i].y, z_value))
            # print("====")
            recognition_result_hand_landmarks_3d.append(image_points_list)        

        # self.gesture_image_publisher.publish(
        #     self.bridge.cv2_to_imgmsg(annotated_image, "rgb8")
        # )

        # print("==================================")
        # print(recognition_result_hand_landmarks_3d)
        # print("----------------------------------")
        # print(recognition_result.hand_landmarks)
        # print("==================================")

        frame = self.frame_adder.add_frame(recognition_result_hand_landmarks_3d)
        self.frame_publisher.publish(frame.to_ros())