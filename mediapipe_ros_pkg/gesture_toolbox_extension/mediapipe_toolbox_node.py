import rclpy

from mediapipe_ros_pkg.gesture_toolbox_extension.mediapipe_toolbox_publisher import MediaPipePublisherExtended
from mediapipe_ros_pkg.realsense_subscriber import RealsenseSubsctiber


def main(args=None):
    rclpy.init(args=args)

    mediapipe_publisher = MediaPipePublisherExtended()
    realsense_subscriber = RealsenseSubsctiber(mediapipe_publisher.forward)

    rclpy.spin(realsense_subscriber)

    realsense_subscriber.destroy_node()
    mediapipe_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
