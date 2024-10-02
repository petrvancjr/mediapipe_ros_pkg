import rclpy

from mediapipe_ros_pkg.mediapipe_publisher import MediaPipePublisher
from mediapipe_ros_pkg.realsense_subscriber import RealsenseSubsctiber


def main(args=None):
    rclpy.init(args=args)

    mediapipe_publisher = MediaPipePublisher()
    realsense_subscriber = RealsenseSubsctiber(mediapipe_publisher.forward)

    rclpy.spin(realsense_subscriber)

    realsense_subscriber.destroy_node()
    mediapipe_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
