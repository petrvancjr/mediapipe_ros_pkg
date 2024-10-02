from pathlib import Path

import rclpy

from mediapipe_ros.mediapipe_publisher import MediaPipePublisher
from mediapipe_ros.realsense_subscriber import RealsenseSubsctiber

# TODO parameterize
model_path = Path("/home/ws/models/gesture_recognizer.task")


def main():
    rclpy.init(args=None)

    # TODO parameterize
    mediapipe_publisher = MediaPipePublisher(
        model_path=model_path,
    )

    realsense_subscriber = RealsenseSubsctiber(mediapipe_publisher.forward)
    rclpy.spin(realsense_subscriber)

    realsense_subscriber.destroy_node()
    mediapipe_publisher.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
