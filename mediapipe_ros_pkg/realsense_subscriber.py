import rclpy
from rclpy.node import Node
from realsense2_camera_msgs.msg import RGBD


class RealsenseSubsctiber(Node):
    def __init__(self, callback):
        super().__init__("realsense_listener")

        self.subscription = self.create_subscription(
            RGBD, "/camera/camera/rgbd", self.listener_callback, 10
        )
        self.subscription

        self.callback = callback

    def listener_callback(self, rgbd_msg):
        self.callback(rgbd_msg)


def main(args=None):
    rclpy.init(args=args)

    subscriber = RealsenseSubsctiber()
    rclpy.spin(subscriber)

    subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
