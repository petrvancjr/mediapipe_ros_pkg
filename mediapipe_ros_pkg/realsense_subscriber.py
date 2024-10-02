import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class RealsenseSubsctiber(Node):
    def __init__(self, callback):
        super().__init__("realsense_listener")

        self.subscription = self.create_subscription(
            Image, "/camera/camera/color/image_raw", self.listener_callback, 10
        )
        self.subscription

        self.callback = callback

    def listener_callback(self, msg):
        # self.get_logger().info(
        #     f"Received an image with width={msg.width}, height={msg.height}"
        # )
        self.callback(msg)


def main(args=None):
    rclpy.init(args=args)

    subscriber = RealsenseSubsctiber()
    rclpy.spin(subscriber)

    subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
