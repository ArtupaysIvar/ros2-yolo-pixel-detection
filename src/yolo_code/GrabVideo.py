#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np
from ultralytics import YOLO

from pixel_msgs.msg import PixelCoordinates
from ament_index_python.packages import get_package_share_directory


class GrabVideo(Node):

    def __init__(self):
        super().__init__('grab_video')

        # --------------------
        # Parameters
        # --------------------
        self.declare_parameter('image_topic', '/camera')
        self.declare_parameter('publish_topic', '/pixel_coordinates')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('inference_rate_hz', 10.0)
        self.declare_parameter('debug_gui', False)

        image_topic = self.get_parameter('image_topic').value
        publish_topic = self.get_parameter('publish_topic').value
        self.conf_thresh = self.get_parameter('confidence_threshold').value
        rate_hz = self.get_parameter('inference_rate_hz').value
        self.debug_gui = self.get_parameter('debug_gui').value

        # --------------------
        # ROS interfaces
        # --------------------
        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            qos_profile_sensor_data
        )

        self.pub = self.create_publisher(
            PixelCoordinates,
            publish_topic,
            10
        )

        # --------------------
        # YOLO model
        # --------------------
        model_path = os.path.join(
            get_package_share_directory('vidgrabber_pkg'),
            'models',
            'yolov8n.pt'
        )

        self.get_logger().info(f'Loading YOLO model: {model_path}')
        self.model = YOLO(yolov8n.pt)

        # --------------------
        # State
        # --------------------
        self.latest_frame = None
        self.latest_msg = None

        # --------------------
        # Timer for inference
        # --------------------
        self.timer = self.create_timer(
            1.0 / rate_hz,
            self.process_frame
        )

        self.get_logger().info('GrabVideo node started.')

    # ============================================================
    # Callbacks
    # ============================================================

    def image_callback(self, msg: Image):
        """Lightweight image receiver."""
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.latest_msg = msg
        except Exception as e:
            self.get_logger().warn(f'CV bridge failed: {e}')

    def process_frame(self):
        """Heavy YOLO inference (runs at fixed rate)."""
        if self.latest_frame is None:
            return

        results = self.model.predict(
            self.latest_frame,
            conf=self.conf_thresh,
            classes=[0],   # person only
            verbose=False
        )

        detections = results[0].boxes
        if detections is None or len(detections) == 0:
            return

        # --------------------
        # Select BEST detection
        # --------------------
        best_box = max(detections, key=lambda b: b.conf[0])

        x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().astype(int)
        confidence = float(best_box.conf[0])

        u = int((x1 + x2) / 2)
        v = int((y1 + y2) / 2)

        # --------------------
        # Publish message
        # --------------------
        msg_out = PixelCoordinates()
        msg_out.header = self.latest_msg.header
        msg_out.u = u
        msg_out.v = v
        msg_out.confidence = confidence

        self.pub.publish(msg_out)

        self.get_logger().info(
            f'Person detected @ ({u}, {v}) | conf={confidence:.2f}'
        )

        # --------------------
        # Optional debug GUI
        # --------------------
        if self.debug_gui:
            frame = self.latest_frame.copy()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (u, v), 4, (0, 0, 255), -1)
            cv2.imshow('YOLO Detection', frame)
            cv2.waitKey(1)


# ============================================================
# Main
# ============================================================

def main(args=None):
    rclpy.init(args=args)
    node = GrabVideo()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
