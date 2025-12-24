# https://docs.ultralytics.com/guides/isolating-segmentation-objects/#how-can-i-crop-isolated-objects-to-their-bounding-boxes-using-ultralytics-yolo11
# Import the necessary libraries
import rclpy # Python library for ROS 2
from rclpy.node import Node # Handles the creation of nodes

from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images

import cv2 # OpenCV library
import numpy as np
from ultralytics import YOLO # YOLO library

from pixel_msgs import PixelCoordinates

class GrabVideo(Node):
  """
  Create an ImageSubscriber class, which is a subclass of the Node class.
  """
  def __init__(self):
    """
    Class constructor to set up the node
    """
    # Initiate the Node class's constructor and give it a name
    super().__init__('GrabVideo')

    self.pub = self.create_publisher(
            PixelCoordinates,
            '/pixel_coordinates',
            10
        )
    # Create the subscriber. This subscriber will receive an Image
    # from the video_frames topic. The queue size is 10 messages.
    self.subscription = self.create_subscription(
      Image, 
      '/camera', 
      self.image_callback, 
      10)
    # self.subscription # prevent unused variable warning
      
    # Used to convert between ROS and OpenCV images
    self.bridge = CvBridge()
    # Load the YOLOv8 model
    self.model = YOLO('yolov8m.pt')


  def image_callback(self, msg):
    """
    Callback function.
    """
    # pixel_detections = PeopleDetected()
    # pixel_detections.header = msg.header  

    frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    results = self.model.predict(frame, conf=0.5, classes=[0])
    detections = results[0]

    
    if detections.boxes is None:
        return

    for box in detections.boxes:
        # Bounding box (pixel coordinates)
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        # Bottom midpoint (important!)
        u = int((x1 + x2) / 2)
        # v = int(y2)
        v = int((y1 + y2) / 2)

        confidence = float(box.conf[0])
        # class_id = int(box.cls[0])

        self.get_logger().info(
            f"Person detected: "
            f"x1={x1}, y1={y1}, x2={x2}, y2={y2} | "
            f"foot_pixel=({u}, {v}) | conf={confidence:.2f}"
        )
        # Fill ROS message
        msg_out = PixelCoordinates()
        # msg_out.frame_
        msg_out.header = msg.header
        msg_out.u = u
        msg_out.v = v
        msg_out.confidence = confidence

        self.pub.publish(msg_out)

        # (Optional) draw
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.circle(frame, (u, v), 5, (0, 0, 255), -1)

    cv2.imshow("YOLO Detection", frame)
    cv2.waitKey(1)
  
def main(args=None):
  
  # Initialize the rclpy library
  rclpy.init(args=args)
  # Create the node
  node = GrabVideo()
  # Spin the node so the callback function is called.
  rclpy.spin(node)
  # Destroy the node explicitly
  # (optional - otherwise it will be done automatically
  # when the garbage collector destroys the node object)
  node.destroy_node()
  # Shutdown the ROS client library for Python
  rclpy.shutdown()
  
if __name__ == '__main__':
  main()
