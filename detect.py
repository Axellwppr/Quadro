import os
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy, QoSProfile
from rclpy.callback_groups import ReentrantCallbackGroup
import numpy as np
import onnxruntime as ort
from utils import *
import tf2_ros


class quadroDetect(Node):
    """
    该节点负责处理蓝色球的识别
    首先加载预训练的yolov5模型，并订阅相机的图像话题
    当检测到球体后，通过深度图像的对应区域，结合相机内参矩阵，计算出球体的平面坐标
    最后将base_footprint坐标转换到odom坐标系下，储存到进程共享的status字典中，触发导航节点规划向球靠近的路径，完成路径的规划并执行
    """

    def __init__(self, ctrl_queue, status):
        super().__init__("quadro_detect")
        self.bridge = CvBridge()
        self.id = 0
        self.ctrl_queue = ctrl_queue
        self.status = status
        self.box = None
        self.count = 0
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.vision_callback_group = ReentrantCallbackGroup()

        self.sub_cam = self.create_subscription(
            Image,
            "/mi1036022/camera/color/image_raw",
            self.cam_callback,
            QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            ),
        )
        self.sub_depth = self.create_subscription(
            Image,
            "/mi1036022/camera/aligned_depth_to_color/image_raw",
            self.depth_callback,
            QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            ),
        )
        self.model = ort.InferenceSession(
            "best-sim.onnx", providers=["CUDAExecutionProvider"]
        )
        self.input_name = self.get_input_name()
        self.output_name = self.get_output_name()

    def get_transform(self, main, sub, time):
        transform_stamped = self.tf_buffer.lookup_transform(main, sub, time)
        translation = transform_stamped.transform.translation
        rotation = transform_stamped.transform.rotation
        return translation, rotation

    def get_input_name(self):
        """获取输入节点名称"""
        input_name = []
        for node in self.model.get_inputs():
            input_name.append(node.name)

        return input_name

    def get_output_name(self):
        """获取输出节点名称"""
        output_name = []
        for node in self.model.get_outputs():
            output_name.append(node.name)

        return output_name

    def get_input_feed(self, image_numpy):
        """获取输入numpy"""
        input_feed = {}
        for name in self.input_name:
            input_feed[name] = image_numpy
        return input_feed

    def depth_callback(self, my_img):
        if self.box is None:
            return
        (left, top, width, height) = self.box
        self.box = None
        print("depth box=", left, top, width, height)
        try:
            depth_image = self.bridge.imgmsg_to_cv2(my_img, "16UC1")
        except CvBridgeError as e:
            print(e)
            return
        mid_w = left + width // 2
        mid_h = top + height // 2
        radius = min(5, width // 2, height // 2)

        start_h = max(0, mid_h - radius)
        end_h = min(depth_image.shape[0], mid_h + radius)
        start_w = max(0, mid_w - radius)
        end_w = min(depth_image.shape[1], mid_w + radius)
        region = depth_image[start_h:end_h, start_w:end_w]

        valid_pixels = region[region > 0]
        if valid_pixels.size > 0:
            ball_center_z = np.mean(valid_pixels)
        else:
            ball_center_z = 0
            return
        ball_center_z = depth_image[mid_h, mid_w]
        scale = 1000
        K = np.array(
            [
                [386.64501953125, 0, 320.5454406738281],
                [0, 386.64501953125, 238.55165100097656],
                [0, 0, 1],
            ]
        )
        ball_center_z /= scale
        X = (mid_w - K[0, 2]) * ball_center_z / K[0, 0]
        Y = (mid_h - K[1, 2]) * ball_center_z / K[1, 1]

        dx = ball_center_z
        dy = -X
        # print("dx,dy=", dx, dy)
        try:
            translation, rotation = self.get_transform(
                "odom", "base_footprint", rclpy.time.Time()
            )

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            print(e)
            return
        q = (rotation.w, rotation.x, rotation.y, rotation.z)
        rotated_point = qv_rotate(q, (dx, dy, 0))
        if self.status["stage"] == "exploring":
            self.status["stage"] = "targeting"
        self.status["target"] = (
            rotated_point[0] + translation.x,
            rotated_point[1] + translation.y,
            0,
        )
        self.status["target_z"] = ball_center_z

    def unwrap_detection(self, image_width, image_height, output_data):
        confidences = []
        boxes = []
        x_factor = image_width / 640
        y_factor = image_height / 640

        mask = (output_data[:, 4] >= 0.8) & (output_data[:, 5] > 0.6)

        selected_rows = output_data[mask]
        confidences = selected_rows[:, 4].tolist()
        x, y, w, h = (
            selected_rows[:, 0],
            selected_rows[:, 1],
            selected_rows[:, 2],
            selected_rows[:, 3],
        )

        left = ((x - 0.5 * w) * x_factor).astype(int)
        top = ((y - 0.5 * h) * y_factor).astype(int)
        width = (w * x_factor).astype(int)
        height = (h * y_factor).astype(int)

        boxes = np.column_stack([left, top, width, height]).tolist()
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.3)
        result_confidences = []
        result_boxes = []

        for i in indexes:
            result_confidences.append(confidences[i])
            result_boxes.append(boxes[i])
        if len(result_confidences) == 0:
            return None
        else:
            max_conf_idx = np.argmax(confidences)
            max_box = boxes[max_conf_idx]
            return max_box

    def cam_callback(self, my_img):
        if self.status["stage"] == "start_nav" or self.status["stage"] == "ready":
            return
        self.count += 1
        if self.count < 4:
            return
        self.count = 0
        try:
            srcimg = self.bridge.imgmsg_to_cv2(my_img, "rgb8")
            h, w = srcimg.shape[:2]
            img = cv2.resize(srcimg, (640, 640))
            image_height, image_width, _ = srcimg.shape
        except CvBridgeError as e:
            print(e)

        # 预处理图片使其符合onnx神经网络的输入要求
        img = img.transpose(2, 0, 1)
        img = img.astype(dtype=np.float32)  # onnx模型的类型是type: float32[ , , , ]
        img /= 255.0
        img = np.expand_dims(img, axis=0)  # [3, 640, 640]扩展为[1, 3, 640, 640]
        # img尺寸(1, 3, 640, 640)
        input_feed = self.get_input_feed(img)  # dict:{ input_name: input_value }
        pred = self.model.run(None, input_feed)[0][0]  # <class 'numpy.ndarray'>
        # print(pred.shape)
        ball = self.unwrap_detection(image_width, image_height, pred)
        if ball is not None:
            self.box = ball

        return


def quadroDetectMain(ctrl_queue=None, status=None):
    print("quadroDetectMain", os.getpid(), os.getppid())
    rclpy.init()
    quadroDetectNode = quadroDetect(ctrl_queue, status)
    try:
        rclpy.spin(quadroDetectNode)
    except KeyboardInterrupt:
        pass
    quadroDetectNode.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    quadroDetectMain()
