import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import time
import cv2
from cv_bridge import CvBridge, CvBridgeError
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy, QoSProfile
from rclpy.callback_groups import ReentrantCallbackGroup
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import heapq


def depth_image_to_point_cloud(depth, scale, K):
    """
    辅助函数，将深度图像转换为点云
    由于为了减小运算量，图像前期先进行的压缩，因此在转换为点云时，需要进行相应的映射处理
    """
    u = np.arange(0, depth.shape[1])
    v = np.arange(0, depth.shape[0])

    u, v = np.meshgrid(u, v)
    u = u.astype(float)
    v = v.astype(float)

    Z = depth.astype(float) / scale
    X = (2 * u - K[0, 2]) * Z / K[0, 0]
    Y = (2 * v - K[1, 2]) * Z / K[1, 1]

    X = np.ravel(X)
    Y = np.ravel(Y)
    Z = np.ravel(Z)

    valid = Z > 0
    valid2 = valid & (-2 < Y) & (Y < 0) & (Z < 3) & (X > -2) & (X < 2)

    angles = np.unique(np.degrees(np.arctan2(X[valid], Z[valid])).astype(int))
    points = np.transpose(np.vstack((X[valid2], Z[valid2])))

    return points, angles


def calculate_and_create_feasibility_matrix(points, full_angles, X, Z):
    """
    辅助函数，计算并创建可行域矩阵
    """
    angles = (np.degrees(np.arctan2(points[:, 0], points[:, 1]))).astype(int)
    distances = np.sum(points**2, axis=1)

    unique_angles, indices = np.unique(angles, return_inverse=True)
    min_distance_per_angle = np.full_like(unique_angles, np.inf, dtype=float)

    np.minimum.at(min_distance_per_angle, indices, distances)

    angles = (np.degrees(np.arctan2(X, Z))).astype(int)
    distances = X**2 + Z**2

    feasibility_matrix = np.zeros_like(X, dtype=int)
    for i in range(len(X)):
        for j in range(len(X[0])):
            angle = angles[i][j]
            index = np.searchsorted(unique_angles, angle)

            if index != len(unique_angles) and unique_angles[index] == angle:
                if distances[i][j] < min_distance_per_angle[index]:
                    feasibility_matrix[i][j] = 1
            elif (
                index + 1 < len(unique_angles)
                and index != 0
                and unique_angles[index] != angle
            ):
                index2 = np.searchsorted(full_angles, angle)
                if index2 != len(full_angles) and full_angles[index2] == angle:
                    feasibility_matrix[i][j] = 1
                    continue
                min_distance_nearby = min(
                    min_distance_per_angle[index - 1], min_distance_per_angle[index + 1]
                )
                if distances[i][j] < min_distance_nearby:
                    feasibility_matrix[i][j] = 1
            elif -43 < angle < 43:
                feasibility_matrix[i][j] = 1
    return feasibility_matrix


def get_nearest_black_distance(feasibility_matrix):
    """
    辅助函数，计算每个白色点到最近的黑色点的距离
    为了节约运算资源，此处使用了小根堆优化的dijkstra算法
    """
    dx = [0, 1, 0, -1]
    dy = [1, 0, -1, 0]
    height, width = feasibility_matrix.shape
    ans = np.zeros_like(feasibility_matrix, dtype=int)
    cache = [[[] for _ in range(width)] for _ in range(height)]
    flag = np.zeros_like(feasibility_matrix, dtype=bool)
    heap = []

    for x in range(height):
        for y in range(width):
            if feasibility_matrix[x][y] == 1:
                ans[x][y] = -1
                for i in range(4):
                    nx, ny = x + dx[i], y + dy[i]
                    if (
                        nx >= 0
                        and nx < height
                        and ny >= 0
                        and ny < width
                        and feasibility_matrix[nx][ny] == 0
                    ):
                        if not flag[x][y]:
                            heapq.heappush(heap, (1, x, y))
                        ans[x][y] = 1
                        cache[x][y].append((nx, ny))
                        flag[x][y] = True

    while heap:
        _, x, y = heapq.heappop(heap)
        for i in range(4):
            nx, ny = x + dx[i], y + dy[i]
            if (
                nx >= 0
                and nx < height
                and ny >= 0
                and ny < width
                and feasibility_matrix[nx][ny] == 1
                and (not flag[nx][ny])
            ):
                for bx, by in cache[x][y]:
                    distance_sq = (bx - nx) ** 2 + (by - ny) ** 2
                    if distance_sq < ans[nx][ny] or ans[nx][ny] == -1:
                        ans[nx][ny] = min(distance_sq, 10)
                        cache[nx][ny] = []
                        cache[nx][ny].append((bx, by))
                        if not flag[nx][ny]:
                            heapq.heappush(heap, (distance_sq, nx, ny))
                            flag[nx][ny] = True
                    elif distance_sq == ans[nx][ny]:
                        cache[nx][ny].append((bx, by))
                        if not flag[nx][ny]:
                            heapq.heappush(heap, (distance_sq, nx, ny))
                            flag[nx][ny] = True
    return ans


def find_best_dest(map):
    """
    辅助函数，找到最佳目标点
    """
    coords = np.where(map >= 10)

    if coords[0].size == 0:
        return -1, -1

    max_coord_1st_dim = np.max(coords[0])
    filtered_coords = coords[1][coords[0] == max_coord_1st_dim]

    value = 0
    choice = -1
    for y in filtered_coords:
        new_val = abs(y - 20)
        if choice == -1 or new_val < value:
            choice = y
            value = new_val
    return max_coord_1st_dim, choice


class quadroVision(Node):
    """
    该节点负责处理深度图像识别，利用深度图像对图像中的障碍物进行识别
    这一方法用于局部避障，避免机器狗撞到障碍物
    这一方法缺乏全局意识，是在全局地图方案被采用之前的一版方案。在全局地图方案被采用之后，这一方案改为备用
    具体流程为：
    1. 将深度图像转换为点云
    2. 通过点云计算可行域矩阵，即哪些点可以被机器狗所到达
    3. 通过可行域矩阵计算每个白色点到最近的黑色点的距离
    4. 通过最近黑色点的距离，找到最佳目标点
    5. 根据目标点的位置，计算机器狗所需的运动速度
    """

    def __init__(self, ctrl_queue):
        super().__init__("quadro_vision")
        self.bridge = CvBridge()
        self.id = 0
        self.ctrl_queue = ctrl_queue
        self.vision_callback_group = ReentrantCallbackGroup()
        self.sub_cam = self.create_subscription(
            Image,
            "/mi1036022/camera/depth/image_rect_raw",
            self.cam_callback,
            QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            ),
        )
        plt.ion()

    def cam_callback(self, my_img):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(my_img, "16UC1")
            depth_image = cv2.resize(
                depth_image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST
            )
        except CvBridgeError as e:
            print(e)
            return
        scale = 1000
        K = np.array(
            [
                [386.64501953125, 0, 320.5454406738281],
                [0, 386.64501953125, 238.55165100097656],
                [0, 0, 1],
            ]
        )

        points, full_angles = depth_image_to_point_cloud(depth_image, scale, K)

        x = np.linspace(-2, 2, 40)
        z = np.linspace(0, 1.5, 15)
        X, Z = np.meshgrid(x, z)
        feasibility_matrix = calculate_and_create_feasibility_matrix(
            points, full_angles, X, Z
        )

        test = get_nearest_black_distance(feasibility_matrix)
        targetx, targety = find_best_dest(test)

        if targetx < 4:
            self.ctrl_queue.put(["gait", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            rx = targetx - 0
            ry = targety - 20
            length = sqrt(rx**2 + ry**2)
            rx = rx / length * 0.2
            ry = ry / length * 0.6

            self.ctrl_queue.put(["gait", rx, -ry, 0.0, 0.0, 0.0, 0.0])
        return


def quadroVisionMain(ctrl_queue, status):
    print("quadroVisionMain", os.getpid(), os.getppid())
    rclpy.init()
    quadroVisionNode = quadroVision(ctrl_queue)
    try:
        rclpy.spin(quadroVisionNode)
    except KeyboardInterrupt:
        pass
    quadroVisionNode.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    quadroVisionMain()
