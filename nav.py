import os
import time
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import cv2
import numpy as np
import tf2_ros
from rclpy.callback_groups import ReentrantCallbackGroup
import heapq
from utils import *
from concurrent.futures import ThreadPoolExecutor


class quadroNav(Node):
    """
    该节点负责处理机器狗的导航，通过接收全局的地图消息，并结合目前狗所处的控制状态(status.stage)做出导航响应
    同时该节点也负责处理机器狗的定位，通过接收来自tf的消息，通过一定的数学运算，获取机器狗在全局地图中的位置和朝向，并将其更新在全局状态中，便于其他主控节点使用

    该节点的实现逻辑有一些复杂，主要是两个函数在发挥作用：
        1. update_map
            该函数接收来自全局地图的消息，并查询机器人当前所处的位置，使用堆优化的dijkstra算法，计算地图的单源最短路径
            如果当前处于探索状态（没有找到球），会寻找最值得探索的前沿（Frontier），并利用dijkstra算法的预处理信息得到一条最短路径，并储存在path中
            如果当前处于追球状态（找到球），会计算球的位置，并利用dijkstra算法的预处理信息得到一条到达球的最短路径，并储存在path中
            由于路径规划计算量较大，需要较长时间（实测0.8s），因此路径规划只在一下情况触发
                1. 机器狗刚刚启动
                2. 机器狗找到球
                3. 机器狗路径跟踪失败
                4. 距离上一次路径规划超过若干秒
                5. 机器狗到达上一条路径的终点
        2. update_path
            该函数根据path中的路径，结合机器人当前的定位信息，通过一定的策略计算机器狗的线速度和角速度，并将其储存在进程共享的ctrl_queue中，由控制节点读取并执行
    """

    def __init__(self, ctrl_queue, status):
        super().__init__("quadro_Nav")
        self.ctrl_queue = ctrl_queue
        self.status = status
        self.nav_callback_group = ReentrantCallbackGroup()
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            "/mi1036022/global_costmap/costmap",
            self.update_map,
            6,
            callback_group=self.nav_callback_group,
        )
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.timing_update = self.create_timer(
            0.2, self.update_path, callback_group=self.nav_callback_group
        )
        self.count = 0
        self.res = 0.5
        self.origin = []

        self.path = []
        self.isNav = False
        self.lasttime = 0
        self.lastTime = 0
        self.renav = False
        self.stage_cache = None

    def get_transform(self, main, sub, time):
        transform_stamped = self.tf_buffer.lookup_transform(main, sub, time)
        translation = transform_stamped.transform.translation
        rotation = transform_stamped.transform.rotation
        return translation, rotation

    def point_odom_to_world(self, point):
        """odom坐标系下的点转换到map坐标系下"""
        translation, rotation = self.get_transform("map", "odom", self.lasttime)
        q = (rotation.w, rotation.x, rotation.y, rotation.z)
        rotated_point = qv_rotate(q, point)
        transformed_point = (
            translation.x + rotated_point[0],
            translation.y + rotated_point[1],
        )
        return transformed_point

    def path_world_to_odom(self, path):
        """map坐标系下的路径转换到odom坐标系下"""
        new_path = []
        translation, rotation = self.get_transform("odom", "map", self.lasttime)

        def point_world_to_odom(point):
            q = (rotation.w, rotation.x, rotation.y, rotation.z)
            rotated_point = qv_rotate(q, point)
            transformed_point = (
                translation.x + rotated_point[0],
                translation.y + rotated_point[1],
            )
            return transformed_point

        for i in range(len(path)):
            new_path.append(
                point_world_to_odom(
                    (
                        path[i][1] * self.res + self.origin.x,
                        path[i][0] * self.res + self.origin.y,
                        0,
                    )
                )
            )
        return new_path

    def update_path(self):
        try:
            translation, rotation = self.get_transform(
                "odom", "base_footprint", rclpy.time.Time()
            )
        except Exception:
            return
        if self.status is None:
            return

        self.status["translation"] = (translation.x, translation.y, translation.z)
        self.status["rotation"] = (rotation.w, rotation.x, rotation.y, rotation.z)
        dv = qv_rotate(
            (rotation.w, rotation.x, rotation.y, rotation.z), (1.0, 0.0, 0.0)
        )
        dvx, dvy = normalize(dv[0], dv[1])
        self.status["dir"] = (dvx, dvy)

        if (
            not (self.status is None)
            and self.status["stage"] != "exploring"
            and self.status["stage"] != "targeting"
        ):
            return

        if self.isNav:
            # 重新规划路径需要时间，趁机环视四周
            self.ctrl_queue.put(["gait", 0.0, 0.0, 0.0, 0.0, 0.0, 0.2])
            return

        if (
            self.stage_cache == "exploring" or self.stage_cache is None
        ) and self.status["stage"] == "targeting":
            # 发现球，立刻重新规划路径
            self.stage_cache = self.status["stage"]
            self.renav = True
            self.ctrl_queue.put(["gait", 0.0, 0.0, 0.0, 0.0, 0.0, 0.2])
            return
        self.stage_cache = self.status["stage"]

        if len(self.path):
            dx = translation.x
            dy = translation.y

            # 检测是否到达路径终点
            def end_of_path():
                return (dx - self.path[-1][0]) ** 2 + (
                    dy - self.path[-1][1]
                ) ** 2 < 0.04

            if self.status["stage"] == "exploring" and end_of_path():
                self.ctrl_queue.put(["gait", 0.0, 0.0, 0.0, 0.0, 0.0, 0.2])
                print("reached!")
                self.renav = True
                return

            # 检测是否到达球
            def end_of_ball():
                return (dx - self.status["target"][0]) ** 2 + (
                    dy - self.status["target"][1]
                ) ** 2 < 0.07

            if self.status["stage"] == "targeting" and end_of_ball():
                if self.status["target_z"] >= 1.5:
                    self.status["stage"] = "done_nav"
                    self.status["target"] = None
                    self.status["target_z"] = None
                    self.stage_cache = None
                    self.path = []
                else:
                    self.status["stage"] = "targeted"
                    self.ctrl_queue.put(["sit"])
                return

            # 计算路径跟踪方向
            index, min_point, min_distance = follow_path(self.path, np.array([dx, dy]))

            if min_distance < 0.1:
                directions = (
                    self.path[index + 1][0] - dx,
                    self.path[index + 1][1] - dy,
                )
                print("following")
            elif min_distance < 0.2:
                directions = (min_point[0] - dx, min_point[1] - dy)
                print("redirecting")
            else:
                # 路径跟踪失败，重新规划路径
                print("error nav!")
                self.ctrl_queue.put(["gait", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                self.renav = True
                return

            # 计算坐标轴旋转和偏移
            try:
                translation, rotation = self.get_transform(
                    "base_footprint", "odom", rclpy.time.Time()
                )
            except Exception:
                return
            newDir = (directions[0], directions[1], 0)
            runDir = qv_rotate((rotation.w, rotation.x, rotation.y, rotation.z), newDir)

            ppx, ppy = normalize(runDir[0], runDir[1])

            # 计算运动方向与正前方的夹角
            theta_target = np.arctan2(ppx, -ppy)
            theta_error = np.abs(theta_target - 0.5 * np.pi)
            if -0.5 * np.pi <= theta_target <= 0.5 * np.pi:
                rot_ang_z = -1  # 向右转
            else:
                rot_ang_z = 1  # 向左转

            if theta_error < 0.17:  # 偏航角小于5度
                # 前进的同时旋转
                ppx *= 0.3
                ppy *= 0.3
                self.ctrl_queue.put(["gait", ppx, ppy, 0.0, 0.0, 0.0, 0.0])
            else:
                # 只旋转
                self.ctrl_queue.put(["gait", 0.0, 0.0, 0.0, 0.0, 0.0, rot_ang_z * 0.4])

    def update_map(self, mapmsg):
        if (
            not (self.status is None)
            and self.status["stage"] != "exploring"
            and self.status["stage"] != "targeting"
        ):
            return
        print(self.status["stage"])
        nowTime = time.time()
        nowtimeT = rclpy.time.Time()

        # 判断是否需要重新规划路径
        if (
            self.renav == False
            and self.lastTime != 0
            and nowTime - self.lastTime < 20
            and len(self.path)
        ):
            return
        self.lastTime = nowTime
        self.renav = False

        tmap = mapmsg.data
        w = mapmsg.info.width
        h = mapmsg.info.height
        # 预处理地图，包括设定阈值、卷积去除孤立点等
        tmap = np.array(tmap)
        tmap = tmap.reshape((h, w))
        map = np.zeros_like(tmap)
        map[tmap < 70] = 0
        map[tmap == -1] = -1
        map[tmap >= 70] = 100
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        neighbourhood_counts = cv2.filter2D((map == 0).astype(np.uint8), -1, kernel)
        map[(map == -1) & (neighbourhood_counts >= 5)] = 0

        # 换算机器人坐标到地图中，注意地图是row-major的，因此需要交换计算得到的x,y
        res = mapmsg.info.resolution
        self.res = res
        try:
            translation, rotation = self.get_transform(
                "map", "base_footprint", nowtimeT
            )
        except Exception:
            return
        self.isNav = True
        origin = mapmsg.info.origin.position
        self.origin = origin
        dx = (int)((translation.x - origin.x) / res)
        dy = (int)((translation.y - origin.y) / res)

        # 堆优化的dijkstra算法计算最短距离场
        distance = shortest_distances(map, (dy, dx))

        # 更新时间戳，保证路径规划的连续性
        self.lasttime = nowtimeT

        if self.status["stage"] == "exploring":
            # 寻找前沿点
            fronts, frontier = get_frontier(map, distance)
            if len(fronts) > 0:
                best = fronts[0]
                points = np.argwhere(distance >= 0)
                distances = np.linalg.norm(
                    points - np.array([best[3], best[2]]), axis=1
                )
                min_index = np.argmin(distances)
                ft_dest = points[min_index]
                path = backward(distance, ft_dest)
                # show_path(map, path) # 将地图和路径转存为图像，用于调试
                self.path = self.path_world_to_odom(path)
            else:
                # 如果找不到前沿点，还没找到球，说明狗遍历完了地图，但还是没看到球，那就重新开始
                self.path = []
                self.status["stage"] = "ready"
                self.status["target"] = None
                self.status["target_z"] = None
                self.stage_cache = None
        elif self.status["stage"] == "targeting":
            x, y = self.point_odom_to_world(self.status["target"])
            ddy = (int)((y - origin.y) / res)
            ddx = (int)((x - origin.x) / res)
            points = np.argwhere(distance >= 0)
            distances = np.linalg.norm(points - np.array([ddy, ddx]), axis=1)
            min_index = np.argmin(distances)
            ft_dest = points[min_index]
            path = backward(distance, ft_dest)
            # show_path(map, path)
            self.path = self.path_world_to_odom(path)
        self.isNav = False


def quadroNavMain(ctrl_queue=None, status=None):
    rclpy.init()
    quadroNavNode = quadroNav(ctrl_queue, status)
    try:
        rclpy.spin(quadroNavNode)
    except KeyboardInterrupt:
        pass
    quadroNavNode.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    quadroNavMain()
