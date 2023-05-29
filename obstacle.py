import os
import rclpy
from rclpy.node import Node
from ception_msgs.srv import SensorDetectionNode
from ception_msgs.msg import Around


class quadroObstacle(Node):
    """
    该节点负责处理机器狗的避障，通过接收来自ObstacleDetection节点的消息，判断一定范围内是否有障碍物，如果有障碍物，则向主控节点发送避障指令
    但是ObstacleDetection并不是默认启动的，在翻阅了相应的源代码之后，发现需要通过调用obstacle_detection服务，向ObstacleDetection节点发送指令，才能使其启动。因此初始化时，需要向obstacle_detection服务发送指令，使其启动
    """

    def __init__(self, ctrl_queue, status):
        super().__init__("quadro_obstacle")
        self.ctrl_queue = ctrl_queue
        self.status = status
        self.isEmergency = False
        self.emergencyExitCount = 0

        # 向obstacle_detection服务发送指令，使其启动
        activateObstacleCli = self.create_client(
            SensorDetectionNode, "/mi1036022/obstacle_detection"
        )
        activateObstacleCli.wait_for_service()
        activateObstacleReq = SensorDetectionNode.Request()
        activateObstacleReq.command = 4
        activateObstacleFuture = activateObstacleCli.call_async(activateObstacleReq)
        rclpy.spin_until_future_complete(self, activateObstacleFuture)
        try:
            activateObstacleRes = activateObstacleFuture.result()
        except Exception as e:
            print("Service call failed %r" % (e,))
        else:
            print("Service call successed %r" % (activateObstacleRes,))

        self.obstacleSubsciber = self.create_subscription(
            Around, "/mi1036022/ObstacleDetection", self.obstacleCallback, 6
        )
        return

    def obstacleCallback(self, msg):
        distance = msg.front_distance.range_info.range
        if distance < 0.3:
            flag = (self.status["stage"] == "exploring") or (
                self.status["stage"] == "targeting"
            )
            if not flag:
                return
            print("obstacle detected! motion stop!")
            if not self.isEmergency:
                self.isEmergency = True
                self.emergencyExitCount = 0
                self.ctrl_queue.put(["obstacle"])
        else:
            if self.isEmergency:
                self.emergencyExitCount += 1
                if self.emergencyExitCount > 2:
                    self.emergencyExitCount = 0
                    self.isEmergency = False
                    self.ctrl_queue.put(["no_obstacle"])
        return


def quadroObstacleMain(ctrl_queue, status):
    print("quadroObstacleMain", os.getpid(), os.getppid())
    rclpy.init()
    quadroObstacleNode = quadroObstacle(ctrl_queue, status)
    try:
        rclpy.spin(quadroObstacleNode)
    except KeyboardInterrupt:
        pass
    quadroObstacleNode.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    quadroObstacleMain()
