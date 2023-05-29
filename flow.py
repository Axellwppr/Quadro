import os
import time
import rclpy
from rclpy.node import Node


class quadroFlow(Node):
    """
    该节点负责处理机器狗的流程控制，按照预定的流程控制机器狗的运动，通过status.stage键值控制不同的主控状态
    不同的stage意义和控制流程如下：
    1. ready: 机器狗准备就绪，等待开始导航
    2. start_nav: 开始导航，此时机器狗会进入导航模式，然后等待导航初始化完成
    3. done_nav: 导航初始化完成
    4. start_first_run: 开始初始地图构建，此时机器狗会向前走一段距离，然后向左转90度，再向右转90度。主要是为了让狗出了窝先瞅一眼周围的状态，看看有没有球，以及为后面的建图做准备
    5. done_first_run: 初始地图构建完成
    6. exploring: 机器狗在未找到球的情况下，开始探索地图，此时机器狗会在地图中按照基于前沿的算法（Frontier Base Exploration）进行探索
    7. targeting: 当机器狗发现球时，会进入目标追踪模式，此时机器狗会沿着靠近球的最短路移动，直到接近到指定范围后趴下
    8. targeted: 机器狗已经找到并趴下
    """

    def __init__(self, ctrl_queue, status):
        super().__init__("quadro_Flow")
        self.ctrl_queue = ctrl_queue
        self.status = status
        self.lock = False
        self.checker = self.create_timer(0.1, self.checkCallback)

    def await_motion(self):
        self.status["motion"] = True
        time.sleep(0.5)
        while True:
            if (not self.status["motion"]) and (self.status["dir"] is not None):
                break
            self.status["motion"] = False
            time.sleep(4)

    def await_angle(self, expect_angle_sin):
        prevdir = self.status["dir"]
        while True:
            time.sleep(0.05)
            nowdir = self.status["dir"]
            print(nowdir)
            calc = prevdir[0] * nowdir[1] - prevdir[1] * nowdir[0]
            print("calc:", calc)
            if (expect_angle_sin > 0 and calc > expect_angle_sin) or (
                expect_angle_sin < 0 and calc < expect_angle_sin
            ):
                break

    def await_full_angle(self):
        self.await_angle(0.94)
        self.await_angle(0.94)
        self.await_angle(0.94)
        self.await_angle(0.94)
        self.await_angle(0.94)

    def checkCallback(self):
        if self.lock:
            return
        self.lock = True
        if self.status["target"] is not None and self.status["stage"] != "targeting":
            self.status["stage"] = "targeting"
        if self.status["stage"] == "ready":
            self.status["stage"] = "start_nav"
            time.sleep(1)
            print("start nav")
            self.ctrl_queue.put(["mode", 14, 5])
            self.await_motion()
            # self.ctrl_queue.put(['mode', 14, 5])
            # self.await_motion()
            print("nav done")
            time.sleep(0.5)
            self.status["stage"] = "done_nav"
        elif self.status["stage"] == "done_nav":
            self.status["stage"] = "start_first_run"
            # self.ctrl_queue.put(['gait', 0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
            # time.sleep(4)
            self.ctrl_queue.put(["gait", 0.0, 0.0, 0.0, 0.0, 0.0, 0.4])
            # ang_z>0 右旋（向左转）
            self.await_full_angle()
            self.ctrl_queue.put(["gait", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.status["stage"] = "done_first_run"
        elif self.status["stage"] == "done_first_run":
            self.status["stage"] = "exploring"
            # processed by nav.py
        self.lock = False


def quadroFlowMain(ctrl_queue, status):
    print("quadroFlowMain", os.getpid(), os.getppid())
    rclpy.init()
    quadroFlowNode = quadroFlow(ctrl_queue, status)
    try:
        rclpy.spin(quadroFlowNode)
    except KeyboardInterrupt:
        pass
    quadroFlowNode.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    quadroFlowMain()
