import os
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
import time
from motion_msgs.action import ChangeMode, ChangeGait, ExtMonOrder
from motion_msgs.msg import SE3VelocityCMD, Frameid


class quadroControl(Node):
    """
    该节点负责处理控制指令，包括：
    1. 机器人模式切换
    2. 机器人动作控制
    通过进程间通信，接收来自各个主控节点的指令，然后通过调用action接口，向机器人发送指令
    这么做可以免去在各个主控节点中调用action接口的麻烦，同时也可以避免不同主控节点之间发布相互冲突的指令，保证机器狗运动的稳定性

    可接受的指令为以下格式：
    ['指令代号', '指令参数1', ...]
    有以下指令可以传入：
    1. 'sit' 使机器狗坐下
    2. 'stand' 使机器狗站起
    3. 'gait' 使机器狗开始行走，后面的参数为速度，格式为(x, y, z, ang_x, ang_y, ang_z)
    4. 'mode' 使机器狗进入指定模式，后面的参数为模式代号和模式类型，格式为(control_mode, mode_type)
    5. 'order' 使机器狗执行指定动作，后面的参数为动作代号，格式为(order)
    6. 'obstacle' 使机器狗进入被动避障（急停）
    7. 'no_obstacle' 使机器狗退出被动避障
    8. 'hold' 使机器狗保持当前运动状态

    不同模式之间存在相互依赖和冲突关系，具体如下：
    'gait'行走指令，仅在机器狗处于'stand'站立模式和'mode'导航模式时有效，否则会被忽略。且机器狗在'stand'模式下需要保持5秒以上才能进入'gait'模式(为了避免还没站起来就走路)
    'obstacle'避障指令具有最高优先级，会使机器狗立即进入被动避障模式，此时所有涉及平移运动的指令均会被忽略

    同时，在动作状态切换时，机器狗会根据接受到的状态回调，判断是否已经完成状态切换，如果完成则会向主控节点发送状态切换完成的消息（motion键值），以便其他主控节点了解当前机器狗的状态，从而辅助相应的决策

    控制、通信部分代码适配自示例程序
    """

    def __init__(self, ctrl_queue, status):
        super().__init__("quadro_control")
        self.ctrl_queue = ctrl_queue
        self.status = status
        self.action_change_mode = ActionClient(
            self, ChangeMode, "/mi1036022/checkout_mode"
        )
        self.action_change_gait = ActionClient(
            self, ChangeGait, "/mi1036022/checkout_gait"
        )
        self.action_change_monorder = ActionClient(
            self, ExtMonOrder, "/mi1036022/exe_monorder"
        )
        self.pub_vel_cmd = self.create_publisher(
            SE3VelocityCMD, "/mi1036022/body_cmd", 6
        )
        self.isEmergency = False
        self.nowMode = "stand"
        self.speed = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.zeroSpeed = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.lastTime = time.time()
        self.control_callback_group = ReentrantCallbackGroup()
        self.timer = self.create_timer(0.05, self.mainCallback)
        self.onlineTime = time.time()
        self.global_sit = False
        return

    def refresh(self, atOnce=False):
        """
        该函数负责向机器狗发送运动指令，每隔0.1秒发送一次保证机器狗保持在所需运动状态，如果atOnce为True，则立即发送
        """
        nowTime = time.time()
        if atOnce or nowTime - self.lastTime > 0.1:
            self.lastTime = nowTime
            if self.nowMode == "gait":
                self.publish(*self.speed)
            if self.nowMode == "stand":
                self.publish(*self.zeroSpeed)

    def mainCallback(self):
        if self.global_sit == True:
            return
        while True:
            if not self.ctrl_queue.empty():
                msg = self.ctrl_queue.get()
                if msg[0] == "mode":
                    self.send_goal_mode(msg[1], msg[2])
                    self.speed = self.zeroSpeed
                    self.nowMode = "mode"
                    self.refresh(True)
                elif msg[0] == "obstacle":
                    self.isEmergency = True
                    self.send_goal_gait(7)
                    self.speed = self.zeroSpeed
                    self.nowMode = "gait"
                    self.refresh(True)
                    continue
                elif msg[0] == "no_obstacle":
                    self.isEmergency = False
                    continue
                if self.isEmergency:
                    flag = False
                    if (
                        msg[0] == "gait"
                        and -0.01 <= msg[1] <= 0.01
                        and -0.01 <= msg[2] <= 0.01
                    ):
                        flag = True
                    if msg[0] == "sit":
                        flag = True
                    if flag == False:
                        continue

                if msg[0] == "hold":
                    if self.nowMode == "gait":
                        msg[0] = "stand"
                    else:
                        continue
                if msg[0] == "stand":
                    self.send_goal_mode(3, 0)
                    self.speed = self.zeroSpeed
                    self.nowMode = msg[0]
                    self.refresh(True)
                elif msg[0] == "sit":
                    self.global_sit = True
                    self.send_goal_mode(0, 0)
                    self.speed = self.zeroSpeed
                    self.nowMode = msg[0]
                    self.refresh(True)
                elif msg[0] == "gait":
                    now_time = time.time() - self.onlineTime
                    if (
                        (self.nowMode == "stand" and now_time > 5)
                        or self.nowMode == "gait"
                        or self.nowMode == "mode"
                    ):
                        if self.nowMode == "stand" or self.nowMode == "mode":
                            self.send_goal_gait(7)
                        self.speed = msg[1:]
                        self.nowMode = msg[0]
                        self.refresh(True)
                elif msg[0] == "order":
                    self.send_goal_order(msg[1])
                    self.speed = self.zeroSpeed
                    self.nowMode = msg[0]
                    self.refresh(True)
            if self.ctrl_queue.empty():
                break
        self.refresh()

    def send_goal_mode(self, control_mode=3, mode_type=0):
        print(control_mode, mode_type, "mode")
        goal_msg = ChangeMode.Goal()
        goal_msg.modestamped.timestamp = self.get_clock().now().to_msg()
        goal_msg.modestamped.control_mode = control_mode
        goal_msg.modestamped.mode_type = mode_type
        self.action_change_mode.wait_for_server()
        self.response = self.action_change_mode.send_goal_async(
            goal_msg, feedback_callback=self.mode_call
        )
        return

    def send_goal_gait(self, gait=7):
        goal_msg = ChangeGait.Goal()
        goal_msg.motivation = 253
        goal_msg.gaitstamped.timestamp = self.get_clock().now().to_msg()
        goal_msg.gaitstamped.gait = gait
        self.action_change_gait.wait_for_server()
        self.response = self.action_change_gait.send_goal_async(
            goal_msg, feedback_callback=self.gait_call
        )
        return

    def send_goal_order(self, order=18):
        # print(order)
        goal_msg = ExtMonOrder.Goal()
        goal_msg.orderstamped.timestamp = self.get_clock().now().to_msg()
        goal_msg.orderstamped.id = order
        self.action_change_monorder.wait_for_server()
        self.response = self.action_change_monorder.send_goal_async(
            goal_msg, feedback_callback=self.order_call
        )
        return

    def publish(self, x, y, z, ang_x, ang_y, ang_z):
        my_msg = SE3VelocityCMD()
        my_frameid = Frameid()
        my_frameid.id = 1
        my_msg.sourceid = 2
        my_msg.velocity.frameid = my_frameid
        my_msg.velocity.timestamp = self.get_clock().now().to_msg()
        if x < -0.2:
            x = -0.2
        if x > 0.2:
            x = 0.2
        if y < -0.2:
            y = -0.2
        if y > 0.2:
            y = 0.2
        if ang_z < -0.3:
            ang_z = -0.3
        if ang_z > 0.3:
            ang_z = 0.3
        my_msg.velocity.linear_x = x
        my_msg.velocity.linear_y = y
        my_msg.velocity.linear_z = 0.0
        my_msg.velocity.angular_x = 0.0
        my_msg.velocity.angular_y = 0.0
        my_msg.velocity.angular_z = ang_z
        self.pub_vel_cmd.publish(my_msg)
        # print(str(my_msg))
        return

    def mode_call(self, msg):
        self.status["motion"] = True
        print(msg)
        return

    def gait_call(self, msg):
        # print(msg)
        return

    def order_call(self, msg):
        # print(msg)
        return


def quadroControlMain(ctrl_queue, status):
    print("quadroControlMain", os.getpid(), os.getppid())
    rclpy.init()
    quadroControlNode = quadroControl(ctrl_queue, status)
    try:
        rclpy.spin(quadroControlNode)
    except KeyboardInterrupt:
        pass
    quadroControlNode.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    quadroControlMain()
