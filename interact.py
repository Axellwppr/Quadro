import rclpy
from rclpy.node import Node
from interaction_msgs.msg import Touch


class quadroInteract(Node):
    """
    该节点负责处理机器狗的交互，实现长按机器狗头部，机器狗跳舞的功能:D
    通过接收来自TouchState节点的消息，判断是否有人触摸机器狗，如果有人触摸，则向控制节点发送跳舞指令
    """

    def __init__(self, ctrl_queue):
        super().__init__("quadro_interact")
        self.ctrl_queue = ctrl_queue
        # Create a subscriber to the Touch message
        self.touchSubscriber = self.create_subscription(
            Touch, "/mi1036022/TouchState", self.touchCallback, 6
        )
        return

    def touchCallback(self, msg):
        print("dance")
        state = msg.touch_state
        if state == 7:
            self.ctrl_queue.put(["order", 15])
        return


def interact(ctrl_queue=None, status=None):
    rclpy.init()
    quadroInteractNode = quadroInteract(ctrl_queue)
    rclpy.spin(quadroInteractNode)
    quadroInteractNode.destroy_node(quadroInteractNode)
    rclpy.shutdown()


if __name__ == "__main__":
    interact()
