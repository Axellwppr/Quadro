from multiprocessing import Manager, get_context
from control import quadroControlMain

# from vision import quadroVisionMain
from obstacle import quadroObstacleMain
from detect import quadroDetectMain
from nav import quadroNavMain
from flow import quadroFlowMain

components = [
    [quadroControlMain, True],  # 运动控制模块，负责接受其他进程的指令，控制机器人运动、模式切换
    # [quadroVisionMain, True],# 视觉模块，负责处理深度图像信息，控制机器人主动避障，由于已经采用了路径规划和被动避障，该模块在决赛任务中不需要
    [quadroObstacleMain, True],  # 避障模块，负责处理障碍物信息，控制机器人被动避障
    [quadroDetectMain, True],  # 检测模块，负责处视觉识别蓝色球，并计算相应的坐标
    [quadroNavMain, True],  # 导航模块，负责规划机器人的运动轨迹，并将轨迹转化为运动指令
    [quadroFlowMain, True],  # 流程控制模块，负责控制机器人的运动流程，如先起立，然后旋转一周，然后开始探索地图...
]

procs = []
ctxs = []


def main():
    """
    主控入口，负责创建进程，创建进程通信队列queue和状态字典status
    通过控制components中的元素来控制进程的创建，以实现不同功能的组合
    """
    with Manager() as manager:
        queue = manager.Queue()
        status = manager.dict()
        status["stage"] = "ready"
        status["target"] = None
        status["target_z"] = None
        status["dir"] = None
        print(status)
        for component in components:
            if component[1]:
                ctx = get_context("spawn")
                procs.append(
                    ctx.Process(
                        target=component[0],
                        args=(
                            queue,
                            status,
                        ),
                    )
                )
                ctxs.append(ctx)
        # queue.put(['mode', 14, 5])
        # queue.put(['sit'])
        # queue.put(['stand'])

        for p in procs:
            p.start()

        for p in procs:
            p.join()


if __name__ == "__main__":
    main()
