import heapq
import numpy as np
import cv2

"""
该文件中包含一些辅助函数类，包括处理坐标系旋转的四元数模块，以及路径规划算法、调试函数等
"""


# 定义四元数乘法
def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z


# 定义四元数共轭
def q_conjugate(q):
    w, x, y, z = q
    return (w, -x, -y, -z)


# 定义四元数旋转函数
def qv_rotate(q, v):
    q2 = (0.0,) + v
    return q_mult(q_mult(q, q2), q_conjugate(q))[1:]


def qv_rotate_inv(q, v):
    q2 = (0.0,) + v
    return q_mult(q_mult(q_conjugate(q), q2), q)[1:]


directions = np.array(
    [
        [0, 1],
        [0, -1],
        [1, 0],
        [-1, 0],
        [1, 1],
        [1, -1],
        [-1, 1],
        [-1, -1],
        [1, 2],
        [2, 1],
        [2, -1],
        [1, -2],
        [-1, 2],
        [-2, 1],
        [-2, -1],
        [-1, -2],
        [1, 3],
        [3, 1],
        [3, -1],
        [1, -3],
        [-1, 3],
        [-3, 1],
        [-3, -1],
        [-1, -3],
    ]
)
euclidean_distances = np.linalg.norm(directions, axis=1)


def shortest_distances(map, start):
    """堆优化的Dijkstra算法，用于计算地图上某点到所有点的最短距离"""
    # 初始化距离矩阵，所有距离初始值设为无穷大
    distances = np.full(map.shape, np.inf, dtype=float)
    visited = np.zeros(map.shape, dtype=bool)
    distances[tuple(start)] = 0

    heap = [(0, tuple(start))]

    # 堆优化的Dijkstra算法，并在一些遍历中用了numpy并行处理
    while heap:
        current_distance, current_pos = heapq.heappop(heap)
        current_pos = np.array(current_pos)

        if current_distance > distances[current_pos[0]][current_pos[1]]:
            continue
        visited[current_pos[0]][current_pos[1]] = True

        new_positions = current_pos + directions

        valid_indices = np.all(
            (new_positions >= 0) & (new_positions < np.array(map.shape)), axis=1
        ) & (map[tuple(new_positions.T)] <= 0)

        valid_new_distances = current_distance + euclidean_distances[valid_indices]
        valid_new_positions = new_positions[valid_indices]
        for i in range(valid_new_positions.shape[0]):
            if map[valid_new_positions[i][0]][valid_new_positions[i][1]] == 0:
                if (
                    valid_new_distances[i]
                    < distances[valid_new_positions[i][0]][valid_new_positions[i][1]]
                ):
                    distances[valid_new_positions[i][0]][
                        valid_new_positions[i][1]
                    ] = valid_new_distances[i]
                    heapq.heappush(
                        heap, (valid_new_distances[i], tuple(valid_new_positions[i]))
                    )
            elif map[valid_new_positions[i][0]][valid_new_positions[i][1]] == -1:
                if (
                    valid_new_distances[i]
                    < distances[valid_new_positions[i][0]][valid_new_positions[i][1]]
                ):
                    visited[valid_new_positions[i][0]][valid_new_positions[i][1]] = True
                    distances[valid_new_positions[i][0]][
                        valid_new_positions[i][1]
                    ] = valid_new_distances[i]
    distances[visited == False] = -1
    return distances


def backward(dis_map, end):
    """根据最短距离矩阵，从终点开始反向搜索，得到最短路径"""
    now_value = dis_map[end[0]][end[1]]
    now_pos = end
    path = [end]
    eps = 1e-7

    while now_value > eps:
        new_positions = now_pos + directions
        new_values = now_value - euclidean_distances
        valid_indices = np.all(
            (new_positions >= 0) & (new_positions < np.array(dis_map.shape)), axis=1
        ) & (
            (new_values - eps <= dis_map[tuple(new_positions.T)])
            & (dis_map[tuple(new_positions.T)] <= new_values + eps)
        )
        now_pos = new_positions[valid_indices][0]
        now_value = new_values[valid_indices][0]
        path.append(now_pos)

    def check_obstacle(positions):
        x, y = np.floor(positions).astype(int).T
        out_of_bounds = (
            (x < 0) | (y < 0) | (x >= dis_map.shape[1]) | (y >= dis_map.shape[0])
        )
        is_obstacle = dis_map[x, y] == -1
        return out_of_bounds | is_obstacle

    def path_optimization(finalPath):
        """路径优化，去除冗余点，保证运动路径的流畅和连续"""
        optimized_path = [finalPath[-1]]
        parent_index = finalPath.shape[0] - 2
        while parent_index >= 0:
            if parent_index == 0 or check_line_collision(
                optimized_path[-1], finalPath[parent_index - 1]
            ):
                optimized_path.append(finalPath[parent_index])
            parent_index -= 1
        return optimized_path

    def check_line_collision(pos1, pos2):
        dpos = pos2 - pos1
        step = int(np.max(np.abs(dpos)) * 20)
        dpos = dpos.astype(float) / step
        points = pos1 + dpos * np.arange(step)[:, None]
        return np.any(check_obstacle(points))

    finalPath = path_optimization(np.array(path))
    # finalPath = finalPath.reverse()
    print("path", finalPath)
    return finalPath


def show_map(map_display, mouse_callback=None):
    """显示地图，可以传入鼠标回调函数"""
    cv2.namedWindow("map", 0)
    cv2.resizeWindow("map", 600, 600)
    cv2.imshow("map", map_display)
    if mouse_callback is not None:
        cv2.setMouseCallback("map", mouse_callback)
    cv2.waitKey(0)


def expand_map(map):
    """将地图转换为彩色图像，方便调试"""
    h, w = map.shape
    new_map = np.zeros((h, w, 3), dtype=np.uint8)
    new_map[map == 0] = [255, 255, 255]
    new_map[map == -1] = [0, 0, 0]
    new_map[map > 0] = [100, 100, 200]
    return new_map


def show_path(map, path):
    map_display = expand_map(map)
    for i in range(len(path) - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        step = max(abs(dx), abs(dy))
        dx /= step
        dy /= step
        for j in range(int(step)):
            x = int(path[i][0] + dx * j)
            y = int(path[i][1] + dy * j)
            if map[x][y] == 0:
                map_display[x][y] = [0, 255, 0]
            else:
                map_display[x][y] = [0, 0, 255]
        map_display[int(path[i][0])][int(path[i][1])] = [255, 0, 255]
    cv2.imwrite("./map.png", map_display)


def follow_path(path, pos):
    min_point = path[0]
    min_distance = (min_point[0] - pos[0]) ** 2 + (min_point[1] - pos[1]) ** 2
    index = 0
    for i in range(len(path) - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        step = max(abs(dx), abs(dy)) * 20
        dx /= step
        dy /= step
        for j in range(int(step)):
            x = path[i][0] + dx * j
            y = path[i][1] + dy * j
            distance = (x - pos[0]) ** 2 + (y - pos[1]) ** 2
            if distance < min_distance:
                min_distance = distance
                min_point = [x, y]
                index = i

    for i in range(len(path) - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        step = max(abs(dx), abs(dy)) * 20
        dx /= step
        dy /= step
        for j in range(int(step)):
            x = path[i][0] + dx * j
            y = path[i][1] + dy * j
            distance = (x - pos[0]) ** 2 + (y - pos[1]) ** 2
            if distance - min_distance < 0.02:
                min_point = [x, y]
                index = i
    return index, min_point, min_distance


def get_frontier(map, distance):
    kernel = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]
    )
    free_space = map == 0
    unknown_space = map == -1
    frontier = cv2.filter2D((free_space).astype(float), -1, kernel) * unknown_space
    frontier[frontier > 0] = 1
    frontier[distance == -1] = 0
    num_features, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (frontier > 0).astype("uint8"), connectivity=8, centroids=True
    )

    frontiers = []
    min_area = 10
    for i in range(1, num_features):
        if stats[i, cv2.CC_STAT_AREA] >= min_area and (
            stats[i, cv2.CC_STAT_WIDTH] > 7 or stats[i, cv2.CC_STAT_HEIGHT] > 7
        ):
            sum = np.sum(distance[labels == i])
            frontiers.append(
                (
                    stats[i, cv2.CC_STAT_AREA],
                    sum / stats[i, cv2.CC_STAT_AREA],
                    centroids[i][0],
                    centroids[i][1],
                )
            )
    frontiers_s = sorted(frontiers, key=lambda x: x[0] + x[1] * 0.1)
    return frontiers_s, frontier


def normalize(dx, dy):
    m = np.math.sqrt(dx * dx + dy * dy)
    return dx / m, dy / m
