"""
A* (A-Star) 算法迷宫求解
使用启发式函数的最优路径搜索算法
"""
import matplotlib.pyplot as plt
import heapq


def heuristic(a, b):
    """
    曼哈顿距离启发式函数
    
    Args:
        a: 坐标点1 (row, col)
        b: 坐标点2 (row, col)
    
    Returns:
        曼哈顿距离
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(maze, start, end):
    """
    A*算法 - 使用启发式函数的最优路径搜索
    
    Args:
        maze: 二维数组，0表示通路，1表示墙壁
        start: 起点坐标 (row, col)
        end: 终点坐标 (row, col)
    
    Returns:
        path: 路径列表，如果无解返回None
        explored_cells: 搜索过的所有格子列表
    """
    rows, cols = len(maze), len(maze[0])
    
    # g_score: 从起点到当前节点的实际代价
    g_score = {(i, j): float('inf') for i in range(rows) for j in range(cols)}
    g_score[start] = 0
    
    # f_score: g_score + 启发式估计值
    f_score = {(i, j): float('inf') for i in range(rows) for j in range(cols)}
    f_score[start] = heuristic(start, end)
    
    # 优先队列: (f_score, 计数器, 位置)
    # 计数器用于打破平局，避免比较元组时出错
    counter = 0
    pq = [(f_score[start], counter, start)]
    
    # 前驱节点字典：用于重建路径
    prev = {(i, j): None for i in range(rows) for j in range(cols)}
    visited = set()
    explored_cells = []
    
    # 四个方向：右、下、左、上
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    while pq:
        _, _, (row, col) = heapq.heappop(pq)
        
        # 跳过已访问的节点
        if (row, col) in visited:
            continue
        
        # 标记为已访问
        visited.add((row, col))
        explored_cells.append((row, col))
        
        # 找到终点
        if (row, col) == end:
            # 重建路径
            path = []
            node = end
            while node is not None:
                path.append(node)
                node = prev[node]
            return path[::-1], explored_cells
        
        # 探索相邻节点
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            # 检查边界、是否为通路、是否已访问
            if (0 <= new_row < rows and 0 <= new_col < cols and 
                maze[new_row][new_col] == 0 and 
                (new_row, new_col) not in visited):
                
                tentative_g = g_score[(row, col)] + 1
                
                # 如果找到更短的路径，更新分数
                if tentative_g < g_score[(new_row, new_col)]:
                    g_score[(new_row, new_col)] = tentative_g
                    f_score[(new_row, new_col)] = tentative_g + heuristic((new_row, new_col), end)
                    prev[(new_row, new_col)] = (row, col)
                    counter += 1
                    heapq.heappush(pq, (f_score[(new_row, new_col)], counter, (new_row, new_col)))
    
    return None, explored_cells  # 无解


def visualize_maze_with_path(maze, path=None, explored_cells=None, algorithm_name="A*"):
    """
    可视化迷宫及路径，并对搜索过的格子染色
    
    Args:
        maze: 迷宫二维数组
        path: 路径列表
        explored_cells: 搜索过的格子列表
        algorithm_name: 算法名称
    """
    plt.figure(figsize=(len(maze[0]) + 2, len(maze) + 2))
    
    # 创建迷宫副本用于着色
    maze_copy = [row[:] for row in maze]
    
    # 标记搜索过的格子（浅蓝色）
    if explored_cells:
        for row, col in explored_cells:
            if maze_copy[row][col] == 0:  # 只标记可通行的格子
                maze_copy[row][col] = 0.3  # 浅灰色表示已搜索
    
    # 标记路径（深蓝色）
    if path:
        for row, col in path:
            if maze_copy[row][col] == 0 or maze_copy[row][col] == 0.3:
                maze_copy[row][col] = 0.6  # 深灰色表示路径
    
    plt.imshow(maze_copy, cmap='Blues', interpolation='nearest', vmin=0, vmax=1)
    
    # 绘制路径
    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_y, path_x, marker='o', markersize=10, color='red', linewidth=2, label='Path')
        
        # 标记起点和终点
        start = path[0]
        end = path[-1]
        plt.plot(start[1], start[0], 'go', markersize=15, label='Start')
        plt.plot(end[1], end[0], 'rs', markersize=15, label='End')
    
    # 设置坐标轴刻度和边框
    plt.xticks(range(len(maze[0])))
    plt.yticks(range(len(maze)))
    plt.gca().set_xticks([x - 0.5 for x in range(1, len(maze[0]))], minor=True)
    plt.gca().set_yticks([y - 0.5 for y in range(1, len(maze))], minor=True)
    plt.grid(which="minor", color="black", linestyle='-', linewidth=2)
    
    plt.title(f'{algorithm_name} - Path Length: {len(path) if path else 0}, Explored: {len(explored_cells) if explored_cells else 0}', fontsize=12)
    plt.legend(loc='upper right')
    plt.axis('on')
    plt.tight_layout()
    plt.show()


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 提供迷宫的二维数组 (0: 通路, 1: 墙壁)
    maze = [
        [0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0]
    ]
    
    # 定义起点和终点
    start = (0, 0)
    end = (4, 4)
    
    print("=" * 60)
    print("A* (A-Star) 算法迷宫求解")
    print("=" * 60)
    print(f"起点: {start}")
    print(f"终点: {end}")
    print()
    
    # 执行A*算法
    path, explored_cells = astar(maze, start, end)
    
    # 输出结果
    if path:
        print(f"✅ 找到路径！")
        print(f"路径长度: {len(path)}")
        print(f"路径: {path}")
        print(f"搜索过的格子数: {len(explored_cells)}")
    else:
        print("❌ 未找到路径")
    
    print("\n" + "=" * 60)
    print("开始可视化...")
    print("=" * 60)
    
    # 可视化结果
    visualize_maze_with_path(maze, path, explored_cells, "A*")
    
    print("\n完成！")
