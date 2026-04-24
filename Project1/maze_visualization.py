"""
迷宫问题求解：BFS、DFS、Dijkstra、A* 算法实现及可视化
"""
import matplotlib.pyplot as plt
import heapq
from collections import deque


def bfs(maze, start, end):
    """广度优先搜索 (BFS) - 适用于无权图的最短路径"""
    rows, cols = len(maze), len(maze[0])
    queue = deque([(start, [start])])
    visited = {start}
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 右、下、左、上
    
    while queue:
        (row, col), path = queue.popleft()
        
        if (row, col) == end:
            return path
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            if (0 <= new_row < rows and 0 <= new_col < cols and 
                maze[new_row][new_col] == 0 and 
                (new_row, new_col) not in visited):
                
                visited.add((new_row, new_col))
                queue.append(((new_row, new_col), path + [(new_row, new_col)]))
    
    return None  # 无解


def dfs(maze, start, end):
    """深度优先搜索 (DFS) - 不保证最短路径"""
    rows, cols = len(maze), len(maze[0])
    stack = [(start, [start])]
    visited = set()
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 右、下、左、上
    
    while stack:
        (row, col), path = stack.pop()
        
        if (row, col) in visited:
            continue
        
        visited.add((row, col))
        
        if (row, col) == end:
            return path
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            if (0 <= new_row < rows and 0 <= new_col < cols and 
                maze[new_row][new_col] == 0 and 
                (new_row, new_col) not in visited):
                
                stack.append(((new_row, new_col), path + [(new_row, new_col)]))
    
    return None  # 无解


def dijkstra(maze, start, end):
    """Dijkstra算法 - 适用于带权图（这里权重均为1）"""
    rows, cols = len(maze), len(maze[0])
    
    # 距离字典
    dist = {(i, j): float('inf') for i in range(rows) for j in range(cols)}
    dist[start] = 0
    
    # 前驱节点字典，用于重建路径
    prev = {(i, j): None for i in range(rows) for j in range(cols)}
    
    # 优先队列: (距离, 位置)
    pq = [(0, start)]
    visited = set()
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    while pq:
        current_dist, (row, col) = heapq.heappop(pq)
        
        if (row, col) in visited:
            continue
        
        visited.add((row, col))
        
        if (row, col) == end:
            # 重建路径
            path = []
            node = end
            while node is not None:
                path.append(node)
                node = prev[node]
            return path[::-1]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            if (0 <= new_row < rows and 0 <= new_col < cols and 
                maze[new_row][new_col] == 0 and 
                (new_row, new_col) not in visited):
                
                new_dist = current_dist + 1  # 权重为1
                
                if new_dist < dist[(new_row, new_col)]:
                    dist[(new_row, new_col)] = new_dist
                    prev[(new_row, new_col)] = (row, col)
                    heapq.heappush(pq, (new_dist, (new_row, new_col)))
    
    return None  # 无解


def heuristic(a, b):
    """曼哈顿距离启发式函数"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(maze, start, end):
    """A*算法 - 使用启发式函数的最优路径搜索"""
    rows, cols = len(maze), len(maze[0])
    
    # g_score: 从起点到当前节点的实际代价
    g_score = {(i, j): float('inf') for i in range(rows) for j in range(cols)}
    g_score[start] = 0
    
    # f_score: g_score + 启发式估计值
    f_score = {(i, j): float('inf') for i in range(rows) for j in range(cols)}
    f_score[start] = heuristic(start, end)
    
    # 优先队列: (f_score, 计数器, 位置) - 计数器用于打破平局
    counter = 0
    pq = [(f_score[start], counter, start)]
    
    # 前驱节点字典
    prev = {(i, j): None for i in range(rows) for j in range(cols)}
    visited = set()
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    while pq:
        _, _, (row, col) = heapq.heappop(pq)
        
        if (row, col) in visited:
            continue
        
        visited.add((row, col))
        
        if (row, col) == end:
            # 重建路径
            path = []
            node = end
            while node is not None:
                path.append(node)
                node = prev[node]
            return path[::-1]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            if (0 <= new_row < rows and 0 <= new_col < cols and 
                maze[new_row][new_col] == 0 and 
                (new_row, new_col) not in visited):
                
                tentative_g = g_score[(row, col)] + 1
                
                if tentative_g < g_score[(new_row, new_col)]:
                    g_score[(new_row, new_col)] = tentative_g
                    f_score[(new_row, new_col)] = tentative_g + heuristic((new_row, new_col), end)
                    prev[(new_row, new_col)] = (row, col)
                    counter += 1
                    heapq.heappush(pq, (f_score[(new_row, new_col)], counter, (new_row, new_col)))
    
    return None  # 无解


def get_explored_cells_bfs(maze, start, end):
    """获取BFS搜索过的所有格子"""
    rows, cols = len(maze), len(maze[0])
    queue = deque([start])
    visited = {start}
    explored = [start]
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    while queue:
        row, col = queue.popleft()
        
        if (row, col) == end:
            return explored
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            if (0 <= new_row < rows and 0 <= new_col < cols and 
                maze[new_row][new_col] == 0 and 
                (new_row, new_col) not in visited):
                
                visited.add((new_row, new_col))
                explored.append((new_row, new_col))
                queue.append((new_row, new_col))
    
    return explored


def get_explored_cells_dfs(maze, start, end):
    """获取DFS搜索过的所有格子"""
    rows, cols = len(maze), len(maze[0])
    stack = [start]
    visited = set()
    explored = []
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    while stack:
        row, col = stack.pop()
        
        if (row, col) in visited:
            continue
        
        visited.add((row, col))
        explored.append((row, col))
        
        if (row, col) == end:
            return explored
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            if (0 <= new_row < rows and 0 <= new_col < cols and 
                maze[new_row][new_col] == 0 and 
                (new_row, new_col) not in visited):
                
                stack.append((new_row, new_col))
    
    return explored


def visualize_maze_with_path(maze, path=None, explored_cells=None, algorithm_name=""):
    """可视化迷宫及路径，并对搜索过的格子染色"""
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
    if path:
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
    print("迷宫问题求解 - 四种算法对比")
    print("=" * 60)
    
    # BFS
    print("\n【BFS - 广度优先搜索】")
    path_bfs = bfs(maze, start, end)
    explored_bfs = get_explored_cells_bfs(maze, start, end)
    if path_bfs:
        print(f"找到路径，长度: {len(path_bfs)}")
        print(f"路径: {path_bfs}")
        print(f"搜索过的格子数: {len(explored_bfs)}")
    else:
        print("未找到路径")
    
    # DFS
    print("\n【DFS - 深度优先搜索】")
    path_dfs = dfs(maze, start, end)
    explored_dfs = get_explored_cells_dfs(maze, start, end)
    if path_dfs:
        print(f"找到路径，长度: {len(path_dfs)}")
        print(f"路径: {path_dfs}")
        print(f"搜索过的格子数: {len(explored_dfs)}")
    else:
        print("未找到路径")
    
    # Dijkstra
    print("\n【Dijkstra算法】")
    path_dijkstra = dijkstra(maze, start, end)
    if path_dijkstra:
        print(f"找到路径，长度: {len(path_dijkstra)}")
        print(f"路径: {path_dijkstra}")
    else:
        print("未找到路径")
    
    # A*
    print("\n【A*算法】")
    path_astar = astar(maze, start, end)
    if path_astar:
        print(f"找到路径，长度: {len(path_astar)}")
        print(f"路径: {path_astar}")
    else:
        print("未找到路径")
    
    print("\n" + "=" * 60)
    print("开始可视化...")
    print("=" * 60)
    
    # 可视化各个算法的结果
    visualize_maze_with_path(maze, path_bfs, explored_bfs, "BFS")
    visualize_maze_with_path(maze, path_dfs, explored_dfs, "DFS")
    visualize_maze_with_path(maze, path_dijkstra, explored_bfs, "Dijkstra")  # Dijkstra探索区域与BFS相同
    visualize_maze_with_path(maze, path_astar, explored_bfs, "A*")  # A*探索区域通常更少
    
    print("\n完成！")
