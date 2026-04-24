import sys


def solve():
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    n = int(input_data[0])
    m = int(input_data[1])

    INF = float("inf")
    # 邻接矩阵初始化为正无穷
    g = [[INF] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        g[i][i] = 0

    idx = 2
    for _ in range(m):
        u = int(input_data[idx])
        v = int(input_data[idx + 1])
        w = int(input_data[idx + 2])
        # 若存在重边，只保留最短的一条
        if w < g[u][v]:
            g[u][v] = w
        idx += 3

    dist = [INF] * (n + 1)
    dist[1] = 0
    st = [False] * (n + 1)

    # 迭代n次，每次确定一个最短距离的点
    for _ in range(n):
        t = -1
        # 在未确定最短路的点集中，寻找距离目前已知最短的一个点
        for j in range(1, n + 1):
            if not st[j] and (t == -1 or dist[j] < dist[t]):
                t = j

        # 所有点如果不连通了，可以提前结束
        if t == -1:
            break

        st[t] = True

        # 用新找到的距离起点的最短的点t更新其它所有点距离起点的最短距离
        for j in range(1, n + 1):
            if dist[t] + g[t][j] < dist[j]:
                dist[j] = dist[t] + g[t][j]

    if dist[n] == INF:
        print("-1")
    else:
        print(dist[n])


if __name__ == "__main__":
    solve()
