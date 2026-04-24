import sys
from collections import deque


def solve():
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    n = int(input_data[0])
    m = int(input_data[1])

    adj = [[0] * (n + 1) for _ in range(n + 1)]
    idx = 2
    for _ in range(m):
        u = int(input_data[idx])
        v = int(input_data[idx + 1])
        adj[u][v] = 1  # 1表示有边
        idx += 2

    # dist 数组记录从1号点到每个点的记录，初始化为-1表示未访问
    dist = [-1] * (n + 1)
    dist[1] = 0

    q = deque([1])

    while q:
        u = q.popleft()

        # 一旦求出到 n 的距离，直接退出
        if u == n:
            break

        for v in range(1, n + 1):
            if adj[u][v] == 1 and dist[v] == -1:  # 若有边且未访问过
                dist[v] = dist[u] + 1
                q.append(v)

    print(dist[n])


if __name__ == "__main__":
    solve()
