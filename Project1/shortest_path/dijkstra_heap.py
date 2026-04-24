import sys
import heapq


def solve():
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    n = int(input_data[0])
    m = int(input_data[1])

    INF = float("inf")
    g = [[] for _ in range(n + 1)]

    idx = 2
    for _ in range(m):
        u = int(input_data[idx])
        v = int(input_data[idx + 1])
        w = int(input_data[idx + 2])
        g[u].append((v, w))
        idx += 3

    dist = [INF] * (n + 1)
    dist[1] = 0
    st = [False] * (n + 1)

    # 优先队列 (距离，节点 id)
    pq = [(0, 1)]

    while pq:
        # 每次弹出当前距离 1 号点最近的点 u
        d, u = heapq.heappop(pq)

        # 堆中可能保留着更新前的冗余记录，若已被确定最短路径，则忽略
        if st[u]:
            continue

        st[u] = True

        # 一旦求出到 n 的最短路，提前返回
        if u == n:
            break

        # 遍历 u 的所有邻接边
        for v, w in g[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))

    if dist[n] == INF:
        print("-1")
    else:
        print(dist[n])


if __name__ == "__main__":
    solve()
