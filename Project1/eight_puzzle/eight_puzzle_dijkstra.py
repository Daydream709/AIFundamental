import sys
import heapq


TARGET = '12345678x'
NEIGHBORS = {
    0: (1, 3),
    1: (0, 2, 4),
    2: (1, 5),
    3: (0, 4, 6),
    4: (1, 3, 5, 7),
    5: (2, 4, 8),
    6: (3, 7),
    7: (4, 6, 8),
    8: (5, 7),
}


def inversion_parity(state: str) -> int:
    nums = [ch for ch in state if ch != 'x']
    inv = 0
    length = len(nums)
    for i in range(length):
        for j in range(i + 1, length):
            if nums[i] > nums[j]:
                inv += 1
    return inv & 1


def dijkstra_min_steps(start: str) -> int:
    if start == TARGET:
        return 0

    # dist 记录从起点到每个状态的最短距离
    dist = {start: 0}
    # 优先队列：(距离，状态)
    pq = [(0, start)]

    while pq:
        d, state = heapq.heappop(pq)

        # 如果当前距离大于已知的最短距离，跳过
        if d > dist.get(state, float('inf')):
            continue

        # 找到目标状态，返回距离
        if state == TARGET:
            return d

        x_idx = state.index('x')
        state_list = list(state)

        # 遍历所有可能的移动
        for nxt in NEIGHBORS[x_idx]:
            # 交换 x 和相邻位置
            state_list[x_idx], state_list[nxt] = state_list[nxt], state_list[x_idx]
            next_state = ''.join(state_list)
            # 恢复状态，继续下一次交换
            state_list[x_idx], state_list[nxt] = state_list[nxt], state_list[x_idx]

            # 新的距离（每次移动权重为 1）
            new_dist = d + 1

            # 如果找到更短的路径，更新
            if new_dist < dist.get(next_state, float('inf')):
                dist[next_state] = new_dist
                heapq.heappush(pq, (new_dist, next_state))

    return -1


def solve() -> None:
    tokens = sys.stdin.read().split()
    if len(tokens) != 9:
        return

    start = ''.join(tokens)
    
    # 检查是否有解：逆序对奇偶性必须相同
    if inversion_parity(start) != 0:
        print(-1)
        return

    print(dijkstra_min_steps(start))


if __name__ == '__main__':
    solve()
