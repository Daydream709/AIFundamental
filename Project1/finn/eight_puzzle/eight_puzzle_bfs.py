import sys
from collections import deque

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
    # 统计逆序对数量
    for i in range(length):
        for j in range(i + 1, length):
            if nums[i] > nums[j]:
                inv += 1
    return inv & 1


def bfs_min_steps(start: str) -> int:
    if start == TARGET:
        return 0

    queue = deque([start])
    dist = {start: 0}

    while queue:
        state = queue.popleft()
        step = dist[state]
        x_idx = state.index('x')

        state_list = list(state)
        # 尝试与所有相邻位置交换
        for nxt in NEIGHBORS[x_idx]:
            # 交换空格与相邻数字
            state_list[x_idx], state_list[nxt] = state_list[nxt], state_list[x_idx]
            next_state = ''.join(state_list)
            # 恢复状态，以便进行下一次交换
            state_list[x_idx], state_list[nxt] = state_list[nxt], state_list[x_idx]

            if next_state in dist:
                continue  # 已访问过的状态跳过

            if next_state == TARGET:
                return step + 1

            dist[next_state] = step + 1
            queue.append(next_state)

    return -1


def solve() -> None:
    tokens = sys.stdin.read().split()
    if len(tokens) != 9:
        return

    start = ''.join(tokens)

    if inversion_parity(start) != 0:
        print(-1)
        return

    print(bfs_min_steps(start))


if __name__ == '__main__':
    solve()