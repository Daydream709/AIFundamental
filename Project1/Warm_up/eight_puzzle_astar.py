import sys
import heapq

TARGET = '12345678x'
TARGET_POS = {ch: idx for idx, ch in enumerate(TARGET)}

# 对应“x”的移动方向：u/d/l/r
MOVES = (
    (-1, 0, 'u'),
    (1, 0, 'd'),
    (0, -1, 'l'),
    (0, 1, 'r'),
)


def inversion_parity(state: str) -> int:
    nums = [ch for ch in state if ch != 'x']
    inv = 0
    length = len(nums)
    for i in range(length):
        for j in range(i + 1, length):
            if nums[i] > nums[j]:
                inv += 1
    return inv & 1


def manhattan(state: str) -> int:
    total = 0
    for i, ch in enumerate(state):
        if ch == 'x':
            continue
        target_idx = TARGET_POS[ch]
        total += abs(i // 3 - target_idx // 3) + abs(i % 3 - target_idx % 3)
    return total


def reconstruct_path(parent: dict[str, tuple[str, str]], end_state: str) -> str:
    actions = []
    cur = end_state
    while cur in parent:
        prev, action = parent[cur]
        actions.append(action)
        cur = prev
    actions.reverse()
    return ''.join(actions)


def astar(start: str) -> str:
    if start == TARGET:
        return ''

    g_score = {start: 0}
    parent = {}

    # 堆元素：(f, g, state)
    heap = [(manhattan(start), 0, start)]

    while heap:
        f_val, g_val, state = heapq.heappop(heap)

        # 跳过过期状态
        if g_val != g_score.get(state, -1):
            continue

        if state == TARGET:
            return reconstruct_path(parent, state)

        x_idx = state.index('x')
        x_row, x_col = divmod(x_idx, 3)
        state_list = list(state)

        for dr, dc, action in MOVES:
            nr, nc = x_row + dr, x_col + dc
            if nr < 0 or nr >= 3 or nc < 0 or nc >= 3:
                continue

            nxt = nr * 3 + nc
            state_list[x_idx], state_list[nxt] = state_list[nxt], state_list[x_idx]
            next_state = ''.join(state_list)
            state_list[x_idx], state_list[nxt] = state_list[nxt], state_list[x_idx]

            new_g = g_val + 1
            if new_g < g_score.get(next_state, 10**9):
                g_score[next_state] = new_g
                parent[next_state] = (state, action)
                heapq.heappush(heap, (new_g + manhattan(next_state), new_g, next_state))

    return 'unsolvable'


def solve() -> None:
    tokens = sys.stdin.read().split()
    if len(tokens) != 9:
        return

    start = ''.join(tokens)
    if inversion_parity(start) != 0:
        print('unsolvable')
        return

    print(astar(start))


if __name__ == '__main__':
    solve()
