import sys
sys.setrecursionlimit(200000)


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


def dfs_can_reach(state: str, visited: set) -> bool:
    # 找到目标状态，返回 True
    if state == TARGET:
        return True
    
    # 标记当前状态为已访问
    visited.add(state)
    
    x_idx = state.index('x')
    state_list = list(state)
    
    # 尝试所有可能的移动
    for nxt in NEIGHBORS[x_idx]:
        # 交换 x 和相邻位置
        state_list[x_idx], state_list[nxt] = state_list[nxt], state_list[x_idx]
        next_state = ''.join(state_list)
        # 恢复状态，继续下一次交换
        state_list[x_idx], state_list[nxt] = state_list[nxt], state_list[x_idx]
        
        # 如果未访问过该状态，继续 DFS
        if next_state not in visited:
            if dfs_can_reach(next_state, visited):
                return True
    
    return False


def solve() -> None:
    tokens = sys.stdin.read().split()
    if len(tokens) != 9:
        return

    start = ''.join(tokens)
    
    visited = set()
    can_reach = dfs_can_reach(start, visited)
    
    print(1 if can_reach else 0)


if __name__ == '__main__':
    solve()
