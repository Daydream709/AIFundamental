from collections import deque


# 判断当前状态是否安全
def is_valid(state):
    f, w, g, c = state
    # 狼和羊在一起，且人不在
    if w == g and f != w:
        return False
    # 羊和菜在一起，且人不在
    if g == c and f != g:
        return False
    return True


# 格式化输出
def format_state(state):
    left = "".join([n for i, n in enumerate(["F", "W", "G", "C"]) if state[i] == 0])
    right = "".join([n for i, n in enumerate(["F", "W", "G", "C"]) if state[i] == 1])
    return f"{left:<4} || {right}"


# 获取所有合法的下一步状态
def get_successors(state):
    successors = []
    f, w, g, c = state
    new_f = 1 - f  # 人必须过河

    # 1. 人自己
    if is_valid((new_f, w, g, c)):
        successors.append((new_f, w, g, c))
    # 2. 人带狼
    if f == w and is_valid((new_f, new_f, g, c)):
        successors.append((new_f, new_f, g, c))
    # 3. 人带羊
    if f == g and is_valid((new_f, w, new_f, c)):
        successors.append((new_f, w, new_f, c))
    # 4. 人带菜
    if f == c and is_valid((new_f, w, g, new_f)):
        successors.append((new_f, w, g, new_f))

    return successors


def solve_bfs():
    start = (0, 0, 0, 0)
    goal = (1, 1, 1, 1)

    queue = deque([[start]])
    visited = {start}

    step = 0
    while queue:
        step += 1
        print(f"\n--- BFS 第 {step} 步 ---")

        # 展示当前队列中每个准备探索的路径的末尾状态
        queue_states = "  =>  ".join([format_state(path[-1]) for path in queue])
        print(f"当前队列: [ {queue_states} ]")

        curr_path = queue.popleft()
        curr_state = curr_path[-1]

        # 如果找到目标状态，输出完整路径
        if curr_state == goal:
            print("\n成功找到解, 最终流程为:")
            for i, p in enumerate(curr_path):
                print(f"步骤 {i}: {format_state(p)}")
            return

        # 扩展下一层节点
        for next_state in get_successors(curr_state):
            if next_state not in visited:
                visited.add(next_state)
                new_path = list(curr_path)
                new_path.append(next_state)
                queue.append(new_path)


if __name__ == "__main__":
    solve_bfs()
