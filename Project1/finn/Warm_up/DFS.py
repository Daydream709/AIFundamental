# 判断当前状态是否安全
def is_valid(state):
    f, w, g, c = state
    if w == g and f != w:
        return False
    if g == c and f != g:
        return False
    return True


# 格式化状态输出
def format_state(state):
    left = "".join([n for i, n in enumerate(["F", "W", "G", "C"]) if state[i] == 0])
    right = "".join([n for i, n in enumerate(["F", "W", "G", "C"]) if state[i] == 1])
    return f"{left:<4} || {right}"


# 获取所有合法的下一步状态
def get_successors(state):
    successors = []
    f, w, g, c = state
    new_f = 1 - f

    if is_valid((new_f, w, g, c)):
        successors.append((new_f, w, g, c))
    if f == w and is_valid((new_f, new_f, g, c)):
        successors.append((new_f, new_f, g, c))
    if f == g and is_valid((new_f, w, new_f, c)):
        successors.append((new_f, w, new_f, c))
    if f == c and is_valid((new_f, w, g, new_f)):
        successors.append((new_f, w, g, new_f))
    return successors


step = 0


def dfs(curr_state, goal, path, visited):
    global step
    step += 1

    print(f"\n--- DFS 第 {step} 次递归展开 ---")
    print("当前递归栈:")
    for level, p in enumerate(path):
        print(f"  L{level}: {format_state(p)}")

    if curr_state == goal:
        print("\n成功找到解, 最终流程为:")
        for i, p in enumerate(path):
            print(f"步骤 {i}: {format_state(p)}")
        return True

    for next_state in get_successors(curr_state):
        if next_state not in visited:
            visited.add(next_state)
            path.append(next_state)

            # 向下一层递归
            if dfs(next_state, goal, path, visited):
                return True

            # 回溯
            path.pop()
            visited.remove(next_state)

    return False


def solve_dfs():
    global step
    step = 0
    start = (0, 0, 0, 0)
    goal = (1, 1, 1, 1)
    visited = {start}
    dfs(start, goal, [start], visited)


if __name__ == "__main__":
    solve_dfs()
