import sys


def inversion_parity(state: str) -> int:
    nums = [ch for ch in state if ch != 'x']
    inv = 0
    length = len(nums)
    for i in range(length):
        for j in range(i + 1, length):
            if nums[i] > nums[j]:
                inv += 1
    return inv & 1


def solve() -> None:
    tokens = sys.stdin.read().split()
    if len(tokens) != 9:
        return

    start = ''.join(tokens)
    # 3x3 八数码：逆序对为偶数 <=> 有解
    print(1 if inversion_parity(start) == 0 else 0)


if __name__ == '__main__':
    solve()
