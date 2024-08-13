# nビットデータを受け取ってpopulation countを行うスクリプト

def population_count(data: int, n: int) -> int:
    count = 0
    for i in range(n):
        if (data & (1 << i)) != 0:
            count += 1
    return count


if __name__ == "__main__":
    data = 0b101011
    n = 6
    expected = 4
    actual = population_count(data, n)
    assert expected == actual, f"{expected}=, {actual=}"
