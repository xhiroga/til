# Brian Kernighanのアルゴリズム

def population_count(data: int) -> int:
    count = 0
    while data != 0:
        data = data & (data - 1)
        count += 1
    return count

if __name__ == "__main__":
    data = 0b101011
    n = 6
    expected = 4
    actual = population_count(data)
    assert expected == actual, f"{expected}=, {actual=}"
