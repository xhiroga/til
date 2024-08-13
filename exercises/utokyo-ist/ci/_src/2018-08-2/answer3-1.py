# ルックアップテーブルを分割する

lookup_table = {
    0b0: 0, 0b1: 1, 0b10: 1, 0b11: 2, 0b100: 1, 0b101: 2, 0b110: 2, 0b111: 3,
    0b1000: 1, 0b1001: 2, 0b1010: 2, 0b1011: 3, 0b1100: 2, 0b1101: 3, 0b1110: 3, 0b1111: 4,
}

def population_count(data: int) -> int:
    lower_half = data & 0b1111
    upper_half = data >> 4

    return lookup_table[lower_half] + lookup_table[upper_half]

if __name__ == "__main__":
    data = 0b10101101
    n = 8
    expected = 5
    actual = population_count(data)
    assert expected == actual, f"{expected}=, {actual=}"
