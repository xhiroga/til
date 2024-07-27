def luhn_algorithm(number: int) -> int:
    """
    クレジットカードの番号のチェックディジットの検証などに使われるLuhnアルゴリズムの実装。
    """
    digits = [int(i) for i in str(number)]
    total_sum = 0
    double_weight = False

    while digits:
        digit = digits.pop()
        if double_weight:
            digit *= 2
            if digit > 9:
                digit -= 9
        total_sum += digit
        double_weight = not double_weight

    return total_sum % 10


if __name__ == "__main__":
    # https://mp-faq.gmo-pg.com/s/article/D00861
    correct = 0
    cards = [
        ("4111111111111111", correct), ("375987000000088", correct), ("4000000000111114", correct), ("4000000000111139", 9)
    ]
    for card, remainder in cards:
        calculated = luhn_algorithm(int(card))
        assert remainder == calculated, f"{remainder=}, {calculated=}"
