def modules_10(number: int):
    digits = [int(i) for i in str(number)]
    sum = 0
    do_weight = True
    while len(digits) > 0:
        digit = digits.pop(-1)
        if do_weight:
            weighted_digit = digit * 2
        else:
            weighted_digit = digit
        if weighted_digit > 10:
            weighted_digit = weighted_digit - 9
        print(weighted_digit)
        sum = sum + weighted_digit
        do_weight = not do_weight
    print(sum)
    return sum % 10


if __name__ == "__main__":
    check_digit = 10 - modules_10(7992739871)
    if check_digit == 3:
        print("success!")
