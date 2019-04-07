def compare2color(first, second):
    print(first[0:2], first[2:4], first[4:6])
    print(int(first[0:2], 16) / int(second[0:2], 16), int(first[2:4], 16) / int(second[2:4], 16), int(first[4:6], 16)/int(second[4:6], 16))
    print(second[0:2], second[2:4], second[4:6])
