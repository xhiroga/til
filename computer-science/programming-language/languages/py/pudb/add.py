"""
pudb3 add.py
"""


def add(x, y):
    result = x + y
    return result


def main():
    x = 1
    y = 2
    z = add(x, y)
    print(f'{x} + {y} ={z}')


if __name__ == '__main__':
    main()
