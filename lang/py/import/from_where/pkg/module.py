import sys


def execute():
    print(__name__)
    print(sys.path)

    from awesome import awesome
    awesome()
