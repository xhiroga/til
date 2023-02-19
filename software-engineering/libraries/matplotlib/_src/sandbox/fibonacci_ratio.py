import numpy as np
import matplotlib.pyplot as plt


# 例題: フィボナッチ数列を一つ前の数で割り算した値は1.6くらいに収束する気がするが、本当か？
def main():
    x = np.arange(0, 30)
    fib = []
    ratio = []

    for n in x:
        if n <= 1:
            fib.append(n)
            ratio.append(n)
        else:
            fib.append(fib[n - 1] + fib[n - 2])
            ratio.append(fib[n] / fib[n - 1])

    plt.plot(x, ratio)
    plt.show()


if __name__ == "__main__":
    main()
