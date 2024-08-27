import math

def calc_uncertainty(data: list[float]) -> tuple[float, float, float]:
    mean = sum(data) / len(data)
    standard_deviation = math.sqrt(sum([(x - mean)**2 for x in data]) / (len(data)))
    uncertainty = standard_deviation / math.sqrt(len(data))
    return (mean, standard_deviation, uncertainty )

if __name__ == '__main__':
    data = [10, 10.2, 9.8, 10.1, 9.9]
    mean, standard_deviation, uncertainty = calc_uncertainty(data)
    print(f"{mean=}, {standard_deviation=}, {uncertainty}")
