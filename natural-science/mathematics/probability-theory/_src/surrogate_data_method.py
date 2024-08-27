from random import shuffle

def calculate_total_absolute_differences(temps: list[int]) -> int:
    return sum(abs(temps[i] - temps[i-1]) for i in range(len(temps)))

def perform_surrogate_data_method(temps: list[int], alpha: float = 0.05, num_surrogates: int = 100) -> bool:
    original_statistic = calculate_total_absolute_differences(temps)
    surrogate_statistics = []

    for _ in range(num_surrogates):
        surrogate = temps.copy()
        shuffle(surrogate)
        surrogate_statistics.append(calculate_total_absolute_differences(surrogate))

    extreme_count = sum(1 for stat in surrogate_statistics if stat <= original_statistic)
    p_value = (extreme_count + 1) / (num_surrogates + 1)

    print(f"{original_statistic=}, {surrogate_statistics=}, {extreme_count=}, {alpha=}, {p_value=}")
    return p_value < alpha  # 帰無仮説を棄却する場合はTrue

if __name__ == "__main__":
    data = [27, 32, 31, 33, 32, 30, 28, 27]
    result = perform_surrogate_data_method(data)
    print(f"Null hypothesis {'rejected' if result else 'not rejected'}")
