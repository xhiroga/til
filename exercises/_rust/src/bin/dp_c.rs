use proconio::input;
use std::cmp::max;

fn solve(
    i: usize,        // 1 <= i <= n
    activity: usize, // 0 <= activity <= 2
    abc: &[(u32, u32, u32)],
    dp: &mut Vec<[Option<u32>; 3]>,
) -> u32 {
    if i == 0 {
        return 0;
    }

    if let Some(value) = dp[i - 1][activity] {
        return value;
    }

    let cost = match activity {
        // Rustでは、tupleの要素に対して動的にアクセスできない。
        0 => abc[i - 1].0 + max(solve(i - 1, 1, abc, dp), solve(i - 1, 2, abc, dp)),
        1 => abc[i - 1].1 + max(solve(i - 1, 0, abc, dp), solve(i - 1, 2, abc, dp)),
        2 => abc[i - 1].2 + max(solve(i - 1, 0, abc, dp), solve(i - 1, 1, abc, dp)),
        _ => unreachable!(),
    };

    dp[i - 1][activity] = Some(cost);
    cost
}

fn wrapper(n: usize, abc: &[(u32, u32, u32)]) -> u32 {
    let mut dp = vec![[None; 3]; n];

    solve(n, 0, abc, &mut dp)
        .max(solve(n, 1, abc, &mut dp))
        .max(solve(n, 2, abc, &mut dp))
}

fn main() {
    input! {
        n: usize,
        abc: [(u32, u32, u32); n],
    }
    let output = wrapper(n, &abc);
    println!("{}", output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample1() {
        assert_eq!(
            wrapper(3, &vec![(10, 40, 70), (20, 50, 80), (30, 60, 90)]),
            210
        )
    }

    #[test]
    fn sample2() {
        assert_eq!(wrapper(1, &vec![(100, 10, 1)]), 100)
    }

    #[test]
    fn sample3() {
        assert_eq!(
            wrapper(
                7,
                &vec![
                    (6, 7, 8),
                    (8, 8, 3),
                    (2, 5, 2),
                    (7, 8, 6),
                    (4, 6, 8),
                    (2, 3, 4),
                    (7, 5, 1)
                ]
            ),
            46
        )
    }
}
