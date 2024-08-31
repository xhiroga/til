use proconio::input;
use std::cmp;

fn solve(n: usize, h: &Vec<i32>, dp: &mut Vec<Option<i32>>) -> i32 {
    match dp[n - 1] {
        Some(value) => return value,
        None => {}
    };

    let result = if n == 2 {
        (h[1] - h[0]).abs()
    } else if n == 3 {
        cmp::min(
            (h[2] - h[1]).abs() + (h[1] - h[0]).abs(),
            (h[2] - h[0]).abs(),
        )
    } else {
        cmp::min(
            (h[n - 1] - h[n - 2]).abs() + solve(n - 1, h, dp),
            (h[n - 1] - h[n - 3]).abs() + solve(n - 2, h, dp),
        )
    };

    dp[n - 1] = Some(result);
    result
}

fn wrapper(n: usize, h: Vec<i32>) -> i32 {
    let mut dp = vec![None; n];
    solve(n, &h, &mut dp)
}

fn main() {
    input! {
        n: usize,
        h: [i32; n],
    }
    let output = wrapper(n, h);
    println!("{}", output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample1() {
        assert_eq!(wrapper(4, vec![10, 30, 40, 20]), 30);
    }

    #[test]
    fn sample2() {
        assert_eq!(wrapper(2, vec![10, 10]), 0);
    }

    #[test]
    fn sample3() {
        assert_eq!(wrapper(6, vec![30, 10, 60, 10, 60, 50]), 40);
    }
}
