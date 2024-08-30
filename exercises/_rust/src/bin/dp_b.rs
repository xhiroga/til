use proconio::input;
use std::cmp::min;

// dp[j-1]: 1,j間の最小コスト（jは1オリジン）
fn solve(j: usize, k: usize, h: &[i32], dp: &mut Vec<Option<i32>>) -> i32{
    if let Some(value) = dp[j-1] {
        return value;
    }
    
    if k == 0 {
        unreachable!("k must be 1 <= k <= 100");
    }

    // j=2, k=3の場合、l=1のみ試す
    // j=3, k=2の場合、l=1,2を試す
    // j=10, k=20の場合、l=l=1,2,...,9を試す
    // つまり, ストライドは min(k, j-1) となる。
    let cost = (1..=min(k, j-1))
        .map(|l| (h[j-1] - h[j-l-1]).abs() + solve(j-l, k, h, dp))
        .min()
        .unwrap();

    dp[j-1] = Some(cost);
    cost
}

fn wrapper(n: usize, k: usize, h: &[i32]) -> i32{
    let mut dp = vec![None; n];
    dp[0] = Some(0);
    solve(n, k, h, &mut dp)
}

fn main(){
    input! {
        n: usize,
        k: usize,
        h: [i32; n],
    }
    let output = wrapper(n, k, &h);
    println!("{}", output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample1() {
        assert_eq!(wrapper(5, 3, &vec![10, 30, 40, 50, 20]), 30)
    }

    #[test]
    fn sample2() {
        assert_eq!(wrapper(3, 1, &vec![10, 20, 10]), 20)
    }

    #[test]
    fn sample3() {
        assert_eq!(wrapper(2, 100, &vec![10, 10]), 0)
    }
}
