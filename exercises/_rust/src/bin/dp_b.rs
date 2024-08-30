use proconio::input;

// dp[j-1]: 1,j間の最小コスト（jは1オリジン）
fn solve(j: usize, k: usize, h: &[i32], dp: &mut Vec<Option<i32>>) -> i32{
    match dp[j-1]{
        Some(value) => return value,
        None => {}
    }
    
    if k == 0 {
        unreachable!("k must be 1 <= k <= 100");
    }

    let mut min_cost = i32::MAX;
    for l in 1..k+1 {
        let current = if l < j {
            (h[j-1] - h[j-l-1]).abs() + solve(j-l, k, h, dp)
        } else {
            i32::MAX
        };
        min_cost = min_cost.min(current);
    }

    dp[j-1] = Some(min_cost);
    min_cost
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
