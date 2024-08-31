use proconio::input;

fn main() {
    input! {
        x: i32,
    }
    let output: i32;
    if x == 0 {
        output = 1;
    } else {
        output = 0;
    }

    println!("{}", output);
}
