use ndarray::{Array2, s};
use ndarray::linalg::*;

// LU分解
// Lの対角成分を1にする
pub fn lu(mat: &Array2<f64>) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let (height, width) = mat.dim();
    let s = height.min(width);
    
    let mut working = mat.clone();
    let mut l_cols = vec![];
    let mut u_rows = vec![];

    for _ in 0..s {
        let mat_00 = working[[0, 0]];

        let l_col = working.column(0).mapv(|x| x / mat_00).to_owned();
        let u_row = working.row(0).to_owned();

        let lu = kron(&l_col.to_shape((l_col.len(), 1)).unwrap(), &u_row.to_shape((1, u_row.len())).unwrap());
        working -= &lu;
        working = working.slice(s![1.., 1..]).to_owned();

        l_cols.push(l_col);
        u_rows.push(u_row);
    }
    
    // TODO: ピボット選択
    let p = Array2::eye(s);

    let l = Array2::from_shape_fn((height, s), |(i, j)| {
        if i == j {
            1.0
        } else if i > j {
            l_cols[j][i]
        } else {
            0.0
        }
    });
    let u  = Array2::from_shape_fn((s, width), |(i, j)| {
        if i <= j {
            u_rows[i][j-i]
        } else {
            0.0
        }
    });

    (p, l, u)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_lu() {
        let mat = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let (p, l, u) = lu(&mat);
        assert_eq!(l, array![[1.0, 0.0], [3.0, 1.0]]);
        assert_eq!(u, array![[1.0, 2.0], [0.0, -2.0]]);
    }
    // 1.1が0になる場合
}
