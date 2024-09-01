use log::warn;
use ndarray::{Array2, ShapeError};

// 行列式
pub fn det(mat: Array2<f64>) -> Result<f64, ShapeError> {
    // 正方行列でない場合はエラー
    let shape = mat.shape();
    if shape[0] != shape[1] {
        warn!("Expected square matrix, got {}x{}", shape[0], shape[1]);
        return Err(ShapeError::from_kind(
            ndarray::ErrorKind::IncompatibleShape
        ));
    }

    // 行列式を計算
    let determinant = todo!();

    Ok(determinant)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_det() {
        let mat = array![[1.0, 2.0], [3.0, 4.0]];
        let res = det(mat).unwrap();
        assert_eq!(res, -2.0);
    }
}