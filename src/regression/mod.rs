pub mod errors;
pub mod solvers;


use num::Float;
use faer::{Mat, MatRef};
use faer_traits::RealField;
use errors::RegressionErrors;


pub trait Regression<T: RealField + Float> {
    fn fitted_values(&self) -> MatRef<T>;
    fn has_bias(&self) -> bool; 

    /// Returns the bias term if present, otherwise 0. 
    /// This is the last column of the fitted values.
    fn bias(&self) -> T {
        if self.has_bias() {
            let idx = self.fitted_values().nrows() - 1;
            *self.fitted_values().get(idx, 0)
        } else {
            T::zero()
        }
    }

    #[allow(non_snake_case)]
    fn add_bias(&self, X: MatRef<T>) -> Mat<T> {
        let (n_rows, n_cols) = (X.nrows(), X.ncols());
        // A column of ones as the initial bias value
        let bias = Mat::<T>::ones(n_rows, 1);
        let mut X_biased = Mat::<T>::zeros(n_rows, n_cols + 1);
        X_biased.as_mut().submatrix_mut(0, 0, n_rows, n_cols).copy_from(X);
        X_biased.as_mut().col_mut(n_cols).copy_from(bias.as_ref().col(0));
        X_biased
    }

    /// Returns true if the model has been fit, false otherwise. 
    /// This is true if the fitted values matrix is non-empty. 
    fn is_fit(&self) -> bool {
        let shape = self.fitted_values().shape();
        shape.0 > 0 && shape.1 > 0
    }

    /// Returns an error if the model has not been fit. 
    fn check_is_fit(&self) -> Result<(), RegressionErrors> {
        if self.is_fit() {
            Ok(())
        } else {
            Err(RegressionErrors::MatNotLearnedYet)
        }
    }

    /// Returns the coefficients of the model. 
    /// This is the same as the fitted values, but without the bias term if present. 
    fn coefficients(&self) -> MatRef<T> {
        if self.has_bias() {
            // Last rows is the bias term
            let n = self.fitted_values().nrows() - 1;
            self.fitted_values().get(0..n, ..)
        } else {
            self.fitted_values()
        }
    }

    fn coeffs_as_vec(&self) -> Result<Vec<T>, RegressionErrors> {
        match self.check_is_fit() {
            Ok(_) => Ok(self
                .coefficients()
                .col(0)
                .iter()
                .copied()
                .collect::<Vec<_>>()),
            Err(e) => Err(e),
        }
    }

    #[allow(non_snake_case)]
    fn fit_unchecked(&mut self, X: MatRef<T>, y: MatRef<T>);


    #[allow(non_snake_case)]
    fn fit(&mut self, X: MatRef<T>, y: MatRef<T>) -> Result<(), RegressionErrors> {
        if X.nrows() != y.nrows() {
            return Err(RegressionErrors::DimensionMismatch);
        } else if X.nrows() < X.ncols() || X.nrows() == 0 || y.nrows() == 0 {
            return Err(RegressionErrors::NotEnoughData);
        }
        self.fit_unchecked(X, y);
        Ok(())
    }

    #[allow(non_snake_case)]
    fn predict(&self, X: MatRef<T>) -> Result<Mat<T>, RegressionErrors> {
        if X.ncols() != self.coefficients().nrows() {
            Err(RegressionErrors::DimensionMismatch)
        } else if !self.is_fit() {
            Err(RegressionErrors::MatNotLearnedYet)
        } else {
            let mut result = X * self.coefficients();
            let bias = self.bias();

            if self.has_bias() && self.bias().abs() > T::epsilon() {
                // Add the bias term
                for i in 0..result.nrows() {
                    let entry = result.get_mut(i, 0);
                    *entry = *entry + bias;
                }
            }
            Ok(result)
        }
    }
}

