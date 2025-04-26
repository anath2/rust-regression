#![allow(non_snake_case)]
use num::Float;
use faer::{Mat, MatRef};
use faer_traits::RealField;

use crate::regression::{
    errors::RegressionErrors,
    solvers::irls,
    Regression
};

use crate::glm::functions::{LinkFunction, VarianceFunction};

/// Generalized Linear Model (GLM) family trait
pub trait Family<T: RealField + Float> : Regression<T> {
    fn link(&self, mu: T) -> T;
    fn inv_link(&self, eta: T) -> T;
    fn link_deriv(&self, mu: T) -> T;
    fn variance(&self, mu: T) -> T;

    fn glm_predict(&self, x: MatRef<T>) -> Result<Mat<T>, RegressionErrors> {
        // Generic prediction
        match <Self as Regression<T>>::predict(self, x) {
            Ok(yhat) => {
                // For GLM, apply the inverse link function elementwise to yhat
                let (nrows, ncols) = (yhat.nrows(), yhat.ncols());
                let mut predicted = Mat::zeros(nrows, ncols);
                for i in 0..nrows {
                    for j in 0..ncols {
                        predicted[(i, j)] = self.inv_link(yhat[(i, j)]);
                    }
                }
                Ok(predicted)
            },
            Err(e) => Err(e)
        }
    }
}

#[derive(Debug)]
pub struct Gaussian<T: RealField + Float> {
    pub coefficients: Mat<T>,
    pub has_bias: bool
}

impl<T: RealField + Float> Gaussian<T> {
    pub fn new(has_bias: bool) -> Self {
        Gaussian { coefficients: Mat::zeros(0, 0), has_bias: has_bias }
    }
}

impl<T: RealField + Float> Family<T> for Gaussian<T> {
    fn link(&self, mu: T) -> T {
        LinkFunction::Identity.link(mu)
    }
    fn inv_link(&self, eta: T) -> T {
        LinkFunction::Identity.inv_link(eta)
    }
    fn link_deriv(&self, mu: T) -> T {
        LinkFunction::Identity.link_deriv(mu)
    }
    fn variance(&self, mu: T) -> T {
        VarianceFunction::Gaussian.variance(mu)
    }
}

impl<T: RealField + Float> Regression<T> for Gaussian<T> {
    fn fitted_values(&self) -> MatRef<T> {
        self.coefficients.as_ref()
    }

    fn has_bias(&self) -> bool {
        self.has_bias
    }

    fn fit_unchecked(&mut self, X: MatRef<T>, y: MatRef<T>) {
        let inv_link = |eta: T| self.inv_link(eta);
        let link_deriv = |mu: T| self.link_deriv(mu);
        let variance = |mu: T| self.variance(mu);

        if self.has_bias {
            let (n_rows, n_cols) = (X.nrows(), X.ncols());

            // Add bias
            let bias = Mat::<T>::ones(n_rows, 1);
            let mut X_biased = Mat::<T>::zeros(n_rows, n_cols + 1);
            X_biased.as_mut().submatrix_mut(0, 0, n_rows, n_cols).copy_from(X);
            X_biased.as_mut().col_mut(n_cols).copy_from(bias.as_ref().col(0));

            self.coefficients = irls(
                X_biased.as_ref(),
                y,
                inv_link,
                link_deriv,
                variance
            );
        } else {
            self.coefficients = irls(
                X,
                y,
                inv_link,
                link_deriv,
                variance
            );
        }
    }
}

#[derive(Debug)]
pub struct Binomial<T: RealField + Float> {
    pub coefficients: Mat<T>,
    pub has_bias: bool
}

impl<T: RealField + Float> Binomial<T> {
    pub fn new(has_bias: bool) -> Self {
        Binomial { coefficients: Mat::zeros(0, 0), has_bias }
    }
}

impl<T: RealField + Float> Family<T> for Binomial<T> {
    fn link(&self, mu: T) -> T {
        LinkFunction::Logit.link(mu)
    }
    fn inv_link(&self, eta: T) -> T {
        LinkFunction::Logit.inv_link(eta)
    }
    fn link_deriv(&self, mu: T) -> T {
        LinkFunction::Logit.link_deriv(mu)
    }
    fn variance(&self, mu: T) -> T {
        VarianceFunction::Binomial.variance(mu)
    }
}

impl<T: RealField + Float> Regression<T> for Binomial<T> {
    fn fitted_values(&self) -> MatRef<T> {
        self.coefficients.as_ref()
    }
    fn has_bias(&self) -> bool {
        self.has_bias
    }
    fn fit_unchecked(&mut self, X: MatRef<T>, y: MatRef<T>) {
        let inv_link = |eta: T| self.inv_link(eta);
        let link_deriv = |mu: T| self.link_deriv(mu);
        let variance = |mu: T| self.variance(mu);
        if self.has_bias {
            let (n_rows, n_cols) = (X.nrows(), X.ncols());
            let bias = Mat::<T>::ones(n_rows, 1);
            let mut X_biased = Mat::<T>::zeros(n_rows, n_cols + 1);
            X_biased.as_mut().submatrix_mut(0, 0, n_rows, n_cols).copy_from(X);
            X_biased.as_mut().col_mut(n_cols).copy_from(bias.as_ref().col(0));
            self.coefficients = irls(
                X_biased.as_ref(),
                y,
                inv_link,
                link_deriv,
                variance
            );
            println!("Model inputs: {:#?}", X_biased);
            
        } else {
            self.coefficients = irls(
                X,
                y,
                inv_link,
                link_deriv,
                variance
            );
        }
    }
}

#[derive(Debug)]
pub struct Poisson<T: RealField + Float> {
    pub coefficients: Mat<T>,
    pub has_bias: bool
}

impl<T: RealField + Float> Poisson<T> {
    pub fn new(has_bias: bool) -> Self {
        Poisson { coefficients: Mat::zeros(0, 0), has_bias }
    }
}

impl<T: RealField + Float> Family<T> for Poisson<T> {
    fn link(&self, mu: T) -> T {
        LinkFunction::Log.link(mu)
    }
    fn inv_link(&self, eta: T) -> T {
        LinkFunction::Log.inv_link(eta)
    }
    fn link_deriv(&self, mu: T) -> T {
        LinkFunction::Log.link_deriv(mu)
    }
    fn variance(&self, mu: T) -> T {
        VarianceFunction::Poisson.variance(mu)
    }
}

impl<T: RealField + Float> Regression<T> for Poisson<T> {
    fn fitted_values(&self) -> MatRef<T> {
        self.coefficients.as_ref()
    }
    fn has_bias(&self) -> bool {
        self.has_bias
    }
    fn fit_unchecked(&mut self, X: MatRef<T>, y: MatRef<T>) {
        let inv_link = |eta: T| self.inv_link(eta);
        let link_deriv = |mu: T| self.link_deriv(mu);
        let variance = |mu: T| self.variance(mu);
        if self.has_bias {
            let (n_rows, n_cols) = (X.nrows(), X.ncols());
            let bias = Mat::<T>::ones(n_rows, 1);
            let mut X_biased = Mat::<T>::zeros(n_rows, n_cols + 1);
            X_biased.as_mut().submatrix_mut(0, 0, n_rows, n_cols).copy_from(X);
            X_biased.as_mut().col_mut(n_cols).copy_from(bias.as_ref().col(0));
            self.coefficients = irls(
                X_biased.as_ref(),
                y,
                inv_link,
                link_deriv,
                variance
            );
        } else {
            self.coefficients = irls(
                X,
                y,
                inv_link,
                link_deriv,
                variance
            );
        }
    }
}

#[derive(Debug)]
pub struct Gamma<T: RealField + Float> {
    pub coefficients: Mat<T>,
    pub has_bias: bool
}

impl<T: RealField + Float> Gamma<T> {
    pub fn new(has_bias: bool) -> Self {
        Gamma { coefficients: Mat::zeros(0, 0), has_bias }
    }
}

impl<T: RealField + Float> Family<T> for Gamma<T> {
    fn link(&self, mu: T) -> T {
        LinkFunction::Log.link(mu)
    }
    fn inv_link(&self, eta: T) -> T {
        LinkFunction::Log.inv_link(eta)
    }
    fn link_deriv(&self, mu: T) -> T {
        LinkFunction::Log.link_deriv(mu)
    }
    fn variance(&self, mu: T) -> T {
        VarianceFunction::Gamma.variance(mu)
    }
}

impl<T: RealField + Float> Regression<T> for Gamma<T> {
    fn fitted_values(&self) -> MatRef<T> {
        self.coefficients.as_ref()
    }
    fn has_bias(&self) -> bool {
        self.has_bias
    }
    fn fit_unchecked(&mut self, X: MatRef<T>, y: MatRef<T>) {
        let inv_link = |eta: T| self.inv_link(eta);
        let link_deriv = |mu: T| self.link_deriv(mu);
        let variance = |mu: T| self.variance(mu);
        if self.has_bias {
            let (n_rows, n_cols) = (X.nrows(), X.ncols());
            let bias = Mat::<T>::ones(n_rows, 1);
            let mut X_biased = Mat::<T>::zeros(n_rows, n_cols + 1);
            X_biased.as_mut().submatrix_mut(0, 0, n_rows, n_cols).copy_from(X);
            X_biased.as_mut().col_mut(n_cols).copy_from(bias.as_ref().col(0));
            self.coefficients = irls(
                X_biased.as_ref(),
                y,
                inv_link,
                link_deriv,
                variance
            );
        } else {
            self.coefficients = irls(
                X,
                y,
                inv_link,
                link_deriv,
                variance
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::{self, Rng};
    use faer::{Mat, MatRef};
    use faer_traits::RealField;
    use num::Float;
    use crate::regression::Regression;
    use crate::glm::family::{Gaussian, Binomial, LinkFunction, Poisson, Gamma};
    use rand_distr::{Poisson as PoissonDist, Gamma as GammaDist, Distribution};

    fn make_gaussian_regression_dataset(n_samples: usize) -> (Mat<f64>, Mat<f64>) {
        // Simple linear relation: y = 2x + 1 + noise

        let mut X = Mat::zeros(n_samples, 1);
        let mut y = Mat::zeros(n_samples, 1);
        let mut rng = rand::thread_rng();

        for i in 0..n_samples {
            let noise: f64 = rng.gen_range(-1.0..1.0);
            let xi: f64 = rng.gen_range(0.0..10.0);
            X[(i, 0)] = xi; // Data
            y[(i, 0)] = 2.0 * xi + 1.0 + noise;
        }
        (X, y)
    }

    fn check_approx_equal<T: RealField + Float>(a: MatRef<T>, b: MatRef<T>, tol: T) -> bool {
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                let diff = a[(i, j)] - b[(i, j)].abs(); 
                    if diff > tol {
                        return false;
                    }
                } 
            } 
        true
    }

    #[test]
    fn test_gaussian_regression() {
        let (X, y) = make_gaussian_regression_dataset(100);
        // Use the Gaussian family struct to fit
        let mut model = Gaussian::new(true);
        model.fit_unchecked(X.as_ref(), y.as_ref());
        let preds = model.fitted_values();
        // For Gaussian, fitted_values returns coefficients; to get predictions:
        let pred_y = if model.has_bias() {
            // Add bias column to X
            let n_rows = X.nrows();
            let n_cols = X.ncols();
            let bias: Mat<f64> = Mat::ones(n_rows, 1);
            let mut X_biased = Mat::zeros(n_rows, n_cols + 1);
            X_biased.as_mut().submatrix_mut(0, 0, n_rows, n_cols).copy_from(X.as_ref());
            X_biased.as_mut().col_mut(n_cols).copy_from(bias.as_ref().col(0));
            X_biased.as_ref() * preds
        } else {
            X.as_ref() * preds
        };
        assert_eq!(pred_y.nrows(), 100);
        assert!(check_approx_equal(pred_y.as_ref(), y.as_ref(), 2.0));
    }

    // Helper for binomial dataset
    fn make_binomial_regression_dataset(n_samples: usize) -> (Mat<f64>, Mat<f64>) {
        // Logistic regression: y ~ Bernoulli(sigmoid(2x - 1))
        let mut X = Mat::zeros(n_samples, 1);
        let mut y = Mat::zeros(n_samples, 1);
        let mut rng = rand::thread_rng();
        for i in 0..n_samples {
            let xi = rng.gen_range(-2.0..2.0);
            X[(i, 0)] = xi;
            let p = 1.0 / (1.0 + (-2.0 * xi + 1.0).exp());
            let random_val: f64 = rng.r#gen();
            if random_val < p {
                y[(i, 0)] = 0.0;    
            } else {
                y[(i, 0)] = 1.0;
            }
        }
        (X, y)
    }

    #[test]
    fn test_binomial_regression() {
        let (X, y) = make_binomial_regression_dataset(100);
        let mut model = Binomial::new(true);
        model.fit_unchecked(X.as_ref(), y.as_ref());
        let preds = model.fitted_values();
        // Add bias column to X for predictions
        let n_rows = X.nrows();
        let n_cols = X.ncols();
        let bias: Mat<f64> = Mat::ones(n_rows, 1);
        let mut X_biased = Mat::zeros(n_rows, n_cols + 1);
        X_biased.as_mut().submatrix_mut(0, 0, n_rows, n_cols).copy_from(X.as_ref());
        X_biased.as_mut().col_mut(n_cols).copy_from(bias.as_ref().col(0));
        let logits = X_biased.as_ref() * preds;
        // Use inv_link (sigmoid) to get probabilities

        let mut probabilities = logits.clone();
        for i in 0..probabilities.nrows() {
            for j in 0..probabilities.ncols() {
                probabilities[(i, j)] = LinkFunction::Logit.inv_link(logits[(i, j)]);
            }
        }

        // Check that probabilities are in [0,1]
        for i in 0..probabilities.nrows() {
            let p = probabilities[(i, 0)];
            assert!(p >= 0.0 && p <= 1.0);
        }
        // Optionally, check prediction accuracy (not strict)
        let y_true = y;
        let mut y_pred = Mat::zeros(n_rows, 1);
        for i in 0..n_rows {
            y_pred[(i, 0)] = if probabilities[(i, 0)] > 0.5 { 1.0 } else { 0.0 };
        }

        let mut correct = 0;
        for i in 0..y_true.nrows() {
            if (y_true[(i, 0)] - y_pred[(i, 0)]).abs() < 1e-8 {
                correct += 1;
            }
        }
        let accuracy = correct as f64 / y_true.nrows() as f64;
        assert!(accuracy > 0.7, "Binomial regression accuracy too low: {}", accuracy);
    }

    // Helper for Poisson dataset
    fn make_poisson_regression_dataset(n_samples: usize) -> (Mat<f64>, Mat<f64>) {
        // Poisson regression: y ~ Poisson(exp(0.5 * x + 1))
        let mut X = Mat::zeros(n_samples, 1);
        let mut y = Mat::zeros(n_samples, 1);
        let mut rng = rand::thread_rng();
        for i in 0..n_samples {
            let xi = rng.gen_range(0.0..2.0);
            X[(i, 0)] = xi;
            let lambda = (0.5 * xi + 1.0).exp();
            let poisson = PoissonDist::new(lambda).unwrap();
            y[(i, 0)] = rng.sample(poisson) as f64;
        }
        (X, y)
    }

    #[test]
    fn test_poisson_regression() {
        let (X, y) = make_poisson_regression_dataset(100);
        let mut model = Poisson::new(true);
        model.fit_unchecked(X.as_ref(), y.as_ref());
        let preds = model.fitted_values();
        // Add bias column to X for predictions
        let n_rows = X.nrows();
        let n_cols = X.ncols();
        let bias: Mat<f64> = Mat::ones(n_rows, 1);
        let mut X_biased = Mat::zeros(n_rows, n_cols + 1);
        X_biased.as_mut().submatrix_mut(0, 0, n_rows, n_cols).copy_from(X.as_ref());
        X_biased.as_mut().col_mut(n_cols).copy_from(bias.as_ref().col(0));
        let eta = X_biased.as_ref() * preds;
        // Use inv_link (exp) to get lambda (mean)
        let mut lambda = eta.clone();
        for i in 0..lambda.nrows() {
            for j in 0..lambda.ncols() {
                lambda[(i, j)] = LinkFunction::Log.inv_link(eta[(i, j)]);
            }
        }
        // Check that lambda is positive
        for i in 0..lambda.nrows() {
            assert!(lambda[(i, 0)] > 0.0);
        }
        // Optionally, check mean prediction error
        let y_true = y;
        let y_pred = lambda;
        let mut error = 0.0;
        for i in 0..y_true.nrows() {
            error += (y_true[(i, 0)] - y_pred[(i, 0)]).abs();
        }
        let mean_error = error / y_true.nrows() as f64;
        assert!(mean_error < 2.0, "Poisson regression mean error too high: {}", mean_error);
    }

    // Helper for Gamma dataset
    fn make_gamma_regression_dataset(n_samples: usize) -> (Mat<f64>, Mat<f64>) {
        // Gamma regression: y ~ Gamma(shape, scale), scale = exp(0.5 * x + 1)
        let mut X = Mat::zeros(n_samples, 1);
        let mut y = Mat::zeros(n_samples, 1);
        let mut rng = rand::thread_rng();
        let shape = 2.0; // shape parameter
        for i in 0..n_samples {
            let xi = rng.gen_range(0.0..2.0);
            X[(i, 0)] = xi;
            let scale = (0.5 * xi + 1.0).exp();
            let gamma = GammaDist::new(shape, scale).unwrap();
            y[(i, 0)] = rng.sample(gamma);
        }
        (X, y)
    }

    #[test]
    fn test_gamma_regression() {
        let (X, y) = make_gamma_regression_dataset(100);
        let mut model = Gamma::new(true);
        model.fit_unchecked(X.as_ref(), y.as_ref());
        let preds = model.fitted_values();
        // Add bias column to X for predictions
        let n_rows = X.nrows();
        let n_cols = X.ncols();
        let bias: Mat<f64> = Mat::ones(n_rows, 1);
        let mut X_biased = Mat::zeros(n_rows, n_cols + 1);
        X_biased.as_mut().submatrix_mut(0, 0, n_rows, n_cols).copy_from(X.as_ref());
        X_biased.as_mut().col_mut(n_cols).copy_from(bias.as_ref().col(0));
        let eta = X_biased.as_ref() * preds;
        // Use inv_link (exp) to get predicted mean
        let mut mu = eta.clone();
        for i in 0..mu.nrows() {
            for j in 0..mu.ncols() {
                mu[(i, j)] = LinkFunction::Log.inv_link(eta[(i, j)]);
            }
        }
        // Check that mu is positive
        for i in 0..mu.nrows() {
            assert!(mu[(i, 0)] > 0.0);
        }
        // Optionally, check mean prediction error
        let y_true = y;
        let y_pred = mu;
        let mut error = 0.0;
        for i in 0..y_true.nrows() {
            error += (y_true[(i, 0)] - y_pred[(i, 0)]).abs();
        }
        let mean_error = error / y_true.nrows() as f64;
        assert!(mean_error < 6.0, "Gamma regression mean error too high: {}", mean_error);
    }
}
