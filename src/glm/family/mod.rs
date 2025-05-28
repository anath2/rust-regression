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

    #[allow(non_snake_case)]
    fn fit_unchecked(&mut self, X: MatRef<T>, y: MatRef<T>) {
        let inv_link = |eta: T| self.inv_link(eta);
        let link_deriv = |mu: T| self.link_deriv(mu);
        let variance = |mu: T| self.variance(mu);

        if self.has_bias {
            let X_biased = self.add_bias(X);
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

    #[allow(non_snake_case)]
    fn fit_unchecked(&mut self, X: MatRef<T>, y: MatRef<T>) {
        let inv_link = |eta: T| self.inv_link(eta);
        let link_deriv = |mu: T| self.link_deriv(mu);
        let variance = |mu: T| self.variance(mu);
        if self.has_bias {
            let X_biased = self.add_bias(X);
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

    #[allow(non_snake_case)]
    fn fit_unchecked(&mut self, X: MatRef<T>, y: MatRef<T>) {
        let inv_link = |eta: T| self.inv_link(eta);
        let link_deriv = |mu: T| self.link_deriv(mu);
        let variance = |mu: T| self.variance(mu);
        if self.has_bias {
            let X_biased = self.add_bias(X);
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

    #[allow(non_snake_case)]
    fn fit_unchecked(&mut self, X: MatRef<T>, y: MatRef<T>) {
        let inv_link = |eta: T| self.inv_link(eta);
        let link_deriv = |mu: T| self.link_deriv(mu);
        let variance = |mu: T| self.variance(mu);
        if self.has_bias {
            let X_biased = self.add_bias(X);
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
#[allow(non_snake_case)]
mod tests {
    use rand::{self, Rng};
    use faer::{Mat, MatRef};
    use faer_traits::RealField;
    use num::Float;
    use crate::glm::family::Family;
    use crate::regression::Regression;
    use crate::glm::family::{Gaussian, Binomial, LinkFunction, Poisson, Gamma};
    use rand_distr::{Poisson as PoissonDist, Gamma as GammaDist};

    // Helper for binomial dataset
    use std::fs::{self, File};
    use std::path::Path;
    use std::io::Write;
    use serde::Deserialize;
    use std::io::{Seek, SeekFrom};

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

    fn download_iris_dataset() -> std::io::Result<String> {
        let data_dir = Path::new("data");
        let iris_path = data_dir.join("iris.csv");
        
        if !data_dir.exists() {
            fs::create_dir(data_dir)?;
        }

        if !iris_path.exists() {
            let url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data";
            let response = reqwest::blocking::get(url)
                .unwrap()
                .text()
                .unwrap();
            
            let mut file = File::create(&iris_path)?;
            file.write_all(response.as_bytes())?;
        }
        
        Ok(iris_path.to_str().unwrap().to_string())
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

    // Helper for iris dataset
    #[derive(Debug, Deserialize)]
    struct IrisRecord {
        #[serde(rename = "0")]
        sepal_length: f64,
        #[serde(rename = "1")]
        sepal_width: f64,
        #[serde(rename = "2")]
        petal_length: f64,
        #[serde(rename = "3")]
        petal_width: f64,
        #[serde(rename = "4")]
        species: String,
    }

    fn make_binomial_regression_dataset() -> (Mat<f64>, Mat<f64>) {
        // Load Iris dataset and create binary classification (setosa vs others)
        let iris_path = download_iris_dataset().unwrap();
        let mut file = File::open(&iris_path).unwrap();

        let nrows = {
            let mut rdr = csv::ReaderBuilder::new()
                .has_headers(false)
                .from_reader(&mut file);
            let count = rdr.records().count();
            count
        };

        file.seek(SeekFrom::Start(0)).unwrap();
        
        let mut X: Mat<f64> = Mat::zeros(nrows, 4);
        let mut y: Mat<f64> = Mat::zeros(nrows, 1);

        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_reader(file);

        for (i, result) in rdr.deserialize().enumerate() {
            let record: IrisRecord = result.unwrap();
            // Skip empty rows (the dataset has some empty lines at the end)
            if record.species.is_empty() {
                continue;
            }

            X[(i, 0)] = record.sepal_length;
            X[(i, 1)] = record.sepal_width;
            X[(i, 2)] = record.petal_length;
            X[(i, 3)] = record.petal_width;

            if record.species == "Iris-setosa" {
                y[(i, 0)] = 0.0;
            } else {
                y[(i, 0)] = 1.0;
            }
        }
       
        (X, y)
    }

    #[test]
    fn test_binomial_regression() {
        // Use 0 to get all available samples
        let (X, y) = make_binomial_regression_dataset();
        println!("X: {:?}", X);
        println!("y: {:?}", y);
        println!("Dataset size: {} samples, {} features", X.nrows(), X.ncols());

        let mut model = Binomial::new(true);
        model.fit_unchecked(X.as_ref(), y.as_ref());
        //let preds = model.fitted_values();

        let probabilities = model.glm_predict(X.as_ref()).unwrap();
        let n_rows = X.nrows();
         
        let mut correct = 0;

        for i in 0..probabilities.nrows() {
            for j in 0..probabilities.ncols() {
                if probabilities[(i, j)] > 0.5 && y[(i, j)] == 1.0 {
                    correct += 1;
                }

                if probabilities[(i, j)] < 0.5 && y[(i, j)] == 0.0 {
                    correct += 1;
                }
            }
        }

        let accuracy = correct as f64 / n_rows as f64;
        println!("Accuracy: {}", accuracy);

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
