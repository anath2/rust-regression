#![allow(non_snake_case)]
use faer::{Mat, MatRef, linalg::solvers::Solve};
use faer_traits::RealField;
use std::f64::EPSILON;
use num::Float;

/// Ordinary Least Squares using QR decomposition
pub fn ols_qr<T: RealField + Float>(
    X: MatRef<T>, 
    y: MatRef<T>,
) -> Mat<T> {
    let Xt = X.transpose();
    let XtX = Xt * X;
    return XtX.col_piv_qr().solve(Xt * y)
}

pub fn weighted_lstq<T: RealField + Float>(
    X: MatRef<T>,
    y: MatRef<T>,
    w: MatRef<T>,
) -> Mat<T> {
    let n_rows = X.nrows();
    let Xt = X.transpose();
    let mut Diag = Mat::zeros(n_rows, n_rows);

    // Calculate diagonal of weight matrix
    for i in 0..n_rows {
        let diag_val = w[(i, 0)];
        Diag[(i, i)] = diag_val;
    }
    
    return (Xt * Diag.as_ref() * X).col_piv_qr().solve(Xt * Diag.as_ref() * y);    
}

/// Iteratively Reweighted Least Squares (IRLS) solver for GLMs.
pub fn irls<T, ILF, LDF, VF>(
    X: MatRef<T>,
    y: MatRef<T>,
    inv_link: ILF,
    link_deriv: LDF,
    variance: VF,
) -> Mat<T>
where
    T: RealField + Float,
    ILF: Fn(T) -> T,
    LDF: Fn(T) -> T,
    VF: Fn(T) -> T,
{
    let n = X.nrows();
    let p = X.ncols();
    let max_iter = 25;
    let tol = T::from(1e-6).unwrap();

    // Initialize coefficients (beta)
    let mut beta = Mat::zeros(p, 1);
    let mut eta = X * &beta;
    let mut mu = eta.clone();
    // Initialize weights
    let mut w = Mat::ones(n, 1);

    for _ in 0..max_iter {
        let beta_old = beta.clone();

        // Compute mu = inv_link(eta)
        for i in 0..n {
            mu[(i, 0)] = inv_link(eta[(i, 0)]);
        }
        // Compute weights: w_i = 1 / [variance(mu_i) * (link'(mu_i))^2]
        for i in 0..n {
            let mu_i = mu[(i, 0)];
            let var_i = variance(mu_i);
            let link_deriv_i = link_deriv(mu_i);
            w[(i, 0)] = T::one() / (var_i * link_deriv_i * link_deriv_i + T::from(EPSILON).unwrap());
        }
        // Adjusted response z = eta + (y - mu) * link'(mu)
        let mut z = Mat::zeros(n, 1);
        for i in 0..n {
            let mu_i = mu[(i, 0)];
            let link_deriv_i = link_deriv(mu_i);
            z[(i, 0)] = eta[(i, 0)] + (y[(i, 0)] - mu_i) * link_deriv_i;
        }
        // Weighted least squares step
        beta = weighted_lstq(X, z.as_ref(), w.as_ref());
        // Update eta for next iteration
        eta = X * &beta;
        // Check for convergence
        let mut diff = T::zero();
        for j in 0..p {
            diff = diff + (beta[(j, 0)] - beta_old[(j, 0)]).abs();
        }
        if diff < tol {
            break;
        }
    }
    beta
}