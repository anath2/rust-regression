use faer::{Mat, MatRef, linalg::solvers::Solve};
use faer_traits::RealField;
use std::f64::EPSILON;
use num::Float;


/// Ordinary Least Squares using QR decomposition
#[allow(non_snake_case)]
pub fn ols_qr<T: RealField + Float>(
    X: MatRef<T>, 
    y: MatRef<T>,
) -> Mat<T> {
    let Xt = X.transpose();
    let XtX = Xt * X;
    return XtX.col_piv_qr().solve(Xt * y)
}


/// Weighted Least Squares using QR decomposition
#[allow(non_snake_case)]
pub fn weighted_lstq<T: RealField + Float>(
    X: MatRef<T>,
    y: MatRef<T>,
    w: MatRef<T>,
) -> Mat<T> {
    let n_rows = X.nrows();
    let n_cols = X.ncols();
    let Xt = X.transpose();
    let mut Diag = Mat::zeros(n_rows, n_rows);

    // Calculate diagonal of weight matrix
    for i in 0..n_rows {
        let diag_val = w[(i, 0)];
        Diag[(i, i)] = diag_val;
    }
    
    // Add L2 regularization (ridge regression)
    let lambda = T::from(0.01).unwrap();
    let mut XtWX = Xt * Diag.as_ref() * X;
    
    // Add lambda to diagonal of XtWX (except bias term)
    for i in 1..n_cols {
        XtWX[(i, i)] = XtWX[(i, i)] + lambda;
    }
    
    return XtWX.col_piv_qr().solve(Xt * Diag.as_ref() * y);    
}


/// Iteratively Reweighted Least Squares (IRLS) solver for GLMs.
#[allow(non_snake_case)]
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

    // Initialize coefficients (beta) to small random values
    let mut beta = Mat::zeros(p, 1);
    for i in 0..p {
        beta[(i, 0)] = T::from(0.1).unwrap();
    }
    let mut eta = X * &beta;
    let mut mu = Mat::zeros(n, 1);
    // Initialize mu with bounded values to avoid numerical issues
    for i in 0..n {
        mu[(i, 0)] = inv_link(eta[(i, 0)]);
        // Bound mu away from 0 and 1 for numerical stability
        let eps = T::from(1e-6).unwrap();
        if mu[(i, 0)] < eps {
            mu[(i, 0)] = eps;
        } else if mu[(i, 0)] > T::one() - eps {
            mu[(i, 0)] = T::one() - eps;
        }
    }
    // Initialize weights
    let mut w = Mat::ones(n, 1);

    let eps = T::from(1e-6).unwrap();
    let one = T::one();
    
    for iter in 0..max_iter {
        let beta_old = beta.clone();

        // Compute mu = inv_link(eta) with bounds
        for i in 0..n {
            mu[(i, 0)] = inv_link(eta[(i, 0)]);
            // Bound mu away from 0 and 1
            if mu[(i, 0)] < eps {
                mu[(i, 0)] = eps;
            } else if mu[(i, 0)] > one - eps {
                mu[(i, 0)] = one - eps;
            }
        }
        
        // Compute weights with numerical stability
        for i in 0..n {
            let mu_i = mu[(i, 0)];
            let var_i = variance(mu_i).max(eps);  // Ensure variance is positive
            let link_deriv_i = link_deriv(mu_i);
            let denom = var_i * link_deriv_i * link_deriv_i + eps;
            w[(i, 0)] = if denom.is_finite() { one / denom } else { eps };
        }
        
        // Adjusted response z = eta + (y - mu) * link'(mu)
        let mut z = Mat::zeros(n, 1);
        for i in 0..n {
            let mu_i = mu[(i, 0)];
            let link_deriv_i = link_deriv(mu_i);
            let diff = (y[(i, 0)] - mu_i).max(-one).min(one);  // Bound difference
            z[(i, 0)] = eta[(i, 0)] + diff * link_deriv_i;
        }
        
        // Weighted least squares step
        beta = weighted_lstq(X, z.as_ref(), w.as_ref());
        
        // Update eta with bounds
        eta = X * &beta;
        for i in 0..n {
            if !eta[(i, 0)].is_finite() {
                eta[(i, 0)] = T::zero();
            }
        }
        
        // Check for convergence
        let mut diff = T::zero();
        let mut has_nan = false;
        for j in 0..p {
            if !beta[(j, 0)].is_finite() {
                has_nan = true;
                break;
            }
            diff = diff + (beta[(j, 0)] - beta_old[(j, 0)]).abs();
        }
        
        // Reset if we hit NaN
        if has_nan {
            if iter == 0 {
                // If we get NaN on first iteration, try different initial values
                for j in 0..p {
                    beta[(j, 0)] = T::from(0.01).unwrap();
                }
            } else {
                // Otherwise revert to previous iteration
                beta = beta_old;
            }
            eta = X * &beta;
            continue;
        }
        
        if diff < tol {
            println!("Converged after {} iterations", iter);
            break;
        }
    }
    beta
}