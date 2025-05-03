use super::functions::{LinkFunction, VarianceFunction};

#[derive(Clone, Copy, PartialEq)]
pub enum GLMFamily {
    Gaussian,
    Poisson,
    Binomial,
    Gamma,
    Custom(LinkFunction, VarianceFunction),
}

impl GLMFamily {
    pub fn link_function(&self) -> LinkFunction {
        match self {
            GLMFamily::Gaussian => LinkFunction::Identity,
            GLMFamily::Poisson => LinkFunction::Log,
            GLMFamily::Binomial => LinkFunction::Logit,
            GLMFamily::Gamma => LinkFunction::Inverse,
            GLMFamily::Custom(link, _) => *link,
        }
    }
    
    pub fn variance_function(&self) -> VarianceFunction {
        match self {
            GLMFamily::Gaussian => VarianceFunction::Gaussian,
            GLMFamily::Poisson => VarianceFunction::Poisson,
            GLMFamily::Binomial => VarianceFunction::Binomial,
            GLMFamily::Gamma => VarianceFunction::Gamma,
            GLMFamily::Custom(_, var) => *var,
        }
    }
}

impl From<&str> for GLMFamily {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "gaussian" | "normal" => GLMFamily::Gaussian,
            "poisson" => GLMFamily::Poisson,
            "binomial" | "logistic" => GLMFamily::Binomial,
            "gamma" => GLMFamily::Gamma,
            _ => GLMFamily::Gaussian, // Default to Gaussian
        }
    }


}

