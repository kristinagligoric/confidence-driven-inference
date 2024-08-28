import os
import sys
sys.path.insert(1, '../')
import numpy as np
import pdb
import pandas as pd
import scipy
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm
import xgboost as xgb


def safe_log1pexp(x):
    """
    Compute log(1 + exp(x)) in a numerically stable way.
    """
    idxs = x > 10
    out = np.empty_like(x)
    out[idxs] = x[idxs]
    out[~idxs] = np.log1p(np.exp(x[~idxs]))
    return out


def safe_expit(x):
    """Computes the sigmoid function in a numerically stable way."""
    return np.exp(-np.logaddexp(0, -x))


def train_tree(X, Y, eta=0.001, max_depth=3, objective='reg:squarederror', boost_rounds=2000):
    dtrain = xgb.DMatrix(X, label=Y)
    tree = xgb.train({'eta': eta, 'max_depth': max_depth, 'objective': objective}, dtrain, boost_rounds)
    return tree


def tree_predict(tree, X):
    return tree.predict(xgb.DMatrix(X))


def logistic(X, Y):
    regression = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=10000,
        tol=1e-15,
        fit_intercept=False,
    ).fit(X, Y)
    return regression.coef_.squeeze()


def logistic_cov(
    pointest,
    X,
    Y,
    Yhat,
    weights,
    lam=1
):
    n = Y.shape[0]
    d = X.shape[1]

    mu = safe_expit(X @ pointest)
    weights_mat = np.array([weights] * d)

    hessian = np.zeros((d, d))
    grads_hat = np.zeros(X.shape)
    grads = np.zeros(X.shape)
    for i in range(n):
        hessian += (
                1 / n
                * mu[i]
                * (1 - mu[i])
                * np.outer(X[i], X[i])
            )
        grads_hat[i, :] = (
                X[i, :]
                * (mu[i] - Yhat[i])
            )
        grads[i, :] = X[i, :] * (mu[i] - Y[i])

    inv_hessian = np.linalg.inv(hessian).reshape(d, d)
    var = np.cov(grads.T*weights_mat + lam * grads_hat.T  - lam * grads_hat.T * weights_mat).reshape(d, d)
    return inv_hessian @ var @ inv_hessian


def active_logistic_pointestimate(
    X,
    Y,
    Yhat,
    weights,
    lam=None,
    coord=None,
    optimizer_options=None
):
    n = Y.shape[0]
    d = X.shape[1]
    if optimizer_options is None:
        optimizer_options = {"ftol": 1e-15}
    if "ftol" not in optimizer_options.keys():
        optimizer_options["ftol"] = 1e-15

    # Initialize theta
    theta = (
        LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=10000,
            tol=1e-15,
            fit_intercept=False,
        )
        .fit(X[np.where(weights)], Y[np.where(weights)])
        .coef_.squeeze()
    )
    if len(theta.shape) == 0:
        theta = theta.reshape(1)


    def rectified_logistic_loss(_theta):
        return (
            lam
            / n
            * np.sum(
                    -Yhat * (X @ _theta)
                    + safe_log1pexp(X @ _theta)
            )
            - lam
            / n
            * np.sum(weights * (-Yhat * (X @ _theta) + safe_log1pexp(X @ _theta)))
            + 1
            / n
            * np.sum(weights * (-Y * (X @ _theta) + safe_log1pexp(X @ _theta)))
        )
        

    def rectified_logistic_grad(_theta):
        return (
            lam
            / n
            * X.T
            @ (
                safe_expit(X @ _theta) - Yhat
            )
            - lam / n * X.T @ (weights * (safe_expit(X @ _theta) - Yhat))
            + 1 / n * X.T @ (weights * (safe_expit(X @ _theta) - Y))
        )

    pointest = minimize(
        rectified_logistic_loss,
        theta,
        jac=rectified_logistic_grad,
        method="L-BFGS-B",
        tol=optimizer_options["ftol"],
        options=optimizer_options,
    ).x

    return pointest


def opt_logistic_tuning(
    pointest,
    X,
    Y,
    Yhat,
    weights
):
    n = Y.shape[0]
    d = X.shape[1]

    mu = safe_expit(X @ pointest)

    hessian = np.zeros((d, d))
    grads_hat = np.zeros(X.shape)
    grads = np.zeros(X.shape)
    for i in range(n):
        hessian += (
                1 / n
                * mu[i]
                * (1 - mu[i])
                * np.outer(X[i], X[i])
            )
        grads_hat[i, :] = (
                X[i, :]
                * (mu[i] - Yhat[i]) * (weights[i] - 1)
            )
        grads[i, :] = X[i, :] * (mu[i] - Y[i]) * weights[i]

    grads_cent = grads - grads.mean(axis=0)
    grad_hat_cent = grads_hat - grads_hat.mean(axis=0)
    cov_grads = (1 / n) * (
        grads_cent.T @ grad_hat_cent + grad_hat_cent.T @ grads_cent
    )

    var_grads_hat = np.cov(grads_hat.T)
    
    inv_hessian = np.linalg.inv(hessian).reshape(d, d)
    h = inv_hessian[0, :]
    num = h @ cov_grads @ h
    denom = 2 * h @ var_grads_hat @ h
    lam = num / denom
    
    return lam

def opt_mean_tuning(Y, Yhat, weights, sampling_ratio):
    return np.clip(np.mean(Y*Yhat*weights*sampling_ratio)/np.mean(Yhat**2*sampling_ratio), 0, 1)


def odds_ratio_ci(Y0, Yhat0, Y1, Yhat1, weights0, weights1, alpha, lhat0=None, lhat1=None):
    n0 = Y0.shape[0]
    n1 = Y1.shape[0]    
    mu0_hat = np.mean(lhat0*Yhat0 + (Y0 - lhat0*Yhat0)*weights0)
    mu1_hat = np.mean(lhat1*Yhat1 + (Y1 - lhat1*Yhat1)*weights1)
    pointest_log = np.log(mu1_hat/(1-mu1_hat)) - np.log(mu0_hat/(1-mu0_hat))
    var_mu0_hat = np.var(lhat0*Yhat0 + (Y0 - lhat0*Yhat0)*weights0)
    var_mu1_hat = np.var(lhat1*Yhat1 + (Y1 - lhat1*Yhat1)*weights1)
    var0 = var_mu0_hat/((mu0_hat*(1-mu0_hat))**2)
    var1 = var_mu1_hat/((mu1_hat*(1-mu1_hat))**2)
    p0 = n0/(n0+n1)
    p1 = n1/(n0+n1)
    var = 1/p0*var0 + 1/p1*var1
    width_log = norm.ppf(1-alpha/2)*np.sqrt(var/(n0+n1))
    return np.exp(pointest_log - width_log), np.exp(pointest_log + width_log), var


def inv_hessian_col_imputed(X, Yhat):
    n = Yhat.shape[0]
    d = X.shape[1]
    imputed_pointest = logistic(X, (Yhat > 0.5))
    mu = safe_expit(X @ imputed_pointest)
    hessian = np.zeros((d, d))
    for i in range(n):
        hessian += (
            1 / n
            * mu[i]
            * (1 - mu[i])
            * np.outer(X[i], X[i])
        )
    inv_hessian = np.linalg.inv(hessian)
    return inv_hessian[:, 0]

def mean_ESS(Y, varhat):
    return np.mean(Y**2)/(varhat + np.mean(Y)**2)*len(Y)