import numpy as np
import scipy.stats as stats
import time
import matplotlib.pyplot as plt

def montecarlo(sample, proba, with_interval_bounds = False):
    """ This function returns the evolution of the Monte Carlo estimator along with the asymptotic variance

    Args:
        sample (np.ndarray): Sample of simulated variables 
        proba (double): The confidence level associated with the estimator's interval
        with_interval_bounds (bool): If True, it also returns what it takes to compute the upper and lower interval bounds easily 
    Returns : 
        np.ndarray, double, double 
    """
    n = sample.size
    Jn = sample.cumsum() / np.arange(1, n + 1)
    var = sample.var(ddof = 1)
    
    if with_interval_bounds == False:
        return Jn, var
    
    else:
        alpha = 1 - proba
        q = stats.norm().ppf(1 - alpha / 2)
        bound = q * np.sqrt(var / np.arange(1, n + 1))
        return Jn, var, bound
        
        


def montecarlo_estimators(N, d, S0, sigma, r, T, K, w, R,with_timings = False, with_values = False):
    """ This function outputs the simulated paths of 4 Monte Carlo estimators, each one of size N:
        - Naive Monte Carlo (MV)
        - Variate Control with a geometric basket option (VC)
        - New Variate Control studied by Curran(1994) (NVC)
        - New Variate Control + condtional expectation method (NVC_C) (Sun and Xu (2017))
    
    Args:
        N (int): Number of Monte Carlo simulations
        d (int): Number of assets in the basket
        S0 (np.ndarray): Initial prices of the assets (size d)
        sigma (np.ndarray): Annual volatilities of the assets (size d)
        r (double): Annual risk-free rate
        T (int): Time to maturity in years
        K (double): Strike price of the option
        w (np.ndarray): Weights of the basket (size d)
        R (np.ndarray): Correlation matrix of the assets (d x d)
        with_timings (bool) : Indicates whether the user wants to obtain information about the computational cost of each technique
    
    """
    
    timings = np.zeros(4)
    norm = stats.norm()
    
    #Is R inversible?
    
    
    cov = R * T

    det = np.linalg.det(cov)

    if (det == 0):
        print("The Matrix is not inversible and therefore we can't apply the Cholesky algorithm")
        return

    #Cholesky
    L = np.linalg.cholesky(cov)
    
    
    #Simulation of Gaussian variables and corresponding spot prices according to the assumptions in the Black-Scholes model

    mean = np.zeros(d)
    cov_Id = np.eye(d)

    Y = np.random.multivariate_normal(mean,cov_Id, size = N)
    W = Y @ L.T 
    S = S0 * np.exp((r - 0.5 * sigma**2) * T + W * sigma)
    
    
    #Naive Monte Carlo 
    
    start = time.time()
    
    A = S@w
    payoff = np.exp(-r * T) * np.maximum(A - K,0)

    end = time.time()
    timings[0] = end - start
    
    #Variate Control with a geometric basket option (VC)
    
    start = time.time()
    
    mu = (np.log(S0) + (r - (sigma**2)/2) * T) @ w
    diag_var = np.diag(sigma)
    var_VC = (w.T @ diag_var @ R @ diag_var @ w) * T
    std_VC = np.sqrt(var_VC)
    d2 =  - (np.log(K) - mu) / std_VC
    d1 = d2 + std_VC
    closed_formula_GS = np.exp(mu + var_VC / 2 - r * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    c = 1
    
    start3 = time.time()
    
    G = np.exp(np.log(S)@w)
    
    end3 = time.time()
    
    payoff2 = np.exp(-r * T) * np.maximum(G- K, 0)
    payoff_VC = payoff - c * (payoff2 - closed_formula_GS)
    
    end = time.time()
    timings[1] = end - start
    
    #New Variate Control studied by Curran(1994)
    
    start = time.time()
    
    start2 = time.time()
    
    mu_i = np.log(S0) + (r - (sigma**2)/2) * T
    sigma_xi = sigma * (R @ (w * sigma)) * T
    CDF = norm.cdf(d2 + sigma_xi  / std_VC )
    B = (np.exp((mu_i + (sigma**2) * T / 2) ) * CDF)@ w
    C = K * norm.cdf(d2)
    closed_formula_new_variate = np.exp(-r * T) * (B - C)
    
    end2 = time.time()
    
    c = 1
    payoff3 = np.exp(-r * T) * np.where(G < K, 0, A - K)
    payoff_NVC = payoff - c * (payoff3 - closed_formula_new_variate)
    
    end = time.time()
    timings[2] = end - start + end3 - start3
    
    #New Variate Control studied by Curran(1994) + condtional expectation method (NVC_C)
    
    
    w_tilde = w[:d-1]
    S_tilde = S[:,:d-1]
    Y_tilde = Y[:,:-1]
    YLd = Y_tilde @ L[d-1, :d-1]
    
    start = time.time()
    
    k = np.log(K)
    drift = (r - 0.5 * sigma**2) * T
    
    kmw = (k - np.log(S_tilde) @ w_tilde) / w[d-1] - np.log(S0[d-1]) - (r - (sigma[d-1]**2)/2) * T
    ku = (kmw - sigma[d-1] * (Y_tilde @ L[d-1,:d-1])) / (sigma[d-1] * L[d-1,d-1])
    
    Kmw_ws = (K - S_tilde @ w_tilde) / (w[d-1] * S0[d-1])
    lnKmw_ws_modified = np.full_like(Kmw_ws, -np.inf, dtype=float)
    mask = Kmw_ws > 0
    lnKmw_ws_modified[mask] = np.log(Kmw_ws[mask])
    kl = (lnKmw_ws_modified - (r - (sigma[d-1]**2)/2) * T - sigma[d-1] * (YLd)) / (sigma[d-1] * L[d-1,d-1])
    GKA = norm.cdf(ku) - norm.cdf(kl)
    
    Pyi_list = []

    for i in range(d-1):
        Pyi_list.append(np.exp(sigma[i] * W[:,i]) * GKA)
        
    a = sigma[d-1] * L[d-1,d-1]

    Pyi_list.append(np.exp(sigma[d-1] * (YLd) + (a**2)/2) * (norm.cdf(ku - a) - norm.cdf(kl - a)))
    Pyi = np.column_stack(Pyi_list)
    
    
    term_A = np.sum(w * (S0 * np.exp(drift)) * Pyi, axis=1)
    payoff_NVC_C = np.exp(-r * T) * (term_A - K * GKA)  + closed_formula_new_variate
                            
    end = time.time()
    timings[3] = end - start + end2 - start2
    
    
    if with_timings:
        return payoff, payoff_VC, payoff_NVC, payoff_NVC_C, timings
    
    elif with_values:
        return payoff, payoff_VC, payoff_NVC, payoff_NVC_C, A, G, timings
    
    else:
        return payoff, payoff_VC, payoff_NVC, payoff_NVC_C

   
    
            
def plot_variance_ratios(K, ratio, ratio_cost, K_name, L=5):
    """
    Plots the variance and cost-adjusted variance ratios for different estimators as K varies.

    Args :

        K (np.ndarray) :  Array of values
        ratio (np.ndarray) : Variance ratios for the estimators. Shape (len(K), 3)
        ratio_cost (np.ndarray) : Cost-adjusted variance ratios for the estimators. Shape (len(K), 3).
        K_name (string) : name of the variable we are working with
        L (int) : Number of last K values to zoom on. Default is 5.
        
    """
        
        
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    

    axes[0, 0].plot(K, ratio[:,0], label="VC / MC")
    axes[0, 0].plot(K, ratio[:,1], label="NVC / MC")
    axes[0, 0].plot(K, ratio[:,2], label="NVC_C / MC")
    axes[0, 0].set_xlabel(f"{K_name} values")
    axes[0, 0].set_ylabel("Ratio")
    axes[0, 0].set_title(f"Evolution of the variance ratio versus {K_name}")
    axes[0, 0].legend()
    axes[0, 0].grid(True)


    axes[0, 1].plot(K[-L:], ratio[-L:,0], label="VC / MC")
    axes[0, 1].plot(K[-L:], ratio[-L:,1], label="NVC / MC")
    axes[0, 1].plot(K[-L:], ratio[-L:,2], label="NVC_C / MC")
    axes[0, 1].set_xlabel(f"{K_name} values")
    axes[0, 1].set_ylabel("Ratio")
    axes[0, 1].set_title(f"Zoom on the last {L} {K_name} values")
    axes[0, 1].legend()
    axes[0, 1].grid(True)


    axes[1, 0].plot(K, ratio_cost[:,0], label="VC / MC")
    axes[1, 0].plot(K, ratio_cost[:,1], label="NVC / MC")
    axes[1, 0].plot(K, ratio_cost[:,2], label="NVC_C / MC")
    axes[1, 0].set_xlabel(f"{K_name} values")
    axes[1, 0].set_ylabel("Cost-adjusted ratio")
    axes[1, 0].set_title(f"Evolution of the cost-adjusted variance ratio versus {K_name}")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(K[-L:], ratio_cost[-L:,0], label="VC / MC")
    axes[1, 1].plot(K[-L:], ratio_cost[-L:,1], label="NVC / MC")
    axes[1, 1].plot(K[-L:], ratio_cost[-L:,2], label="NVC_C / MC")
    axes[1, 1].set_xlabel(f"{K_name} values")
    axes[1, 1].set_ylabel("Cost-adjusted ratio")
    axes[1, 1].set_title(f"Zoom on the last {L} {K_name} values")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()