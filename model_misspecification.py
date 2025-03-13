import numpy as np

# -----------------------------
# 1. Setup
# -----------------------------
np.random.seed(42)

n = 1000         # sample size
n_sims = 5000   # number of bootstrap replicates

# True data-generating parameters
mu_true = 0.0
sigma_true = 1.0

# Generate one "observed" dataset from the TRUE distribution
observed_data = np.random.normal(loc=mu_true, scale=sigma_true, size=n)

# MLE for normal model: sample mean and sample std
mu_hat = np.mean(observed_data)
sigma_hat = np.std(observed_data, ddof=1)

# -----------------------------
# 2. Parametric Bootstrap (Correct Model)
# -----------------------------
bootstrap_means_correct = np.empty(n_sims)

for b in range(n_sims):
    # Sample new dataset from N(mu_hat, sigma_hat^2)
    bootstrap_sample = np.random.normal(loc=mu_hat, scale=sigma_hat, size=n)
    # Fit normal MLE (sample mean)
    bootstrap_means_correct[b] = np.mean(bootstrap_sample)

# Estimated SE using the parametric bootstrap
boot_se_correct = np.std(bootstrap_means_correct, ddof=1)

# -----------------------------
# 3. Compare with Theoretical SE
# -----------------------------
# The MLE for the mean is ~ N(mu_true, sigma_true^2 / n), so theoretical SE is:
theoretical_se = sigma_true / np.sqrt(n)

print("Parametric Bootstrap (Correct Model)")
print("------------------------------------")
print(f"Observed MLE of mean: {mu_hat:.4f}")
print(f"Bootstrap-based SE:   {boot_se_correct:.4f}")
print(f"Theoretical SE:       {theoretical_se:.4f}")
print()

# -----------------------------
# 4. Misspecified Model Example
#    Suppose the true distribution is Normal,
#    but we *incorrectly* fit (say) a T-distribution with low df
#    to do the bootstrap. This is a contrived example:
#    We'll just pretend we have a "wrong" scale for demonstration.
# -----------------------------
# We'll just pick a "wrong" scale for demonstration. 
# In reality, you'd do something like fit a T( df=3 ) or something else.
wrong_sigma_hat = sigma_hat / 2.0   # "too narrow" model

bootstrap_means_wrong = np.empty(n_sims)
for b in range(n_sims):
    bootstrap_sample_wrong = np.random.normal(loc=mu_hat, 
                                              scale=wrong_sigma_hat, 
                                              size=n)
    bootstrap_means_wrong[b] = np.mean(bootstrap_sample_wrong)

boot_se_wrong = np.std(bootstrap_means_wrong, ddof=1)

print("Parametric Bootstrap (Misspecified Model)")
print("-----------------------------------------")
print(f"Observed MLE of mean:  {mu_hat:.4f}")
print(f"Bootstrap-based SE:    {boot_se_wrong:.4f}")
print(f"Theoretical SE (true): {theoretical_se:.4f}")

