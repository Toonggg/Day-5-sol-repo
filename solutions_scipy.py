import numpy as np
import scipy
import scipy.linalg as alg
import scipy.stats as st
import matplotlib.pyplot as plt
# Linear Algebra
# a) 
A = np.arange(1, 10).reshape(3, 3)
print(A)
print()

# b)
b = np.arange(1, 4).reshape(3, 1)

# c)
# Matrix has zero determinant so can't invert.
# Scipy gives an error. 

# d) 
# See above answer.

# e)
B = np.random.random(size = (3,3))
b_rand = np.random.random(size = (3, 1))
B_inv = alg.inv(B)

x_rand = np.matmul(B_inv, b_rand)
print("b_rand is:", b_rand)
print("B times x_rand:", np.matmul(B, x_rand))
print("B times x_rand is b_rand:", np.isclose(b_rand, np.matmul(B, x_rand)))
print()

# f)
lambda_A = alg.eigvals(A)
eigvec_A = alg.eig(A)
print("Eigenvalues A:\n")
print(lambda_A) 
print("Eigenvectors A:\n")
print(eigvec_A)
print()
# g)
# Since A has determinant zero (manual calculation), the inverse is ill-defined. Might use pinv().... 

# h) 
A_fro = alg.norm(A, 'fro')
print("Frobenius norm A:", A_fro)
A_one = alg.norm(A, 1)
print("One norm A:", A_one)
A_two = alg.norm(A, 2)
print("Two norm A:", A_two) 
A_inf = alg.norm(A, np.inf)
print("Infinity norm A:", A_inf)
A_ninf = alg.norm(A, -np.inf)
print("Negative infinity norm A:", A_ninf)

# Statistics
# a)
k = np.arange(0, 30, step = 1)
pmf_poiss = st.poisson.pmf(k, mu = 0.5)
plt.figure(1)
plt.title('PMF')
plt.xlabel('')
plt.plot(pmf_poiss)
cdf_poiss = st.poisson.cdf(k, mu = 0.5)
plt.figure(2)
plt.title('CDF')
plt.xlabel('')
plt.plot(cdf_poiss)
samples = st.poisson.rvs(mu = 0.5, size = 1000)
plt.figure(3)
plt.title('Histogram of 1000 Poisson random variables')
plt.xlabel('')
plt.ylabel('Counts')
plt.hist(samples)

# b)
x = np.linspace(st.norm.ppf(0.01), st.norm.ppf(0.99), 100)
pdf_norm = st.norm.pdf(x)
plt.figure(4)
plt.title('PDF')
plt.xlabel('')
plt.plot(pdf_norm)
cdf_norm = st.norm.cdf(x)
plt.figure(5)
plt.title('CDF')
plt.xlabel('')
plt.plot(cdf_norm)
samples_norm = st.norm.rvs(size = 1000)
plt.figure(6)
plt.title('Histogram of 1000 normal random variables')
plt.xlabel('')
plt.ylabel('Counts')
plt.hist(samples_norm)
plt.show()

# c)
check_id = st.ttest_ind(st.poisson.cdf(k, mu = 0.5), st.norm.rvs(size = 1000))
print("P-value:", check_id.pvalue) #2 sets of random data comes from different distributions

check_id_norm = st.ttest_ind(st.norm.rvs(size = 1000), st.norm.rvs(size = 1000))
print("P-value:", check_id_norm.pvalue) #2 different samples from a normal distribution come from the same distribution...

check_id_poiss = st.ttest_ind(st.poisson.cdf(k, mu = 0.5), st.poisson.cdf(k, mu = 0.5))
print("P-value:", check_id_poiss.pvalue) #2 different samples from a normal distribution come from the same distribution...