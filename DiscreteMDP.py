import numpy as np
import matplotlib.pyplot as plt

"""---------------- MDP DEFINITION -----------------------------"""
n_states = input("How many states would you like to have in your MDP? ")
gamma = input("Choose gamma: ")
print("The system will generate ",n_states , " states MDP, with reward chosen between -10. and 10. uniformly, and the transition distribution choosen uniformly belween 0. and 1. between all the states")
R = np.random.uniform(-10.,10.,(n_states,1))
P = np.random.uniform(0.,1.,(n_states, n_states))
marg = np.sum(P,axis=1)**(-1)
P = np.matmul(np.diag(marg), P)

n_iteration = input("How many iteration would you like to run for evaluating V? ")

"""---------------- Sampled MDP --------------------------------"""
N_Future_States = 50
RSample = lambda : np.random.normal(R, 0.5)
def PSampleMean() :
    samples = np.array(map(lambda x: np.random.multinomial(N_Future_States, x), P))
    return samples/(N_Future_States+0.)


"""---------------------- OPERATORS --------------------------------------"""
""" Perfect Bellman Operator"""
T = lambda v: R + gamma * np.matmul(P,v)
""" Samples Bellman Operator"""
TSample = lambda v: RSample() + gamma *  np.matmul(PSampleMean(),v)
""" Perfect S operator"""
S = lambda b: gamma * np.matmul(P,b)
""" Samples S operator"""
SSample = lambda b: gamma * np.matmul(PSampleMean(),b)
""" Approximator Operator"""
approx  = lambda f: f + np.random.normal(np.zeros_like(f), .01 * np.abs(f))

"""------------------------ Algorithms -------------------------"""

""" Emulates BPE and RPE together with just function approximator. 
The output is the mean-squared Bellman Error"""
def MixedApproximateIterationBoosting(alpha):
    V = np.zeros((n_states, 1))
    B = approx(T(V) - V)
    for _ in range(n_iteration):
        V = V + B
        B = approx((1. - alpha) * (T(V) - V) + alpha * S(B))
    V = V + B
    return np.mean(np.square(T(V) - V))

""" Emulates BPE and RPE together with just function approximator and samples. 
The output is the mean-squared Bellman Error"""
def MixedApproximateIterationBoostingSamples(alpha):
    V = np.zeros((n_states, 1))
    B = approx(TSample(V) - V)
    for _ in range(n_iteration):
        V = V + B
        B = approx((1. - alpha) * (TSample(V) - V) + alpha * SSample(B))
    V = V + B
    return np.mean(np.square(T(V) - V))



print("\n We will compute the plots comparing the Bellman error at the last iteration, both taking in account of samplings and not.\n In x-axis there will be the mixing parameter alpha. On the y-axis the MS Bellman Error.\n Every single instance of the algorithm is run for 100 times.")
n_alphas = 20

error_perfect_matrix = np.zeros((0,n_alphas*2))
error_samples_matrix = np.zeros((0,n_alphas*2))

alphas = np.concatenate((np.linspace(0.,0.9,n_alphas),np.linspace(0.9,1.,n_alphas)))
for _ in range(100):
    errors_perfect = []
    errors_samples = []
    print '.',
    for alpha in alphas:
        errors_perfect.append(MixedApproximateIterationBoosting(alpha))
        errors_samples.append(MixedApproximateIterationBoostingSamples(alpha))

    error_perfect_matrix = np.concatenate((error_perfect_matrix, [errors_perfect]), axis=0)
    error_samples_matrix = np.concatenate((error_samples_matrix, [errors_samples]), axis=0)

m = np.mean(error_perfect_matrix, axis=0)
c = 2. * np.std(error_perfect_matrix / 10., axis=0)
plt.title("Mixed BPE and RPE with perfect model")
plt.plot(alphas, m, color='b')
plt.fill_between(alphas,m-c, m+c, color='b', alpha=0.25 )
plt.legend(loc='best')
plt.ylabel('MSBE')
plt.xlabel('alpha')
plt.show()

plt.title("Mixed BPE and RPE with sampled reward and future state")
m = np.mean(error_samples_matrix, axis=0)
c = 2. * np.std(error_samples_matrix / 10., axis=0)
plt.plot(alphas, m, color='b')
plt.fill_between(alphas,m-c, m+c, color='b', alpha=0.25 )
plt.legend(loc='best')
plt.ylabel('MSBE')
plt.xlabel('alpha')
plt.show()