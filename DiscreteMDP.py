import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import rc
rc('text', usetex=True)

"""---------------- MDP DEFINITION -----------------------------"""
n_states = input("How many states would you like to have in your MDP? ")
gamma = input("Choose gamma: ")
print("The system will generate ",n_states , " states MDP, with reward chosen between -10. and 10. uniformly, and the transition distribution choosen uniformly belween 0. and 1. between all the states")
R = np.random.uniform(-10.,10.,(n_states,1))
P = np.random.uniform(0.,1.,(n_states, n_states))
marg = np.sum(P,axis=1)**(-1)
P = np.matmul(np.diag(marg), P)

V_opt = np.matmul(np.linalg.inv(np.eye(n_states)- gamma * P),R)
#print R + gamma * np.matmul(P,V_opt) - V_opt

n_iteration = input("How many iteration would you like to run for evaluating V? ")

"""---------------- Sampled MDP --------------------------------"""
N_Future_States = 100
RSample = lambda : np.random.normal(R, 0.02)
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
approx  = lambda f: f + np.random.normal(np.zeros_like(f), .001 * np.abs(f))

"""------------------------ Algorithms -------------------------"""

""" Emulates BPE and RPE together with just function approximator. 
The output is the mean-squared Bellman Error"""
def MixedApproximateIterationBoosting(alpha):
    errors = []
    V = np.zeros((n_states, 1))
    B = approx(T(V) - V)
    for _ in range(n_iteration):
        V = V + B
        B = approx((1. - alpha) * (T(V) - V) + alpha * S(B))
        errors.append(np.mean(np.square(V_opt - V)))
    V = V + B
    errors.append(np.mean(np.square(V_opt - V)))
    return np.mean(np.square(T(V) - V)), np.mean(np.square(V_opt - V)), np.array(errors)

""" Emulates BPE and RPE together with just function approximator and samples. 
The output is the mean-squared Bellman Error"""
def MixedApproximateIterationBoostingSamples(alpha):
    errors = []
    V = np.zeros((n_states, 1))
    B = approx(TSample(V) - V)
    for _ in range(n_iteration):
        V = V + B
        B = approx((1. - alpha) * (TSample(V) - V) + alpha * SSample(B))
        errors.append(np.mean(np.square(V_opt - V)))
    V = V + B
    errors.append(np.mean(np.square(V_opt - V)))
    return np.mean(np.square(T(V) - V)), np.mean(np.square(V_opt - V)),np.array(errors)



print("\n We will compute the plots comparing the Bellman error at the last iteration, both taking in account of samplings and not.\n In x-axis there will be the mixing parameter alpha. On the y-axis the MS Bellman Error.\n Every single instance of the algorithm is run for 100 times.")
n_alphas = 40

error_perfect_matrix = np.zeros((0,n_alphas))
error_samples_matrix = np.zeros((0,n_alphas))
residual_perfect_matrix = np.zeros((0,n_alphas))
residual_samples_matrix = np.zeros((0,n_alphas))
error_tensor_perfect = np.zeros((n_alphas,n_iteration+1, 0))
error_tensor_samples = np.zeros((n_alphas,n_iteration+1, 0))
alphas = np.log(np.linspace(1.,np.exp(1),n_alphas))
for _ in range(500):
    residuals_perfect = []
    residuals_samples = []
    errors_perfect = []
    errors_samples = []
    error_t_perfect = []
    error_t_samples = []
    print '.',
    for alpha in alphas:
        residual, error, errors = MixedApproximateIterationBoosting(alpha)
        errors_perfect.append(error)
        residuals_perfect.append(residual)
        error_t_perfect.append(errors)
        residual, error, errors = MixedApproximateIterationBoostingSamples(alpha)
        errors_samples.append(error)
        residuals_samples.append(residual)
        error_t_samples.append(errors)

    error_perfect_matrix = np.concatenate((error_perfect_matrix, [errors_perfect]), axis=0)
    error_samples_matrix = np.concatenate((error_samples_matrix, [errors_samples]), axis=0)

    residual_perfect_matrix = np.concatenate((residual_perfect_matrix, [residuals_perfect]), axis=0)
    residual_samples_matrix = np.concatenate((residual_samples_matrix, [residuals_samples]), axis=0)

    error_tensor_perfect = np.concatenate((error_tensor_perfect, np.array(error_t_perfect).reshape((n_alphas,n_iteration+1, 1))),axis=2)
    error_tensor_samples = np.concatenate((error_tensor_samples, np.array(error_t_samples).reshape((n_alphas,n_iteration+1, 1))),axis=2)


plt.title("Mixed BPE and RPE with perfect model")
m = np.mean(error_perfect_matrix, axis=0)
c = 2. * np.std(error_perfect_matrix / 10., axis=0)
plt.plot(alphas, m, color='b', label='V_opt-V')
plt.fill_between(alphas,m-c, m+c, color='b', alpha=0.25 )
# m = np.mean(residual_perfect_matrix, axis=0)
# c = 2. * np.std(residual_perfect_matrix / 10., axis=0)
# plt.plot(alphas, m, color='g', label='TV-V')
# plt.fill_between(alphas,m-c, m+c, color='g', alpha=0.25 )
plt.legend(loc='best')
plt.ylabel('MSBE')
plt.xlabel('alpha')
plt.show()

plt.title("Mixed BPE and RPE with sampled reward and future state")
m = np.mean(error_samples_matrix, axis=0)
c = 2. * np.std(error_samples_matrix / 10., axis=0)
plt.plot(alphas, m, color='b', label='V_opt-V')
plt.fill_between(alphas,m-c, m+c, color='b', alpha=0.25 )
# m = np.mean(residual_samples_matrix, axis=0)
# c = 2. * np.std(residual_samples_matrix / 10., axis=0)
# plt.plot(alphas, m, color='g', label='TV-V')
# plt.fill_between(alphas,m-c, m+c, color='g', alpha=0.25 )
plt.legend(loc='best')
plt.ylabel('MSBE')
plt.xlabel('alpha')
plt.show()


y_ticks = range(0, 40, 5) + [39]
x_ticks = range(0,n_iteration+1,10)

fig, ax = plt.subplots()
z = ax.matshow(np.log(np.mean(error_tensor_samples,axis=2)))
ax.set_yticklabels(np.round(alphas[np.array(y_ticks)],3))
ax.set_yticks(y_ticks)
ax.set_xticklabels(x_ticks)
ax.set_xticks(x_ticks)
ax.set_aspect('auto')
plt.ylabel(r'$\alpha$',rotation='horizontal',fontsize=10)
plt.xlabel('$N$')
plt.colorbar(z)
plt.show()

# plt.matshow(np.log(np.mean(error_tensor_samples,axis=2)))
# plt.gca().set_xticklabels(range(n_iteration))
# plt.gca().set_yticklabels(alphas)
# plt.show()