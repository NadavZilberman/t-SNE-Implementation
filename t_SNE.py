import time
import numpy as np
import numpy.typing as npt
from scipy.stats import entropy
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal

class t_SNE:
    def __init__(self, perplexity: float, 
                 num_of_iter: int, 
                 initial_learning_rate: float, 
                 momentum1: list[float], 
                 momentum2: list[float], 
                 switch_momentum_iter: int,
                 exaggeration_coef: float, 
                 exaggeration_interval: int):
        self.perp = perplexity
        self.T = num_of_iter
        self.lr = initial_learning_rate
        self.momentum1 = momentum1
        self.momentum2 = momentum2
        self.switch_momentum_iter = switch_momentum_iter
        self.exaggeration_coef = exaggeration_coef
        self.exaggeration_interval = exaggeration_interval
        self.perp_tol = 1e-3
        self.var_initial_guess = 1e-4

    def compute_pairwise_affinities(self, data: npt.NDArray, sigmas: float | list[float], i: int | None=None):
        if i is None:  # compute entire matrix
            dist_sq = cdist(data, data, metric='sqeuclidean')
            P = np.exp(-dist_sq / (2*sigmas**2))
            np.fill_diagonal(P, 0)
            P /= np.sum(P, axis=1, keepdims=True)
        else:  # compute for certain i
            diff = data[i] - data
            dist_sq = np.linalg.norm(diff, axis=1) ** 2
            P = np.exp(-dist_sq / (2*sigmas**2))
            P /= np.sum(P, keepdims=True)
        return P

    def calc_gradient(self):
        gradient = np.zeros((self.y.shape))
        for i in range(self.y.shape[0]):
            term1 = self.joint_prob_dist[i,:] - self.low_dim_affinities[i,:]
            term2 = self.y[i] - self.y
            term3 = 1/(1 + np.linalg.norm(term2, axis=1))
            gradient[i] = 4 * np.sum((term1 @ term3) * term2, axis=0)
        return gradient
    
    def compute_low_dim_affinities(self, pairwise_dist):
        q = 1/(1 + pairwise_dist)
        np.fill_diagonal(q, 0)
        q = q / (np.sum(q, axis=1, keepdims=True))
        q = np.maximum(q, 1e-9)
        return q
    
    def sample_initial_solution(self, data_size: int):
        mean = np.array([0, 0])
        cov = self.var_initial_guess * np.identity(2)
        initial_guess = multivariate_normal.rvs(mean, cov, size=(data_size))
        return initial_guess
    
    def solve_optimization(self, joint_probability_dist, initial_solution):
        # Optimization process:
        # Early exaggeration: coeff 4, for first 50 iterations
        # 1000 iterations
        # Momentum: 0.5 for 250 first iterations, 0.8 for 250 untill 1000.
        # Learning rate: initiated by 100, changing 
        self.joint_prob_dist = joint_probability_dist * self.exaggeration_coef
        self.y = initial_solution
        momentum = self.momentum1
        previous_diff = 0
        for iter in range(self.T):
            print(f"Optimization: step {iter + 1}")
            if iter == self.exaggeration_interval:
                self.joint_prob_dist /= self.exaggeration_coef
            if iter == self.switch_momentum_iter:
                momentum = self.momentum2

            pairwise_dist = cdist(self.y, self.y, metric="sqeuclidean")
            self.low_dim_affinities = self.compute_low_dim_affinities(pairwise_dist)

            gradient = self.calc_gradient()
            previous_y = self.y
            self.y = self.y - self.lr * gradient + momentum * previous_diff
            previous_diff = self.y - previous_y
            loss = np.sum(self.joint_prob_dist*np.log(self.joint_prob_dist/(self.low_dim_affinities)))
            print(f"Loss: {loss}")
            # FIX: optimization doesn't work.
            
    def find_sigmas_from_perp_binary_search(self, data: npt.NDArray, max_sigma = 100, min_sigma = 0.05):
        """Find the vector of sigmas from the data and perplexity"""
        start = time.time()
        sigmas = np.zeros((data.shape[0]))
        for i in range(data.shape[0]):
            max_sigma = np.inf
            min_sigma = -np.inf
            curr_sigma = np.std(np.linalg.norm(data[i] - data, axis=1))  # initial sigma
            Pi = self.compute_pairwise_affinities(data, curr_sigma, i=i)
            curr_perp = 2 ** entropy(Pi, base=2)
            while abs(curr_perp - self.perp) > self.perp_tol:
                if curr_perp > self.perp + self.perp_tol:
                    max_sigma = curr_sigma
                    if min_sigma == -np.inf:
                        curr_sigma = curr_sigma / 2
                    else:
                        curr_sigma = (curr_sigma + min_sigma) / 2
                elif curr_perp < self.perp - self.perp_tol:
                    min_sigma = curr_sigma
                    if max_sigma == np.inf:
                        curr_sigma = curr_sigma * 2
                    else:
                        curr_sigma = (curr_sigma + max_sigma) / 2
                Pi = self.compute_pairwise_affinities(data, curr_sigma, i=i)
                curr_perp = 2 ** entropy(Pi, base=2)
            sigmas[i] = curr_sigma
            if str(i * 100/data.shape[0])[0] != str((i - 1)*100/data.shape[0])[0]:
                print(f"Found {i*100/data.shape[0]}% of sigmas...")
        print(f"Finished calculating sigmas. Took {time.time() - start} seconds")
        return sigmas

    def run_algorithm(self, data):
        sigmas = self.find_sigmas_from_perp_binary_search(data)
        P = self.compute_pairwise_affinities(data, sigmas)
        joint_probability_dist = (P + P.T) / (2 * data.shape[0])
        joint_probability_dist = np.maximum(joint_probability_dist, 1e-9)
        initial_solution = self.sample_initial_solution(data.shape[0])
        self.solve_optimization(joint_probability_dist, initial_solution)

if __name__ == "__main__":
    t_sne = t_SNE(40, 10, 2, 2, 2, 2)
    t_sne.sample_initial_solution(1000)