import time
from matplotlib import pyplot as plt
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
                 exaggeration_interval: int,
                 tol: float,
                 const_cost_max_iters: int,
                 adaptive_learning_coeff: float,
                 start_iter_of_adaptive_learning: int,
                 early_comp_end_iter,
                 early_comp_coeff
                 ):

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
        self.tol = tol
        self.const_cost_max_iters = const_cost_max_iters
        self.adaptive_lr_coeff = adaptive_learning_coeff
        self.start_iter_of_adaptive_learning = start_iter_of_adaptive_learning
        self.early_comp_end_iter = early_comp_end_iter
        self.early_comp_coeff = early_comp_coeff

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

    def calc_gradient(self, joint_prob_dist):
        gradient = np.zeros((self.y.shape))
        for i in range(self.y.shape[0]):
            term1 = joint_prob_dist[i,:] - self.low_dim_affinities[i,:]
            term2 = self.y[i] - self.y
            term3 = 1/(1 + np.linalg.norm(term2, axis=1))
            gradient[i] = 4 * term1[:, np.newaxis].T @ (term2 * term3[:, np.newaxis])
            pass
        return gradient
    
    def early_comp_gradient(self):
        return 2 * self.early_comp_coeff * self.y

    def compute_low_dim_affinities(self):
        q = np.zeros((self.y.shape[0], self.y.shape[0]))
        for i in range(0, self.y.shape[0]):
            diff = self.y[i] - self.y
            q[i, :] = 1/(1 + np.linalg.norm(diff, axis=1)**2)
        np.fill_diagonal(q, 0)
        q /= q.sum()
        q = np.maximum(q, 1e-14)
        return q
    
    def sample_initial_solution(self, data_size: int):
        mean = np.array([0, 0])
        cov = self.var_initial_guess * np.identity(2)
        initial_guess = multivariate_normal.rvs(mean, cov, size=(data_size))
        return initial_guess
    
    def solve_optimization(self, joint_probability_dist, initial_solution, labels, plot_save_interval):
        self.y = initial_solution
        momentum = self.momentum1
        previous_diff = 0
        previous_cost = 0
        no_change_in_cost_counter = 0
        for iter in range(self.T):
            print(f"Optimization: step {iter + 1}")
            # determine exaggeration coeff
            if iter < self.exaggeration_interval:
                exag_coef = self.exaggeration_coef
            else:
                exag_coef = 1
            # determine momentum
            if iter == self.switch_momentum_iter:
                momentum = self.momentum2

            # determine (adaptive) learning rate
            if no_change_in_cost_counter == self.const_cost_max_iters:
                if iter > self.start_iter_of_adaptive_learning:
                    print(f"Got constant cost, dividing learning rate by {self.adaptive_lr_coeff}.")
                    self.lr /= self.adaptive_lr_coeff
                no_change_in_cost_counter = 0

            self.low_dim_affinities = self.compute_low_dim_affinities()

            gradient = self.calc_gradient(exag_coef * joint_probability_dist)
            if iter < self.early_comp_end_iter:
                gradient += self.early_comp_gradient()
            previous_y = self.y
            self.y = self.y - self.lr * gradient + momentum * previous_diff
            previous_diff = self.y - previous_y
            cost = np.sum(joint_probability_dist * np.log(joint_probability_dist/(self.low_dim_affinities)))
            # add early compression cost
            if iter < self.early_comp_end_iter:
                cost += self.early_comp_coeff * np.sum(np.power(self.y, 2))
            print(f"Cost: {cost}")
            if np.abs(cost - previous_cost) < self.tol:
                no_change_in_cost_counter += 1
            else:
                no_change_in_cost_counter = 0

            previous_cost = cost
            # save scatter plot of   y
            if iter % plot_save_interval == 0 or iter == self.T - 1:
                plt.figure()
                scatter = plt.scatter(self.y[:, 0], self.y[:, 1], c=labels, cmap='tab10', alpha=0.7)
                plt.colorbar(scatter, label="Label")
                filename = f"out_gif/save_{iter}.png"
                plt.savefig(filename)
                plt.close()

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

    def run_algorithm(self, data, labels, plot_save_interval):
        sigmas = self.find_sigmas_from_perp_binary_search(data)
        P = self.compute_pairwise_affinities(data, sigmas)
        joint_probability_dist = (P + P.T) / (2 * data.shape[0])
        joint_probability_dist = np.maximum(joint_probability_dist, 1e-14)
        initial_solution = self.sample_initial_solution(data_size=data.shape[0])
        self.solve_optimization(joint_probability_dist, initial_solution, labels, plot_save_interval)
