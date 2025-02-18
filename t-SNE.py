import time
import numpy as np
import numpy.typing as npt
from scipy.stats import entropy
from scipy.spatial.distance import cdist, squareform
from data_loader import DataLoader

class t_SNE:
    def __init__(self, perplexity: float, 
                 num_of_iter: int, 
                 initial_learning_rate: float, 
                 momentum: list[float], 
                 exaggeration_coef: float, 
                 exaggeration_interval: int):
        self.perp = perplexity
        self.T = num_of_iter
        self.lr = initial_learning_rate
        self.momentum = momentum
        self.perp_tol = 1e-3

    def compute_pairwise_affinities(self, data: npt.NDArray, sigmas: float | list[float], i: int | None=None):
        if i is None:
            dist_sq = cdist(data, data, metric='sqeuclidean')
            P = np.exp(-dist_sq / (2*sigmas**2))
            np.fill_diagonal(P, 0)
            P /= np.sum(P, axis=1, keepdims=True)
        else:
            diff = data[i] - data
            dist_sq = np.linalg.norm(diff, axis=1) ** 2
            P = np.exp(-dist_sq / (2*sigmas**2))
            P /= np.sum(P, keepdims=True)
        return P
    
    def solve_optimization(self):
        pass

    def optimization_step(self):
        pass

    def sample_initial_solution(self):
        pass


    # def find_sigmas_from_perp_grid_search(self, data: npt.NDArray):
    #     sigmas = np.zeros((data.shape[0]))

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
            if str(i*100/data.shape[0])[0] != str((i-1)*100/data.shape[0])[0]:
                print(f"Found {i*100/data.shape[0]}% of sigmas...")
        print(f"Finished calculating sigmas. Took {time.time() - start} seconds")
        return sigmas

if __name__ == "__main__":
    dataset_name = "mnist"
    num_of_data_points = 3000
    dataloader = DataLoader(dataset_name, num_of_data_points)
    dataloader.load_dataset()
    dataloader.images_to_vectors()
    dataloader.normalize_data()
    dataloader.pca_data()
    t_sne = t_SNE(perplexity=40, num_of_iter=100, initial_learning_rate=100, momentum=0.5, exaggeration_coef=4, exaggeration_interval=100)

    sigmas = t_sne.find_sigmas_from_perp_binary_search(dataloader.data)
    P = t_sne.compute_pairwise_affinities(dataloader.data, sigmas)
    pass