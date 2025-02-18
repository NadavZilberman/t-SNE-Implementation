from keras import datasets
import numpy as np
from sklearn import decomposition

class DataLoader:
    def __init__(self, dataset_name: str = "mnist", num_of_points_to_load: int = 6000):
        self.dataset_name = dataset_name
        self.num_of_points_to_load = num_of_points_to_load

    
    def load_dataset(self):
        if self.dataset_name == "mnist":
            (_, _), (self.data , self.labels) = datasets.mnist.load_data()
        try:
            self.data = self.data[:self.num_of_points_to_load, :, :]
            self.labels = self.labels[:self.num_of_points_to_load]
        except:
            pass
        print(f"Loaded {self.labels.shape[0]} data points from \"{self.dataset_name}\" dataset.")
    
    def images_to_vectors(self) -> None:
        try:
            self.data = self.data.reshape(self.data.shape[0], -1)
        except:
            print("[Error] Wasn't able to reshape data, perhaps it's not images?")

    def normalize_data(self):
        self.data = self.data.astype(np.float32) / np.max(self.data, axis=1, keepdims=True)

    def pca_data(self):
        # initializing the pca
        pca = decomposition.PCA(n_components=30)
        # PCA for dimensionality redcution (non-visualization)
        self.data = pca.fit_transform(self.data)