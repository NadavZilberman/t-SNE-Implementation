from keras import datasets as keras_datasets
import numpy as np
from sklearn import decomposition
import kagglehub
import os
from torchvision import datasets as torch_datasets, transforms
from torch.utils.data import DataLoader as Loader

class DataLoader:
    def __init__(self, dataset_name: str = "mnist", num_of_points_to_load: int = 6000):
        self.dataset_name = dataset_name
        self.num_of_points_to_load = num_of_points_to_load

    
    def load_dataset(self):
        if self.dataset_name == "mnist":
            (_, _), (self.data , self.labels) = keras_datasets.mnist.load_data()
        try:
            self.data = self.data[:self.num_of_points_to_load, :, :]
            self.labels = self.labels[:self.num_of_points_to_load]
        except:
            pass

        if self.dataset_name == "hands":
            # path = kagglehub.dataset_download("dilipkumar2k6/sports-ball")            
            path = kagglehub.dataset_download("ruslanbredun/sign-language-eng-alphabet")
            self.dataset_path = path + "\\Images"
            all_items = os.listdir(self.dataset_path)
            dirs = [d for d in all_items if os.path.isdir(os.path.join(self.dataset_path, d))]
            labels = dirs
            self.data = []
            self.labels = []
            resize_to = 100
            transform = transforms.Compose([
                transforms.Resize((resize_to, resize_to)), 
                transforms.ToTensor(),
                transforms.Grayscale()
            ])
            dataset = torch_datasets.ImageFolder(root=self.dataset_path, transform=transform)
            data_loader = Loader(dataset, shuffle=True)
            for i, (input, label) in enumerate(data_loader):
                if i == self.num_of_points_to_load:
                    break
                image_data = np.reshape(input[0, :, :], (resize_to*resize_to))
                self.data.append(image_data)
                self.labels.append(labels[label])
            self.data = np.array(self.data)
            self.labels = np.array(self.labels)
            self.labels_map, self.numeric_labels = np.unique(self.labels, return_inverse=True)
        print(f"Loaded {self.labels.shape[0]} data points from \"{self.dataset_name}\" dataset.")
    
    def images_to_vectors(self) -> None:
        try:
            self.data = self.data.reshape(self.data.shape[0], -1)
        except:
            print("[Error] Wasn't able to reshape data, perhaps it's not images?")

    def normalize_data(self):
        self.data = self.data.astype(np.float32) / np.max(self.data, axis=1, keepdims=True)

    def pca_data(self, degree):
        # initializing the pca
        pca = decomposition.PCA(n_components=degree)
        # PCA for dimensionality redcution (non-visualization)
        self.data = pca.fit_transform(self.data)
        print(f"Performed PCA on data, new dimension is {degree}")