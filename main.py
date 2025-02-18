from matplotlib import pyplot as plt
from data_loader import DataLoader
from t_SNE import t_SNE
def main():
    # Load data set, normalize, reduce dimensionality
    dataset_name = "mnist"
    num_of_data_points = 1000
    dataloader = DataLoader(dataset_name, num_of_data_points)
    dataloader.load_dataset()
    dataloader.images_to_vectors()
    dataloader.normalize_data()
    dataloader.pca_data()

    # Initiate t-SNE algorithm
    t_sne = t_SNE(perplexity=40, 
                  num_of_iter=1000, 
                  initial_learning_rate=100, 
                  momentum1=0.5,
                  momentum2=0.8,
                  switch_momentum_iter=250,
                  exaggeration_coef=4, 
                  exaggeration_interval=100)

    t_sne.run_algorithm(dataloader.data)
    results = t_sne.y

    plt.figure()
    # plt.scatter(results[:, 0], results[:, 1])
    scatter = plt.scatter(results[:, 0], results[:, 1], c=dataloader.labels, cmap='tab10', alpha=0.7)

    # Add colorbar
    plt.colorbar(scatter, label="Label")
    plt.show()
    
if __name__ == "__main__":
    main()