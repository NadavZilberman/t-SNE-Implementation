import imageio
from matplotlib import pyplot as plt
from data_loader import DataLoader
from t_SNE import t_SNE
def run_on_mnist():
    # Load data set, normalize, reduce dimensionality
    dataset_name = "mnist"
    num_of_data_points = 2000
    dataloader = DataLoader(dataset_name, num_of_data_points)
    dataloader.load_dataset()
    dataloader.images_to_vectors()
    dataloader.normalize_data()
    dataloader.pca_data(degree=30)
    plot_save_interval = 25  # every 25 iterations
    # Initiate t-SNE algorithm
    t_sne = t_SNE(perplexity=20, 
                  num_of_iter=350, 
                  initial_learning_rate=300, 
                  momentum1=0.5,
                  momentum2=0.8,
                  switch_momentum_iter=400,
                  exaggeration_coef=4, 
                  exaggeration_interval=100,
                  tol=1e-4,
                  const_cost_max_iters=5,
                  adaptive_learning_coeff=2,
                  start_iter_of_adaptive_learning=200,
                  early_comp_end_iter=10,
                  early_comp_coeff=1*1e-4)

    t_sne.run_algorithm(dataloader.data, dataloader.labels, plot_save_interval)
    results = t_sne.y

    
    plt.figure()
    # plt.scatter(results[:, 0], results[:, 1])
    scatter = plt.scatter(results[:, 0], results[:, 1], c=dataloader.labels, cmap='tab10', alpha=0.7)

    # Add colorbar
    plt.colorbar(scatter, label="Label")
    plt.show()
    
def run_on_hands_letters():
    dataset_name = "hands"
    num_of_data_points = 24*240
    dataloader = DataLoader(dataset_name, num_of_data_points)
    dataloader.load_dataset()
    dataloader.normalize_data()
    dataloader.pca_data(10)

    plot_save_interval = 25  # every 25 iterations
    # Initiate t-SNE algorithm
    t_sne = t_SNE(perplexity=100, 
                  num_of_iter=2000, 
                  initial_learning_rate=100, 
                  momentum1=0.5,
                  momentum2=0.9,
                  switch_momentum_iter=250,
                  exaggeration_coef=1.2, 
                  exaggeration_interval=100,
                  tol=1e-5,
                  const_cost_max_iters=5,
                  adaptive_learning_coeff=2,
                  start_iter_of_adaptive_learning=200,
                  early_comp_end_iter=50,
                  early_comp_coeff=1e-5)  # 1e-5

    t_sne.run_algorithm(dataloader.data, dataloader.numeric_labels, plot_save_interval)
    results = t_sne.y

    
    plt.figure()
    # plt.scatter(results[:, 0], results[:, 1])
    scatter = plt.scatter(results[:, 0], results[:, 1], c=dataloader.numeric_labels, cmap='tab10', alpha=0.7)

    # Add colorbar
    plt.colorbar(scatter, label="Label")
    plt.show()


def create_gif(total_iterations: int, save_interval: int, duration: float):
    gif_filename = "optimization_process.gif"
    images = []
    for i in range(save_interval, total_iterations, save_interval):
        images.append(imageio.imread(f"out_gif/save_{i}.png"))
    images.append(imageio.imread(f"out_gif/save_{total_iterations - 1}.png"))
    imageio.mimsave(gif_filename, images, duration=duration)

if __name__ == "__main__":
    run_on_mnist()
    # run_on_hands_letters()
    create_gif(total_iterations=1000, save_interval=25, duration=15)