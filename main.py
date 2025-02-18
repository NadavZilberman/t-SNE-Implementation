from data_loader import DataLoader

def main():
    # Load data set
    dataset_name = "mnist"
    num_of_data_points = 6000
    dataloader = DataLoader(dataset_name, num_of_data_points)
    dataloader.load_dataset()
    dataloader.images_to_vectors()

    # Initiate t-SNE algorithm
    
if __name__ == "__main__":
    main()