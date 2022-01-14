import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from model import NeuralNerwork
from engine import Engine
import config
import utils
import prune


def train():
    # Pre-loaded MNIST train-test datasets, stores the samples and corresponding label
    train_dataset = datasets.MNIST(
        root="../inputs/", train=True, transform=transforms.ToTensor(), download=True
    )
    test_dataset = datasets.MNIST(
        root="../inputs/", train=False, transform=transforms.ToTensor(), download=True
    )
    # Initialize DataLoader, each iteration returns a batch of features and labels
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=config.TRAIN_BATCH_SIZE
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=config.TEST_BATCH_SIZE
    )

    # initialize model with 784 features pixels and 10 classes
    # set model as global variable
    global model
    model = NeuralNerwork(784, 10)

    # Define optimization function - Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # initialize Engine class with model, optimizer
    engine = Engine(model, optimizer)

    # for loop each epoch and print Train/Test-Set Accuracy Score
    for epochs in range(config.EPOCHS):
        # initialize model train and eval on dataloader
        train_acc = engine.train_fn(train_loader)
        eval_acc = engine.eval_fn(test_loader)
        print(
            f"Epoch:{epochs+1}/{config.EPOCHS}, Training Set Accuracy: {train_acc*100:0.2f}%, Testing Set Accuracy: {eval_acc*100:0.2f}%"
        )

    # define k% sparsity in a list
    sparsities = [0.0, 0.25, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.97, 0.99]

    # intialize weight and unit pruning from prune module
    df_weight = prune.get_pruning_accuracies(
        model, "weight", sparsities, optimizer, test_loader
    )
    df_unit = prune.get_pruning_accuracies(
        model, "unit", sparsities, optimizer, test_loader
    )

    # plot and save % of weight and unit pruning
    utils.plot_sparsity(df_unit, df_weight)


if __name__ == "__main__":
    train()
