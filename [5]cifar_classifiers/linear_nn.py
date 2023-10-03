import torch
from torch import nn
import numpy as np

from typing import Tuple, Union, List, Callable
from torch.optim import SGD
import torchvision
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm


def linear_model() -> nn.Module:
    """Instantiate a linear model and send it to device."""
    model =  nn.Sequential(nn.Flatten(),nn.Linear(d, 10))
    return model.to(DEVICE)

def train(
    model: nn.Module, optimizer: SGD,
    train_loader: DataLoader, val_loader: DataLoader,
    epochs: int = 20
)-> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Trains a model for the specified number of epochs using the loaders.

    Returns: 
    Lists of training loss, training accuracy, validation loss, validation accuracy for each epoch.
    """

    loss = nn.CrossEntropyLoss()
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for e in tqdm(range(epochs)):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        # Main training loop; iterate over train_loader. The loop
        # terminates when the train loader finishes iterating, which is one epoch.
        for (x_batch, labels) in train_loader:
            x_batch, labels = x_batch.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            labels_pred = model(x_batch)
            batch_loss = loss(labels_pred, labels)
            train_loss = train_loss + batch_loss.item()

            labels_pred_max = torch.argmax(labels_pred, 1)
            batch_acc = torch.sum(labels_pred_max == labels)
            train_acc = train_acc + batch_acc.item()

            batch_loss.backward()
            optimizer.step()
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_acc / (B_SIZE * len(train_loader)))

        # Validation loop; use .no_grad() context manager to save memory.
        model.eval()
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            for (v_batch, labels) in val_loader:
                v_batch, labels = v_batch.to(DEVICE), labels.to(DEVICE)
                labels_pred = model(v_batch)
                v_batch_loss = loss(labels_pred, labels)
                val_loss = val_loss + v_batch_loss.item()

                v_pred_max = torch.argmax(labels_pred, 1)
                batch_acc = torch.sum(v_pred_max == labels)
                val_acc = val_acc + batch_acc.item()
            val_losses.append(val_loss / len(val_loader))
            val_accuracies.append(val_acc / (B_SIZE * len(val_loader)))

    return train_losses, train_accuracies, val_losses, val_accuracies

def parameter_search(
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    model_fn:Callable[[], nn.Module],
    lrs: List[float] = torch.linspace(1e-6, 1e-1, 5),
    epochs: int = 20
) -> float:
    """
    Parameter search for our linear model using SGD.

    Args:
    train_loader: the train dataloader.
    val_loader: the validation dataloader.
    model_fn: a function that, when called, returns a torch.nn.Module.
    lrs: a list of learning rates to try.
    num_iter: the number of iterations to train for.

    Returns:
    The learning rate with the least validation loss.
    NOTE: you may need to modify this function to search over and return
     other parameters beyond learning rate.
    """
    best_loss = torch.tensor(np.inf)
    best_lr = 0.0

    if lrs is None:
        lrs = torch.linspace(10 ** (-6), 10 ** (-1), num_iter)

    for lr in lrs:
        print(f"trying learning rate {lr}")
        model = model_fn()
        optim = SGD(model.parameters(), lr)
        
        train_loss, train_acc, val_loss, val_acc = train(
            model, optim, train_loader, val_loader, epochs
        )

        if min(val_loss) < best_loss:
            best_loss = min(val_loss)
            best_lr = lr
        print(f"Min loss: {min(val_loss)}, Min loss epoch: {np.argmin(val_loss)+1}/{epochs}")
    return best_lr

def evaluate(
    model: nn.Module, loader: DataLoader
) -> Tuple[float, float]:
    """Computes test loss and accuracy of model on loader."""
    loss = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for (batch, labels) in loader:
            batch, labels = batch.to(DEVICE), labels.to(DEVICE)
            y_batch_pred = model(batch)
            batch_loss = loss(y_batch_pred, labels)
            test_loss = test_loss + batch_loss.item()

            pred_max = torch.argmax(y_batch_pred, 1)
            batch_acc = torch.sum(pred_max == labels)
            test_acc = test_acc + batch_acc.item()
        test_loss = test_loss / len(loader)
        test_acc = test_acc / (B_SIZE * len(loader))
        return test_loss, test_acc

def eval_linear_model(best_lr: float = 0.015):
    model = linear_model()
    optimizer = SGD(model.parameters(), best_lr)

    train_loss, train_accuracy, val_loss, val_accuracy = train(
        model, optimizer, train_loader, val_loader, 20
    )

    epochs = range(1, 21)
    plt.plot(epochs, train_accuracy, label="Train Accuracy")
    plt.plot(epochs, val_accuracy, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Logistic Regression Accuracy for CIFAR-10 vs Epoch")
    plt.show()

    test_loss, test_acc = evaluate(model, test_loader)
    print(f"Test Accuracy: {test_acc}")

# assert torch.cuda.is_available(), "GPU is not available, check the directions above (or disable this assertion to use CPU)"
B_SIZE = 128
LRS = [0.001, 0.005, 0.01, 0.025, 0.05]
EPOCHS = 20
TRAINING_SUBSET = 0.25

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)  # this should print out CUDA

train_dataset = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=torchvision.transforms.ToTensor())
orig_len = (len(train_dataset))
train_dataset = torch.utils.data.dataset.Subset(train_dataset, indices=range(int(TRAINING_SUBSET * len(train_dataset))))
partial_len = (len(train_dataset))
train_dataset, val_dataset = random_split(train_dataset, [int(0.9 * len(train_dataset)), int(0.1 * len(train_dataset))])
train_len = (len(train_dataset))
print(f"orig_len={orig_len}, partial_len={partial_len}, train_len={train_len}")

# Create separate dataloaders for the train, test, and validation set
train_loader = DataLoader(train_dataset, batch_size=B_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=B_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=B_SIZE, shuffle=True)

imgs, labels = next(iter(train_loader))
print(f"A single batch of images has shape: {imgs.size()}")
example_image, example_label = imgs[0], labels[0]
c, w, h = example_image.size()
d = c * w * h

best_lr = 0.015
best_lr = parameter_search(train_loader, val_loader, linear_model, LRS, 20)
eval_linear_model(best_lr)

# model = model.to(DEVICE)  # Sending a model to GPU

# for x, y in tqdm(data_loader):
#   x, y = x.to(DEVICE), y.to(DEVICE)  # Sending data to available device