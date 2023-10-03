import torch
from torch import nn
import numpy as np

from typing import Tuple, Union, List, Callable
from torch.optim import SGD
from torch.optim import Adam
import torchvision
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

def part_a_model(M: int = 10) -> nn.Module:
    model = nn.Sequential(nn.Flatten(), nn.Linear(d, M), nn.ReLU(), nn.Linear(M, 10))
    return model.to(DEVICE)

def train(
    model: nn.Module, optimizer: SGD,
    train_loader: DataLoader, val_loader: DataLoader,
    epochs: int = 20,
    tqdm_desc: str = "" 
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
    for e in tqdm(range(epochs), ncols=80, desc=tqdm_desc):
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

def part_a_search(
) -> float:
    best_loss = torch.tensor(np.inf)
    best_lr = 0.0
    best_params = None
    top_three = {}

    for lr in SEARCH_LRS:
        for M in SEARCH_MS:
            print(f"trying lr={lr}, M={M}")
            model = part_a_model(M)
            optim = SGD(model.parameters(), lr, momentum=MMT)
            train_loss, t_acc, val_loss, v_acc = train(
                model, optim, t_search_loader, v_search_loader, SEARCH_EPOCHS, 
                "    "
            )
            if min(val_loss) < best_loss:
                best_loss = min(val_loss)
                best_lr = lr
                best_params = M
            if len(top_three) < 3 or min(top_three) < max(v_acc):
                if len(top_three) == 3:
                    top_three.pop(min(top_three))
                top_three[max(v_acc)] = {"t_acc": t_acc, "v_acc": v_acc, "lr": lr, "M": M}
            print(f"    max(val_acc): {max(v_acc):.3f} in epoch {np.argmax(v_acc)+1}/{SEARCH_EPOCHS}")
        print()
    
    for i, (acc, info) in enumerate(top_three.items()):
        lr, M = info["lr"], info["M"]
        t_acc, v_acc = info["t_acc"], info["v_acc"]
        plt.plot(range(1, len(t_acc)+1), t_acc, label=f"{mname(lr, M)} train", linestyle="-", color=C[i])
        plt.plot(range(1, len(v_acc)+1), v_acc, label=f"{mname(lr, M)} val", linestyle="--", color=C[i])
    plt.title("Top 3 models")
    plt.plot(range(1, len(t_acc)+1), [0.5]*len(t_acc), label="Threshold", linestyle=":", color="k")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    return top_three

def eval_part_a_models(top_three):
    for i, (acc, info) in enumerate(top_three.items()):
        lr, M = info["lr"], info["M"]
        model = part_a_model(M)
        print(f"Evaluating lr={lr}, M={M}")
        optim = SGD(model.parameters(), lr, momentum=MMT)
        train_loss, t_acc, val_loss, v_acc = train(model, optim, 
            # t_eval_loader, v_eval_loader, EVAL_EPOCHS,
            t_search_loader, v_search_loader, SEARCH_EPOCHS,
            "Evaluation"
        )
        test_loss, test_acc = evaluate(model, test_loader)
        print(f"Model {mname(lr, M)} has test accuracy {test_acc:.3f} and max val accuracy {max(v_acc):.3f}")
        plt.title("Top 3 models")
        print(C[i])
        plt.plot(range(1, len(t_acc)+1), t_acc, label=f"{mname(lr, M)} train", linestyle="-", color=C[i])
        plt.plot(range(1, len(v_acc)+1), v_acc, label=f"{mname(lr, M)} val", linestyle="--", color=C[i])
    plt.plot(range(1, len(t_acc)+1), [0.5]*len(t_acc), label="Threshold", linestyle=":", color="k")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

def mname(lr, M):
    return f"(lr={lr}, M={M})"

# assert torch.cuda.is_available(), "GPU is not available, check the directions above (or disable this assertion to use CPU)"
C = ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'brown', 'pink', 'cyan']
B_SIZE = 256
SEARCH_PCT = 0.01
SEARCH_EPOCHS=10
MMT = 0.9
SEARCH_LRS=[0.01, 0.025, 0.050]
SEARCH_MS=[125, 300, 500]
EVAL_EPOCHS=50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)  # this should print out CUDA

train_ds = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=torchvision.transforms.ToTensor())
search_ds = torch.utils.data.dataset.Subset(train_ds, indices=range(int(SEARCH_PCT * len(train_ds))))
t_full_ds, v_full_ds = random_split(train_ds, [int(0.9 * len(train_ds)), int(0.1 * len(train_ds))])
t_search_ds, v_search_ds = random_split(search_ds, [int(0.9 * len(search_ds)), int(0.1 * len(search_ds))])
test_ds = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=torchvision.transforms.ToTensor())
print(f"len(train_ds)={len(train_ds)}, len(t_search_ds)={len(t_search_ds)}, len(t_full_ds)={len(t_full_ds)}\n")

# Create separate dataloaders for the train, test, and validation set
t_eval_loader = DataLoader(t_full_ds, batch_size=B_SIZE, shuffle=True)
v_eval_loader = DataLoader(v_full_ds, batch_size=B_SIZE, shuffle=True)
t_search_loader = DataLoader(t_search_ds, batch_size=B_SIZE, shuffle=True)
v_search_loader = DataLoader(v_search_ds, batch_size=B_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=B_SIZE, shuffle=True)

imgs, labels = next(iter(t_eval_loader))
example_image, example_label = imgs[0], labels[0]
c, w, h = example_image.size()
d = c * w * h

top_three = part_a_search()
eval_part_a_models(top_three)
