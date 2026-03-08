from sklearn.datasets import fetch_openml # type:ignore
from sklearn.model_selection import train_test_split # type:ignore
from sklearn.preprocessing import StandardScaler # type:ignore
import pandas as pd # type:ignore
import torch # type:ignore
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from models.models import CreditRiskANN
import torch.optim as optim
from training.train import train_model
from experiments.evaluate import evaluate
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main() :
    sns.set_style("whitegrid")
    sns.set_context("talk")

    data = fetch_openml("credit-g", as_frame=True)

    X = data.data
    y = data.target

    # encoding
    X = pd.get_dummies(X, dtype=int)

    y = (y=="bad").astype(int) # 0 -> safe , 1 -> risky
    # data splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # feature scaling
    s = StandardScaler()

    X_train = s.fit_transform(X_train)
    X_test = s.transform(X_test)


    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)
    X_train = X_train.to(torch.float32)
    X_test = X_test.to(torch.float32)

    y_train = torch.tensor(y_train.values, dtype = torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)

    input_size = X_train.shape[1]

    y_train = y_train.unsqueeze(1)
    y_test = y_test.unsqueeze(1)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    input_size = X_train.shape[1]


    # medium architecture model.
    print("-"*20+"\tMEDIUM MODEL\t"+"-"*20)
    mediumModel = CreditRiskANN(
        input_size=input_size,
        hidden_layers=[64,32]
    )
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(mediumModel.parameters(), lr=0.001)

    train_losses_m, val_losses_m = train_model(
        mediumModel,
        train_loader,
        test_loader,
        epochs=50,
        criterion=criterion,
        optimizer=optimizer
    )

    metrics = evaluate(mediumModel, X_test, y_test)

    plt.figure()
    plt.plot(train_losses_m, label="Train Loss")
    plt.plot(val_losses_m, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss - Medium architecture")

    plt.legend()
    plt.savefig("plots/training_curve_medium.png")

    print(metrics)
    print('\n'+"-"*80+"\n")
    print("-"*20+"\tSMALL MODEL\t"+"-"*20)
    # Small Architecture model
    smallModel = CreditRiskANN(
        input_size=input_size,
        hidden_layers=[32]
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(smallModel.parameters(), lr=0.001)

    train_losses_s, val_losses_s = train_model(
        smallModel,
        train_loader,
        test_loader,
        epochs=50,
        criterion=criterion,
        optimizer=optimizer
    )

    metrics = evaluate(smallModel, X_test, y_test)
    plt.figure()
    plt.plot(train_losses_s, label="Train Loss")
    plt.plot(val_losses_s, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss - Small architecture")

    plt.legend()

    plt.savefig("plots/training_curve_small.png")

    print(metrics)
    print('\n'+"-"*80+"\n")
    print("-"*20+"\tLARGE MODEL\t"+"-"*20)
    # Large Architecture model
    largeModel = CreditRiskANN(
        input_size=input_size,
        hidden_layers=[128, 64, 32]
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(largeModel.parameters(), lr=0.001)

    train_losses_l, val_losses_l = train_model(
        largeModel,
        train_loader,
        test_loader,
        epochs=50,
        criterion=criterion,
        optimizer=optimizer
    )

    metrics = evaluate(largeModel, X_test, y_test)

    plt.figure()
    plt.plot(train_losses_l, label="Train Loss")
    plt.plot(val_losses_l, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss - Large architecture")

    plt.legend()

    plt.savefig("plots/training_curve_large.png")


    print(metrics)
    print('\n'+"-"*80+"\n")

    plt.figure(figsize=(10,6))
    best_epoch_small = np.argmin(val_losses_s)
    best_epoch_medium = np.argmin(val_losses_m)
    best_epoch_large = np.argmin(val_losses_l)

    plt.scatter(best_epoch_small, val_losses_s[best_epoch_small], color="blue")
    plt.scatter(best_epoch_medium, val_losses_m[best_epoch_medium], color="orange")
    plt.scatter(best_epoch_large, val_losses_l[best_epoch_large], color="green")

    plt.plot(val_losses_s, label="Small model")
    plt.plot(val_losses_m, label="Medium model")
    plt.plot(val_losses_l, label="Large model")

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss vs Epoch for Different Model Capacities")

    plt.legend()
    plt.grid(True)

    plt.savefig("plots/model_capacity_comparison.png", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    main()