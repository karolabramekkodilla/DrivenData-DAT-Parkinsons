import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from zmq import Flag

from model.cnn_model import ProfileMLP

def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

def train_profile_mlp(
    X_train,
    y_train,
    X_valid,
    y_valid,
    input_size,
    epochs=300,
    batch_size=64,
    learning_rate=1e-4,
    weight_decay=1e-4,
    patience=30,
):
    device = get_device()
    model = ProfileMLP(input_size=input_size).to(device)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    X_valid_tensor = torch.tensor(
        X_valid,
        dtype=torch.float32,
    ).to(device)

    y_valid_tensor = torch.tensor(
        y_valid,
        dtype=torch.float32,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_valid_loss = np.inf
    best_state_dict = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            logits = model(batch_X)
            loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()

        model.eval()

        with torch.inference_mode():
            valid_logits = model(X_valid_tensor)
            valid_loss = criterion(
                valid_logits,
                y_valid_tensor,
            ).item()

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_state_dict)
    model.to(device)
    model.eval()

    return model, best_valid_loss

def predict_profile_mlp(model, X, batch_size=256):
    current_device = get_device()
    model.to(current_device)
    model.eval()

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
    )

    loader = DataLoader(
        TensorDataset(X_tensor),
        batch_size=batch_size,
        shuffle=False,
    )

    probabilities = []

    with torch.inference_mode():
        for (batch_X,) in loader:
            batch_X = batch_X.to(current_device)

            logits = model(batch_X)
            batch_probabilities = torch.sigmoid(logits)

            probabilities.append(
                batch_probabilities.cpu().numpy()
            )

    return np.concatenate(probabilities)

from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy


def train_cnn_model(
    model,
    X_train,
    y_train,
    X_valid,
    y_valid,
    epochs=1000,
    batch_size=8,
    learning_rate=1e-4,
    weight_decay=1e-4,
    patience=50,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1)

    X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).view(-1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_valid_loss = np.inf
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        valid_losses = []
        valid_proba = []

        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)
                loss = criterion(logits, y_batch)

                proba = torch.sigmoid(logits)

                valid_losses.append(loss.item())
                valid_proba.extend(proba.cpu().numpy().flatten())

        mean_train_loss = np.mean(train_losses)
        mean_valid_loss = np.mean(valid_losses)

        if mean_valid_loss < best_valid_loss:
            best_valid_loss = mean_valid_loss
            best_model_state = deepcopy(model.state_dict())
            patience_counter = 0
            improved = True
        else:
            patience_counter += 1
            improved = False
        if (epoch + 1) % 10 == 0 or improved:
            print(
                f"Epoch {epoch + 1:03d}/{epochs} | "
                f"train_loss: {mean_train_loss:.4f} | "
                f"valid_loss: {mean_valid_loss:.4f} | "
                f"best: {best_valid_loss:.4f} | "
                f"patience: {patience_counter}/{patience}"
            )
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}. Best valid_loss: {best_valid_loss:.4f}")
            break

    model.load_state_dict(best_model_state)

    model.eval()
    valid_proba = []

    with torch.no_grad():
        for X_batch, _ in valid_loader:
            X_batch = X_batch.to(device)

            logits = model(X_batch)
            proba = torch.sigmoid(logits)

            valid_proba.extend(proba.cpu().numpy().flatten())

    return model, np.array(valid_proba), best_valid_loss

def predict_cnn_model(model, X, batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32)

    if X_tensor.ndim == 4:
        X_tensor = X_tensor.unsqueeze(1)

    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds = []

    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(device)

            logits = model(X_batch)
            proba = torch.sigmoid(logits)

            preds.extend(proba.cpu().numpy().flatten())

    return np.array(preds)