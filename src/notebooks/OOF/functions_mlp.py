import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

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