import torch
import torch.nn as nn
from tqdm import tqdm


def train_autoencoder(model, train_loader, val_loader=None, epochs=10, lr=1e-3, device="cpu"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for clean, corrupted, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            clean = clean.to(device)

            optimizer.zero_grad()
            recon = model(clean)
            loss = criterion(recon, clean)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * clean.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        history["train_loss"].append(train_loss)

        # Validation (optional)
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_running = 0.0
            with torch.no_grad():
                for clean, _, _ in val_loader:
                    clean = clean.to(device)
                    recon = model(clean)
                    loss = criterion(recon, clean)
                    val_running += loss.item() * clean.size(0)

            val_loss = val_running / len(val_loader.dataset)
            history["val_loss"].append(val_loss)

        if val_loss is None:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}")
        else:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f} | Val Loss = {val_loss:.6f}")

    return model, history
