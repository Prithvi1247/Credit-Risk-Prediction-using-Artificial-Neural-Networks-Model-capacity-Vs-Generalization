import torch # type: ignore

def train_model(model, train_loader, val_loader, epochs, criterion, optimizer):

    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        model.train()
        running_loss = 0

        for X_batch, y_batch in train_loader:

            optimizer.zero_grad()

            logits = model(X_batch)

            loss = criterion(logits, y_batch)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # validation
        model.eval()
        val_loss_total = 0

        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                logits = model(X_batch)
                loss = criterion(logits, y_batch)

                val_loss_total += loss.item()

        val_loss = val_loss_total / len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    return train_losses, val_losses