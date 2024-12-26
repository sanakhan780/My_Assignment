import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from Networks import build_advanced_segformer
from landslide_dataset import LandslideDataset
from tools import calculate_loss, calculate_metrics
import os

# Training settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 4
num_epochs = 20
learning_rate = 0.0001
num_classes = 5
input_channels = 14
patience = 5  # Early stopping patience

# Dataset and DataLoader
train_dataset = LandslideDataset("TrainData/train.txt")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

valid_dataset = LandslideDataset("ValidData/valid.txt")
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

# Model, optimizer, and loss function
model = build_advanced_segformer(num_classes=num_classes, num_input_channels=input_channels).to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
best_loss = float('inf')
early_stopping_counter = 0

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        inputs, targets = batch['image'].to(device), batch['mask'].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = calculate_loss(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Validation step
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in valid_loader:
            inputs, targets = batch['image'].to(device), batch['mask'].to(device)
            outputs = model(inputs)
            val_loss += calculate_loss(outputs, targets).item()

    avg_val_loss = val_loss / len(valid_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")

    # Learning rate adjustment and early stopping check
    scheduler.step(avg_val_loss)
    
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        early_stopping_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
    else:
        early_stopping_counter += 1
    
    if early_stopping_counter >= patience:
        print("Early stopping triggered.")
        break

print("Training complete.")