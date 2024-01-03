import torch
from torch import nn

batch_size = 6
lr = 1e-4
num_epochs = 20


device = torch.device("cpu")

train_data, train_labels = [], []

for i in range(5):
    train_data.append(torch.load(f"train_images_{i}.pt"))
    train_labels.append(torch.load(f"train_target_{i}.pt"))


test_data = torch.load("test_images.pt")
test_labels = torch.load("test_target.pt")


train_data = torch.cat(train_data, dim=0)
train_labels = torch.cat(train_labels, dim=0)


train_data = train_data.unsqueeze(1)
train_label = train_labels.unsqueeze(1)


test_dataloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(test_data, test_labels),
    batch_size=batch_size,
)

train_dataloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_data, train_labels),
    batch_size=batch_size,
)


model = nn.Sequential(
    nn.Conv2d(1, 32, 3),  # [B, 1, 28 ,28] -> [B, 32, 26, 26]
    nn.LeakyReLU(),
    nn.Conv2d(32, 64, 3),  # [B, 32, 26, 26] -> [B, 64, 24, 24]
    nn.LeakyReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(64 * 12 * 12, 10),
)

print(test_data.shape)
# print(model(test_data[:1].to(device)).shape)


optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
criterion = torch.nn.CrossEntropyLoss()


model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch} with loss: {loss}")

model.eval()
