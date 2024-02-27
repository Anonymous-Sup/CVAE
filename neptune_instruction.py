import neptune

run = neptune.init_run(
    project="Zhengwei-Lab/NIPSTransferReID",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2ODIwNTQ4Yy0xZDA3LTRhNDctOTRmMy02ZjRlMmMzYmYwZjUifQ==",
)  # your credentials

params = {
    "lr": 1e-2,
    "bs": 128,
    "input_sz": 32 * 32 * 3,
    "n_classes": 10,
}
run["parameters"] = params

transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
trainset = datasets.CIFAR10("./data", transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=params["bs"], shuffle=True
)


class BaseModel(nn.Module):
    def __init__(self, input_sz, hidden_dim, n_classes):
        super(BaseModel, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_sz, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, input):
        x = input.view(-1, 32 * 32 * 3)
        return self.main(x)


model = BaseModel(params["input_sz"], params["input_sz"], params["n_classes"])
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=params["lr"])

for i, (x, y) in enumerate(trainloader, 0):

    optimizer.zero_grad()
    outputs = model.forward(x)
    _, preds = torch.max(outputs, 1)
    loss = criterion(outputs, y)
    acc = (torch.sum(preds == y.data)) / len(x)

    run["train/batch/loss"].append(loss)
    run["train/batch/acc"].append(acc)

    loss.backward()
    optimizer.step()

model.eval()
correct = 0
for X, y in trainloader:
    pred = model(X)
    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
run["valid/acc"] = correct / len(trainloader.dataset)

run.stop()