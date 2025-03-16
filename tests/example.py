import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from CPUOptimizer import CPUOptimizer

def train():
    class MNISTModel(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(784, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 10)

        def forward(self, x):
            x = x.view(-1, 784)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    model = MNISTModel()
    loss_fn = nn.CrossEntropyLoss()

    # Close over the optimizer before it's defined.
    # This is legal python, and is actually required.
    def pipeline_hook(param):
        optimizer.step_param(param)

    # Arguments are the same as torch.optim.Adam/AdamW.
    # Torch's AdamW implementation is substantially different from the original paper,
    # while Adam is the same. We have implemented all of them.
    optimizer = CPUOptimizer(
        model.parameters(),
        step_kind="torch_adamw",      # Or "adam" or "adamw".
        lr=4e-6,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        pipeline_hook=pipeline_hook,  # Or None for a drop-in replacement for Adam without pipelining.
    )

    num_epochs = 50
    train_loader = DataLoader(datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor()), batch_size=64, shuffle=True)
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (inputs, labels) in enumerate(train_loader):

            # Forward as normal
            outputs = model(inputs)

            # Calculate loss as normal
            loss = loss_fn(outputs, labels)
x
            # If you defined a pipeline hook, call this extra method before backward().
            # If you didn't define one, you don't have to call this (but you still can).
            optimizer.begin_step()

            # With a pipeline hook, as grads become available during backward() the optimizer step runs asynchronously on CPU.
            # Overlapping the optimizer step with backward() in this way leads to large speed improvements.
            # If you didn't define a pipeline hook, no changes.
            loss.backward()

            # If you're using a pipeline hook, the step() function waits for the async optimizer step queued by backward() to finish.
            # If you aren't using a pipeline hook, optimizer.step() behaves as normal.
            optimizer.step()

            # Zero grad as normal.
            optimizer.zero_grad()

            # Everything else is unchanged.
            epoch_loss += loss.item()
            print(f'\r\33[2K\033[0;32mEpoch [{epoch+1}/{num_epochs}]\033[0m, \033[0;31mLoss: {loss.item():.4f}\033[0m, \033[0;36mStep [{i+1}/{len(train_loader)}]\033[0m', end="")
        print(f"\r\33[2K\033[0;32mEpoch [{epoch+1}/{num_epochs}]\033[0m, \033[0;31mLoss: {epoch_loss/len(train_loader):.4f}\033[0m")

train()
