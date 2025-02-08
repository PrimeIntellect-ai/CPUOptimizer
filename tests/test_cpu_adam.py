import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from CPUOptimizer.cpu_adam import CPUAdam
from torch.optim import Adam
import math

# pip install numpy torch scikit-learn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def main():
    # Training settings
    batch_size = 32
    epochs = 50
    lr = 0.01
    betas = (0.9, 0.999)
    eps = 1e-8
    
    # Load digits dataset
    digits = load_digits()
    data, target = digits.data, digits.target # type: ignore
    images = StandardScaler().fit_transform(data).astype(np.float32)
    labels = target.astype(np.int64)
    
    
    # Create DataLoader
    images = torch.from_numpy(images)
    labels = torch.from_numpy(labels)
    dataset = torch.utils.data.TensorDataset(images, labels)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize two identical models
    model_cpu = SimpleNet()
    model_torch = SimpleNet()
    model_torch.load_state_dict(model_cpu.state_dict())
    model_cpu.train()
    model_torch.train()

    print("Using vector width:", CPUAdam.vector_width())
    print("Total params:", sum(p.numel() for p in model_cpu.parameters()))
    print("Params:", list(name for name, _ in model_cpu.named_parameters()))
    print()

    # Create optimizers
    def pipeline_hook(param):
        cpu_opt.step_param(param)
    cpu_opt = CPUAdam(model_cpu.parameters(), lr=lr, betas=betas, eps=eps, pipeline_hook=pipeline_hook)
    torch_opt = Adam(model_torch.parameters(), lr=lr, betas=betas, eps=eps)
    
    # Train both models
    max_diff = 0
    for epoch in range(1, epochs + 1):
        sample_number = 0
        for data, target in train_loader:
            sample_number += len(data)
            # Zero gradients
            cpu_opt.zero_grad()
            torch_opt.zero_grad()
            
            # Forward/backward pass
            output_cpu = model_cpu(data)
            loss_cpu = F.nll_loss(output_cpu, target)
            loss_cpu.backward()
            output_torch = model_torch(data)
            loss_torch = F.nll_loss(output_torch, target)
            loss_torch.backward()
                
            # Optimizer steps (The cpu step is done in the backwards hook)
            cpu_opt.step()
            torch_opt.step()
            
            # Verify parameters are close
            for param_cpu, param_torch in zip(model_cpu.parameters(), model_torch.parameters()):    
                diff = torch.max(torch.abs(param_cpu - param_torch)).item()
                if diff > 1e-6:
                    raise AssertionError(f"Parameters diverged! Max difference: {diff}")
                max_diff = max(max_diff, diff)
            
            # Copy CPU parameters to torch model to prevent accumulation
            for param_cpu, param_torch in zip(model_cpu.parameters(), model_torch.parameters()):
                param_torch.data.copy_(param_cpu.data)

            # Print epoch progress
            print(f'\rEpoch: {epoch} [{sample_number}/{len(dataset)}] ', end="")
        print()
    print()
    print("Max diff:", max_diff)

    torch.save(model_cpu.state_dict(), "/tmp/cpu_model.pth")
    model_cpu_state_dict = torch.load("/tmp/cpu_model.pth")

if __name__ == '__main__':
    main()
