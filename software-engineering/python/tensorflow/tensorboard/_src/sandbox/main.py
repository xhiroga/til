import torch
import torch.nn as nn
import torch.profiler
from torch.utils.tensorboard import SummaryWriter


class SimpleModel(nn.Module):
    def __init__(self, input_size=784, hidden_size=500, num_classes=10):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def train_step(model, data, target, criterion, optimizer):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss


def profile_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SimpleModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    writer = SummaryWriter('runs/pytorch_profiler_demo')
    
    batch_size = 32
    num_steps = 100
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=2,
            warmup=3,
            active=5,
            repeat=2
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./runs/pytorch_profiler_demo'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for step in range(num_steps):
            data = torch.randn(batch_size, 784).to(device)
            target = torch.randint(0, 10, (batch_size,)).to(device)
            
            loss = train_step(model, data, target, criterion, optimizer)
            
            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")
                writer.add_scalar('Loss/train', loss.item(), step)
            
            prof.step()
    
    writer.close()
    print("\nProfiling complete!")
    print("To view the results, run: tensorboard --logdir=runs")


def main():
    profile_training()


if __name__ == "__main__":
    main()