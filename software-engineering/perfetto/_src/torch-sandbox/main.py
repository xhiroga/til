import torch
import torch.nn as nn
import torch.profiler
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from datetime import datetime
import time


class ImageProcessor:
    """Image preprocessing with data augmentation"""
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def augment_batch(self, images):
        """Apply data augmentation to a batch"""
        augmented = []
        for img in images:
            # Simulate multiple augmentations
            for _ in range(3):
                aug_img = self.transform(img)
                augmented.append(aug_img)
        return torch.stack(augmented)


class CustomResNet(nn.Module):
    """Custom ResNet with additional layers for demonstration"""
    def __init__(self, num_classes=1000):
        super(CustomResNet, self).__init__()
        # Use pretrained ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=False)
        
        # Replace the final layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Add custom head with multiple layers
        self.custom_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Additional attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Apply attention
        attention_weights = self.attention(features)
        features = features * attention_weights
        
        # Final classification
        output = self.custom_head(features)
        return output


def simulate_data_loading(batch_size):
    """Simulate expensive data loading operations"""
    time.sleep(0.001)  # Simulate I/O
    
    # Generate synthetic images
    images = torch.randn(batch_size, 3, 224, 224)
    
    # Simulate complex preprocessing
    for i in range(batch_size):
        # Random crops and transforms
        h_offset = np.random.randint(0, 32)
        w_offset = np.random.randint(0, 32)
        images[i] = torch.roll(images[i], shifts=(h_offset, w_offset), dims=(1, 2))
    
    labels = torch.randint(0, 1000, (batch_size,))
    return images, labels


def mixed_precision_training_step(model, data, target, criterion, optimizer, scaler):
    """Training step with mixed precision"""
    with torch.cuda.amp.autocast():
        output = model(data)
        loss = criterion(output, target)
    
    # Scales loss and calls backward()
    scaler.scale(loss).backward()
    
    # Gradient clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Updates weights
    scaler.step(optimizer)
    scaler.update()
    
    return loss.item()


def profile_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model and training components
    model = CustomResNet(num_classes=1000).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    # Mixed precision training
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Image processor for augmentation
    processor = ImageProcessor()
    
    batch_size = 16
    num_batches = 10
    
    print(f"Starting profiling with {num_batches} batches...")
    print(f"Mixed Precision Training: {'Enabled' if use_amp else 'Disabled'}")
    
    # Profile with detailed settings
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ] if torch.cuda.is_available() else [torch.profiler.ProfilerActivity.CPU],
        schedule=torch.profiler.schedule(
            skip_first=2,
            wait=1,
            warmup=1,
            active=5,
            repeat=1
        ),
        record_shapes=True,
        profile_memory=True,
        with_flops=True  # Calculate FLOPs
    ) as prof:
        
        for step in range(num_batches):
            # Simulate data loading
            with torch.profiler.record_function("data_loading"):
                data, target = simulate_data_loading(batch_size)
                data, target = data.to(device), target.to(device)
            
            # Data augmentation
            with torch.profiler.record_function("data_augmentation"):
                if step % 3 == 0:  # Augment every 3rd batch
                    augmented_data = processor.augment_batch(data)
                    # Concatenate with original data
                    data = torch.cat([data, augmented_data[:batch_size]], dim=0)
                    target = torch.cat([target, target], dim=0)
            
            # Training step
            with torch.profiler.record_function("training_step"):
                optimizer.zero_grad()
                
                if use_amp and scaler is not None:
                    loss = mixed_precision_training_step(
                        model, data, target, criterion, optimizer, scaler
                    )
                else:
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    loss = loss.item()
            
            # Update learning rate
            with torch.profiler.record_function("lr_scheduler"):
                scheduler.step()
            
            # Periodic evaluation
            if step % 5 == 0:
                with torch.profiler.record_function("evaluation"):
                    model.eval()
                    with torch.no_grad():
                        eval_data, eval_target = simulate_data_loading(8)
                        eval_data = eval_data.to(device)
                        eval_output = model(eval_data)
                        accuracy = (eval_output.argmax(1) == eval_target.to(device)).float().mean()
                    model.train()
                    print(f"Step {step}, Loss: {loss:.4f}, Acc: {accuracy:.3f}")
            else:
                print(f"Step {step}, Loss: {loss:.4f}")
            
            # Notify profiler
            prof.step()
    
    # Create traces directory
    os.makedirs("traces", exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_filename = f"traces/trace_{timestamp}.json"
    
    # Export trace
    prof.export_chrome_trace(trace_filename)
    print(f"\nProfiling complete! Trace saved to '{trace_filename}'")
    print("To view the trace:")
    print("1. Open https://ui.perfetto.dev/ in your browser")
    print(f"2. Drag and drop '{trace_filename}' into the interface")
    
    # Print detailed summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    if torch.cuda.is_available():
        print("\n[GPU Operations - Sorted by CUDA time]")
        print(prof.key_averages(group_by_input_shape=True).table(
            sort_by="cuda_time_total", row_limit=15
        ))
    
    print("\n[CPU Operations - Sorted by CPU time]")
    print(prof.key_averages(group_by_input_shape=True).table(
        sort_by="cpu_time_total", row_limit=15
    ))
    
    # Memory usage summary
    if torch.cuda.is_available():
        print(f"\nPeak GPU Memory Usage: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")
        torch.cuda.reset_peak_memory_stats()


def main():
    print("PyTorch Advanced Profiling Example")
    print("="*60)
    print("This example demonstrates:")
    print("- Custom ResNet model with attention mechanism")
    print("- Mixed precision training (AMP)")
    print("- Data augmentation pipeline")
    print("- Learning rate scheduling")
    print("- Gradient clipping")
    print("- Periodic evaluation")
    print("="*60)
    
    profile_training()


if __name__ == "__main__":
    main()