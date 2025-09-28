"""
🔥 PyTorch GPU Training Module for RTX 4060
Advanced PyTorch neural networks with CUDA acceleration
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import time

# Ensure CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 PyTorch Training Device: {device}")

class PyTorchExoplanetNet(nn.Module):
    """Advanced PyTorch neural network for exoplanet classification"""
    
    def __init__(self, input_dim, num_classes, architecture='deep'):
        super(PyTorchExoplanetNet, self).__init__()
        
        if architecture == 'deep':
            self.layers = nn.ModuleList([
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.2),
                
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                
                nn.Linear(64, num_classes)
            ])
        elif architecture == 'wide':
            self.layers = nn.ModuleList([
                nn.Linear(input_dim, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(0.4),
                
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                nn.Linear(512, num_classes)
            ])
        elif architecture == 'residual':
            self.input_layer = nn.Linear(input_dim, 256)
            self.bn_input = nn.BatchNorm1d(256)
            
            # Residual blocks
            self.res_blocks = nn.ModuleList([
                ResidualBlock(256, 256) for _ in range(4)
            ])
            
            self.output_layer = nn.Linear(256, num_classes)
            
        self.architecture = architecture
    
    def forward(self, x):
        if self.architecture == 'residual':
            x = F.relu(self.bn_input(self.input_layer(x)))
            
            for res_block in self.res_blocks:
                x = res_block(x)
            
            x = self.output_layer(x)
        else:
            for i, layer in enumerate(self.layers):
                x = layer(x)
        
        return x

class ResidualBlock(nn.Module):
    """Residual block for deeper networks"""
    
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        
        # Skip connection
        self.skip = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
    
    def forward(self, x):
        residual = self.skip(x)
        
        out = F.relu(self.bn1(self.linear1(x)))
        out = self.bn2(self.linear2(out))
        
        out += residual
        return F.relu(out)

class AttentionLayer(nn.Module):
    """Attention mechanism for feature importance"""
    
    def __init__(self, input_dim, hidden_dim=64):
        super(AttentionLayer, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        weights = self.attention(x)
        return x * weights

class PyTorchTrainer:
    """PyTorch trainer optimized for RTX 4060"""
    
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.device = device
        
    def train_pytorch_model(self, X_train, y_train, X_val, y_val, X_test, y_test, 
                           architecture='deep', epochs=200, batch_size=512):
        """Train PyTorch model with GPU acceleration"""
        
        print(f"🔥 Training PyTorch {architecture} model on {self.device}")
        
        # Convert to tensors and move to GPU
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(np.argmax(y_train, axis=1)).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(np.argmax(y_val, axis=1)).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.LongTensor(np.argmax(y_test, axis=1)).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        # Create model
        input_dim = X_train.shape[1]
        num_classes = y_train.shape[1]
        model = PyTorchExoplanetNet(input_dim, num_classes, architecture).to(self.device)
        
        print(f"📊 Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
        
        # Training loop
        best_val_acc = 0.0
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss, train_correct = 0.0, 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                
                output = model(data)
                loss = criterion(output, target)
                
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                train_correct += pred.eq(target.view_as(pred)).sum().item()
            
            # Validation phase
            model.eval()
            val_loss, val_correct = 0.0, 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1, keepdim=True)
                    val_correct += pred.eq(target.view_as(pred)).sum().item()
            
            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_acc = train_correct / len(train_loader.dataset)
            val_acc = val_correct / len(val_loader.dataset)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'architecture': architecture
                }, f"{self.models_dir}/pytorch_{architecture}_best.pth")
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        training_time = time.time() - start_time
        print(f"⏱️  Training completed in {training_time:.2f} seconds")
        print(f"🏆 Best Validation Accuracy: {best_val_acc:.4f}")
        
        # Load best model for testing
        checkpoint = torch.load(f"{self.models_dir}/pytorch_{architecture}_best.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Test evaluation
        model.eval()
        test_correct = 0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()
                
                all_preds.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())
        
        test_acc = test_correct / len(test_loader.dataset)
        
        print(f"🎯 Test Accuracy: {test_acc:.4f}")
        
        return {
            'model': model,
            'test_accuracy': test_acc,
            'best_val_accuracy': best_val_acc,
            'training_time': training_time,
            'train_history': {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs
            }
        }
    
    def train_ensemble_pytorch(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Train ensemble of PyTorch models"""
        print("🚀 Training PyTorch Ensemble on RTX 4060...")
        
        architectures = ['deep', 'wide', 'residual']
        models = {}
        
        for arch in architectures:
            print(f"\n{'='*50}")
            print(f"Training {arch.upper()} Architecture")
            print(f"{'='*50}")
            
            result = self.train_pytorch_model(
                X_train, y_train, X_val, y_val, X_test, y_test,
                architecture=arch, epochs=150, batch_size=512
            )
            
            models[arch] = result
        
        return models

# Usage example and integration
if __name__ == "__main__":
    # This will be called from the main training script
    print("🔥 PyTorch RTX 4060 Training Module Ready!")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")