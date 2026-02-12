"""
Trainer Class for Animal CNN
Handles training loop with progress tracking
"""

import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

class Trainer:
    def __init__(self, batch_size: int, learning_rate: float, data: DataLoader, 
                 model, model_path, device):
        self.batch_size = batch_size
        self.data = data
        self.model = model
        self.lr = learning_rate
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model_directory = "models"
        os.makedirs(self.model_directory, exist_ok=True)
        self.model_path = model_path
        self.device = device
    
    def start_training_loop(self, epoch):
        try:
            self.model.train()
            training_losses = []
            correct = 0
            total = 0
            
            # Add tqdm progress bar
            pbar = tqdm(self.data, desc=f'Epoch {epoch} [Train]')
            for batch, (x, y) in enumerate(pbar):
                # Forward pass
                x = x.to(torch.device(self.device))
                y = y.to(torch.device(self.device))
                prediction = self.model(x)
                training_loss = self.loss(prediction, y)
                training_losses.append(training_loss.item())

                # Backward pass
                self.optimizer.zero_grad()
                training_loss.backward()
                self.optimizer.step()

                # Calculate accuracy
                _, predicted = torch.max(prediction.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                
                # Update progress bar
                current_acc = 100.0 * correct / total if total > 0 else 0
                pbar.set_postfix({
                    'loss': f'{training_loss.item():.4f}',
                    'acc': f'{current_acc:.2f}%'
                })

            epoch_acc = 100.0 * correct / total if total > 0 else 0.0
            avg_epoch_loss = sum(training_losses) / len(training_losses) if training_losses else 0.0

            print(f"   Average Training Loss: {avg_epoch_loss:.4f}")
            print(f"   Training Accuracy: {epoch_acc:.2f}%")
            return avg_epoch_loss, training_losses, epoch_acc
        
        except Exception as e:
            print(f"Error in Training Loop Epoch {epoch}: {e}")
            return None, None, None
        
    def save_model(self):
        try:
            final_path = os.path.join(self.model_directory, f"{self.model_path}.pth")
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "num_classes": 3,
                    "classes": ["cat", "dog", "panda"]
                },
                final_path
            )
            return final_path
        except Exception as e:
            print(f"Error in Saving Model: {e}")
            return None