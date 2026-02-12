"""
Evaluator Class for Animal CNN
Handles validation and testing loops
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class Evaluator:
    def __init__(self, batch_size: int, data: DataLoader, model, device):
        self.batch_size = batch_size
        self.data = data
        self.model = model
        self.device = device
        self.loss = torch.nn.CrossEntropyLoss()
        
    def start_evaluation_loop(self, epoch=None, phase="Validation"):
        try:
            self.model.eval()
            correct = 0
            total = 0
            validation_losses = []
            
            # Add tqdm progress bar
            pbar = tqdm(self.data, desc=f'{phase} [Epoch {epoch}]' if epoch else phase)
            for batch, (x, y) in enumerate(pbar):
                with torch.no_grad():
                    x = x.to(torch.device(self.device))
                    y = y.to(torch.device(self.device))
                    prediction = self.model(x)
                    validation_loss = self.loss(prediction, y)
                    validation_losses.append(validation_loss.item())
                    
                    # Calculate accuracy
                    _, predicted = torch.max(prediction.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
                    
                    # Update progress bar
                    current_acc = 100.0 * correct / total if total > 0 else 0
                    pbar.set_postfix({
                        'loss': f'{validation_loss.item():.4f}',
                        'acc': f'{current_acc:.2f}%'
                    })
            
            epoch_acc = 100.0 * correct / total if total > 0 else 0.0
            avg_epoch_loss = sum(validation_losses) / len(validation_losses) if validation_losses else 0.0
            
            if epoch is not None:
                print(f"   Average {phase} Loss: {avg_epoch_loss:.4f}")
                print(f"   {phase} Accuracy: {epoch_acc:.2f}%")
            
            return avg_epoch_loss, validation_losses, epoch_acc
        
        except Exception as e:
            print(f"Error in {phase} Loop: {e}")
            return None, None, None
    
    def get_per_class_accuracy(self, class_names=['cat', 'dog', 'panda']):
        """Calculate per-class accuracy"""
        self.model.eval()
        class_correct = [0] * len(class_names)
        class_total = [0] * len(class_names)
        
        with torch.no_grad():
            for x, y in self.data:
                x = x.to(torch.device(self.device))
                y = y.to(torch.device(self.device))
                outputs = self.model(x)
                _, predicted = torch.max(outputs, 1)
                
                for i in range(len(y)):
                    label = y[i]
                    class_correct[label] += (predicted[i] == label).item()
                    class_total[label] += 1
        
        print("\n📊 Per-Class Accuracy:")
        for i, class_name in enumerate(class_names):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                print(f"   {class_name}: {acc:.2f}%")
        
        return {class_names[i]: 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 
                for i in range(len(class_names))}