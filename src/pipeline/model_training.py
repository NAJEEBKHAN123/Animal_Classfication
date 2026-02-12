"""
Main Training Pipeline for Animal Classification
Teacher's exact workflow, adapted for cat/dog/panda dataset
Run this file to train the model
"""

import torch
import os
import wandb
from datetime import datetime
from src.data.loader import train_loader, val_loader, test_loader
from src.model.cnn import CNN
from src.model.train import Trainer
from src.model.evaluation import Evaluator

def main():
    try:
        # ============= CONFIGURATION =============
        EPOCHS = 50  # Can be increased to 100
        BATCH_SIZE = 32
        LEARNING_RATE = 0.001
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        NUM_CLASSES = 3  # cat, dog, panda
        CLASS_NAMES = ['cat', 'dog', 'panda']
        
        # Experiment config
        config = {
            "Epochs": EPOCHS,
            "Batch Size": BATCH_SIZE,
            "Learning Rate": LEARNING_RATE,
            "Device": DEVICE,
            "Model": "TeacherCNN",
            "Classes": NUM_CLASSES,
            "Class Names": CLASS_NAMES,
            "Dataset": "Animal Dataset (5,848 images)",
            "Input Size": "128x128"
        }

        # ============= WANDB INITIALIZATION =============
        wandb.init(
            project="Animal-Classifier-CNN",
            config=config,
            name=f'Animal-Exp-{datetime.now().strftime("%d_%m_%Y_%H_%M")}',
            tags=["teacher-cnn", "animals", "cat-dog-panda"]
        )
        
        print("\n" + "="*60)
        print("🐾 ANIMAL CLASSIFICATION TRAINING")
        print("="*60)
        print(f"Using device: {DEVICE}")
        if DEVICE == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Classes: {CLASS_NAMES}")
        print(f"Epochs: {EPOCHS}")
        print("="*60 + "\n")

        # ============= MODEL INITIALIZATION =============
        torch.set_default_device(DEVICE)
        my_model = CNN(num_classes=NUM_CLASSES).to(DEVICE)
        
        print(f"🧠 Model created: TeacherCNN")
        print(f"   Parameters: {sum(p.numel() for p in my_model.parameters()):,}")
        print(f"   Output classes: {NUM_CLASSES}")

        # ============= TRAINER & EVALUATOR =============
        model_trainer = Trainer(
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            data=train_loader,
            model=my_model,
            model_path="animal_cnn_best",
            device=DEVICE
        )

        model_evaluator = Evaluator(
            batch_size=BATCH_SIZE,
            data=val_loader,
            model=my_model,
            device=DEVICE
        )
        
        test_evaluator = Evaluator(
            batch_size=BATCH_SIZE,
            data=test_loader,
            model=my_model,
            device=DEVICE
        )

        # ============= TRAINING LOOP =============
        BEST_ACCURACY = 0
        BEST_EPOCH = 0

        for epoch in range(EPOCHS):
            print(f"\n{'='*50}")
            print(f"📌 EPOCH {epoch+1}/{EPOCHS}")
            print(f"{'='*50}")
            
            # Training phase
            avg_train_loss, _, train_acc = model_trainer.start_training_loop(epoch+1)

            # Validation phase
            avg_val_loss, _, val_acc = model_evaluator.start_evaluation_loop(epoch+1, phase="Validation")
            
            # Log to wandb
            wandb.log({
                "Training Loss": avg_train_loss,
                "Validation Loss": avg_val_loss,
                "Training Accuracy": train_acc,
                "Validation Accuracy": val_acc,
                "Epoch": epoch + 1,
                "Learning Rate": model_trainer.optimizer.param_groups[0]['lr']
            })
            
            # Save best model
            if val_acc > BEST_ACCURACY:
                BEST_ACCURACY = val_acc
                BEST_EPOCH = epoch + 1
                final_model_path = model_trainer.save_model()
                
                if final_model_path:
                    print(f"\n🏆 NEW BEST MODEL!")
                    print(f"   Accuracy: {val_acc:.2f}%")
                    print(f"   Epoch: {epoch+1}")
                    print(f"   Saved to: {final_model_path}")
                    
                    # Log model to wandb
                    wandb.log_model(
                        final_model_path, 
                        "animal_classifier_cnn",
                        aliases=[f"epoch-{epoch+1}", f"acc-{val_acc:.2f}", "best"]
                    )
            
            # Early stopping notification
            if epoch + 1 - BEST_EPOCH > 15:
                print(f"\n⚠️  No improvement for 15 epochs. Best accuracy: {BEST_ACCURACY:.2f}% at epoch {BEST_EPOCH}")

        # ============= FINAL EVALUATION =============
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETED!")
        print("="*60)
        print(f"\n🏆 Best Validation Accuracy: {BEST_ACCURACY:.2f}% at epoch {BEST_EPOCH}")
        
        # Load best model for testing
        print("\n📊 Evaluating best model on test set...")
        checkpoint = torch.load('models/animal_cnn_best.pth', map_location=DEVICE)
        my_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Test evaluation
        _, _, test_acc = test_evaluator.start_evaluation_loop(phase="Test")
        print(f"\n🎯 Test Accuracy: {test_acc:.2f}%")
        
        # Per-class accuracy
        per_class_acc = test_evaluator.get_per_class_accuracy(CLASS_NAMES)
        
        # Log final metrics
        wandb.log({
            "Best Validation Accuracy": BEST_ACCURACY,
            "Test Accuracy": test_acc,
            "Best Epoch": BEST_EPOCH,
            "Cat Accuracy": per_class_acc['cat'],
            "Dog Accuracy": per_class_acc['dog'],
            "Panda Accuracy": per_class_acc['panda']
        })
        
        # Save final model with metadata
        final_save_path = f'models/animal_cnn_final_acc{test_acc:.1f}.pth'
        torch.save({
            'model_state_dict': my_model.state_dict(),
            'val_accuracy': BEST_ACCURACY,
            'test_accuracy': test_acc,
            'epochs_trained': EPOCHS,
            'best_epoch': BEST_EPOCH,
            'classes': CLASS_NAMES,
            'num_classes': NUM_CLASSES,
            'input_size': 128
        }, final_save_path)
        
        print(f"\n💾 Final model saved to: {final_save_path}")
        
        # Close wandb run
        wandb.finish()
        
        print("\n🎯 TRAINING PIPELINE COMPLETE!")
        print(f"   Best Validation Accuracy: {BEST_ACCURACY:.2f}%")
        print(f"   Test Accuracy: {test_acc:.2f}%")
        print(f"   Model saved in: models/")

    except Exception as e:
        print(f"❌ Error in Training Pipeline: {e}")
        wandb.finish(exit_code=1)
        raise e

if __name__ == "__main__":
    # Try environment variable first, then prompt user
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
    else:
        print("\n⚠️  WANDB_API_KEY not found in environment variables")
        print("Options:")
        print("  1. Enter your API key now")
        print("  2. Press Enter to continue without WandB")
        print("  3. Type 'skip' to disable WandB for this run\n")
        
        user_input = input("Enter API key (or press Enter to skip): ").strip()
        
        if user_input and user_input.lower() != 'skip':
            wandb.login(key=user_input)
        else:
            print("✅ Continuing without WandB logging")
            # Disable wandb by setting mode to disabled
            os.environ['WANDB_MODE'] = 'disabled'
    
    main()