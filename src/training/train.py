"""
Enhanced Training Script for SQL Query Analyzer
Trains multi-output model for query classification and table ranking
"""
import sys
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import config
from model.data_processor import EnhancedDataProcessor
from model.model import EnhancedSQLAnalyzer, MultiOutputLoss, create_model


class EnhancedTrainer:
    """Trainer for multi-output SQL analyzer"""
    
    def __init__(self, model, train_loader, val_loader, processor):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.processor = processor
        self.device = config.DEVICE
        
        # Loss function
        self.criterion = MultiOutputLoss()
        
        # Optimizer with different learning rates
        bert_params = list(self.model.bert.parameters())
        head_params = []
        for name, param in self.model.named_parameters():
            if 'bert' not in name:
                head_params.append(param)
        
        self.optimizer = AdamW([
            {'params': bert_params, 'lr': config.LEARNING_RATE},
            {'params': head_params, 'lr': config.LEARNING_RATE * 5}
        ], weight_decay=config.WEIGHT_DECAY)
        
        # Scheduler
        total_steps = len(train_loader) * config.NUM_EPOCHS
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=[config.LEARNING_RATE, config.LEARNING_RATE * 5],
            total_steps=total_steps,
            pct_start=0.1
        )
        
        # History
        self.history = {
            'train_loss': [], 'val_loss': [],
            'complexity_acc': [], 'category_acc': [], 'table_count_acc': [],
            'keywords_f1': [], 'subcategory_f1': []
        }
        self.best_val_score = 0.0
        self.patience_counter = 0
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        all_losses = {k: 0 for k in ['complexity', 'keywords', 'category', 'subcategory', 'table_count']}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        
        for batch in pbar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            labels = {
                'complexity': batch['complexity_label'].to(self.device),
                'keywords': batch['keyword_labels'].to(self.device),
                'category': batch['category_label'].to(self.device),
                'subcategory': batch['subcategory_labels'].to(self.device),
                'table_count': batch['table_count_label'].to(self.device)
            }
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            
            # Compute loss
            loss, losses = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            for k, v in losses.items():
                all_losses[k] += v.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        n = len(self.train_loader)
        return {
            'total': total_loss / n,
            **{k: v / n for k, v in all_losses.items()}
        }
    
    @torch.no_grad()
    def validate(self):
        """Validate model"""
        self.model.eval()
        
        total_loss = 0
        
        # Predictions and labels
        complexity_preds, complexity_labels = [], []
        category_preds, category_labels = [], []
        table_count_preds, table_count_labels = [], []
        keywords_preds, keywords_labels = [], []
        subcategory_preds, subcategory_labels = [], []
        
        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            labels = {
                'complexity': batch['complexity_label'].to(self.device),
                'keywords': batch['keyword_labels'].to(self.device),
                'category': batch['category_label'].to(self.device),
                'subcategory': batch['subcategory_labels'].to(self.device),
                'table_count': batch['table_count_label'].to(self.device)
            }
            
            outputs = self.model(input_ids, attention_mask)
            loss, _ = self.criterion(outputs, labels)
            total_loss += loss.item()
            
            # Single-label predictions
            complexity_preds.extend(torch.argmax(outputs['complexity'], dim=1).cpu().numpy())
            complexity_labels.extend(labels['complexity'].cpu().numpy())
            
            category_preds.extend(torch.argmax(outputs['category'], dim=1).cpu().numpy())
            category_labels.extend(labels['category'].cpu().numpy())
            
            table_count_preds.extend(torch.argmax(outputs['table_count'], dim=1).cpu().numpy())
            table_count_labels.extend(labels['table_count'].cpu().numpy())
            
            # Multi-label predictions (threshold at 0.5)
            keywords_preds.extend((torch.sigmoid(outputs['keywords']) > 0.5).cpu().numpy())
            keywords_labels.extend(labels['keywords'].cpu().numpy())
            
            subcategory_preds.extend((torch.sigmoid(outputs['subcategory']) > 0.5).cpu().numpy())
            subcategory_labels.extend(labels['subcategory'].cpu().numpy())
        
        # Calculate metrics
        complexity_acc = np.mean(np.array(complexity_preds) == np.array(complexity_labels))
        category_acc = np.mean(np.array(category_preds) == np.array(category_labels))
        table_count_acc = np.mean(np.array(table_count_preds) == np.array(table_count_labels))
        
        # Multi-label F1
        def multi_label_f1(preds, labels):
            preds = np.array(preds)
            labels = np.array(labels)
            
            # Per-sample F1
            intersection = (preds & labels.astype(bool)).sum(axis=1)
            pred_sum = preds.sum(axis=1)
            label_sum = labels.sum(axis=1)
            
            precision = np.divide(intersection, pred_sum, where=pred_sum > 0, out=np.zeros_like(intersection, dtype=float))
            recall = np.divide(intersection, label_sum, where=label_sum > 0, out=np.zeros_like(intersection, dtype=float))
            
            f1 = np.divide(2 * precision * recall, precision + recall, 
                          where=(precision + recall) > 0, 
                          out=np.zeros_like(precision))
            return np.mean(f1)
        
        keywords_f1 = multi_label_f1(keywords_preds, keywords_labels)
        subcategory_f1 = multi_label_f1(subcategory_preds, subcategory_labels)
        
        return {
            'loss': total_loss / len(self.val_loader),
            'complexity_acc': complexity_acc,
            'category_acc': category_acc,
            'table_count_acc': table_count_acc,
            'keywords_f1': keywords_f1,
            'subcategory_f1': subcategory_f1,
            'combined': (complexity_acc + category_acc + table_count_acc + keywords_f1 + subcategory_f1) / 5
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_accuracy': self.best_val_score,
            'history': self.history
        }
        
        if is_best:
            torch.save(checkpoint, config.BEST_MODEL_PATH)
            print(f"  ✓ Best model saved to {config.BEST_MODEL_PATH}")
    
    def train(self):
        """Full training loop"""
        print(f"\n{'='*60}")
        print("Starting Enhanced Training")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {config.NUM_EPOCHS}")
        print(f"Batch size: {config.BATCH_SIZE}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(config.NUM_EPOCHS):
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            epoch_time = time.time() - epoch_start
            
            # Update history
            self.history['train_loss'].append(train_metrics['total'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['complexity_acc'].append(val_metrics['complexity_acc'])
            self.history['category_acc'].append(val_metrics['category_acc'])
            self.history['table_count_acc'].append(val_metrics['table_count_acc'])
            self.history['keywords_f1'].append(val_metrics['keywords_f1'])
            self.history['subcategory_f1'].append(val_metrics['subcategory_f1'])
            
            # Print summary
            print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS} ({epoch_time:.1f}s):")
            print(f"  Train Loss: {train_metrics['total']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Complexity Acc: {val_metrics['complexity_acc']:.4f}")
            print(f"  Category Acc: {val_metrics['category_acc']:.4f}")
            print(f"  Table Count Acc: {val_metrics['table_count_acc']:.4f}")
            print(f"  Keywords F1: {val_metrics['keywords_f1']:.4f}")
            print(f"  Subcategory F1: {val_metrics['subcategory_f1']:.4f}")
            print(f"  Combined: {val_metrics['combined']:.4f}")
            
            # Check for improvement
            if val_metrics['combined'] > self.best_val_score:
                print(f"  ✓ New best: {val_metrics['combined']:.4f}")
                self.best_val_score = val_metrics['combined']
                self.save_checkpoint(epoch, is_best=True)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                print(f"  No improvement. Patience: {self.patience_counter}/{config.EARLY_STOPPING_PATIENCE}")
            
            if self.patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping after {epoch+1} epochs")
                break
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Best combined score: {self.best_val_score:.4f}")
        
        # Save history
        with open(config.TRAINING_HISTORY_PATH, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"History saved to {config.TRAINING_HISTORY_PATH}")
        
        return self.history


def main():
    print(f"\n{'='*60}")
    print("SQL Query Analyzer - Enhanced Training")
    print(f"{'='*60}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {config.DEVICE}")
    
    # Set seeds
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    # Load data
    print("\n[1/4] Loading and processing data...")
    processor = EnhancedDataProcessor()
    df = processor.load_and_process_data()
    labels = processor.prepare_labels(df)
    split_data = processor.split_data(df, labels)
    
    # Create dataloaders
    print("\n[2/4] Creating dataloaders...")
    dataloaders = processor.create_dataloaders(split_data)
    processor.save_encoders()
    
    # Create model
    print("\n[3/4] Creating model...")
    model = create_model()
    
    # Train
    print("\n[4/4] Starting training...")
    trainer = EnhancedTrainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        processor=processor
    )
    
    trainer.train()
    
    print(f"\n✓ Training complete! Model saved to: {config.BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
