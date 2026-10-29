"""
FIXES for Model Collapse Issue
Problems identified:
1. Learning rate too low (1e-5) - model not learning
2. Class imbalance (30% adverse events) - model ignores minority
3. Loss weights may be imbalanced
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.tracking import SwanLabTracker

from legacy.model import LatentODEModel, compute_elbo_loss, compute_classification_loss
from legacy.preprocessing import create_dataloaders
from legacy.preprocessing import collate_fn
from torch.utils.data import DataLoader, WeightedRandomSampler

class ImprovedTrainer:
    def __init__(self, model, train_loader, val_loader, device='cuda', lr=5e-4, tracking_output='results/legacy_training'):
        if not str(device).startswith('cuda') or not torch.cuda.is_available():
            raise RuntimeError('CUDA training is required; CPU fallback is disabled')
        self.tracking_output = Path(tracking_output)
        self.tracker = None
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # FIXED: Higher learning rate
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100, eta_min=1e-6)
        
        # Calculate class weights for imbalanced data
        self.class_weight = self.compute_class_weights()

        # Replace train_loader with a balanced sampler to mitigate class imbalance
        try:
            dataset = self.train_loader.dataset
            # Extract labels from dataset sequences if available
            labels = []
            if hasattr(dataset, 'sequences'):
                labels = [int(s['label']) for s in dataset.sequences]
            else:
                # Fallback: iterate once over loader
                for b in self.train_loader:
                    labels.extend(b['labels'].numpy().flatten().tolist())

            labels = np.array(labels)
            # Inverse frequency weights per sample
            class_counts = np.bincount(labels.astype(int))
            class_weights = 1.0 / (class_counts + 1e-6)
            sample_weights = class_weights[labels.astype(int)]

            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            orig_bs = self.train_loader.batch_size if hasattr(self.train_loader, 'batch_size') else 32
            self.train_loader = DataLoader(dataset, batch_size=orig_bs, sampler=sampler, collate_fn=collate_fn)
            print(f"[INFO] Replaced train DataLoader with WeightedRandomSampler (balanced batches)")
        except Exception as e:
            print(f"[WARN] Could not create balanced sampler: {e}")

        self.history = {'train_loss': [], 'val_loss': [], 'val_auroc': [], 'val_sensitivity': []}
        self.best_val_auroc = -np.inf
        # Early stopping (increase patience)
        self.early_stop_patience = 20
        self._epochs_since_improvement = 0
        
    def compute_class_weights(self):
        """Compute inverse frequency weights for imbalanced classes"""
        all_labels = []
        for batch in self.train_loader:
            all_labels.extend(batch['labels'].numpy().flatten())
        
        all_labels = np.array(all_labels)
        n_samples = len(all_labels)
        n_pos = all_labels.sum()
        n_neg = n_samples - n_pos
        
        # Inverse frequency
        weight_pos = n_samples / (2 * n_pos) if n_pos > 0 else 1.0
        weight_neg = n_samples / (2 * n_neg) if n_neg > 0 else 1.0
        
        print(f"\n[INFO] Class distribution:")
        print(f"  - Negative samples: {n_neg} ({n_neg/n_samples*100:.1f}%)")
        print(f"  - Positive samples: {n_pos} ({n_pos/n_samples*100:.1f}%)")
        print(f"  - Class weights: neg={weight_neg:.2f}, pos={weight_pos:.2f}")
        
        return torch.FloatTensor([weight_neg, weight_pos]).to(self.device)
    
    def weighted_classification_loss(self, outputs, labels):
        """Weighted BCE loss for imbalanced classes"""
        alpha = outputs['alpha']
        beta = outputs['beta']
        probs = alpha / (alpha + beta)
        
        # Apply class weights
        weights = torch.where(labels > 0.5, 
                             self.class_weight[1], 
                             self.class_weight[0])
        
        loss = -weights * (labels * torch.log(probs + 1e-7) + 
                          (1 - labels) * torch.log(1 - probs + 1e-7))
        
        return loss.mean()
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        total_class = 0
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch in pbar:
            vitals = batch['vitals'].to(self.device)
            times = batch['times'].to(self.device)
            masks = batch['masks'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward
            outputs = self.model(vitals, times, masks)
            
            # Get targets
            targets = vitals[:, -1, :]
            mask_final = masks[:, -1]
            
            # FIXED: Better loss balancing
            elbo_dict = compute_elbo_loss(outputs, targets, mask_final, beta=0.01)  # Reduced KL weight
            class_loss = self.weighted_classification_loss(outputs, labels)
            
            # FIXED: Higher weight on classification
            loss = 0.3 * elbo_dict['total'] + 2.0 * class_loss  # Emphasize classification
            
            # Backward
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            if self.tracker is not None:
                self.tracker.batch(loss, self.optimizer.param_groups[0]['lr'], reconstruction=elbo_dict['recon'], kl=elbo_dict['kl'], classification=class_loss, gradient_norm=grad_norm)
            
            # Track
            total_loss += loss.item()
            total_recon += elbo_dict['recon'].item()
            total_kl += elbo_dict['kl'].item()
            total_class += class_loss.item()
            n_batches += 1
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'class': f"{class_loss.item():.4f}"
            })
        
        avg_loss = total_loss / n_batches
        self.history['train_loss'].append(avg_loss)
        
        return {
            'loss': avg_loss,
            'recon': total_recon / n_batches,
            'kl': total_kl / n_batches,
            'class': total_class / n_batches
        }
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_probs = []
        all_labels = []
        all_preds = []
        n_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                vitals = batch['vitals'].to(self.device)
                times = batch['times'].to(self.device)
                masks = batch['masks'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(vitals, times, masks)
                
                targets = vitals[:, -1, :]
                mask_final = masks[:, -1]
                
                elbo_dict = compute_elbo_loss(outputs, targets, mask_final, beta=0.01)
                class_loss = self.weighted_classification_loss(outputs, labels)
                loss = 0.3 * elbo_dict['total'] + 2.0 * class_loss
                
                total_loss += loss.item()
                n_batches += 1
                
                # Get probabilities
                alpha = outputs['alpha']
                beta = outputs['beta']
                probs = (alpha / (alpha + beta)).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                all_probs.extend(probs.flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
                all_preds.extend(preds.flatten())
        
        avg_loss = total_loss / n_batches
        
        # Compute metrics
        from sklearn.metrics import roc_auc_score
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        
        try:
            auroc = roc_auc_score(all_labels, all_probs)
        except:
            auroc = 0.5
        
        # Sensitivity
        tp = ((all_labels == 1) & (all_preds == 1)).sum()
        fn = ((all_labels == 1) & (all_preds == 0)).sum()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        self.history['val_loss'].append(avg_loss)
        self.history['val_auroc'].append(auroc)
        self.history['val_sensitivity'].append(sensitivity)
        
        return {'loss': avg_loss, 'auroc': auroc, 'sensitivity': sensitivity}
    
    def save_checkpoint(self, epoch, metrics, path='models/best_model.pt'):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history
        }, path)
        print(f"  [SAVED] {path}")
    
    def train(self, n_epochs=100, save_every=20):
        print("\n" + "="*60)
        print("IMPROVED TRAINING WITH FIXES")
        print("="*60)
        print("\nFixes applied:")
        print("  ✓ Learning rate: 5e-4 (was 1e-5)")
        print("  ✓ Class-weighted loss")
        print("  ✓ Higher classification weight (2.0x)")
        print("  ✓ Reduced KL weight (0.01)")
        print("  ✓ Better gradient clipping (5.0)")
        
        config = {'seed': str(torch.initial_seed()), 'epochs': n_epochs, 'learning_rate': self.optimizer.param_groups[0]['lr'], 'batch_size': self.train_loader.batch_size, 'device': str(self.device)}
        with SwanLabTracker(self.tracking_output, config, 'legacy', self.model, config['seed']) as tracker:
            self.tracker = tracker
            for epoch in range(1, n_epochs + 1):
                # Train
                train_metrics = self.train_epoch(epoch)
            
                # Validate
                val_metrics = self.validate()
            
                # Scheduler step
                self.scheduler.step()
                tracker.epoch(epoch, train_metrics, val_metrics, self.optimizer.param_groups[0]['lr'])
            
                # Print
                if epoch % 5 == 0 or epoch == 1:
                    print(f"\nEpoch {epoch}/{n_epochs}")
                    print(f"  Train - Loss: {train_metrics['loss']:.4f}, Class: {train_metrics['class']:.4f}")
                    print(f"  Val   - Loss: {val_metrics['loss']:.4f}, AUROC: {val_metrics['auroc']:.4f}, Sens: {val_metrics['sensitivity']:.4f}")

                # Save best based on AUROC (not loss)
                if val_metrics['auroc'] > self.best_val_auroc:
                    self.best_val_auroc = val_metrics['auroc']
                    self.save_checkpoint(epoch, val_metrics, 'models/best_model.pt')
                    self._epochs_since_improvement = 0
                    if epoch % 5 == 0 or epoch == 1:
                        print("  [NEW BEST AUROC]")
                else:
                    self._epochs_since_improvement += 1

                # Early stopping check
                if self._epochs_since_improvement >= self.early_stop_patience:
                    print(f"\n[EARLY STOPPING] No improvement for {self.early_stop_patience} epochs. Stopping training.")
                    break
            
                # Early warning if model is stuck
                if epoch == 20 and val_metrics['sensitivity'] == 0:
                    print("\n⚠️  WARNING: Model not predicting any positive cases!")
                    print("  Consider: Increase classification weight further")
            
                # Save periodic
                if epoch % save_every == 0:
                    self.save_checkpoint(epoch, val_metrics, f'models/checkpoint_epoch_{epoch}.pt')
        
            # If a best checkpoint exists, restore it so the returned model is the best seen
            best_path = Path('models/best_model.pt')
            if best_path.exists():
                try:
                    chk = torch.load(best_path, map_location=self.device)
                    self.model.load_state_dict(chk['model_state_dict'])
                    print(f"\n[RESTORED] Loaded best model from {best_path} (epoch {chk.get('epoch', 'unknown')})")
                except Exception as e:
                    print(f"\n[WARN] Could not restore best model: {e}")

            print("\n" + "="*60)
            print("TRAINING COMPLETE")
            print("="*60)
            print(f"Best Val AUROC: {self.best_val_auroc:.4f}")
        
            # Save history
            with open('results/training_history.json', 'w') as f:
                json.dump(self.history, f, indent=2)
            print("[OK] Saved training history")
        
            tracker.final({'best_validation_auroc': self.best_val_auroc})
            return self.history


def main():
    print("="*60)
    print("RETRAINING WITH IMPROVED CONFIGURATION")
    print("="*60)
    
    # Config
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA training is required; CPU fallback is disabled')
    device = 'cuda'
    print(f"\n[OK] Device: {device}")
    
    # Load data
    print("\n[1/3] Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(batch_size=16)  # Smaller batch
    
    # Create model (fresh initialization)
    print("\n[2/3] Initializing fresh model...")
    model = LatentODEModel(
        input_dim=6,
        hidden_dim=64,
        latent_dim=32,
        output_dim=6
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - Parameters: {total_params:,}")
    
    # Create improved trainer
    print("\n[3/3] Setting up improved trainer...")
    trainer = ImprovedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=5e-4  # FIXED: Much higher learning rate
    )
    
    # Train
    history = trainer.train(n_epochs=100, save_every=20)
    
    print("\n" + "="*60)
    print("RETRAINING COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Run: python src/evaluate.py")
    print("2. Check if AUROC > 0.70")
    print("3. If still low, try:")
    print("   - Increase classification weight to 5.0")
    print("   - Use focal loss for hard examples")
    print("   - Add data augmentation")


if __name__ == "__main__":
    main()
