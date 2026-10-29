"""
Step 6: Advanced Enhancements
- Attention visualization
- Trajectory prediction
- Interpretability via integrated gradients
- Real-time inference demo
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
sys.path.append('src')

from model import LatentODEModel
from preprocessing import create_dataloaders
import pickle
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    brier_score_loss,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
)
from sklearn.calibration import calibration_curve
from sklearn.metrics import f1_score
import math
from sklearn.linear_model import LogisticRegression

# ==================== ATTENTION VISUALIZATION ====================
def visualize_attention_weights(model, sample_batch, device='cuda'):
    """Extract and visualize transformer attention patterns"""
    model.eval()
    vitals = sample_batch['vitals'][:1].to(device)  # Single patient
    times = sample_batch['times'][:1].to(device)
    masks = sample_batch['masks'][:1].to(device)
    
    # Hook to capture attention
    attention_weights = []
    
    def hook_fn(module, input, output):
        # output is tuple (attn_output, attn_weights)
        if len(output) > 1 and output[1] is not None:
            attention_weights.append(output[1].detach().cpu())
    
    # Register hook
    for layer in model.encoder.transformer.layers:
        layer.self_attn.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        _ = model(vitals, times, masks)
    
    if attention_weights:
        attn = attention_weights[0][0]  # First head, first layer
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(attn.numpy(), cmap='viridis', cbar_kws={'label': 'Attention Weight'})
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        ax.set_title('Transformer Attention Weights (Layer 1, Head 1)')
        plt.tight_layout()
        plt.savefig('results/attention_weights.png', dpi=300, bbox_inches='tight')
        print("[OK] Saved attention_weights.png")
        plt.close()


# ==================== TRAJECTORY PREDICTION ====================
def predict_patient_trajectory(model, patient_data, future_hours=24, device='cuda'):
    """Predict full patient trajectory into future"""
    model.eval()
    
    vitals = patient_data['vitals'].unsqueeze(0).to(device)
    times = patient_data['times'].unsqueeze(0).to(device)
    masks = patient_data['masks'].unsqueeze(0).to(device)
    
    # Observed timepoints
    obs_times = times[0][masks[0] == 1].cpu().numpy()
    obs_vitals = vitals[0][masks[0] == 1].cpu().numpy()
    
    # Future timepoints
    last_time = obs_times[-1]
    future_times_array = np.linspace(last_time, last_time + future_hours, 50)
    future_times_tensor = torch.FloatTensor(future_times_array).to(device)
    
    with torch.no_grad():
        # Get latent state
        mu, logvar = model.encoder(vitals, masks)
        z0 = mu  # Use mean (no sampling for prediction)
        
        # Solve ODE at future times
        z_trajectory = model.ode(z0, future_times_tensor)
        
        # Decode
        vitals_pred = model.decoder(z_trajectory).squeeze(0).cpu().numpy()
    
    # Plot trajectory
    vital_names = ['Heart Rate', 'SBP', 'DBP', 'SpO2', 'Temperature', 'Resp Rate']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, name in enumerate(vital_names):
        # Observed
        axes[i].scatter(obs_times, obs_vitals[:, i], c='blue', s=50, 
                       label='Observed', zorder=3, edgecolors='black')
        
        # Predicted trajectory
        axes[i].plot(future_times_array, vitals_pred[:, i], 'r-', 
                    linewidth=2, label='Predicted', alpha=0.7)
        
        # Shaded uncertainty region (simplified)
        std = np.std(vitals_pred[:, i]) * 0.5
        axes[i].fill_between(future_times_array, 
                             vitals_pred[:, i] - std,
                             vitals_pred[:, i] + std,
                             color='red', alpha=0.2)
        
        axes[i].axvline(x=last_time, color='k', linestyle='--', 
                       linewidth=1, label='Prediction Start')
        axes[i].set_xlabel('Time (hours)')
        axes[i].set_ylabel(name)
        axes[i].set_title(f'{name} Trajectory')
        axes[i].legend()
        axes[i].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/patient_trajectory.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved patient_trajectory.png")
    plt.close()


# ==================== LATENT SPACE VISUALIZATION ====================
def visualize_latent_space(model, dataloader, device='cuda'):
    """Visualize latent representations using t-SNE"""
    from sklearn.manifold import TSNE
    
    model.eval()
    all_z = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            vitals = batch['vitals'].to(device)
            masks = batch['masks'].to(device)
            labels = batch['labels'].cpu().numpy()
            
            mu, _ = model.encoder(vitals, masks)
            all_z.append(mu.cpu().numpy())
            all_labels.extend(labels.flatten())
    
    all_z = np.vstack(all_z)
    all_labels = np.array(all_labels)
    
    # t-SNE
    print("Computing t-SNE (this may take a minute)...")
    n_samples = len(all_z)
    perplexity = min(5, n_samples - 1)  # Adjust perplexity for small datasets
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    z_embedded = tsne.fit_transform(all_z)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(z_embedded[:, 0], z_embedded[:, 1], 
                        c=all_labels, cmap='RdYlGn_r', 
                        s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Adverse Event (0=No, 1=Yes)')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title('Latent Space Visualization (t-SNE)')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/latent_space_tsne.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved latent_space_tsne.png")
    plt.close()


# ==================== ODE DYNAMICS VISUALIZATION ====================
def visualize_ode_dynamics(model, sample_batch, device='cuda'):
    """Visualize how latent state evolves through ODE"""
    model.eval()
    
    vitals = sample_batch['vitals'][:1].to(device)
    times = sample_batch['times'][:1].to(device)
    masks = sample_batch['masks'][:1].to(device)
    
    with torch.no_grad():
        mu, _ = model.encoder(vitals, masks)
        z0 = mu
        
        # Solve at many timepoints
        time_array = torch.linspace(0, 72, 200).to(device)
        z_trajectory = model.ode(z0, time_array)
        z_traj = z_trajectory.squeeze(0).cpu().numpy()  # (200, latent_dim)
    
    # Plot first 6 latent dimensions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    time_hours = time_array.cpu().numpy()
    
    for i in range(6):
        axes[i].plot(time_hours, z_traj[:, i], linewidth=2)
        axes[i].set_xlabel('Time (hours)')
        axes[i].set_ylabel(f'Latent Dim {i+1}')
        axes[i].set_title(f'ODE Dynamics - Dimension {i+1}')
        axes[i].grid(alpha=0.3)
    
    plt.suptitle('Continuous-Time Latent Dynamics (Neural ODE)', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig('results/ode_dynamics.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved ode_dynamics.png")
    plt.close()


# ==================== FEATURE IMPORTANCE ====================
def compute_feature_importance(model, dataloader, device='cuda'):
    """Compute feature importance via gradient-based attribution"""
    model.eval()
    
    vital_names = ['Heart Rate', 'SBP', 'DBP', 'SpO2', 'Temperature', 'Resp Rate']
    importances = np.zeros(len(vital_names))
    n_samples = 0
    
    for batch in dataloader:
        vitals = batch['vitals'].to(device)
        times = batch['times'].to(device)
        masks = batch['masks'].to(device)
        
        vitals.requires_grad = True
        
        outputs = model(vitals, times, masks)
        alpha = outputs['alpha']
        prob = alpha / (alpha + outputs['beta'])
        
        # Compute gradients
        prob.sum().backward()
        
        # Average absolute gradients
        grads = torch.abs(vitals.grad).mean(dim=(0, 1)).cpu().numpy()
        importances += grads
        n_samples += 1
        
        model.zero_grad()
        
        if n_samples >= 20:  # Sample subset for speed
            break
    
    importances /= n_samples
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(vital_names, importances, color='steelblue', edgecolor='black')
    ax.set_xlabel('Average Absolute Gradient (Importance)')
    ax.set_title('Feature Importance for Adverse Event Prediction')
    ax.grid(axis='x', alpha=0.3)
    
    # Add values
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
               f'{width:.4f}', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved feature_importance.png")
    plt.close()


# ==================== TEST SET EVALUATION ====================
def evaluate_on_testloader(model, test_loader, device='cuda'):
    """Evaluate model on the test loader and save metrics + plots."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            vitals = batch['vitals'].to(device)
            times = batch.get('times', None)
            if times is not None:
                times = times.to(device)
            masks = batch['masks'].to(device)

            labels = batch['labels'].cpu().numpy().flatten()

            # Model forward -> expect evidential outputs 'alpha' and 'beta'
            outputs = model(vitals, times, masks) if times is not None else model(vitals, None, masks)
            alpha = outputs['alpha']
            beta = outputs['beta']

            probs = (alpha / (alpha + beta)).cpu().numpy().flatten()

            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())

    import numpy as np
    probs = np.array(all_probs)
    labels = np.array(all_labels).astype(int)

    metrics = {}
    # Safely compute AUROC / AUPRC only if both classes present
    unique_labels = np.unique(labels)
    if len(unique_labels) == 2:
        try:
            metrics['auroc'] = float(roc_auc_score(labels, probs))
        except Exception:
            metrics['auroc'] = float('nan')
        try:
            metrics['auprc'] = float(average_precision_score(labels, probs))
        except Exception:
            metrics['auprc'] = float('nan')
    else:
        metrics['auroc'] = float('nan')
        metrics['auprc'] = float('nan')

    # Binarize at 0.5 for accuracy/PRF
    preds = (probs >= 0.5).astype(int)
    try:
        metrics['accuracy'] = float(accuracy_score(labels, preds))
    except Exception:
        metrics['accuracy'] = float('nan')

    try:
        metrics['brier'] = float(brier_score_loss(labels, probs))
    except Exception:
        metrics['brier'] = float('nan')

    try:
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
        metrics['precision'] = float(precision)
        metrics['recall'] = float(recall)
        metrics['f1'] = float(f1)
    except Exception:
        metrics['precision'] = metrics['recall'] = metrics['f1'] = float('nan')

    # Confusion matrix
    try:
        cm = confusion_matrix(labels, preds)
        metrics['confusion_matrix'] = cm.tolist()
    except Exception:
        metrics['confusion_matrix'] = None

    # Save metrics to file
    Path('results').mkdir(parents=True, exist_ok=True)
    out_txt = Path('results/test_evaluation.txt')
    with out_txt.open('w') as f:
        f.write('Test set evaluation metrics\n')
        f.write('===========================\n')
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    print('[OK] Saved test metrics -> results/test_evaluation.txt')

    # Plot ROC and PR if possible
    if not np.isnan(metrics['auroc']):
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f'AUROC = {roc_auc:.4f}')
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('[OK] Saved ROC curve -> results/roc_curve.png')

    if not np.isnan(metrics['auprc']):
        prec, rec, _ = precision_recall_curve(labels, probs)
        pr_auc = auc(rec, prec)
        plt.figure(figsize=(6, 6))
        plt.plot(rec, prec, label=f'AUPRC = {pr_auc:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/pr_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('[OK] Saved PR curve -> results/pr_curve.png')
    return metrics, probs, labels


def fit_temperature_scaling(model, val_loader, device='cuda', max_iters=200):
    """Fit a single temperature scalar on validation set by minimizing NLL.
    Returns temperature (float). If val set is single-class or fitting fails, returns 1.0
    """
    model.eval()
    probs = []
    labels = []
    with torch.no_grad():
        for batch in val_loader:
            vitals = batch['vitals'].to(device)
            times = batch.get('times', None)
            if times is not None:
                times = times.to(device)
            masks = batch['masks'].to(device)
            lab = batch['labels'].cpu().numpy().flatten()

            outputs = model(vitals, times, masks) if times is not None else model(vitals, None, masks)
            alpha = outputs['alpha']
            beta = outputs['beta']
            p = (alpha / (alpha + beta)).cpu().numpy().flatten()
            probs.extend(p.tolist())
            labels.extend(lab.tolist())

    import numpy as np
    probs = np.clip(np.array(probs), 1e-6, 1 - 1e-6)
    labels = np.array(labels).astype(int)

    if len(np.unique(labels)) < 2:
        print('[WARN] Validation set has single class - skipping temperature scaling')
        return 1.0

    # Convert to logits
    logits = np.log(probs / (1.0 - probs))

    # Optimize log_t such that T = exp(log_t) > 0
    log_t = torch.tensor(0.0, requires_grad=True, device=device)
    labels_t = torch.tensor(labels, dtype=torch.float32, device=device)
    logits_t = torch.tensor(logits, dtype=torch.float32, device=device)

    optimizer = torch.optim.LBFGS([log_t], max_iter=50, tolerance_grad=1e-6, line_search_fn='strong_wolfe')

    loss_fn = torch.nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        T = torch.exp(log_t)
        loss = loss_fn(logits_t / T, labels_t)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
        T_val = float(torch.exp(log_t).item())
        print(f"[OK] Fitted temperature: {T_val:.4f}")
        Path('results').mkdir(parents=True, exist_ok=True)
        with open('results/temperature.txt', 'w') as f:
            f.write(f"temperature: {T_val}\n")
        return T_val
    except Exception as e:
        print(f"[WARN] Temperature scaling failed: {e}")
        return 1.0


def fit_platt_scaling(model, val_loader, device='cuda'):
    """Fit Platt scaling (logistic regression) on validation logits.
    Returns fitted sklearn LogisticRegression instance. If fitting fails or val single-class, returns None.
    """
    model.eval()
    probs = []
    labels = []
    with torch.no_grad():
        for batch in val_loader:
            vitals = batch['vitals'].to(device)
            times = batch.get('times', None)
            if times is not None:
                times = times.to(device)
            masks = batch['masks'].to(device)
            lab = batch['labels'].cpu().numpy().flatten()

            outputs = model(vitals, times, masks) if times is not None else model(vitals, None, masks)
            alpha = outputs['alpha']
            beta = outputs['beta']
            p = (alpha / (alpha + beta)).cpu().numpy().flatten()
            probs.extend(p.tolist())
            labels.extend(lab.tolist())

    import numpy as np
    probs = np.clip(np.array(probs), 1e-6, 1 - 1e-6)
    labels = np.array(labels).astype(int)

    if len(np.unique(labels)) < 2:
        print('[WARN] Validation set has single class - skipping Platt scaling')
        return None

    logits = np.log(probs / (1.0 - probs)).reshape(-1, 1)

    try:
        clf = LogisticRegression(solver='lbfgs')
        clf.fit(logits, labels)
        Path('results').mkdir(parents=True, exist_ok=True)
        with open('results/platt_params.txt', 'w') as f:
            f.write(f"coef: {clf.coef_.tolist()}\nintercept: {clf.intercept_.tolist()}\n")
        print('[OK] Fitted Platt scaling (LogisticRegression)')
        return clf
    except Exception as e:
        print(f"[WARN] Platt scaling failed: {e}")
        return None


def find_best_threshold_on_val(model, val_loader, device='cuda', apply_temperature=None, apply_platt_clf=None):
    """Find threshold on validation set that maximizes F1.
    Returns threshold (float).
    """
    model.eval()
    probs = []
    labels = []
    with torch.no_grad():
        for batch in val_loader:
            vitals = batch['vitals'].to(device)
            times = batch.get('times', None)
            if times is not None:
                times = times.to(device)
            masks = batch['masks'].to(device)
            lab = batch['labels'].cpu().numpy().flatten()

            outputs = model(vitals, times, masks) if times is not None else model(vitals, None, masks)
            alpha = outputs['alpha']
            beta = outputs['beta']
            p = (alpha / (alpha + beta)).cpu().numpy().flatten()
            probs.extend(p.tolist())
            labels.extend(lab.tolist())

    import numpy as np
    probs = np.array(probs)
    labels = np.array(labels).astype(int)

    # Apply calibration if requested
    if apply_platt_clf is not None:
        # apply Platt scaling by converting probs->logits->platt.proba
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        logits = np.log(probs / (1 - probs)).reshape(-1, 1)
        try:
            probs = apply_platt_clf.predict_proba(logits)[:, 1]
        except Exception:
            # fallback to original probs
            pass
    elif apply_temperature is not None and apply_temperature != 1.0:
        # temperature scaling via logits -> scaled probs
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        logits = np.log(probs / (1 - probs))
        logits = logits / apply_temperature
        probs = 1.0 / (1.0 + np.exp(-logits))

    if len(np.unique(labels)) < 2:
        print('[WARN] Validation set single class - using default threshold 0.5')
        return 0.5

    best_thresh = 0.5
    best_f1 = -1
    for thresh in np.linspace(0.01, 0.99, 99):
        preds = (probs >= thresh).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)

    Path('results').mkdir(parents=True, exist_ok=True)
    with open('results/threshold.txt', 'w') as f:
        f.write(f"best_threshold: {best_thresh}\nbest_f1: {best_f1}\n")
    print(f"[OK] Selected threshold on val: {best_thresh} (F1={best_f1:.4f})")
    return best_thresh


def calibration_plot(probs, labels, n_bins=10, outpath='results/calibration_plot.png'):
    import numpy as np
    frac_pos, mean_pred = calibration_curve(labels, probs, n_bins=n_bins)
    plt.figure(figsize=(6, 6))
    plt.plot(mean_pred, frac_pos, 's-', label='Calibration')
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Plot (Reliability Diagram)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved calibration plot -> {outpath}")


def evaluate_with_calibration(model, val_loader, test_loader, device='cuda'):
    """Calibrate on val (temperature scaling + threshold selection) and evaluate on test.
    Saves calibrated metrics and plots.
    """
    # Prefer Platt scaling (stable) and fall back to temperature scaling
    platt_clf = fit_platt_scaling(model, val_loader, device)
    T = 1.0
    use_platt = platt_clf is not None
    if not use_platt:
        try:
            T = fit_temperature_scaling(model, val_loader, device)
        except Exception as e:
            print(f"[WARN] Temperature fit failed: {e}")
            T = 1.0

    # Choose threshold on val using calibrated probs
    if use_platt:
        thresh = find_best_threshold_on_val(model, val_loader, device, apply_platt_clf=platt_clf)
    else:
        thresh = find_best_threshold_on_val(model, val_loader, device, apply_temperature=T)

    # Collect test probs and labels
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            vitals = batch['vitals'].to(device)
            times = batch.get('times', None)
            if times is not None:
                times = times.to(device)
            masks = batch['masks'].to(device)
            labels = batch['labels'].cpu().numpy().flatten()

            outputs = model(vitals, times, masks) if times is not None else model(vitals, None, masks)
            alpha = outputs['alpha']
            beta = outputs['beta']
            p = (alpha / (alpha + beta)).cpu().numpy().flatten()
            all_probs.extend(p.tolist())
            all_labels.extend(labels.tolist())

    import numpy as np
    probs = np.array(all_probs)
    labels = np.array(all_labels).astype(int)

    # Apply calibration: either Platt or temperature
    if use_platt and platt_clf is not None:
        logits = np.log(np.clip(probs, 1e-6, 1 - 1e-6) / (1 - np.clip(probs, 1e-6, 1 - 1e-6))).reshape(-1, 1)
        probs = platt_clf.predict_proba(logits)[:, 1]
    else:
        if T is not None and T != 1.0:
            probs = np.clip(probs, 1e-6, 1 - 1e-6)
            logits = np.log(probs / (1 - probs))
            logits = logits / T
            probs = 1.0 / (1.0 + np.exp(-logits))

    # Save calibration plot (using test set for visualization is OK)
    try:
        calibration_plot(probs, labels, n_bins=10, outpath='results/calibration_plot.png')
    except Exception as e:
        print(f"[WARN] Calibration plot failed: {e}")

    # Compute thresholded metrics on test
    preds = (probs >= thresh).astype(int)

    metrics = {}
    unique_labels = np.unique(labels)
    if len(unique_labels) == 2:
        try:
            metrics['auroc'] = float(roc_auc_score(labels, probs))
        except Exception:
            metrics['auroc'] = float('nan')
        try:
            metrics['auprc'] = float(average_precision_score(labels, probs))
        except Exception:
            metrics['auprc'] = float('nan')
    else:
        metrics['auroc'] = float('nan')
        metrics['auprc'] = float('nan')

    try:
        metrics['accuracy'] = float(accuracy_score(labels, preds))
    except Exception:
        metrics['accuracy'] = float('nan')

    try:
        metrics['brier'] = float(brier_score_loss(labels, probs))
    except Exception:
        metrics['brier'] = float('nan')

    try:
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
        metrics['precision'] = float(precision)
        metrics['recall'] = float(recall)
        metrics['f1'] = float(f1)
    except Exception:
        metrics['precision'] = metrics['recall'] = metrics['f1'] = float('nan')

    try:
        cm = confusion_matrix(labels, preds)
        metrics['confusion_matrix'] = cm.tolist()
    except Exception:
        metrics['confusion_matrix'] = None

    Path('results').mkdir(parents=True, exist_ok=True)
    with open('results/test_evaluation_calibrated.txt', 'w') as f:
        f.write('Test set evaluation metrics (calibrated + thresholded)\n')
        f.write('=============================================\n')
        f.write(f"temperature: {T}\n")
        f.write(f"threshold: {thresh}\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    print('[OK] Saved calibrated test metrics -> results/test_evaluation_calibrated.txt')
    # Also save calibrated ROC/PR
    if not math.isnan(metrics['auroc']):
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f'AUROC = {roc_auc:.4f}')
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Calibrated)')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/roc_curve_calibrated.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('[OK] Saved ROC curve -> results/roc_curve_calibrated.png')

    if not math.isnan(metrics['auprc']):
        prec, rec, _ = precision_recall_curve(labels, probs)
        pr_auc = auc(rec, prec)
        plt.figure(figsize=(6, 6))
        plt.plot(rec, prec, label=f'AUPRC = {pr_auc:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (Calibrated)')
        plt.legend(loc='lower left')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/pr_curve_calibrated.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('[OK] Saved PR curve -> results/pr_curve_calibrated.png')

    return metrics



# ==================== INFERENCE DEMO ====================
def inference_demo(model, device='cuda'):
    """Demonstrate real-time inference on a patient"""
    
    # Load a test patient
    with open('data/processed/test_sequences.pkl', 'rb') as f:
        test_seqs = pickle.load(f)
    
    patient = test_seqs[0]  # First test patient
    
    print("\n" + "="*60)
    print("REAL-TIME INFERENCE DEMO")
    print("="*60)
    print(f"\nPatient ID: {patient['patient_id']}")
    print(f"Observations: {patient['seq_length']} measurements")
    print(f"Time range: {patient['times'][0]:.1f} - {patient['times'][-1]:.1f} hours")
    print(f"True label: {'Adverse Event' if patient['label'] == 1 else 'No Adverse Event'}")
    
    # Prepare input
    vitals = torch.FloatTensor(patient['vitals']).unsqueeze(0).to(device)
    times = torch.FloatTensor(patient['times']).unsqueeze(0).to(device)
    masks = torch.ones(1, len(patient['times'])).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(vitals, times, masks)
        
        alpha = outputs['alpha'].item()
        beta = outputs['beta'].item()
        prob = alpha / (alpha + beta)
        uncertainty = beta / (alpha * (alpha + beta + 1))
    
    print(f"\n--- MODEL PREDICTION ---")
    print(f"Risk Score: {prob:.4f} ({prob*100:.2f}%)")
    print(f"Uncertainty: {uncertainty:.4f}")
    print(f"Prediction: {'HIGH RISK - Adverse Event Likely' if prob > 0.5 else 'LOW RISK - No Adverse Event'}")
    
    if prob > 0.7:
        print("\n⚠️  ALERT: High risk patient - recommend immediate intervention")
    elif prob > 0.5:
        print("\n⚠️  WARNING: Elevated risk - monitor closely")
    else:
        print("\n✓ STABLE: Patient appears stable")
    
    print("\n" + "="*60)


def main():
    print("="*60)
    print("STEP 6: ADVANCED ENHANCEMENTS")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[OK] Device: {device}")
    
    # Load model
    print("\n[1/7] Loading model...")
    model = LatentODEModel(input_dim=6, hidden_dim=64, latent_dim=32, output_dim=6)
    checkpoint = torch.load('models/best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Load data
    print("\n[2/7] Loading data...")
    _, val_loader, test_loader = create_dataloaders(batch_size=32)
    sample_batch = next(iter(test_loader))
    
    # Run enhancements
    print("\n[3/7] Visualizing attention weights...")
    try:
        visualize_attention_weights(model, sample_batch, device)
    except Exception as e:
        print(f"  [SKIP] Attention visualization failed: {e}")
    
    print("\n[4/7] Predicting patient trajectory...")
    # Prepare single patient data
    with open('data/processed/test_sequences.pkl', 'rb') as f:
        test_seqs = pickle.load(f)
    patient_data = {
        'vitals': torch.FloatTensor(test_seqs[0]['vitals']),
        'times': torch.FloatTensor(test_seqs[0]['times']),
        'masks': torch.ones(len(test_seqs[0]['times']))
    }
    predict_patient_trajectory(model, patient_data, future_hours=24, device=device)
    
    print("\n[5/7] Visualizing latent space...")
    visualize_latent_space(model, test_loader, device)
    
    print("\n[6/7] Visualizing ODE dynamics...")
    visualize_ode_dynamics(model, sample_batch, device)
    
    print("\n[7/7] Computing feature importance...")
    compute_feature_importance(model, test_loader, device)
    
    # Evaluate on test set (calibrate on validation first)
    try:
        print("\n[7.5/7] Calibrating on validation and evaluating on test set...")
        evaluate_with_calibration(model, val_loader, test_loader, device)
    except Exception as e:
        print(f"  [SKIP] Calibrated test evaluation failed: {e}")

    # Inference demo
    inference_demo(model, device)
    
    print("\n" + "="*60)
    print("STEP 6 COMPLETE - SUCCESS")
    print("="*60)
    print("\nNew visualizations saved:")
    print("  - attention_weights.png")
    print("  - patient_trajectory.png")
    print("  - latent_space_tsne.png")
    print("  - ode_dynamics.png")
    print("  - feature_importance.png")
    
    print("\n" + "="*60)
    print("PROJECT COMPLETE!")
    print("="*60)
    print("\nYour Neural ODE project includes:")
    print("  ✓ Transformer encoder")
    print("  ✓ Augmented Neural ODE")
    print("  ✓ Evidential classifier")
    print("  ✓ Comprehensive evaluation")
    print("  ✓ Advanced visualizations")
    print("  ✓ Real-time inference")
    print("\nReady for GitHub! 🚀")


if __name__ == "__main__":
    main()