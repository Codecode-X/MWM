"""
Step 3: Augmented Neural ODE with Transformer Encoder & Evidential Classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import math

# ==================== TRANSFORMER ENCODER ====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)

class TransformerEncoder(nn.Module):
    """Advanced set-based encoder for irregular time series"""
    def __init__(self, input_dim, hidden_dim, latent_dim, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim*4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x, mask):
        """
        x: (batch, seq_len, input_dim)
        mask: (batch, seq_len) - 1=real, 0=padding
        """
        x = self.input_proj(x)
        x = self.input_dropout(x)
        x = self.pos_encoding(x)
        
        # Transformer expects mask: True=ignore, False=attend
        attn_mask = (mask == 0)
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        
        # Pool over sequence (mean of non-padded)
        mask_expanded = mask.unsqueeze(-1)
        x_pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        
        mu = self.mu_layer(x_pooled)
        logvar = self.logvar_layer(x_pooled)
        return mu, logvar

# ==================== AUGMENTED NEURAL ODE ====================
class ODEFunc(nn.Module):
    """ODE dynamics function with augmented dimensions"""
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, t, z):
        """dz/dt = f(z, t)"""
        return self.net(z)

class AugmentedNeuralODE(nn.Module):
    """Neural ODE with augmented dimensions for better expressivity"""
    def __init__(self, latent_dim, hidden_dim, augment_dim=5):
        super().__init__()
        self.latent_dim = latent_dim
        self.augment_dim = augment_dim
        self.total_dim = latent_dim + augment_dim
        
        self.ode_func = ODEFunc(self.total_dim, hidden_dim)
    
    def forward(self, z0, t):
        """
        z0: (batch, latent_dim)
        t: (batch, n_timepoints) or scalar
        """
        batch_size = z0.shape[0]
        
        # Augment with zeros
        aug = torch.zeros(batch_size, self.augment_dim, device=z0.device)
        z0_aug = torch.cat([z0, aug], dim=-1)
        
        # Solve ODE
        if isinstance(t, torch.Tensor) and t.dim() > 0:
            t_unique = torch.unique(t.flatten()).sort()[0]
            z_t = odeint(self.ode_func, z0_aug, t_unique, method='dopri5', rtol=1e-3, atol=1e-4)
            z_t = z_t.permute(1, 0, 2)  # (batch, time, dim)
        else:
            z_t = odeint(self.ode_func, z0_aug, torch.tensor([0, t], device=z0.device), method='dopri5')
            z_t = z_t[-1]  # Take final state
        
        # Remove augmented dimensions
        return z_t[..., :self.latent_dim]

# ==================== DECODER ====================
class Decoder(nn.Module):
    """Decode latent state to vital signs"""
    def __init__(self, latent_dim, output_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, z):
        return self.net(z)

# ==================== EVIDENTIAL CLASSIFIER ====================
class EvidentialClassifier(nn.Module):
    """Evidential Deep Learning for uncertainty quantification"""
    def __init__(self, latent_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4)  # evidence for [alpha0, alpha1, beta, v]
        )
    
    def forward(self, z):
        """
        Returns evidence parameters for Beta distribution
        alpha: evidence for class 1
        beta: evidence for class 0
        """
        out = self.net(z)
        alpha = F.softplus(out[:, 0:1]) + 1  # >1
        beta = F.softplus(out[:, 1:2]) + 1   # >1
        return alpha, beta
    
    def predict(self, z):
        """Get probability and uncertainty"""
        alpha, beta = self.forward(z)
        prob = alpha / (alpha + beta)
        uncertainty = beta / (alpha * (alpha + beta + 1))
        return prob, uncertainty

# ==================== COMPLETE MODEL ====================
class LatentODEModel(nn.Module):
    """Complete Latent ODE architecture"""
    def __init__(self, input_dim=6, hidden_dim=64, latent_dim=32, output_dim=6, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Components
        self.encoder = TransformerEncoder(input_dim, hidden_dim, latent_dim, dropout=dropout)
        self.ode = AugmentedNeuralODE(latent_dim, hidden_dim, augment_dim=5)
        self.decoder = Decoder(latent_dim, output_dim, hidden_dim, dropout=dropout)
        self.classifier = EvidentialClassifier(latent_dim, hidden_dim, dropout=dropout)
    
    def reparameterize(self, mu, logvar):
        """VAE reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, vitals, times, mask, predict_horizon=12.0):
        """
        vitals: (batch, seq_len, input_dim)
        times: (batch, seq_len)
        mask: (batch, seq_len)
        """
        batch_size = vitals.shape[0]
        
        # Encode to latent
        mu, logvar = self.encoder(vitals, mask)
        z0 = self.reparameterize(mu, logvar)
        
        # Evolve through ODE
        t_final = times.max() + predict_horizon
        z_final = self.ode(z0, t_final)
        
        # Decode to vitals
        vitals_recon = self.decoder(z_final)
        
        # Classify
        alpha, beta = self.classifier(z_final)
        
        return {
            'vitals_recon': vitals_recon,
            'alpha': alpha,
            'beta': beta,
            'mu': mu,
            'logvar': logvar,
            'z0': z0,
            'z_final': z_final
        }
    
    def predict_trajectory(self, vitals, times, mask, future_times):
        """Predict full trajectory at specific timepoints"""
        mu, logvar = self.encoder(vitals, mask)
        z0 = self.reparameterize(mu, logvar)
        
        # Solve at all timepoints
        all_times = torch.cat([torch.tensor([0.0], device=vitals.device), future_times])
        z_trajectory = self.ode(z0, all_times)
        
        vitals_trajectory = self.decoder(z_trajectory)
        return vitals_trajectory

# ==================== LOSS FUNCTIONS ====================
def compute_elbo_loss(outputs, targets, masks, beta=0.01, lambda_ev=0.1):
    """
    ELBO loss with evidential classification
    """
    # Reconstruction loss (MSE on non-padded)
    vitals_recon = outputs['vitals_recon']
    
    # Expand masks to match targets if needed
    if masks.dim() == 1:
        masks = masks.unsqueeze(-1)
    
    recon_loss = F.mse_loss(vitals_recon * masks, 
                             targets * masks, 
                             reduction='sum') / masks.sum()
    
    # KL divergence
    mu, logvar = outputs['mu'], outputs['logvar']
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]
    
    # Evidential loss (negative log-likelihood)
    alpha, beta_param = outputs['alpha'], outputs['beta']
    # Simplified evidential loss
    ev_loss = -torch.mean(torch.log(alpha + beta_param))
    
    total_loss = recon_loss + beta * kl_loss + lambda_ev * ev_loss
    
    return {
        'total': total_loss,
        'recon': recon_loss,
        'kl': kl_loss,
        'evidential': ev_loss
    }

def compute_classification_loss(outputs, labels):
    """Binary cross-entropy for classification"""
    alpha, beta = outputs['alpha'], outputs['beta']
    prob = alpha / (alpha + beta)
    bce_loss = F.binary_cross_entropy(prob, labels, reduction='mean')
    return bce_loss


if __name__ == "__main__":
    print("="*60)
    print("STEP 3: MODEL ARCHITECTURE TEST")
    print("="*60)
    
    # Test model
    model = LatentODEModel(input_dim=6, hidden_dim=64, latent_dim=32, output_dim=6)
    
    # Dummy data
    batch_size, seq_len, input_dim = 16, 30, 6
    vitals = torch.randn(batch_size, seq_len, input_dim)
    times = torch.linspace(0, 48, seq_len).unsqueeze(0).repeat(batch_size, 1)
    mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    outputs = model(vitals, times, mask)
    
    print("\n[OK] Model initialized")
    print(f"  - Encoder: Transformer (4 heads, 2 layers)")
    print(f"  - ODE: Augmented Neural ODE (5 extra dims)")
    print(f"  - Decoder: MLP")
    print(f"  - Classifier: Evidential")
    
    print("\nOutput shapes:")
    for key, val in outputs.items():
        if isinstance(val, torch.Tensor):
            print(f"  - {key}: {val.shape}")
    
    # Test loss
    targets = torch.randn(batch_size, input_dim)
    labels = torch.randint(0, 2, (batch_size, 1)).float()
    
    loss_dict = compute_elbo_loss(outputs, targets, mask[:, -1])
    class_loss = compute_classification_loss(outputs, labels)
    
    print("\nLoss computation:")
    print(f"  - Total ELBO: {loss_dict['total'].item():.4f}")
    print(f"  - Reconstruction: {loss_dict['recon'].item():.4f}")
    print(f"  - KL Divergence: {loss_dict['kl'].item():.4f}")
    print(f"  - Classification: {class_loss.item():.4f}")
    
    print("\n" + "="*60)
    print("STEP 3 COMPLETE - SUCCESS")
    print("="*60)
    print("\nModel ready for training!")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")