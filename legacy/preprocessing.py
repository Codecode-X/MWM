"""
Neural ODE for ICU Patient Monitoring - Step 2: Data Preprocessing
Advanced preprocessing for irregular time series with missing data
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path

class ICUDataPreprocessor:
    """Advanced preprocessing for irregular ICU time series"""
    
    def __init__(self, data_path='data/raw/synthetic_icu_data.csv'):
        self.data_path = data_path
        self.vital_columns = ['heart_rate', 'sbp', 'dbp', 'spo2', 'temperature', 'respiratory_rate']
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load raw ICU data"""
        df = pd.read_csv(self.data_path)
        print(f"[OK] Loaded {len(df)} measurements from {df['patient_id'].nunique()} patients")
        return df
    
    def normalize_vitals(self, df):
        """Normalize vital signs using z-score normalization"""
        # Deprecated: kept for backward compatibility but not used in pipeline
        df_copy = df.copy()
        df_copy[self.vital_columns] = self.scaler.transform(df_copy[self.vital_columns])
        print("[OK] Normalized vital signs (z-score) - using existing scaler")
        return df_copy
    
    def create_sequences(self, df):
        """
        Convert irregular time series to sequence format
        Returns: List of patient sequences with variable lengths
        """
        sequences = []
        patient_ids = df['patient_id'].unique()
        
        for pid in patient_ids:
            patient_data = df[df['patient_id'] == pid].sort_values('time_hours')
            
            # Extract features
            times = patient_data['time_hours'].values
            vitals = patient_data[self.vital_columns].values
            label = patient_data['adverse_event'].iloc[0]
            
            sequences.append({
                'patient_id': pid,
                'times': times,
                'vitals': vitals,
                'label': label,
                'seq_length': len(times)
            })
        
        print(f"[OK] Created {len(sequences)} patient sequences")
        print(f"    - Avg length: {np.mean([s['seq_length'] for s in sequences]):.1f} measurements")
        print(f"    - Min/Max length: {min([s['seq_length'] for s in sequences])}/{max([s['seq_length'] for s in sequences])}")
        
        return sequences
    
    def add_time_features(self, sequences):
        """
        Add advanced temporal features:
        - Time deltas (irregular sampling intervals)
        - Time since admission
        - Sine/cosine encoding for periodicity
        """
        for seq in sequences:
            times = seq['times']
            
            # Time deltas (intervals between measurements)
            deltas = np.diff(times, prepend=0)
            
            # Normalize time to [0, 1]
            time_normalized = times / times[-1] if times[-1] > 0 else times
            
            # Periodic encoding (24-hour cycle)
            time_sin = np.sin(2 * np.pi * times / 24)
            time_cos = np.cos(2 * np.pi * times / 24)
            
            seq['time_deltas'] = deltas
            seq['time_normalized'] = time_normalized
            seq['time_sin'] = time_sin
            seq['time_cos'] = time_cos
        
        print("[OK] Added temporal features (deltas, periodic encoding)")
        return sequences
    
    def split_data(self, sequences, train_ratio=0.7, val_ratio=0.15, random_state=42):
        """Split into train/val/test sets"""
        n = len(sequences)
        indices = np.arange(n)
        
        # Stratified split by label
        labels = [s['label'] for s in sequences]
        
        train_idx, temp_idx = train_test_split(
            indices, train_size=train_ratio, random_state=random_state, stratify=labels
        )
        
        temp_labels = [labels[i] for i in temp_idx]
        val_ratio_adjusted = val_ratio / (1 - train_ratio)
        
        val_idx, test_idx = train_test_split(
            temp_idx, train_size=val_ratio_adjusted, random_state=random_state, stratify=temp_labels
        )
        
        train_seqs = [sequences[i] for i in train_idx]
        val_seqs = [sequences[i] for i in val_idx]
        test_seqs = [sequences[i] for i in test_idx]
        
        print(f"[OK] Data split: Train={len(train_seqs)}, Val={len(val_seqs)}, Test={len(test_seqs)}")
        print(f"    - Train adverse rate: {np.mean([s['label'] for s in train_seqs]):.2%}")
        print(f"    - Val adverse rate: {np.mean([s['label'] for s in val_seqs]):.2%}")
        print(f"    - Test adverse rate: {np.mean([s['label'] for s in test_seqs]):.2%}")
        
        return train_seqs, val_seqs, test_seqs

    def fit_and_apply_scaler(self, train_seqs, val_seqs, test_seqs):
        """
        Fit the scaler on training sequences only and apply to all splits.
        This prevents data leakage from validation/test into the normalization.
        """
        # Collect all vitals from training sequences
        all_train_vitals = np.vstack([s['vitals'] for s in train_seqs]) if len(train_seqs) > 0 else np.empty((0, len(self.vital_columns)))

        if len(all_train_vitals) > 0:
            self.scaler.fit(all_train_vitals)
        else:
            # Fallback: fit on all data if train empty (shouldn't happen)
            all_vitals = np.vstack([s['vitals'] for s in (train_seqs + val_seqs + test_seqs)])
            self.scaler.fit(all_vitals)

        # Apply transform in-place
        for seqs in (train_seqs, val_seqs, test_seqs):
            for s in seqs:
                s['vitals'] = self.scaler.transform(s['vitals'])

        print('[OK] Fitted scaler on training set and normalized all sequences')
    
    def save_processed_data(self, train_seqs, val_seqs, test_seqs):
        """Save preprocessed sequences"""
        output_dir = Path('data/processed')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'train_sequences.pkl', 'wb') as f:
            pickle.dump(train_seqs, f)
        
        with open(output_dir / 'val_sequences.pkl', 'wb') as f:
            pickle.dump(val_seqs, f)
        
        with open(output_dir / 'test_sequences.pkl', 'wb') as f:
            pickle.dump(test_seqs, f)
        
        with open(output_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save metadata
        metadata = {
            'n_features': len(self.vital_columns),
            'vital_columns': self.vital_columns,
            'train_size': len(train_seqs),
            'val_size': len(val_seqs),
            'test_size': len(test_seqs)
        }
        
        with open(output_dir / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"[OK] Saved processed data to {output_dir}")
    
    def process_pipeline(self):
        """Complete preprocessing pipeline"""
        print("="*60)
        print("STEP 2: DATA PREPROCESSING PIPELINE")
        print("="*60 + "\n")
        
        # Load
        df = self.load_data()

        # Create sequences from raw (unnormalized) data
        sequences = self.create_sequences(df)

        # Add temporal features
        sequences = self.add_time_features(sequences)

        # Split into train/val/test
        train_seqs, val_seqs, test_seqs = self.split_data(sequences)

        # Fit scaler on train and normalize all splits (prevents leakage)
        self.fit_and_apply_scaler(train_seqs, val_seqs, test_seqs)

        # Save processed data
        self.save_processed_data(train_seqs, val_seqs, test_seqs)
        
        print("\n" + "="*60)
        print("STEP 2 COMPLETE - SUCCESS")
        print("="*60)
        print("\nProcessed data ready for Neural ODE training!")
        
        return train_seqs, val_seqs, test_seqs


class ICUDataset(Dataset):
    """PyTorch Dataset for irregular time series"""
    
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Convert to tensors
        times = torch.FloatTensor(seq['times'])
        vitals = torch.FloatTensor(seq['vitals'])
        time_deltas = torch.FloatTensor(seq['time_deltas'])
        label = torch.FloatTensor([seq['label']])
        
        return {
            'patient_id': seq['patient_id'],
            'times': times,
            'vitals': vitals,
            'time_deltas': time_deltas,
            'label': label,
            'seq_length': seq['seq_length']
        }


def collate_fn(batch):
    """
    Custom collate function for variable-length sequences
    Pads sequences to max length in batch
    """
    max_len = max([item['seq_length'] for item in batch])
    batch_size = len(batch)
    n_features = batch[0]['vitals'].shape[1]
    
    # Initialize padded tensors
    times_padded = torch.zeros(batch_size, max_len)
    vitals_padded = torch.zeros(batch_size, max_len, n_features)
    time_deltas_padded = torch.zeros(batch_size, max_len)
    masks = torch.zeros(batch_size, max_len)  # 1 = real data, 0 = padding
    labels = torch.zeros(batch_size, 1)
    
    for i, item in enumerate(batch):
        seq_len = item['seq_length']
        times_padded[i, :seq_len] = item['times']
        vitals_padded[i, :seq_len, :] = item['vitals']
        time_deltas_padded[i, :seq_len] = item['time_deltas']
        masks[i, :seq_len] = 1
        labels[i] = item['label']
    
    return {
        'times': times_padded,
        'vitals': vitals_padded,
        'time_deltas': time_deltas_padded,
        'masks': masks,
        'labels': labels
    }


def create_dataloaders(batch_size=32):
    """Create train/val/test dataloaders"""
    
    # Load preprocessed data
    with open('data/processed/train_sequences.pkl', 'rb') as f:
        train_seqs = pickle.load(f)
    with open('data/processed/val_sequences.pkl', 'rb') as f:
        val_seqs = pickle.load(f)
    with open('data/processed/test_sequences.pkl', 'rb') as f:
        test_seqs = pickle.load(f)
    
    # Create datasets
    train_dataset = ICUDataset(train_seqs)
    val_dataset = ICUDataset(val_seqs)
    test_dataset = ICUDataset(test_seqs)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    print(f"[OK] Created dataloaders (batch_size={batch_size})")
    print(f"    - Train batches: {len(train_loader)}")
    print(f"    - Val batches: {len(val_loader)}")
    print(f"    - Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Run preprocessing
    preprocessor = ICUDataPreprocessor()
    train_seqs, val_seqs, test_seqs = preprocessor.process_pipeline()
    
    # Test dataloader
    print("\n" + "="*60)
    print("Testing DataLoader...")
    print("="*60)
    train_loader, val_loader, test_loader = create_dataloaders(batch_size=16)
    
    # Show sample batch
    batch = next(iter(train_loader))
    print("\nSample Batch:")
    print(f"  - Times shape: {batch['times'].shape}")
    print(f"  - Vitals shape: {batch['vitals'].shape}")
    print(f"  - Masks shape: {batch['masks'].shape}")
    print(f"  - Labels shape: {batch['labels'].shape}")
    print(f"  - Positive samples: {batch['labels'].sum().item():.0f}/{len(batch['labels'])}")