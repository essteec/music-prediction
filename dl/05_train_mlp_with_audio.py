"""
Train MLP with Audio Embeddings (Phase 3)

Compare different audio embedding techniques:
- Baseline (Phase 1B): 798 features (23 audio + 5 text + 2 sentiment + 768 MPNet)
- + VGGish (128d)
- + Mel Stats (512d)
- + MERT (768d)
- + PANNs (2048d)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
from models import MusicMLP
from metrics import compute_metrics, print_metrics


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def align_audio_embeddings(split_ids, emb_ids, emb_data):
    """Align audio embeddings to the split IDs. Use zeros for missing."""
    emb_dim = emb_data.shape[1]
    aligned = np.zeros((len(split_ids), emb_dim), dtype=np.float32)
    
    # Create a mapping from ID to index in the embedding array
    # emb_ids is a numpy array of strings
    emb_id_to_idx = {str(id_val): idx for idx, id_val in enumerate(emb_ids)}
    
    found_count = 0
    for idx, split_id in enumerate(split_ids):
        split_id_str = str(split_id)
        if split_id_str in emb_id_to_idx:
            aligned[idx] = emb_data[emb_id_to_idx[split_id_str]]
            found_count += 1
            
    print(f"      Matched {found_count}/{len(split_ids)} ({found_count/len(split_ids)*100:.2f}%)")
    return aligned


def load_data_with_audio(audio_model, batch_size=256, num_workers=4):
    """Load data with MPNet + Audio embeddings."""
    print(f"\nLoading features with {audio_model} audio embeddings:")
    
    feat_dir = Path('ml/features')
    emb_dir = Path('data/embeddings')
    audio_emb_dir = emb_dir / 'audio'
    proc_dir = Path('data/processed')
    
    # Define audio model specifics
    audio_config = {
        'vggish': ('vggish_embeddings_128d', 128),
        'mel_stats': ('mel_stats_embeddings_512d', 512),
        'mert': ('mert_embeddings_768d', 768),
        'panns': ('panns_embeddings_2048d', 2048),
        'all': ('all', 128+512+768+2048)
    }
    
    if audio_model != 'none':
        if audio_model == 'all':
            audio_embs = {}
            for m in ['vggish', 'mel_stats', 'mert', 'panns']:
                prefix = audio_config[m][0]
                print(f"  Loading {m} embeddings...")
                emb_data = np.load(audio_emb_dir / f"{prefix}.npy", mmap_mode='r')
                emb_ids = np.load(audio_emb_dir / f"{prefix}_ids.npy")
                audio_embs[m] = (emb_data, emb_ids)
        else:
            prefix = audio_config[audio_model][0]
            print(f"  Loading {audio_model} embeddings...")
            emb_data = np.load(audio_emb_dir / f"{prefix}.npy", mmap_mode='r')
            emb_ids = np.load(audio_emb_dir / f"{prefix}_ids.npy")
    
    splits = {}
    for split_name in ['train', 'val', 'test']:
        print(f"\n  {split_name.capitalize()} split:")
        
        # Load base features
        audio = np.load(feat_dir / f'X_{split_name}_audio.npy')
        text = np.load(feat_dir / f'X_{split_name}_text_stats.npy')
        sentiment = np.load(feat_dir / f'X_{split_name}_sentiment.npy')
        mpnet = np.load(emb_dir / f'mpnet_lyrics_768d_{split_name}.npy')
        
        # Load split IDs
        df_split = pd.read_csv(proc_dir / f'{split_name}.csv', usecols=['id'])
        split_ids = df_split['id'].values
        
        features_list = [audio, text, sentiment, mpnet]
        
        if audio_model != 'none':
            if audio_model == 'all':
                for m in ['vggish', 'mel_stats', 'mert', 'panns']:
                    print(f"    Aligning {m}...")
                    m_data, m_ids = audio_embs[m]
                    aligned_audio = align_audio_embeddings(split_ids, m_ids, m_data)
                    features_list.append(aligned_audio)
            else:
                print(f"    Aligning {audio_model}...")
                aligned_audio = align_audio_embeddings(split_ids, emb_ids, emb_data)
                features_list.append(aligned_audio)
        
        # Concatenate all features
        X = np.hstack(features_list)
        
        # Load targets
        y_valence = np.load(feat_dir / f'y_{split_name}_valence.npy')
        y_energy = np.load(feat_dir / f'y_{split_name}_energy.npy')
        y_dance = np.load(feat_dir / f'y_{split_name}_danceability.npy')
        y_pop = np.load(feat_dir / f'y_{split_name}_popularity.npy')
        
        # Stack targets: [valence, energy, danceability, popularity]
        y = np.stack([y_valence, y_energy, y_dance, y_pop], axis=1)
        
        print(f"    Total features: {X.shape[1]}")
        print(f"    Targets: {y.shape}")
        
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        shuffle = (split_name == 'train')
        loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            generator=torch.Generator().manual_seed(42)
        )
        splits[split_name] = loader
        
    return splits['train'], splits['val'], splits['test'], X.shape[1]


def train_epoch(model, train_loader, criterion, optimizer, device, weights=None):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        if weights is not None:
            loss = (loss * weights).mean()
        else:
            loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(model, data_loader, criterion, device, weights=None):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            if weights is not None:
                loss = (loss * weights).mean()
            else:
                loss = loss.mean()
            total_loss += loss.item()
            all_predictions.append(predictions.cpu())
            all_targets.append(y_batch.cpu())
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    avg_loss = total_loss / len(data_loader)
    metrics = compute_metrics(targets, predictions)
    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_model', type=str, choices=['none', 'vggish', 'mel_stats', 'mert', 'panns', 'all'], required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()
    
    set_seed(42)
    print("=" * 70)
    print(f"Phase 3: Training MLP with {args.audio_model} Audio Embeddings")
    print("=" * 70)
    
    config = {
        'seed': 42,
        'batch_size': args.batch_size,
        'learning_rate': 0.001,
        'num_epochs': args.epochs,
        'patience': args.patience,
        'dropout': 0.5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'audio_model': args.audio_model
    }
    
    train_loader, val_loader, test_loader, input_size = load_data_with_audio(
        audio_model=args.audio_model,
        batch_size=config['batch_size'],
        num_workers=4
    )
    config['input_size'] = input_size
    
    print("\n" + "=" * 70)
    print("Creating Model")
    print("=" * 70)
    device = torch.device(config['device'])
    model = MusicMLP(input_size=input_size, num_targets=4, dropout=config['dropout'])
    model = model.to(device)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Input size: {input_size}")
    print(f"Parameters: {model.count_parameters():,}")
    
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    loss_weights = torch.tensor([2.0, 1.0, 2.0, 0.5]).to(device)
    
    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['num_epochs']):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, loss_weights)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, loss_weights)
        
        val_r2_avg = np.mean([m['r2'] for m in val_metrics.values()])
        print(f"Epoch {epoch+1:3d}/{config['num_epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val R²: {val_r2_avg:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_path = f'models/checkpoints/mlp_audio_{args.audio_model}_best.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'config': config
            }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
                
    print("\n" + "=" * 70)
    print("Final Evaluation on Test Set")
    print("=" * 70)
    checkpoint = torch.load(f'models/checkpoints/mlp_audio_{args.audio_model}_best.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, test_metrics = evaluate(model, test_loader, criterion, device, loss_weights)
    print_metrics(test_metrics, "Test Metrics")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'results/dl_metrics/mlp_audio_{args.audio_model}_{timestamp}.csv'
    
    all_metrics = []
    for target, values in test_metrics.items():
        all_metrics.append({
            'timestamp': timestamp,
            'model': f'MLP_Audio_{args.audio_model}',
            'phase': '3',
            'split': 'test',
            'target': target,
            'input_features': input_size,
            **values
        })
    df = pd.DataFrame(all_metrics)
    df.to_csv(results_file, index=False)
    print(f"\n✓ Results saved to: {results_file}")

if __name__ == '__main__':
    main()
