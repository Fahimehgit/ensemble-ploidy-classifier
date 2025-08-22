#!/usr/bin/env python3
import os
import numpy as np
import torch
import json
import glob
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd

# Fixed indices
GLOBAL_TEST_LIST = [484, 22, 633, 594, 43, 623, 325, 248, 573, 543, 513, 515, 328, 20, 629, 496, 208, 286, 96, 420, 474, 270, 645, 280, 346, 460, 266, 104, 300, 129, 352, 145, 146]
GLOBAL_TRUE_VAL_LIST = [2, 171, 55, 506, 544, 407, 333, 46, 612, 371, 399, 546, 186, 422, 440, 419, 275, 249, 176, 250, 398, 3, 461, 362, 503, 175, 306, 521, 117, 550, 198, 447, 598]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels, indices):
        self.sequences = np.array(sequences)[indices.astype(int)]
        self.labels = np.array(labels)[indices.astype(int)]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return seq, label

def load_sequences_and_labels(json_file, json_dir):
    sequences, labels = [], []
    with open(os.path.join(json_dir, json_file), 'r') as f:
        data = json.load(f)
    for species, info in data.items():
        full_seq = [info['reference_tokens']] + info['auxiliary_tokens']
        sequences.append(full_seq)
        labels.append(info['label'])
    return sequences, labels

def evaluate_model(model, loader):
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch_sequences, batch_labels in loader:
            batch_sequences = batch_sequences.to(device)
            batch_labels = batch_labels.to(device)
            batch_sequences = batch_sequences.view(batch_sequences.size(0), 1, 351, -1)
            outputs = model(batch_sequences)
            y_true.extend(batch_labels.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())

    auc_roc = roc_auc_score(y_true, y_pred)
    acc = accuracy_score(y_true, np.round(y_pred))
    return auc_roc, acc

def main():
    print("🚀 Starting Probe Ranking with Fixed Indices")
    
    # Use the correct JSON directory path
    json_dir = '/orange/juannanzhou/ploidy/json_probe15/'
    
    print(f"🔍 Checking if directory exists: {json_dir}")
    print(f"Directory exists: {os.path.exists(json_dir)}")
    
    if not os.path.exists(json_dir):
        print(f"❌ JSON directory not found: {json_dir}")
        return
    
    print(f"📁 Using JSON directory: {json_dir}")
    
    # Find all JSON files
    all_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])
    print(f"📊 Found {len(all_files)} JSON files")
    
    # Find trained models
    models_dir = "organized_project/models"
    model_files = []
    if os.path.exists(models_dir):
        model_files = glob.glob(os.path.join(models_dir, "best_model_probe_*.pt"))
    print(f"🧠 Found {len(model_files)} trained model files")
    
    # Combine test and validation indices
    test_indices = np.array(GLOBAL_TEST_LIST + GLOBAL_TRUE_VAL_LIST, dtype=int)
    
    results = []
    
    for probe_idx, json_file in enumerate(all_files[:5]):  # Test with first 5
        print(f"\n🔍 Processing probe {probe_idx} - {json_file}")
        
        try:
            sequences, labels = load_sequences_and_labels(json_file, json_dir)
            
            max_index = len(sequences) - 1
            valid_test_indices = test_indices[test_indices <= max_index]
            
            if len(valid_test_indices) == 0:
                print(f"  ⚠️  No valid test indices for probe {probe_idx}")
                continue
            
            test_dataset = CustomDataset(sequences, labels, valid_test_indices)
            test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
            
            # Find and load model
            model_loaded = False
            for model_file in model_files:
                if f"probe_{probe_idx}" in model_file:
                    try:
                        model = torch.load(model_file, map_location=device)
                        model = model.to(device)
                        model.eval()
                        model_loaded = True
                        print(f"  ✅ Loaded model: {model_file}")
                        break
                    except Exception as e:
                        print(f"  ❌ Failed to load {model_file}: {e}")
                        continue
            
            if not model_loaded:
                print(f"  ⚠️  No trained model found for probe {probe_idx}")
                continue
            
            test_auc_roc, test_acc = evaluate_model(model, test_loader)
            print(f"  📊 Test AUC-ROC: {test_auc_roc:.4f}, Test Accuracy: {test_acc:.4f}")
            
            results.append({
                'probe_idx': probe_idx,
                'json_file': json_file,
                'test_auc_roc': test_auc_roc,
                'test_accuracy': test_acc
            })
            
        except Exception as e:
            print(f"  ❌ Error processing probe {probe_idx}: {e}")
            continue
    
    if results:
        df = pd.DataFrame(results)
        ranked_df = df.sort_values(by=['test_accuracy', 'test_auc_roc'], ascending=[False, False])
        
        os.makedirs('organized_project/results', exist_ok=True)
        ranked_df.to_csv('organized_project/results/probe_rankings_test.csv', index=False)
        
        print(f"\n🏆 Top Probes by Test Accuracy:")
        for i, row in ranked_df.iterrows():
            print(f"  {i+1}. Probe {row['probe_idx']} - Accuracy: {row['test_accuracy']:.4f}, AUC-ROC: {row['test_auc_roc']:.4f}")
        
        print(f"\n✅ Results saved to organized_project/results/probe_rankings_test.csv")
    else:
        print("❌ No results obtained!")

if __name__ == "__main__":
    main()
