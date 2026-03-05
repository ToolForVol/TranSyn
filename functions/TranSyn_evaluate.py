"""
@description: TranSyn testing
@author: chen ye
@email: q23101020@stu.ahu.edu.cn
"""


import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report,
    fbeta_score, matthews_corrcoef
)


def direct_inference(model, dataloader, save_preds_path=None, device='cuda'):
    """
    Desc:
        Direct inference without metric calculated
    """
    print("[INFO] Performing inference...")
    model.eval()
    all_preds, all_probs, all_vids = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            vid = batch['vid']
            dna_feature = batch["gpn_msa"].to(device)
            rna_feature = batch["calm_diff"].to(device)
            bio_feature = batch["biological_feature"].to(device)

            logits = model(dna_feature, rna_feature, bio_feature, train=False)
            probs = torch.sigmoid(logits).squeeze(1)
            preds = (probs > 0.5).long()

            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
            all_vids.append(vid)

    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()
    y_vids = [x for sub in all_vids for x in sub]

    # save pred result for every sample
    preds_df = pd.DataFrame({
        'vid': y_vids,
        'pred_prob': y_prob,
        'pred_label': y_pred
    })
    if save_preds_path is not None:
        preds_df.to_csv(save_preds_path, sep='\t', index=False, float_format="%.4f")
    return preds_df


def test_model_detail(model, dataloader, savepath=None, save_preds_path=None, device='cuda', feature_name=None):
    """
    Desc:
        Inference with metric calculated
    """
    print("[INFO] Performing testing...")
    model.eval()
    all_preds, all_probs, all_labels, all_vids = [], [], [], []

    with torch.no_grad():
        for batch in dataloader:
            vid = batch['vid']
            dna_feature = batch["gpn_msa"].to(device)
            rna_feature = batch["calm_diff"].to(device)
            bio_feature = batch["biological_feature"].to(device)
            y = batch['label'].unsqueeze(1).to(device, dtype=torch.float32)

            logits = model(dna_feature, rna_feature, bio_feature, train=False)
            probs = torch.sigmoid(logits).squeeze(1)
            preds = (probs > 0.5).long()

            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
            all_labels.append(y.cpu())
            all_vids.append(vid)

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()
    y_vids = [x for sub in all_vids for x in sub]

    # save pred result for every sample
    preds_df = pd.DataFrame({
        'vid': y_vids,
        'label': y_true.squeeze(),
        'pred_prob': y_prob,
        'pred_label': y_pred
    })
    if save_preds_path is not None:
        preds_df.to_csv(save_preds_path, sep='\t', index=False, float_format="%.4f")

    # calculate binary classification metrics
    acc  = round(accuracy_score(y_true, y_pred), 4)
    pre  = round(precision_score(y_true, y_pred), 4)
    rec  = round(recall_score(y_true, y_pred), 4)
    f1   = round(f1_score(y_true, y_pred), 4)
    f2   = round(fbeta_score(y_true, y_pred, beta=2), 4)    
    mcc  = round(matthews_corrcoef(y_true, y_pred), 4)       
    auc  = round(roc_auc_score(y_true, y_prob), 4)
    aupr = round(average_precision_score(y_true, y_prob), 4)

    df = pd.DataFrame({
        "Accuracy":  [float(f"{acc:.4f}")],
        "Precision": [float(f"{pre:.4f}")],
        "Recall":    [float(f"{rec:.4f}")],
        "F1-score":  [float(f"{f1:.4f}")],
        "F2-score":  [float(f"{f2:.4f}")],  # ★
        "MCC":       [float(f"{mcc:.4f}")], # ★
        "AUC":       [float(f"{auc:.4f}")],
        "AUPR":      [float(f"{aupr:.4f}")]
    })

    print(df.T)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    if savepath is not None:
        df.to_csv(savepath, sep='\t', index=False, float_format="%.4f")

    return auc, aupr, f2, mcc



def test_model_ablation(model, 
                        dataloader, 
                        savepath=None, 
                        save_preds_path=None, 
                        device='cuda', 
                        feature_name=None,
                        mode='dna'):
    """
    Desc:
        Inference for ablation study
    """
    print("[INFO] Performing testing...")
    model.eval()
    all_preds, all_probs, all_labels, all_vids = [], [], [], []

    with torch.no_grad():
        for batch in dataloader:
            vid = batch['vid']
            dna_feature = batch["gpn_msa"].to(device)
            rna_feature = batch["calm_diff"].to(device)
            bio_feature = batch["biological_feature"].to(device)
            y = batch['label'].unsqueeze(1).to(device, dtype=torch.float32)
            logits = model(dna_feature, rna_feature, bio_feature, train=False)
                 
            probs = torch.sigmoid(logits).squeeze(1)
            preds = (probs > 0.5).long()

            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
            all_labels.append(y.cpu())
            all_vids.append(vid)

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()
    y_vids = [x for sub in all_vids for x in sub]

    preds_df = pd.DataFrame({
        'vid': y_vids,
        'label': y_true.squeeze(),
        'pred_prob': y_prob,
        'pred_label': y_pred
    })
    if save_preds_path is not None:
        preds_df.to_csv(save_preds_path, sep='\t', index=False, float_format="%.4f")

    acc  = round(accuracy_score(y_true, y_pred), 4)
    pre  = round(precision_score(y_true, y_pred), 4)
    rec  = round(recall_score(y_true, y_pred), 4)
    f1   = round(f1_score(y_true, y_pred), 4)
    f2   = round(fbeta_score(y_true, y_pred, beta=2), 4)   
    mcc  = round(matthews_corrcoef(y_true, y_pred), 4)     
    auc  = round(roc_auc_score(y_true, y_prob), 4)
    aupr = round(average_precision_score(y_true, y_prob), 4)

    df = pd.DataFrame({
        "Accuracy":  [float(f"{acc:.4f}")],
        "Precision": [float(f"{pre:.4f}")],
        "Recall":    [float(f"{rec:.4f}")],
        "F1-score":  [float(f"{f1:.4f}")],
        "F2-score":  [float(f"{f2:.4f}")], 
        "MCC":       [float(f"{mcc:.4f}")], 
        "AUC":       [float(f"{auc:.4f}")],
        "AUPR":      [float(f"{aupr:.4f}")]
    })

    print(df.T)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    if savepath is not None:
        df.to_csv(savepath, sep='\t', index=False, float_format="%.4f")

    return auc, aupr, f2, mcc


def plot_losses_multiple(all_train_losses, all_val_losses, save_path='./models/kfold_loss_plot.png'):
	plt.figure(figsize=(12, 6))
	for i, (train, val) in enumerate(zip(all_train_losses, all_val_losses)):
		plt.plot(train, label=f'Fold {i+1} Train Loss', color='#c71e1d', linestyle='-')
		plt.plot(val, label=f'Fold {i+1} Val Loss', color='#18a1cd', linestyle='--')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title('Train / Val Loss per Fold')
	plt.legend()
	plt.tight_layout()
	plt.savefig(save_path)
	plt.close()

def plot_losses_single(train_losses, val_losses, save_path='./models/loss_plot.png'):
	plt.figure(figsize=(12, 6))
	plt.plot(train_losses, label='Train Loss', color='#c71e1d', linestyle='-')
	plt.plot(val_losses, label=f'Val Loss', color='#18a1cd', linestyle='--')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title('Train / Val Loss per Fold')
	plt.legend()
	plt.tight_layout()
	plt.savefig(save_path)
	plt.close()