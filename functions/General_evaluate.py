"""
@description: General testing functions
@author: chen ye
@email: q23101020@stu.ahu.edu.cn
"""


import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
	accuracy_score, precision_score, recall_score, f1_score,
	roc_auc_score, average_precision_score, classification_report
)


def test_model_detail(model, dataloader, savepath=None, save_preds_path=None, device='cuda', feature_name=None):
	"""
	Args:
		model, the final trained model
		dataloader, the test data loader
		savepath, the metric save path
		save_preds_path, the per sample prediction save path
		device, gpu/cpu
		feature_name, feature name, e.g. calm_diff | gpn_msa
	"""
	print("[INFO] Performing testing...")
	model.eval()
	all_preds = []
	all_probs = []
	all_labels = []
	all_vids = []
	with torch.no_grad():
		for batch in dataloader:
			vid= batch['vid']
			X = batch[feature_name].to(device)
			y = batch['label'].unsqueeze(1).to(device, dtype=torch.float32)
			logits = model(X)
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
	# 1 save prediction for every samples
	preds_df = pd.DataFrame({
		'vid': y_vids,
		'label': y_true.squeeze(),
		'pred_prob': y_prob,
		'pred_label': y_pred
	})
	if save_preds_path is not None:
		preds_df.to_csv(save_preds_path, sep='\t', index=False, float_format="%.4f")
	# 2 calculate metrics
	acc = round(accuracy_score(y_true, y_pred), 4)
	pre = round(precision_score(y_true, y_pred), 4)
	rec = round(recall_score(y_true, y_pred), 4)
	f1 = round(f1_score(y_true, y_pred), 4)
	auc = round(roc_auc_score(y_true, y_prob), 4)
	aupr = round(average_precision_score(y_true, y_prob), 4)
	df = pd.DataFrame({
		"Accuracy": [float(f"{acc:.4f}")],
		"Precision": [float(f"{pre:.4f}")],
		"Recall": [float(f"{rec:.4f}")],
		"F1-score": [float(f"{f1:.4f}")],
		"AUC": [float(f"{auc:.4f}")],
		"AUPR": [float(f"{aupr:.4f}")]
	})
	print(df.T)
	print("\nClassification Report:")
	print(classification_report(y_true, y_pred))
	if savepath is not None:
		df.to_csv(savepath, sep='\t', index=False, float_format="%.4f")
	return auc, aupr