"""
@description: General training functions
@author: chen ye
@email: q23101020@stu.ahu.edu.cn
"""


from functions.FocalLoss import FocalLoss
from functions.TranSyn_BSS import BatchSpectralShrinkage
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange


def train_model(model,
				train_loader, 
				val_loader, 
				epochs, 
				lr, 
				patience, 
				save_path=None, 
				loss_path=None, 
				device='cuda',
				weight_decay=None,
				strategy_mode='tsrs', 
				tsrs_trade_off=None,
				mute=False,
				loss='bce',  
				optim_name='adam',
				feature_name=None):
	print(f"[Check] model on device: {next(model.parameters()).device}")

	# loss function
	if loss == 'bce':
		criterion = nn.BCEWithLogitsLoss()
	elif loss == 'focal':
		criterion = FocalLoss()
	else:
		raise ValueError(f"Unknown loss type: {loss}. Choose 'bce' or 'focal'.")
	# optimizer
	if optim_name == 'adamw':
		optimizer = optim.AdamW(
			filter(lambda p: p.requires_grad, model.parameters()), 
			lr=lr, 
			weight_decay=weight_decay if weight_decay is not None else 0.0
		)
	else:
		optimizer = optim.Adam(
			filter(lambda p: p.requires_grad, model.parameters()), 
			lr=lr, 
			weight_decay=weight_decay if weight_decay is not None else 0.0
		)
	# best result
	best_val_loss = float('inf')
	epochs_without_improvement = 0
	best_model_state = model.state_dict()
	train_losses = []
	val_losses = []
	# start epoch
	for epoch in trange(epochs):
		model.train()
		running_loss = 0.0
		# train every batch
		for batch in train_loader:
			features = batch[feature_name].to(device)
			labels = batch['label'].unsqueeze(1).to(device, dtype=torch.float32)
			optimizer.zero_grad()
			if strategy_mode == 'tsrs':
				outputs, noise_outputs = model(features, use_tsrs=True)
				noise_loss = torch.mean(torch.stack(noise_outputs)) # calculate trsr loss
				cls_loss = criterion(outputs, labels)
				loss =  cls_loss + tsrs_trade_off * noise_loss
			else:
				outputs = model(features)
				loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
		avg_train_loss = running_loss / len(train_loader)
		# evaludate after every epoch
		model.eval()
		val_loss = 0.0
		with torch.no_grad():
			for val_batch in val_loader:
				features = val_batch[feature_name].to(device)
				labels = val_batch['label'].unsqueeze(1).to(device, dtype=torch.float32)
				outputs = model(features)
				loss = criterion(outputs, labels)
				val_loss += loss.item()
		# calculate avg train & eval loss
		avg_val_loss = val_loss / len(val_loader)
		train_losses.append(avg_train_loss)
		val_losses.append(avg_val_loss)
		if not mute:
			print(f"[INFO] Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
		# save best result
		if avg_val_loss < best_val_loss:
			best_val_loss = avg_val_loss
			epochs_without_improvement = 0
			best_model_state = model.state_dict()
			torch.save(best_model_state, save_path)
		else:
			epochs_without_improvement += 1
			if epochs_without_improvement >= patience:
				print("[Warning] Early stopping due to no improvement")
				break
	# reload best model from save path
	model.load_state_dict(torch.load(save_path))
	# save the train loss information
	if loss_path is not None:
		losses_dict = {
			'train_losses': torch.tensor(train_losses),
			'val_losses': torch.tensor(val_losses),
		}
		torch.save(losses_dict, loss_path)
	return model, train_losses, val_losses