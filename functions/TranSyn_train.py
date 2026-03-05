"""
@description: General training
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
				save_path='./models/best_source_model.pt', 
				loss_path=None, 
				device='cuda',
				loss='bce',  
				optim_name='adam',
				trsr_trade_off=0.1,
				transfer=True):
	print(f"[Check] model on device: {next(model.parameters()).device}")
	# loss function
	if loss == 'bce':
		criterion = nn.BCEWithLogitsLoss()
	elif loss == 'focal':
		criterion = FocalLoss()
	else:
		raise ValueError(f"Unknown loss type: {loss}. Choose 'bce' or 'fl'.")
	# optimizer
	if optim_name == 'adamw':
		optimizer = optim.AdamW(
			filter(lambda p: p.requires_grad, model.parameters()), 
			lr=lr, 
		)
	else:
		optimizer = optim.Adam(
			filter(lambda p: p.requires_grad, model.parameters()), 
			lr=lr, 
		)
	best_val_loss = float('inf')
	epochs_without_improvement = 0
	best_epoch = 0
	best_model_state = model.state_dict()
	train_losses = []
	val_losses = []
	# start training
	for epoch in trange(epochs):
		model.train()
		running_loss = 0.0
		# train every batch
		for batch in train_loader:
			dna_feature = batch["gpn_msa"].to(device)
			rna_feature = batch["calm_diff"].to(device)
			bio_feature = batch["biological_feature"].to(device)
			labels = batch['label'].unsqueeze(1).to(device, dtype=torch.float32)
			optimizer.zero_grad()
			if transfer:
				outputs, dna_output, dna_noise_outputs, rna_output, rna_noise_outputs = model(dna_feature, rna_feature, bio_feature)
				cls_loss = criterion(outputs, labels)
				dna_noise_loss = torch.mean(torch.stack(dna_noise_outputs))
				rna_noise_loss = torch.mean(torch.stack(rna_noise_outputs))
				loss =  cls_loss + trsr_trade_off * dna_noise_loss + trsr_trade_off * rna_noise_loss
			else:
				outputs = model(dna_feature, rna_feature, bio_feature, train=False)
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
				dna_feature = val_batch["gpn_msa"].to(device)
				rna_feature = val_batch["calm_diff"].to(device)
				bio_feature = val_batch["biological_feature"].to(device)
				labels = val_batch['label'].unsqueeze(1).to(device, dtype=torch.float32)
				outputs = model(dna_feature, rna_feature, bio_feature, train=False)
				loss = criterion(outputs, labels)
				val_loss += loss.item()
		# calculate avg train & eval loss
		avg_val_loss = val_loss / len(val_loader)
		train_losses.append(avg_train_loss)
		val_losses.append(avg_val_loss)
		print(f"[INFO] Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
		# save best result
		if avg_val_loss < best_val_loss:
			best_val_loss = avg_val_loss
			best_epoch = epoch
			epochs_without_improvement = 0
			best_model_state = model.state_dict()
			torch.save(best_model_state, save_path)
		else:
			epochs_without_improvement += 1
			if epochs_without_improvement >= patience:
				print("[Warning] Early stopping due to no improvement")
				break
	# 4 reload best model from save path
	model.load_state_dict(torch.load(save_path))
	# 5 save the train loss information
	if loss_path is not None:
		losses_dict = {
			'train_losses': torch.tensor(train_losses),
			'val_losses': torch.tensor(val_losses),
		}
		torch.save(losses_dict, loss_path)
	return model, best_val_loss, best_epoch



def train_model_ablation(model,
						 train_loader, 
						 val_loader, 
						 epochs, 
						 lr, 
						 patience, 
						 save_path='./models/best_source_model.pt', 
						 loss_path=None, 
						 device='cuda',
						 loss='bce', 
						 optim_name='adam',
						 trsr_trade_off=0.1,
						 mode='dna'):
	"""
	Desc:
		train for ablation study
	"""
	print(f"[Check] model on device: {next(model.parameters()).device}")
	# loss function
	if loss == 'bce':
		criterion = nn.BCEWithLogitsLoss()
	elif loss == 'focal':
		criterion = FocalLoss()
	else:
		raise ValueError(f"Unknown loss type: {loss}. Choose 'bce' or 'fl'.")
	# optimizer
	if optim_name == 'adamw':
		optimizer = optim.AdamW(
			filter(lambda p: p.requires_grad, model.parameters()), 
			lr=lr, 
		)
	else:
		optimizer = optim.Adam(
			filter(lambda p: p.requires_grad, model.parameters()), 
			lr=lr, 
		)
	best_val_loss = float('inf')
	epochs_without_improvement = 0
	best_epoch = 0
	best_model_state = model.state_dict()
	train_losses = []
	val_losses = []
	# start training
	for epoch in trange(epochs):
		model.train()
		running_loss = 0.0
		# train every batch
		for batch in train_loader:
			dna_feature = batch["gpn_msa"].to(device)
			rna_feature = batch["calm_diff"].to(device)
			bio_feature = batch["biological_feature"].to(device)
			labels = batch['label'].unsqueeze(1).to(device, dtype=torch.float32)
			optimizer.zero_grad()

			if mode in {'dna', 'rna', 'dna+bio', 'rna+bio'}: 
				outputs, deep_repr, noise_outputs = model(dna_feature, rna_feature, bio_feature, train=True)
				cls_loss = criterion(outputs, labels)
				noise_loss = torch.mean(torch.stack(noise_outputs))
				loss =  cls_loss + trsr_trade_off * noise_loss

			elif mode in {'dna+rna', 'no_fusion'}: 
				outputs, dna_output, dna_noise_outputs, rna_output, rna_noise_outputs = model(dna_feature, rna_feature, bio_feature, train=True)
				cls_loss = criterion(outputs, labels)
				dna_noise_loss = torch.mean(torch.stack(dna_noise_outputs))
				rna_noise_loss = torch.mean(torch.stack(rna_noise_outputs))
				loss =  cls_loss + trsr_trade_off * dna_noise_loss + trsr_trade_off * rna_noise_loss
				
			elif mode == 'no_trsr': 
				outputs, dna_output, rna_output = model(dna_feature, rna_feature, bio_feature, train=True)
				cls_loss = criterion(outputs, labels)
				dna_bss_loss = bss_module(dna_output)
				rna_bss_loss = bss_module(rna_output)
				loss =  cls_loss + bss_trade_off * dna_bss_loss + bss_trade_off * rna_bss_loss

			else: # no_regularization | bio | no_transfer
				outputs = model(dna_feature, rna_feature, bio_feature, train=True)
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
				dna_feature = val_batch["gpn_msa"].to(device)
				rna_feature = val_batch["calm_diff"].to(device)
				bio_feature = val_batch["biological_feature"].to(device)
				labels = val_batch['label'].unsqueeze(1).to(device, dtype=torch.float32)
				outputs = model(dna_feature, rna_feature, bio_feature, train=False)
				loss = criterion(outputs, labels)
				val_loss += loss.item()
		# calculate avg train & eval loss
		avg_val_loss = val_loss / len(val_loader)
		train_losses.append(avg_train_loss)
		val_losses.append(avg_val_loss)
		print(f"[INFO] Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
		# save best result
		if avg_val_loss < best_val_loss:
			best_val_loss = avg_val_loss
			best_epoch = epoch
			epochs_without_improvement = 0
			best_model_state = model.state_dict()
			torch.save(best_model_state, save_path)
		else:
			epochs_without_improvement += 1
			if epochs_without_improvement >= patience:
				print("[Warning] Early stopping due to no improvement")
				break
	# 4 reload best model from save path
	model.load_state_dict(torch.load(save_path))
	# 5 save the train loss information
	if loss_path is not None:
		losses_dict = {
			'train_losses': torch.tensor(train_losses),
			'val_losses': torch.tensor(val_losses),
		}
		torch.save(losses_dict, loss_path)
	return model, best_val_loss, best_epoch