"""
@description: Main training data for TranSyn
@author: chen ye
@email: q23101020@stu.ahu.edu.cn
"""

# normal pkgs 
import os
import yaml
import time
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# torch pkgs
import torch
from torch.utils.data import DataLoader, ConcatDataset
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # set depends on your device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Debug] Using device: {device}")
print(f"[Debug] CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"[Debug] torch.cuda.current_device() = {torch.cuda.current_device()}")
print(f"[Debug] torch.cuda.get_device_name(0) = {torch.cuda.get_device_name(0)}")

# customized functions
from functions.Dataset import MyDataset, concat_and_loader
from functions.TranSyn_train import train_model
from functions.TranSyn_evaluate import test_model_detail
from functions.Utils import set_full_deterministic_seed

from functions.TranSynAttention import TranSynAttention
from functions.ModelScaffold import model_fn, build_backbone
from functions.DyPositionTransformer import DyPositionTransformer

# load settings
with open('./config/config.yaml', 'r', encoding='utf-8') as file:
	config = yaml.safe_load(file)['transyn']

# ❗ settings to test sample effiency and transfer learning
# SEEDS = [1918, 1917, 74, 310, 913, 11, 7, 20, 616, 404] # repeat 10 times
# pcts = [0.02, 0.05, 0.1, 0.3, 0.5, 0.7, 0.8] # sample efficiency test
# transfer_controls = [True, False] # use pretrain weight or not

# pretrained weight path
# noncoding weight
source_dna_pretrain_path = "./model_weight/pretrained/noncoding_ResNet1D.pt"
# missense weight
source_rna_pretrain_path = "./model_weight/pretrained/missense_Transformer.pt"


print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Task begin...") 
for SEED in SEEDS:
	for TRANSFER in transfer_controls:
		for pct in pcts:
			set_full_deterministic_seed(SEED)

			# load target data
			TARGET_ROOT_PATH = "./data/target/"
			train_data_bn = MyDataset(data_path=f"{TARGET_ROOT_PATH}/train_neg.pth", label=0)
			train_data_pt = MyDataset(data_path=f"{TARGET_ROOT_PATH}/train_pos.pth", label=1)
			train_data = ConcatDataset([train_data_bn, train_data_pt])
			test_data_bn = MyDataset(data_path=f"{TARGET_ROOT_PATH}/test_neg.pth", label=0)
			test_data_pt = MyDataset(data_path=f"{TARGET_ROOT_PATH}/test_pos.pth", label=1)
			test_data = ConcatDataset([test_data_bn, test_data_pt])

			# ini task directory
			task_dir = f"./result/TranSyn/{'transfer' if TRANSFER else 'notransfer'}/SEED_{SEED}/labeled_p{int(pct*100)}"
			print(f"[INFO] Job inited at {task_dir}...")
			os.makedirs(task_dir, exist_ok=True)
			weight_path = f"{task_dir}/model.pt"
			metric_path = f"{task_dir}/metrics.txt"
			pred_path = f"{task_dir}/pred.txt"

			# split data
			train_loader, val_loader = concat_and_loader(train_data, split_rate=pct, batch_size=config['patience'], seed=SEED)
			test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)

			# ini module
			print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Init module skeleton...") 
			
			## DNA module
			tsa_dna_module = build_backbone(name="ResNet1D", seq_len=129, embed_dim=768)
			if TRANSFER:
				source_dna_module = model_fn(backbone_name="ResNet1D", seq_len=129, embed_dim=768).to(device)
				source_dna_module.load_state_dict(torch.load(source_dna_pretrain_path, map_location='cpu'))
				missing_keys, unexpected_keys = tsa_dna_module.load_state_dict(source_dna_module.backbone.state_dict(), strict=False)
				print("DNA backbone load_state_dict:")
				print("  Missing keys:", missing_keys)
				print("  Unexpected keys:", unexpected_keys)
			
			## RNA module
			tsa_rna_module = DyPositionTransformer(
									input_dim=768,
									num_layers=4,
									d_model=256,
									num_heads=4,
									dff=512,
									dropout_rate=0.1,
									head_hidden=128,
							).to(device) 
			if TRANSFER:
				state_dict = torch.load(source_rna_pretrain_path, map_location='cpu')
				missing_keys, unexpected_keys = tsa_rna_module.load_state_dict(state_dict, strict=False)
				print("RNA backbone load_state_dict:")
				print("  Missing keys:", missing_keys)
				print("  Unexpected keys:", unexpected_keys)

			## bio module
			tsa_bio_module = build_backbone(name="MLP", embed_dim=183, output_dim=config['bio_hidden']).to(device) 
			
			## final model
			my_model = TranSynAttention(
						dna_module = tsa_dna_module,
						rna_module = tsa_rna_module,
						bio_module = tsa_bio_module,
						hidden_layer = config['head_hidden'],
						dropout = config['dropout_rate']
					).to(device) 

			# training
			print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Start training...") 
			model, best_val_loss, best_epoch = train_model(
															model=my_model,
															train_loader=train_loader, 
															val_loader=val_loader, 
															epochs=config['epochs'], 
															lr=config['lr'], 
															patience=config['patience'],
															save_path=weight_path, 
															loss=config['loss_fn'],                    
															optim_name=config['optimizer'],                    
															trsr_trade_off=config['trade_off_trsr'],
															transfer=TRANSFER
														)

			# evaluation
			print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Finished! Evaludating...")
			test_model_detail(
								model=model, 
								dataloader=test_loader, 
								savepath=metric_path, 
								save_preds_path=pred_path, 
								device=device
							)
			print(f"[SUCCESS] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Task completed") 