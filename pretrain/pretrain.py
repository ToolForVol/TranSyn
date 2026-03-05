"""
@desc: Code for pretrain the network
@author: Chen ye
@email: q23101020@stu.ahu.edu.cn
"""

# torch related pkgs
import torch
from torch.utils.data import DataLoader, ConcatDataset

# normal pkgs
import os
import sys
import yaml
import random
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# training settings
with open('./config/config.yaml', 'r', encoding='utf-8') as file:
	config = yaml.safe_load(file)['baseline']

# GPU settings (depend on your device)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Debug] Using device: {device}")
print(f"[Debug] CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"[Debug] torch.cuda.current_device() = {torch.cuda.current_device()}")
print(f"[Debug] torch.cuda.get_device_name(0) = {torch.cuda.get_device_name(0)}")

# customized codes
from functions.Dataset import MyDataset, LMDBDataset, concat_and_loader
from functions.ModelScaffold import model_fn, build_backbone
from functions.General_train import train_model
from functions.General_evaluate import test_model_detail, plot_losses_single
from functions.DyPositionTransformer import DyPositionTransformer
from functions.Utils import set_full_deterministic_seed

# control the random seed
SEED = 913
set_full_deterministic_seed(SEED)
feature_name = 'calm_diff_feature' # | 'gpn_msa'

# ====================== [Start] ======================

# 1 Laad in the target data (for zeroshot test)
print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Loading target data...")
TARGET_ROOT_PATH = "./data/target/"
tgt_test_bn = MyDataset(f"{TARGET_ROOT_PATH}/test_neg.pth", label=0)
tgt_test_pt = MyDataset(f"{TARGET_ROOT_PATH}/test_pos.pth", label=1)
tgt_test = ConcatDataset([tgt_test_bn, tgt_test_pt])
tgt_test_loader = DataLoader(tgt_test, batch_size=32, shuffle=False, num_workers=4)

# 2 Load in the source data (for pretrain)
print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Loading source data...")
SOURCE_ROOT_PATH = "./data/source/missense/" # NOTE depend on your path
# SOURCE_ROOT_PATH = "./data/source/noncoding/" # NOTE depend on your path
src_train_pt = LMDBDataset(
	lmdb_path=f"{SOURCE_ROOT_PATH}/pt/",
	feature_keys=[feature_name],
	preload=False,
    label=1
)
src_train_bn = LMDBDataset(
	lmdb_path=f"{SOURCE_ROOT_PATH}/bn/",
	feature_keys=[feature_name],
	preload=False,
    label=0
)
src_train_data = ConcatDataset([src_train_pt, src_train_bn])
src_train_loader, src_val_loader = concat_and_loader(src_train_data, split_rate=0.8, batch_size=64) # 20% data for early stop

# 3 where the output result are saved
SAVE_DIR = f"./result/pretrained/missense/"
# SAVE_DIR = f"./result/pretrained/noncoding/"
os.makedirs(SAVE_DIR, exist_ok=True)
src_path = os.path.join(SAVE_DIR, f"missense_Transformer.pt")
# src_path = os.path.join(SAVE_DIR, f"noncoding_ResNet1D.pt")
metric_path = src_path.replace(".pt", "_metric.txt")
save_preds_path = src_path.replace(".pt", "_pred.txt")

# 4 Build the model
my_model = DyPositionTransformer(
            input_dim=768,
            num_layers=4,
            d_model=256,
            num_heads=4,
            dff=512,
            dropout_rate=0.1,
            head_hidden=128,
    ).to(device)  
# my_model = model_fn(backbone_name="ResNet1D", seq_len=129, embed_dim=768).to(device)
# 5 Begin to train
print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Begin to train... ")
model, _, _ = train_model(my_model,
                          src_train_loader, 
                          src_val_loader, 
                          config['epochs'], 
                          float(config['lr']), 
                          config['patience'],
                          save_path=src_path, 
                          device='cuda',
                          strategy_mode='plain', # no transfer strategy is used
                          feature_name=feature_name)
# 6 Zeroshot evaluate
test_model_detail(model=model, 
                  dataloader=tgt_test_loader, 
                  savepath=metric_path, 
                  save_preds_path=save_preds_path, 
                  device=device, 
                  feature_name=feature_name)