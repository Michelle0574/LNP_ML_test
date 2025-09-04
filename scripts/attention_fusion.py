# scripts/attention_fusion.py
import os
import json
import math
import numpy as np
# Compat for RDKit on NumPy>=1.20
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'):   np.int = int
if not hasattr(np, 'bool'):  np.bool = bool

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.info')
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.error')
import pandas as pd
from typing import List, Dict, Tuple, Optional
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from smiles_features import load_smiles_features_npz, _calc_one

def _device_of(t: torch.Tensor) -> torch.device:
	return t.device if isinstance(t, torch.Tensor) else torch.device('cpu')

def load_attention_config(cfg_path: Optional[str]) -> dict:
	# Defaults
	cfg = {
		'd_model': 128,
		'nhead': 4,
		'enc_layers': 1,
		'cross_layers': 2,
		'per_dim_tokens_thresh': 64,
		'lr': 1e-4,
		'weight_decay': 1e-4,
		'lambda_clf': 0.1
	}
	if cfg_path and os.path.exists(cfg_path):
		try:
			user = json.load(open(cfg_path, 'r', encoding='utf-8'))
			for k, v in user.items():
				cfg[k] = v
		except Exception:
			pass
	return cfg

class SmilesVectorizer(nn.Module):
	"""
	Build attention-ready tokens for the chemical structure group:
	- desc (standardized) -> Linear -> d_model
	- morgan bits (0/1)   -> Linear -> d_model
	- maccs bits (0/1)    -> Linear -> d_model
	Returns tokens shape [T=3, d_model]
	"""
	def __init__(self, npz_path: str, d_model: int = 128):
		super().__init__()
		if not os.path.exists(npz_path):
			raise FileNotFoundError(f"smiles features cache not found: {npz_path}")
		sm2feat = load_smiles_features_npz(npz_path)
		self.sm2feat = sm2feat
		# Fit standardization for descriptors on cached data
		desc_mat = np.stack([v['desc'] for v in sm2feat.values()], axis=0).astype(np.float64)
		desc_mat = np.nan_to_num(desc_mat, nan=0.0, posinf=1e6, neginf=-1e6)
		desc_mat = np.clip(desc_mat, -1e9, 1e9).astype(np.float32)
		d_desc = desc_mat.shape[1]
		self.register_buffer('desc_mean', torch.from_numpy(desc_mat.mean(axis=0)))
		std = desc_mat.std(axis=0)
		std[std < 1e-6] = 1.0
		self.register_buffer('desc_std', torch.from_numpy(std))

		self.proj_desc   = nn.Linear(d_desc, d_model)
		self.proj_morgan = nn.Linear(1024,  d_model)
		self.proj_maccs  = nn.Linear(167,   d_model)   # RDKit MACCS is 167 bits

		# type embeddings for 3 token types
		self.type_emb = nn.Embedding(3, d_model)

	def forward(self, smiles: str) -> torch.Tensor:
		"""
		Return tokens [3, d_model] for a single SMILES.
		If SMILES missing in cache, compute on-the-fly with RDKit. If invalid, zeros.
		"""
		if smiles in self.sm2feat:
			f = self.sm2feat[smiles]
			desc = torch.from_numpy(f['desc'].astype(np.float32))
			morgan = torch.from_numpy(f['morgan'].astype(np.float32))
			maccs = torch.from_numpy(f['maccs'].astype(np.float32))
		else:
			res = _calc_one(smiles)
			if res is None:
				desc = torch.zeros_like(self.desc_mean)
				morgan = torch.zeros(1024, dtype=torch.float32)
				maccs  = torch.zeros(167,  dtype=torch.float32)
			else:
				d, mg, mc = res
				desc = torch.from_numpy(d.astype(np.float32))
				morgan = torch.from_numpy(mg.astype(np.float32))
				maccs  = torch.from_numpy(mc.astype(np.float32))

		# sanitize then standardize
		desc = torch.nan_to_num(desc, nan=0.0, posinf=1e6, neginf=-1e6)
		desc = torch.clamp(desc, -1e9, 1e9)
		desc = (desc - self.desc_mean) / self.desc_std
		# project
		td = self.proj_desc(desc.unsqueeze(0))     # [1, d_model]
		tm = self.proj_morgan(morgan.unsqueeze(0))
		tc = self.proj_maccs(maccs.unsqueeze(0))
		tokens = torch.cat([td, tm, tc], dim=0)   # [3, d_model]
		# add type embeddings
		types = torch.arange(0, 3, dtype=torch.long, device=tokens.device)
		tokens = tokens + self.type_emb(types)
		return tokens

class PositionalEncoding(nn.Module):
	def __init__(self, d_model: int, max_len: int = 512):
		super().__init__()
		pe = torch.zeros(max_len, d_model)
		pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(pos * div_term)
		pe[:, 1::2] = torch.cos(pos * div_term)
		self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return x + self.pe[:, :x.size(1), :]

class GroupEncoder(nn.Module):
	"""
	Generic per-group self-attention encoder.
	Input: tokens [B, T, Din] or features [B, D] (when as a single vector).
	We provide two adapters:
	- from dense vector [B, D] -> make T=1 token via Linear
	- from per-dim tokens [B, T, 1] -> project each to d_model
	"""
	def __init__(self, in_dim: int, d_model: int = 128, nhead: int = 4, nlayers: int = 1, tokens_mode: str = 'vector'):
		super().__init__()
		self.tokens_mode = tokens_mode  # 'vector' or 'per_dim'
		if tokens_mode == 'vector':
			self.proj = nn.Linear(in_dim, d_model)
			self.tok_T = 1
		else:
			self.proj = nn.Linear(1, d_model)
			self.tok_T = in_dim

		enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, batch_first=True)
		self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
		self.pool = nn.AdaptiveAvgPool1d(1)

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		# x: [B, D] if vector mode; else [B, D] interpreted as D tokens each 1-dim
		if self.tokens_mode == 'vector':
			tok = self.proj(x).unsqueeze(1)   # [B, 1, d_model]
		else:
			tok = self.proj(x.unsqueeze(-1))  # [B, D, d_model]
		h = self.encoder(tok)                 # [B, T, d_model]
		# mean pooling over tokens -> group vector
		g = h.mean(dim=1)                     # [B, d_model]
		return h, g

class CrossAttnBlock(nn.Module):
	def __init__(self, d_model: int = 128, nhead: int = 4):
		super().__init__()
		self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
		self.ff = nn.Sequential(
			nn.Linear(d_model, 4*d_model),
			nn.ReLU(),
			nn.Linear(4*d_model, d_model)
		)
		self.ln1 = nn.LayerNorm(d_model)
		self.ln2 = nn.LayerNorm(d_model)

	def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
		a, _ = self.attn(q, kv, kv)
		h = self.ln1(q + a)
		h2 = self.ff(h)
		h = self.ln2(h + h2)
		return h

class AttentionFusionModel(nn.Module):
	def __init__(self, d_model: int, n_reg: int, smiles_vec: SmilesVectorizer,
				 comp_in: int, helper_in: int, exp_in: int, phys_in: int,
				 n_clf: int = 0, per_dim_tokens_thresh: int = 64,
				 nhead: int = 4, enc_layers: int = 1, cross_layers: int = 2):
		super().__init__()
		self.smiles_vec = smiles_vec
		self.n_reg = n_reg
		self.n_clf = n_clf
		self.chem_pe = PositionalEncoding(d_model)

		# groups
		self.comp_enc   = GroupEncoder(comp_in,   d_model, nhead=nhead, nlayers=enc_layers, tokens_mode=('per_dim' if comp_in  <= per_dim_tokens_thresh else 'vector'))
		self.help_enc   = GroupEncoder(helper_in, d_model, nhead=nhead, nlayers=enc_layers, tokens_mode=('per_dim' if helper_in<= per_dim_tokens_thresh else 'vector'))
		self.exp_enc    = GroupEncoder(exp_in,    d_model, nhead=nhead, nlayers=enc_layers, tokens_mode=('per_dim' if exp_in   <= per_dim_tokens_thresh else 'vector'))
		self.phys_enc   = GroupEncoder(phys_in,   d_model, nhead=nhead, nlayers=enc_layers, tokens_mode=('per_dim' if phys_in  <= per_dim_tokens_thresh else 'vector'))

		# cross-attn stack
		self.cross = nn.ModuleList([CrossAttnBlock(d_model, nhead) for _ in range(cross_layers)])

		# fuse: concat group vectors + pooled cross outputs
		fuse_in = d_model * 6
		self.head_reg = nn.Sequential(
			nn.Linear(fuse_in, 2*d_model),
			nn.ReLU(),
			nn.Linear(2*d_model, n_reg)
		)
		self.head_clf = None
		if self.n_clf > 0:
			self.head_clf = nn.Sequential(
				nn.Linear(fuse_in, 2*d_model),
				nn.ReLU(),
				nn.Linear(2*d_model, n_clf)
			)

	def forward(self, smiles: List[str], comp_x: torch.Tensor, helper_x: torch.Tensor,
				exp_x: torch.Tensor, phys_x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
		B = comp_x.size(0)
		device = comp_x.device

		chem_tokens = []
		for s in smiles:
			t = self.smiles_vec(s).to(device)
			chem_tokens.append(t.unsqueeze(0))
		chem_tokens = torch.cat(chem_tokens, dim=0)
		chem_tokens = self.chem_pe(chem_tokens)

		comp_h, comp_g = self.comp_enc(comp_x)
		help_h, help_g = self.help_enc(helper_x)
		exp_h,  exp_g  = self.exp_enc(exp_x)
		phys_h, phys_g = self.phys_enc(phys_x)

		q = torch.cat([comp_h, exp_h], dim=1)
		kv = torch.cat([chem_tokens, phys_h], dim=1)
		x = q
		for blk in self.cross:
			x = blk(x, kv)
		x_pool = x.mean(dim=1)

		fuse = torch.cat([comp_g, help_g, exp_g, phys_g, x_pool, chem_tokens.mean(dim=1)], dim=1)
		reg = self.head_reg(fuse)
		clf = self.head_clf(fuse) if self.head_clf is not None else None
		return reg, clf

def _read_set(base_dir: str, prefix: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	y = pd.read_csv(os.path.join(base_dir, f'{prefix}.csv'))
	x = pd.read_csv(os.path.join(base_dir, f'{prefix}_extra_x.csv'))
	m = pd.read_csv(os.path.join(base_dir, f'{prefix}_metadata.csv'))
	return y, x, m

def _select_columns(df: pd.DataFrame, names: List[str]) -> np.ndarray:
	cols = [c for c in names if c in df.columns]
	if len(cols) == 0:
		return np.zeros((len(df), 0), dtype=np.float32)
	return df[cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(np.float32)

def _select_prefix(df: pd.DataFrame, prefixes: List[str]) -> np.ndarray:
	cols = [c for c in df.columns for p in prefixes if c.startswith(p)]
	cols = sorted(list(set(cols)))
	if len(cols) == 0:
		return np.zeros((len(df), 0), dtype=np.float32)
	return df[cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(np.float32)

def _standardize_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	if x.shape[1] == 0:
		return np.zeros((0,), dtype=np.float32), np.ones((0,), dtype=np.float32)
	mean = np.nanmean(x, axis=0).astype(np.float32)
	std  = np.nanstd(x, axis=0).astype(np.float32)
	std[std == 0] = 1.0
	return mean, std

def _standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
	if x.shape[1] == 0:
		return x
	return (x - mean) / std

def build_inputs_from_split(cv_dir: str) -> Dict[str, any]:
	"""
	Read CV fold directory and prepare tensors and column selections.
	Targets are standardized (per-column z-score) on train set.
	"""
	tr_y, tr_x, tr_m = _read_set(cv_dir, 'train')
	va_y, va_x, va_m = _read_set(cv_dir, 'valid')
	te_y, te_x, te_m = _read_set(cv_dir, 'test')

	# optional classification sets
	clf_exists = os.path.exists(os.path.join(cv_dir, 'train_clf.csv'))
	if clf_exists:
		tr_yc, _, _ = _read_set(cv_dir, 'train_clf')
		va_yc, _, _ = _read_set(cv_dir, 'valid_clf')
		te_yc, _, _ = _read_set(cv_dir, 'test_clf')
		clf_cols = [c for c in tr_yc.columns if c.lower() != 'smiles']
	else:
		tr_yc = va_yc = te_yc = None
		clf_cols = []

	# targets
	reg_cols = [c for c in tr_y.columns if c.lower() != 'smiles']
	assert len(reg_cols) > 0, f"No regression targets found in {cv_dir}/train.csv"

	# compute target normalization on train
	tr_y_mat = tr_y[reg_cols].apply(pd.to_numeric, errors='coerce').to_numpy(np.float32)
	y_mean = np.nanmean(tr_y_mat, axis=0).astype(np.float32)
	y_std  = np.nanstd(tr_y_mat, axis=0).astype(np.float32)
	y_std[y_std < 1e-6] = 1.0

	# groups
	comp_cols = [
		'Cationic_Lipid_Mass_Ratio','Phospholipid_Mass_Ratio','Cholesterol_Mass_Ratio','PEG_Lipid_Mass_Ratio',
		'Cationic_Lipid_to_mRNA_weight_ratio',
		'Cationic_Lipid_Mol_Ratio','Phospholipid_Mol_Ratio','Cholesterol_Mol_Ratio','PEG_Lipid_Mol_Ratio'
	]
	phys_cols = ['size','molecular_weight','PDI','MolWt','Tail_length','Tail_count','Num_tails']
	help_prefix = ['Helper_lipid_ID_']
	exp_prefix  = ['Route_of_administration_','Cargo_type_','Model_type_','Batch_or_individual_or_barcoded_','Purity_']

	def group_xy(y_df, x_df, m_df, yc_df=None):
		smiles = y_df['smiles'].astype(str).tolist()
		joined = pd.concat([x_df, m_df], axis=1)
		comp = _select_columns(joined, comp_cols)
		helpv = _select_prefix(joined, help_prefix)
		expc  = _select_prefix(joined, exp_prefix)
		phys  = _select_columns(joined, phys_cols)
		yraw  = y_df[reg_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(np.float32)
		yvals = (yraw - y_mean) / y_std
		if yc_df is not None and len(clf_cols) > 0:
			ycls = yc_df[clf_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(np.float32)
		else:
			ycls = np.zeros((len(y_df), 0), dtype=np.float32)
		return smiles, comp, helpv, expc, phys, yvals, ycls

	tr_s, tr_comp, tr_help, tr_exp, tr_phys, tr_yv, tr_yc_np = group_xy(tr_y, tr_x, tr_m, tr_yc)
	va_s, va_comp, va_help, va_exp, va_phys, va_yv, va_yc_np = group_xy(va_y, va_x, va_m, va_yc)
	te_s, te_comp, te_help, te_exp, te_phys, te_yv, te_yc_np = group_xy(te_y, te_x, te_m, te_yc)

	# standardize numeric groups on train
	comp_mean, comp_std = _standardize_fit(tr_comp)
	phys_mean, phys_std = _standardize_fit(tr_phys)
	tr_comp = _standardize_apply(tr_comp, comp_mean, comp_std)
	va_comp = _standardize_apply(va_comp, comp_mean, comp_std)
	te_comp = _standardize_apply(te_comp, comp_mean, comp_std)
	tr_phys = _standardize_apply(tr_phys, phys_mean, phys_std)
	va_phys = _standardize_apply(va_phys, phys_mean, phys_std)
	te_phys = _standardize_apply(te_phys, phys_mean, phys_std)

	meta = {
		'reg_cols': reg_cols,
		'clf_cols': clf_cols,
		'comp_in': tr_comp.shape[1],
		'helper_in': tr_help.shape[1],
		'exp_in': tr_exp.shape[1],
		'phys_in': tr_phys.shape[1],
		'comp_mean': comp_mean, 'comp_std': comp_std,
		'phys_mean': phys_mean, 'phys_std': phys_std,
		'y_mean': y_mean, 'y_std': y_std
	}
	def pack(smiles, comp, helpv, expc, phys, y, yc):
		return {
			'smiles': smiles,
			'comp': torch.from_numpy(comp),
			'help': torch.from_numpy(helpv),
			'exp': torch.from_numpy(expc),
			'phys': torch.from_numpy(phys),
			'y': torch.from_numpy(y),
			'y_clf': torch.from_numpy(yc)
		}
	return {
		'train': pack(tr_s, tr_comp, tr_help, tr_exp, tr_phys, tr_yv, tr_yc_np),
		'valid': pack(va_s, va_comp, va_help, va_exp, va_phys, va_yv, va_yc_np),
		'test':  pack(te_s, te_comp, te_help, te_exp, te_phys, te_yv, te_yc_np),
		'meta': meta
	}

def train_cv(split_folder: str, cv_index: int, npz_path: str, epochs: int = 20, d_model: int = 128, device: str = 'cpu', cfg_path: Optional[str] = None) -> None:
	base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
	cv_dir = os.path.join(base, 'data', 'crossval_splits', split_folder, f'cv_{cv_index}')
	out_dir = os.path.join(cv_dir + '_attn')
	os.makedirs(out_dir, exist_ok=True)

	# load config (defaults + overrides)
	default_cfg_path = os.path.join(base, 'data', 'args_files', 'attention_config.json')
	cfg = load_attention_config(cfg_path or default_cfg_path)

	bundle = build_inputs_from_split(cv_dir)
	meta = bundle['meta']
	reg_cols = meta['reg_cols']; clf_cols = meta['clf_cols']

	d_model = int(cfg.get('d_model', d_model))
	smiles_vec = SmilesVectorizer(npz_path=npz_path, d_model=d_model).to(device)
	model = AttentionFusionModel(
		d_model=d_model, n_reg=len(reg_cols), smiles_vec=smiles_vec,
		comp_in=meta['comp_in'], helper_in=meta['helper_in'],
		exp_in=meta['exp_in'], phys_in=meta['phys_in'],
		n_clf=len(clf_cols),
		per_dim_tokens_thresh=int(cfg.get('per_dim_tokens_thresh', 64)),
		nhead=int(cfg.get('nhead', 4)),
		enc_layers=int(cfg.get('enc_layers', 1)),
		cross_layers=int(cfg.get('cross_layers', 2))
	).to(device)

	opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get('lr', 1e-4)), weight_decay=float(cfg.get('weight_decay', 1e-4)))
	bce = nn.BCEWithLogitsLoss()
	lambda_clf = float(cfg.get('lambda_clf', 0.1))

	def _to_device(batch):
		return {
			'smiles': batch['smiles'],
			'comp': batch['comp'].to(device),
			'help': batch['help'].to(device),
			'exp': batch['exp'].to(device),
			'phys': batch['phys'].to(device),
			'y': batch['y'].to(device),
			'y_clf': batch['y_clf'].to(device)
		}

	tr, va = _to_device(bundle['train']), _to_device(bundle['valid'])
	y_std_t = torch.tensor(meta['y_std'], dtype=torch.float32, device=device)
	scale = torch.mean(y_std_t**2)

	best_val = float('inf')
	best_epoch = -1

	pbar = tqdm(total=epochs, desc=f"[cv{cv_index}] epochs", leave=True)
	for ep in range(1, epochs + 1):
        # ---- train ----
		model.train()
		reg, clf = model(tr['smiles'], tr['comp'], tr['help'], tr['exp'], tr['phys'])
		loss_reg = F.mse_loss(reg, tr['y'])
		loss_clf = bce(clf, tr['y_clf']) if (clf is not None and tr['y_clf'].shape[1] > 0) else torch.tensor(0.0, device=device)
		loss = loss_reg + lambda_clf * loss_clf
		opt.zero_grad(); loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
		opt.step()

        # ---- valid ----
		with torch.no_grad():
			model.eval()
			vreg, vclf = model(va['smiles'], va['comp'], va['help'], va['exp'], va['phys'])
			val_reg = F.mse_loss(vreg, va['y'])
			val_clf = bce(vclf, va['y_clf']) if (vclf is not None and va['y_clf'].shape[1] > 0) else torch.tensor(0.0, device=device)
			val_loss = (val_reg + lambda_clf * val_clf).item()

		# de-normalized RMSE (approx)
		rmse_tr = torch.sqrt(loss_reg * scale).item()
		rmse_va = torch.sqrt(val_reg * scale).item()

		# ---- save last.pt every epoch ----
		last_ckpt = {
			'epoch': ep,
			'state_dict': model.state_dict(),
			'optimizer': opt.state_dict(),
			'reg_cols': reg_cols,
			'clf_cols': clf_cols,
			'd_model': d_model,
			'config': cfg,
			'val_loss': val_loss,
			'rmse_va': rmse_va
		}
		torch.save(last_ckpt, os.path.join(out_dir, 'last.pt'))

		# ---- save best.pt when improved ----
		if val_loss < best_val:
			best_val = val_loss
			best_epoch = ep
			torch.save(last_ckpt, os.path.join(out_dir, 'best.pt'))

		# one-line log per epoch
		pbar.update(1)
		tqdm.write(
			f"[cv{cv_index}] epoch {ep}/{epochs} | "
			f"loss={loss.item():.4f} va_loss={val_loss:.4f} | "
			f"rmse={rmse_tr:.2f} va_rmse={rmse_va:.2f} | "
			f"bce={loss_clf.item():.4f} va_bce={val_clf.item():.4f} | "
			f"best@{best_epoch}={best_val:.4f}"
		)
	pbar.close()

	# keep a copy as model.pt (alias of best.pt or last.pt)
	src = 'best.pt' if os.path.exists(os.path.join(out_dir, 'best.pt')) else 'last.pt'
	ck = torch.load(os.path.join(out_dir, src), map_location='cpu')
	torch.save(ck, os.path.join(out_dir, 'model.pt'))
	print(f"[cv{cv_index}] saved best.pt (epoch {best_epoch}) and last.pt to: {out_dir}")

def predict_cv(split_folder: str, cv_index: int, npz_path: str, out_csv: Optional[str] = None, device: str = 'cpu', cfg_path: Optional[str] = None) -> pd.DataFrame:
	base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
	cv_dir = os.path.join(base, 'data', 'crossval_splits', split_folder, f'cv_{cv_index}')
	attn_dir = os.path.join(cv_dir + '_attn')
	# results path aligned without overwriting original model
	results_dir = os.path.join(base, 'results', 'crossval_splits', split_folder, f'cv_{cv_index}_attn')
	os.makedirs(results_dir, exist_ok=True)
	preds_out = os.path.join(results_dir, 'predicted_vs_actual.csv')

	# load config
	default_cfg_path = os.path.join(base, 'data', 'args_files', 'attention_config.json')
	cfg = load_attention_config(cfg_path or default_cfg_path)

	bundle = build_inputs_from_split(cv_dir)
	meta = bundle['meta']
	reg_cols = meta['reg_cols']; clf_cols = meta['clf_cols']
	y_mean = meta['y_mean']; y_std = meta['y_std']

	d_model = int(cfg.get('d_model', 128))
	smiles_vec = SmilesVectorizer(npz_path=npz_path, d_model=d_model).to(device)
	model = AttentionFusionModel(
		d_model=d_model, n_reg=len(reg_cols), smiles_vec=smiles_vec,
		comp_in=meta['comp_in'], helper_in=meta['helper_in'],
		exp_in=meta['exp_in'], phys_in=meta['phys_in'],
		n_clf=len(clf_cols),
		per_dim_tokens_thresh=int(cfg.get('per_dim_tokens_thresh', 64)),
		nhead=int(cfg.get('nhead', 4)),
		enc_layers=int(cfg.get('enc_layers', 1)),
		cross_layers=int(cfg.get('cross_layers', 2))
	).to(device)
	ckpt = torch.load(os.path.join(attn_dir, 'model.pt'), map_location=device)
	model.load_state_dict(ckpt['state_dict']); model.eval()

	# base output with metadata + actuals
	test_csv = os.path.join(cv_dir, 'test.csv')
	meta_csv = os.path.join(cv_dir, 'test_metadata.csv')
	base_df = pd.read_csv(test_csv)
	if os.path.exists(meta_csv):
		md = pd.read_csv(meta_csv)
		base_df = pd.concat([md, base_df], axis=1)

	# optional classification actuals
	test_clf_csv = os.path.join(cv_dir, 'test_clf.csv')
	test_clf_df = pd.read_csv(test_clf_csv) if (len(clf_cols)>0 and os.path.exists(test_clf_csv)) else None

	with torch.no_grad():
		reg, clf = model(
			bundle['test']['smiles'],
			bundle['test']['comp'].to(device),
			bundle['test']['help'].to(device),
			bundle['test']['exp'].to(device),
			bundle['test']['phys'].to(device)
		)
		reg_np = reg.cpu().numpy()
		clf_np = clf.cpu().numpy() if clf is not None and len(clf_cols)>0 else None

	# de-normalize regression predictions to original scale
	reg_denorm = reg_np * y_std.reshape(1, -1) + y_mean.reshape(1, -1)

	# attach predictions using cv_<i>_pred_* names
	for i, c in enumerate(reg_cols):
		base_df[f'cv_{cv_index}_pred_{c}'] = reg_denorm[:, i]
	if clf_np is not None:
		# logits -> probabilities
		probs = 1.0 / (1.0 + np.exp(-clf_np))
		for i, c in enumerate(clf_cols):
			base_df[f'cv_{cv_index}_pred_{c}'] = probs[:, i]

	# write predictions
	base_df.to_csv(preds_out, index=False)
	if out_csv:
		base_df.to_csv(out_csv, index=False)

	# compute test_scores.csv (RMSE for regression; AUC for classification)
	from sklearn.metrics import roc_auc_score
	test_scores_rows = []

	# regression metrics
	for i, c in enumerate(reg_cols):
		y_true = pd.to_numeric(pd.read_csv(test_csv)[c], errors='coerce').to_numpy()
		y_pred = reg_denorm[:, i]
		mask = np.isfinite(y_true) & np.isfinite(y_pred)
		if mask.sum() == 0:
			rmse = float('nan')
		else:
			rmse = float(np.sqrt(np.mean((y_true[mask] - y_pred[mask])**2)))
		test_scores_rows.append([c, rmse, 0.0, rmse])  # Mean, Std(=0), Fold 0

	# classification metrics
	if test_clf_df is not None and clf_np is not None:
		for i, c in enumerate(clf_cols):
			y_true = pd.to_numeric(test_clf_df[c], errors='coerce').to_numpy()
			y_score = probs[:, i]
			mask = np.isfinite(y_true) & np.isfinite(y_score)
			auc = float('nan')
			if mask.sum() > 0:
				u = np.unique(y_true[mask])
				if len(u) >= 2:
					try:
						auc = float(roc_auc_score(y_true[mask], y_score[mask]))
					except Exception:
						auc = float('nan')
			test_scores_rows.append([c, auc, 0.0, auc])

	# write test_scores.csv with headers similar to chemprop
	if len(test_scores_rows) > 0:
		# split into two tables to match headers
		reg_table = [r for r in test_scores_rows if r[1] is not None and not (np.isnan(r[1]) and isinstance(r[1], float))]
		df_reg = pd.DataFrame(reg_table, columns=['Task','Mean rmse','Standard deviation rmse','Fold 0 rmse'])
		clf_table = [r for r in test_scores_rows if isinstance(r[1], float) and not np.isnan(r[1])]
		# try to separate by known targets: if task in reg_cols -> RMSE table; if in clf_cols -> AUC table
		df_rmse = pd.DataFrame([[t, m, s, f] for (t, m, s, f) in test_scores_rows if t in reg_cols],
			columns=['Task','Mean rmse','Standard deviation rmse','Fold 0 rmse'])
		df_auc  = pd.DataFrame([[t, m, s, f] for (t, m, s, f) in test_scores_rows if t in clf_cols],
			columns=['Task','Mean auc','Standard deviation auc','Fold 0 auc'])

		# write separate files for clarity
		if len(df_rmse) > 0:
			df_rmse.to_csv(os.path.join(results_dir, 'test_scores.csv'), index=False)
		if len(df_auc) > 0:
			df_auc.to_csv(os.path.join(results_dir, 'test_scores_clf.csv'), index=False)

		# also dump a compact json
		scores_json = { 'rmse': {r['Task']: r['Fold 0 rmse'] for _, r in df_rmse.iterrows()},
		                'auc':  {r['Task']: r['Fold 0 auc']  for _, r in df_auc.iterrows()} }
		with open(os.path.join(results_dir, 'test_scores.json'), 'w', encoding='utf-8') as f:
			json.dump(scores_json, f, indent=2, ensure_ascii=False)

	return base_df