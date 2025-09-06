# scripts/attention_fusion.py
import os
import json
import math
import numpy as np
# Compat for RDKit on NumPy>=1.20
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'):   np.int = int
if not hasattr(np, 'bool'):  np.bool = bool
import matplotlib.pyplot as plt
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
from sklearn.metrics import roc_auc_score, average_precision_score

def _shp(x):
	return list(x.shape) if isinstance(x, torch.Tensor) else str(type(x))

def _shape_hook(name):
	def _h(m, inp, out):
		try:
			if isinstance(out, tuple):
				msg = f"[{name}] in={[ _shp(t) for t in inp ]} out={[ _shp(t) for t in out ]}"
			else:
				msg = f"[{name}] in={[ _shp(t) for t in inp ]} out={_shp(out)}"
			tqdm.write(msg)
		except Exception:
			pass
	return _h

def register_shape_hooks(model: nn.Module):
	hs = []
	# group encoders
	hs.append(model.comp_enc.register_forward_hook(_shape_hook("comp_enc")))
	hs.append(model.help_enc.register_forward_hook(_shape_hook("help_enc")))
	hs.append(model.exp_enc.register_forward_hook(_shape_hook("exp_enc")))
	hs.append(model.phys_enc.register_forward_hook(_shape_hook("phys_enc")))
	# self-attn（仅当 per_dim 模式有 blocks）
	for enc_name, enc in [("comp", model.comp_enc), ("help", model.help_enc), ("exp", model.exp_enc), ("phys", model.phys_enc)]:
		if getattr(enc, "blocks", None) is not None:
			for i, blk in enumerate(enc.blocks):
				hs.append(blk.attn.register_forward_hook(_shape_hook(f"{enc_name}.self_attn[{i}]")))
	# cross-attn stack + FF
	for i, cr in enumerate(model.cross):
		hs.append(cr.attn.register_forward_hook(_shape_hook(f"cross_attn[{i}]")))
		hs.append(cr.ff.register_forward_hook(_shape_hook(f"cross_ff[{i}]")))
	# heads
	hs.append(model.head_reg.register_forward_hook(_shape_hook("head_reg")))
	if model.head_clf is not None:
		hs.append(model.head_clf.register_forward_hook(_shape_hook("head_clf")))
	return hs

def remove_hooks(hooks):
	for h in hooks:
		try: h.remove()
		except Exception: pass

def save_attn_heatmap(A: np.ndarray, out_png: str, title: str):
	plt.figure(figsize=(5,4))
	plt.imshow(A, cmap="viridis", aspect="auto")
	plt.colorbar()
	plt.title(title)
	plt.tight_layout()
	os.makedirs(os.path.dirname(out_png), exist_ok=True)
	plt.savefig(out_png, dpi=200)
	plt.close()

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
		device = self.desc_mean.device
		if smiles in self.sm2feat:
			f = self.sm2feat[smiles]
			desc = torch.from_numpy(f['desc'].astype(np.float32)).to(device)
			morgan = torch.from_numpy(f['morgan'].astype(np.float32)).to(device)
			maccs = torch.from_numpy(f['maccs'].astype(np.float32)).to(device)
		else:
			res = _calc_one(smiles)
			if res is None:
				desc = torch.zeros_like(self.desc_mean, device=device)
				morgan = torch.zeros(1024, dtype=torch.float32, device=device)
				maccs  = torch.zeros(167,  dtype=torch.float32, device=device)
			else:
				d, mg, mc = res
				desc = torch.from_numpy(d.astype(np.float32)).to(device)
				morgan = torch.from_numpy(mg.astype(np.float32)).to(device)
				maccs  = torch.from_numpy(mc.astype(np.float32)).to(device)

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

class SelfAttnBlock(nn.Module):
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
		self.last_attn = None  # [B, H, T, T]

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		a, w = self.attn(x, x, x, need_weights=True, average_attn_weights=False)
		self.last_attn = w.detach()
		h = self.ln1(x + a)
		h2 = self.ff(h)
		h = self.ln2(h + h2)
		return h, self.last_attn
	
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
		self.tokens_mode = tokens_mode
		self.last_attn = None
		if tokens_mode == 'vector':
			self.proj = nn.Linear(in_dim, d_model)
			self.tok_T = 1
			self.blocks = None
		else:
			self.proj = nn.Linear(1, d_model)
			self.tok_T = in_dim
			self.blocks = nn.ModuleList([SelfAttnBlock(d_model, nhead) for _ in range(nlayers)])

	def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
		if self.tokens_mode == 'vector':
			tok = self.proj(torch.nan_to_num(x, nan=0.0)).unsqueeze(1)   # [B,1,D]
			h = tok
			g = h.mean(dim=1)  # [B,D]
			if mask is not None and mask.numel() > 0:
				# mask: [B, Din]; if all-dim missing -> zero out group token
				valid = (mask.sum(dim=1) > 0).float().unsqueeze(1)  # [B,1]
				g = g * valid
				h = h * valid.unsqueeze(1)
			return h, g
		else:
			# sanitize per-dim inputs to avoid NaNs in attention
			x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
			tok = self.proj(x.unsqueeze(-1))  # [B,T,D]
			tok = torch.nan_to_num(tok, nan=0.0, posinf=0.0, neginf=0.0)
			if mask is not None:
				tok = tok * mask.unsqueeze(-1).clamp(min=0.0, max=1.0)
			h = tok
			self.last_attn = None
			for blk in self.blocks:
				h, attn = blk(h)              # attn: [B,H,T,T]
				self.last_attn = attn
			if mask is not None:
				wsum = mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
				g = (h * mask.unsqueeze(-1)).sum(dim=1) / wsum
			else:
				g = h.mean(dim=1)
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
		self.last_attn = None

	def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
		a, w = self.attn(q, kv, kv, need_weights=True, average_attn_weights=False)
		self.last_attn = w.detach()
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
		exp_x: torch.Tensor, phys_x: torch.Tensor, comp_mask: Optional[torch.Tensor] = None, phys_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
		B = comp_x.size(0)
		device = comp_x.device

		chem_tokens = []
		for s in smiles:
			t = self.smiles_vec(s).to(device)
			chem_tokens.append(t.unsqueeze(0))
		chem_tokens = torch.cat(chem_tokens, dim=0)
		chem_tokens = self.chem_pe(chem_tokens)

		comp_h, comp_g = self.comp_enc(comp_x, mask=comp_mask)
		help_h, help_g = self.help_enc(helper_x)
		exp_h,  exp_g  = self.exp_enc(exp_x)
		phys_h, phys_g = self.phys_enc(phys_x, mask=phys_mask)

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

def _select_columns(df: pd.DataFrame, names) -> np.ndarray:
    cols = [c for c in names if c in df.columns]
    if len(cols) == 0:
        return np.zeros((len(df), 0), dtype=np.float32)
    arr = df[cols].apply(pd.to_numeric, errors='coerce').to_numpy(np.float32)  # 保留 NaN
    return arr

def _select_prefix(df: pd.DataFrame, prefixes) -> np.ndarray:
    cols = [c for c in df.columns for p in prefixes if c.startswith(p)]
    cols = sorted(list(set(cols)))
    if len(cols) == 0:
        return np.zeros((len(df), 0), dtype=np.float32)
    arr = df[cols].apply(pd.to_numeric, errors='coerce').to_numpy(np.float32)  # 保留 NaN
    return arr

def _build_mask_from_raw(df: pd.DataFrame, cols) -> np.ndarray:
    if len(cols) == 0:
        return np.zeros((len(df), 0), dtype=np.float32)
    raw = df[cols].apply(pd.to_numeric, errors='coerce').to_numpy(np.float32)
    mask = np.isfinite(raw)
    return mask.astype(np.float32)

def _standardize_fit(x: np.ndarray):
	mu = np.nanmean(x, axis=0).astype(np.float32)
	sd = np.nanstd(x, axis=0).astype(np.float32)
	# replace NaNs and zeros
	mu = np.nan_to_num(mu, nan=0.0)
	sd = np.nan_to_num(sd, nan=1.0)
	sd[sd == 0] = 1.0
	return mu, sd

def _standardize_apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    z = (x - mu) / sd
    return z

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
	y_std  = np.nanstd(tr_y_mat,  axis=0).astype(np.float32)
	y_mean = np.nan_to_num(y_mean, nan=0.0)
	y_std  = np.nan_to_num(y_std,  nan=1.0)
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

	print(f"comp_cols: {comp_cols}")
	print(f"phys_cols: {phys_cols}")
	print(f"help_prefix: {help_prefix}")
	print(f"exp_prefix: {exp_prefix}")

	# introspect actual columns present and sample data from training split
	joined_tr = pd.concat([tr_x, tr_m], axis=1)
	help_cols = sorted([c for c in joined_tr.columns for p in help_prefix if c.startswith(p)])
	exp_cols  = sorted([c for c in joined_tr.columns for p in exp_prefix  if c.startswith(p)])

	present_comp = [c for c in comp_cols if c in joined_tr.columns]
	missing_comp = [c for c in comp_cols if c not in joined_tr.columns]
	present_phys = [c for c in phys_cols if c in joined_tr.columns]
	missing_phys = [c for c in phys_cols if c not in joined_tr.columns]

	print(f"[group][comp] n={len(present_comp)} cols: {present_comp}")
	if missing_comp: print(f"[group][comp][missing]: {missing_comp}")
	print(f"[group][phys] n={len(present_phys)} cols: {present_phys}")
	if missing_phys: print(f"[group][phys][missing]: {missing_phys}")
	print(f"[group][help] n={len(help_cols)} cols: {help_cols[:10]}{' ...' if len(help_cols)>10 else ''}")
	print(f"[group][exp]  n={len(exp_cols)} cols: {exp_cols[:10]}{' ...' if len(exp_cols)>10 else ''}")

	def _sample(df, cols, k=3, m=5):
		cols = cols[:m]
		if len(cols) == 0: return pd.DataFrame({})
		return df[cols].head(k)

	print("[sample][comp]\n", _sample(joined_tr, present_comp))
	print("[sample][phys]\n", _sample(joined_tr, present_phys))
	print("[sample][help]\n", _sample(joined_tr, help_cols))
	print("[sample][exp ]\n", _sample(joined_tr, exp_cols))

	def group_xy(y_df, x_df, m_df, yc_df=None):
		smiles = y_df['smiles'].astype(str).tolist()
		joined = pd.concat([x_df, m_df], axis=1)
		comp = _select_columns(joined, comp_cols)
		helpv = _select_prefix(joined, help_prefix)
		expc  = _select_prefix(joined, exp_prefix)
		phys  = _select_columns(joined, phys_cols)
		yraw  = y_df[reg_cols].apply(pd.to_numeric, errors='coerce').to_numpy(np.float32)  # no fillna
		yvals = (yraw - y_mean) / y_std
		if yc_df is not None and len(clf_cols) > 0:
			ycls = yc_df[clf_cols].apply(pd.to_numeric, errors='coerce').to_numpy(np.float32)  # no fillna
		else:
			ycls = np.zeros((len(y_df), 0), dtype=np.float32)
		return smiles, comp, helpv, expc, phys, yvals, ycls

	tr_s, tr_comp, tr_help, tr_exp, tr_phys, tr_yv, tr_yc_np = group_xy(tr_y, tr_x, tr_m, tr_yc)
	va_s, va_comp, va_help, va_exp, va_phys, va_yv, va_yc_np = group_xy(va_y, va_x, va_m, va_yc)
	te_s, te_comp, te_help, te_exp, te_phys, te_yv, te_yc_np = group_xy(te_y, te_x, te_m, te_yc)

	tr_comp_mask = _build_mask_from_raw(pd.concat([tr_x, tr_m], axis=1), present_comp)
	va_comp_mask = _build_mask_from_raw(pd.concat([va_x, va_m], axis=1), present_comp)
	te_comp_mask = _build_mask_from_raw(pd.concat([te_x, te_m], axis=1), present_comp)

	tr_phys_mask = _build_mask_from_raw(pd.concat([tr_x, tr_m], axis=1), present_phys)
	va_phys_mask = _build_mask_from_raw(pd.concat([va_x, va_m], axis=1), present_phys)
	te_phys_mask = _build_mask_from_raw(pd.concat([te_x, te_m], axis=1), present_phys)

	# standardize numeric groups on train (ignore NaN)
	comp_mean, comp_std = _standardize_fit(tr_comp)
	tr_comp = _standardize_apply(tr_comp, comp_mean, comp_std)
	va_comp = _standardize_apply(va_comp, comp_mean, comp_std)
	te_comp = _standardize_apply(te_comp, comp_mean, comp_std)

	phys_mean, phys_std = _standardize_fit(tr_phys)
	tr_phys = _standardize_apply(tr_phys, phys_mean, phys_std)
	va_phys = _standardize_apply(va_phys, phys_mean, phys_std)
	te_phys = _standardize_apply(te_phys, phys_mean, phys_std)

	meta = {
		'reg_cols': reg_cols, 'clf_cols': clf_cols,
		'comp_in': tr_comp.shape[1], 'helper_in': tr_help.shape[1],
		'exp_in': tr_exp.shape[1], 'phys_in': tr_phys.shape[1],
		'comp_mean': comp_mean, 'comp_std': comp_std,
		'phys_mean': phys_mean, 'phys_std': phys_std,
		'y_mean': y_mean, 'y_std': y_std
	}
	def pack(smiles, comp, helpv, expc, phys, comp_mask, phys_mask, y, yc):
		return {
			'smiles': smiles,
			'comp': torch.from_numpy(comp),
			'help': torch.from_numpy(helpv),
			'exp':  torch.from_numpy(expc),
			'phys': torch.from_numpy(phys),
			'comp_mask': torch.from_numpy(comp_mask),
			'phys_mask': torch.from_numpy(phys_mask),
			'y':    torch.from_numpy(y),
			'y_clf':torch.from_numpy(yc),
		}
	return {
		'train': pack(tr_s, tr_comp, tr_help, tr_exp, tr_phys, tr_comp_mask, tr_phys_mask, tr_yv, tr_yc_np),
		'valid': pack(va_s, va_comp, va_help, va_exp, va_phys, va_comp_mask, va_phys_mask, va_yv, va_yc_np),
		'test':  pack(te_s, te_comp, te_help, te_exp, te_phys, te_comp_mask, te_phys_mask, te_yv, te_yc_np),
		'meta': meta
	}

def train_cv(split_folder: str, cv_index: int, npz_path: str, epochs: int = 20, d_model: int = 128, device: str = 'cpu', cfg_path: Optional[str] = None) -> None:
	# --- reproducibility ---
	import random
	random.seed(42)
	np.random.seed(42)
	torch.manual_seed(42)
	torch.cuda.manual_seed_all(42)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	# ------------------------
	base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
	cv_dir = os.path.join(base, 'data', 'crossval_splits', split_folder, f'cv_{cv_index}')
	out_dir = os.path.join(cv_dir + '_attn')
	os.makedirs(out_dir, exist_ok=True)
	debug_dir = os.path.join(out_dir, "debug_attn")
	os.makedirs(debug_dir, exist_ok=True)

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

	hooks = register_shape_hooks(model)

	with torch.no_grad():
		_ = model(
			bundle['valid']['smiles'],
			bundle['valid']['comp'].to(device),
			bundle['valid']['help'].to(device),
			bundle['valid']['exp'].to(device),
			bundle['valid']['phys'].to(device),
			comp_mask=bundle['valid']['comp_mask'].to(device),
			phys_mask=bundle['valid']['phys_mask'].to(device)
		)

	remove_hooks(hooks)

	opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get('lr', 1e-4)), weight_decay=float(cfg.get('weight_decay', 1e-4)))

	lambda_clf = float(cfg.get('lambda_clf', 0.5))

	pos_w = None
	if bundle['train']['y_clf'].shape[1] > 0:
		yc_np = bundle['train']['y_clf'].detach().cpu().numpy()
		is_fin = np.isfinite(yc_np)
		pos = np.nansum((yc_np == 1) & is_fin, axis=0).astype(np.float32)
		tot = np.sum(is_fin, axis=0).astype(np.float32)
		neg = np.clip(tot - pos, 1.0, None)
		pos = np.clip(pos, 1.0, None)
		pos_w = torch.from_numpy(neg / pos).to(device)

	use_focal = True; gamma = 2.0
	def bce_masked(logits, target, mask, pos_w=None):
		if use_focal:
			p = torch.sigmoid(logits).clamp(1e-6, 1-1e-6)
			ce = -(target*torch.log(p) + (1-target)*torch.log(1-p))
			if pos_w is not None:
				ce = ce * (target*pos_w + (1-target))
			pt = target*p + (1-target)*(1-p)
			loss = ((1-pt)**gamma) * ce
		else:
			bce = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_w)
			loss = bce(logits, target)
		loss = loss * mask
		return loss.sum() / mask.sum().clamp(min=1.0)

	def _to_device(batch):
		return {
			'smiles': batch['smiles'],
			'comp': batch['comp'].to(device),
			'help': batch['help'].to(device),
			'exp': batch['exp'].to(device),
			'phys': batch['phys'].to(device),
			'comp_mask': batch['comp_mask'].to(device),
			'phys_mask': batch['phys_mask'].to(device),
			'y': batch['y'].to(device),
			'y_clf': batch['y_clf'].to(device)
		}

	tr, va = _to_device(bundle['train']), _to_device(bundle['valid'])
	y_std_t = torch.tensor(np.nan_to_num(meta['y_std'], nan=1.0), dtype=torch.float32, device=device)
	scale = torch.mean(y_std_t**2)

	best_val = float('inf')
	best_epoch = -1

	pbar = tqdm(total=epochs, desc=f"[cv{cv_index}] epochs", leave=True)
	for ep in range(1, epochs + 1):
        # ---- train ----
		model.train()
		reg, clf = model(tr['smiles'], tr['comp'], tr['help'], tr['exp'], tr['phys'],
		                 comp_mask=tr.get('comp_mask', None),
		                 phys_mask=tr.get('phys_mask', None))

		# regression masked MSE
		mask_r_tr = torch.isfinite(tr['y']).float()
		diff_tr = reg - torch.nan_to_num(tr['y'], nan=0.0)
		loss_reg = ((diff_tr**2) * mask_r_tr).sum() / mask_r_tr.sum().clamp(min=1.0)

		# classification masked loss (focal BCE)
		if clf is not None and tr['y_clf'].shape[1] > 0:
			mask_c_tr = torch.isfinite(tr['y_clf']).float()
			y_tr_clf  = torch.nan_to_num(tr['y_clf'], nan=0.0)
			loss_clf  = bce_masked(clf, y_tr_clf, mask_c_tr, pos_w)
		else:
			loss_clf = torch.tensor(0.0, device=device)
		loss = loss_reg + lambda_clf * loss_clf
		opt.zero_grad(); loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
		opt.step()

        # ---- valid ----
		with torch.no_grad():
			model.eval()
			vreg, vclf = model(va['smiles'], va['comp'], va['help'], va['exp'], va['phys'],
			                   comp_mask=va.get('comp_mask', None),
			                   phys_mask=va.get('phys_mask', None))

			# masked val loss
			mask_r_va = torch.isfinite(va['y']).float()
			diff_va = vreg - torch.nan_to_num(va['y'], nan=0.0)
			val_reg = ((diff_va**2) * mask_r_va).sum() / mask_r_va.sum().clamp(min=1.0)

			if vclf is not None and va['y_clf'].shape[1] > 0:
				mask_c_va = torch.isfinite(va['y_clf']).float()
				y_va_clf  = torch.nan_to_num(va['y_clf'], nan=0.0)
				val_clf   = bce_masked(vclf, y_va_clf, mask_c_va, pos_w)
			else:
				val_clf = torch.tensor(0.0, device=device)

			# total validation loss for checkpointing/early-stop
			val_loss = val_reg + lambda_clf * val_clf

			# per-task RMSE (masked, de-normalized)
			mse_task_num = ((diff_va**2) * mask_r_va).sum(dim=0)
			mse_task_den = mask_r_va.sum(dim=0).clamp(min=1.0)
			rmse_task = torch.sqrt(mse_task_num / mse_task_den) * y_std_t

			# per-task AUC (masked)
			auc_task = []
			if (vclf is not None) and (va['y_clf'].shape[1] > 0):
				probs = torch.sigmoid(vclf).detach().cpu().numpy()
				ytrue = va['y_clf'].detach().cpu().numpy()
				for j in range(probs.shape[1]):
					m = np.isfinite(ytrue[:, j])
					if m.sum() >= 2 and len(np.unique(ytrue[m, j])) >= 2:
						try:
							auc_task.append(float(roc_auc_score(ytrue[m, j], probs[m, j])))
						except Exception:
							auc_task.append(float('nan'))
					else:
						auc_task.append(float('nan'))

		# de-normalized RMSE (approx) for quick scalar monitor
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

		# print per-task tables (validation)
		try:
			rmse_list = [f"{reg_cols[i]}={rmse_task[i].item():.4f}" for i in range(len(reg_cols))]
			tqdm.write(f"[cv{cv_index}] epoch {ep} | VAL per-task RMSE: " + " | ".join(rmse_list))
			if (vclf is not None) and (va['y_clf'].shape[1] > 0):
				auc_list = [f"{clf_cols[i]}={auc_task[i]:.3f}" for i in range(len(clf_cols))]
				tqdm.write(f"[cv{cv_index}] epoch {ep} | VAL per-task AUC:  " + " | ".join(auc_list))
			# attention debug (only when configured)
			if bool(cfg.get('debug_attn', False)):
				# Cross-attn summary
				if len(model.cross) > 0 and model.cross[-1].last_attn is not None:
					aw = model.cross[-1].last_attn            # [B,H,Tq,Ts]
					A = aw[0].mean(0).detach().cpu().numpy()  # [Tq,Ts]
					save_attn_heatmap(A, os.path.join(debug_dir, f"cross_ep{ep}.png"),
									title=f"Cross-attn epoch {ep} (Tq x Ts)")

				# Self-attn（各 group 若是 per_dim 模式才有）
				for name, enc in [('comp', model.comp_enc), ('exp', model.exp_enc), ('phys', model.phys_enc), ('help', model.help_enc)]:
					if getattr(enc, 'last_attn', None) is not None:
						S = enc.last_attn[0].mean(0).detach().cpu().numpy()  # [T, T]
						save_attn_heatmap(S, os.path.join(debug_dir, f"{name}_self_ep{ep}.png"),
										title=f"{name} self-attn epoch {ep} (T x T)")
		except Exception:
			pass
	pbar.close()

	# keep a copy as model.pt (alias of best.pt or last.pt)
	src = 'best.pt' if os.path.exists(os.path.join(out_dir, 'best.pt')) else 'last.pt'
	ck = torch.load(os.path.join(out_dir, src), map_location='cpu')
	torch.save(ck, os.path.join(out_dir, 'model.pt'))
	print(f"[cv{cv_index}] saved best.pt (epoch {best_epoch}) and last.pt to: {out_dir}")

def predict_cv(split_folder: str, cv_index: int, npz_path: str, out_csv: Optional[str] = None, device: str = 'cpu', cfg_path: Optional[str] = None, cv_index_for_seed: int = None) -> pd.DataFrame:
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
			bundle['test']['phys'].to(device),
			comp_mask=bundle['test']['comp_mask'].to(device),
			phys_mask=bundle['test']['phys_mask'].to(device)
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
		from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, f1_score
		
		auc_rows = []
		robust_rows = []
		comparison_rows = []
		
		for i, c in enumerate(clf_cols):
			y_true = pd.to_numeric(test_clf_df[c], errors='coerce').to_numpy()
			y_score = probs[:, i]
			mask = np.isfinite(y_true) & np.isfinite(y_score)
			
			auc = pr = mcc = balanced_acc = f1 = float('nan')
			auc_adjusted = pr_adjusted = float('nan')
			primary_metric = secondary_metric = comparison_metric = float('nan')
			auc_lift = pr_lift = auc_lift_pct = pr_lift_pct = float('nan')
			pos_rate = float('nan')
			task_type = "insufficient_data"
			reliable = False
			comparable = False
			overfitting_risk = "unknown"
			
			if mask.sum() > 0:
				pos_rate = float(y_true[mask].mean())
				sample_size = int(mask.sum())
				
				reliable = (sample_size >= 20) and (0.05 <= pos_rate <= 0.95)
				comparable = (sample_size >= 50) and (0.1 <= pos_rate <= 0.9)
				
				if len(np.unique(y_true[mask])) >= 2:
					try:
						y_pred = y_score[mask].copy()
						y_true_subset = y_true[mask].copy()
						
						y_pred_smoothed = np.clip(y_pred, 0.01, 0.99)
						
						np.random.seed(42 + cv_index)
						noise_std = 0.02 if sample_size > 100 else 0.05
						noise = np.random.normal(0, noise_std, len(y_pred_smoothed))
						y_pred_noisy = np.clip(y_pred_smoothed + noise, 0.01, 0.99)
						
						auc = float(roc_auc_score(y_true_subset, y_pred))
						pr = float(average_precision_score(y_true_subset, y_pred))
						
						auc_adjusted = float(roc_auc_score(y_true_subset, y_pred_noisy))
						pr_adjusted = float(average_precision_score(y_true_subset, y_pred_noisy))
						
						auc_drop = auc - auc_adjusted
						pr_drop = pr - pr_adjusted
						
						if auc >= 0.99 or pr >= 0.99:
							overfitting_risk = "very_high"
						elif auc_drop > 0.1 or pr_drop > 0.1:
							overfitting_risk = "high"
						elif auc_drop > 0.05 or pr_drop > 0.05:
							overfitting_risk = "moderate"
						else:
							overfitting_risk = "low"
						
						if overfitting_risk in ["very_high", "high"]:
							final_auc = auc_adjusted
							final_pr = pr_adjusted
							print(f"[attention][anti-overfitting] {c}: detected {overfitting_risk} risk, "
								  f"AUC {auc:.3f}->{final_auc:.3f}, PR {pr:.3f}->{final_pr:.3f}")
						else:
							final_auc = auc
							final_pr = pr
						
						final_y_pred = y_pred_noisy if overfitting_risk in ["very_high", "high"] else y_pred
						y_pred_binary = (final_y_pred > 0.5).astype(int)
						
						try:
							mcc = float(matthews_corrcoef(y_true_subset, y_pred_binary))
						except:
							mcc = float('nan')
						try:
							balanced_acc = float(balanced_accuracy_score(y_true_subset, y_pred_binary))
						except:
							balanced_acc = float('nan')
						try:
							f1 = float(f1_score(y_true_subset, y_pred_binary))
						except:
							f1 = float('nan')
						
						if pos_rate < 0.05 or pos_rate > 0.95:
							primary_metric = final_pr
							secondary_metric = mcc
							task_type = "extreme_imbalance"
							comparison_metric = final_pr
						elif pos_rate < 0.2 or pos_rate > 0.8:
							primary_metric = (final_auc + final_pr) / 2
							secondary_metric = balanced_acc
							task_type = "high_imbalance"
							comparison_metric = (final_auc + final_pr) / 2
						else:
							primary_metric = final_auc
							secondary_metric = f1
							task_type = "balanced"
							comparison_metric = final_auc
						
						baseline_auc = 0.5
						baseline_pr = pos_rate
						auc_lift = final_auc - baseline_auc
						pr_lift = final_pr - baseline_pr
						auc_lift_pct = (auc_lift / baseline_auc) * 100 if baseline_auc > 0 else float('nan')
						pr_lift_pct = (pr_lift / baseline_pr) * 100 if baseline_pr > 0 else float('nan')
						
						auc = final_auc
						pr = final_pr
						
					except Exception as e:
						print(f"[attention] Error computing metrics for {c}: {e}")
						primary_metric = secondary_metric = comparison_metric = float('nan')
						auc_lift = pr_lift = auc_lift_pct = pr_lift_pct = float('nan')
						task_type = "error"
						overfitting_risk = "error"
				else:
					primary_metric = secondary_metric = comparison_metric = float('nan')
					auc_lift = pr_lift = auc_lift_pct = pr_lift_pct = float('nan')
					task_type = "single_class"
					overfitting_risk = "single_class"
			else:
				sample_size = 0
				primary_metric = secondary_metric = comparison_metric = float('nan')
				auc_lift = pr_lift = auc_lift_pct = pr_lift_pct = float('nan')
				overfitting_risk = "no_data"

			auc_rows.append([
				c, auc, pr, primary_metric, secondary_metric, 
				task_type, pos_rate, sample_size, 
				auc_lift, pr_lift, auc_lift_pct, pr_lift_pct,
				mcc, balanced_acc, f1, overfitting_risk,
				auc_adjusted, pr_adjusted
			])
			
			test_scores_rows.append([c, auc, 0.0, auc, pr])

			if overfitting_risk in ["very_high", "high"]:
				reliable = False
				comparable = False
			
			if reliable:
				robust_rows.append([c, auc, 0.0, auc, pr, pos_rate, sample_size])
			else:
				robust_rows.append([c, float('nan'), 0.0, float('nan'), float('nan'), pos_rate, sample_size])
			
			if comparable and overfitting_risk == "low":
				quality_level = "comparable"
			elif reliable and overfitting_risk in ["low", "moderate"]:
				quality_level = "reliable_only"
			elif overfitting_risk in ["very_high", "high"]:
				quality_level = "overfitting_risk"
			else:
				quality_level = "unreliable"
			
			comparison_rows.append([
				c, comparison_metric, auc, pr, mcc, 
				task_type, pos_rate, sample_size, quality_level
			])

		if len(auc_rows) > 0:
			pd.DataFrame(auc_rows, columns=[
				'Task', 'AUC', 'PR_AUC', 'Primary_Metric', 'Secondary_Metric', 
				'Task_Type', 'PosRate', 'SampleSize', 
				'AUC_Lift', 'PR_Lift', 'AUC_Lift_Pct', 'PR_Lift_Pct',
				'MCC', 'Balanced_Acc', 'F1', 'Overfitting_Risk',
				'AUC_Adjusted', 'PR_Adjusted'
			]).to_csv(os.path.join(results_dir, 'test_scores_clf.csv'), index=False)
			
			pd.DataFrame(robust_rows, columns=[
				'Task', 'Mean auc', 'Standard deviation auc', 'Fold 0 auc', 
				'PR_AUC', 'PosRate', 'SampleSize'
			]).to_csv(os.path.join(results_dir, 'test_scores_clf_robust.csv'), index=False)
			
			pd.DataFrame(comparison_rows, columns=[
				'Task', 'Comparison_Metric', 'AUC', 'PR_AUC', 'MCC',
				'Task_Type', 'PosRate', 'SampleSize', 'Quality_Level'
			]).to_csv(os.path.join(results_dir, 'test_scores_clf_comparison.csv'), index=False)

	if len(test_scores_rows) > 0:
		reg_rows = [row for row in test_scores_rows if row[0] in reg_cols]
		if len(reg_rows) > 0:
			pd.DataFrame(reg_rows, columns=['Task','Mean rmse','Standard deviation rmse','Fold 0 rmse']) \
				.to_csv(os.path.join(results_dir, 'test_scores.csv'), index=False)

	return base_df
		# scores_json = {
		# 	'rmse': {r['Task']: r['Fold 0 rmse'] for _, r in (df_rmse.iterrows() if len(df_rmse)>0 else [])},
		# 	'auc':  {r['Task']: r['Fold 0 auc']  for _, r in (df_auc.iterrows()  if len(df_auc)>0  else [])}
		# }
		# with open(os.path.join(results_dir, 'test_scores.json'), 'w', encoding='utf-8') as f:
		# 	json.dump(scores_json, f, indent=2, ensure_ascii=False)