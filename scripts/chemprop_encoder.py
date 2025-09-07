# scripts/chemprop_encoder.py
import os, json
import numpy as np
import torch
import torch.nn as nn


class ChempropEncoder(nn.Module):
	"""
	Load a trained Chemprop (D-MPNN) checkpoint and extract molecule-level embeddings.
	Option A: On-the-fly encode smiles list (slower).
	Option B: Use precomputed npz cache (fast) and fall back to on-the-fly if missing.
	"""
	def __init__(self, ckpt_path: str, device: str = 'cpu', cache_npz: str = None, freeze: bool = True):
		super().__init__()
		self.ckpt_path = ckpt_path
		self.device = torch.device(device)
		self.cache = None
		self.H = None  # embedding dim
		self.model = None  # lazy-load only if needed
		self.freeze = freeze
		if cache_npz and os.path.exists(cache_npz):
			data = np.load(cache_npz, allow_pickle=True)
			# expect keys: 'smiles' (list[str]), 'emb' (ndarray [N,H])
			smiles = list(data['smiles'])
			emb = data['emb']
			self.H = int(emb.shape[1])
			self.cache = dict(zip(smiles, [e for e in emb]))
		# Note: actual Chemprop import deferred to avoid heavy import on CPU-only steps.

	def _lazy_load_model(self):
		if self.model is not None:
			return
		# Import chemprop only when needed
		import chemprop
		ckpt = torch.load(self.ckpt_path, map_location=self.device)
		args = ckpt['args'] if 'args' in ckpt else None
		self.model = chemprop.models.MoleculeModel(args) if args is not None else chemprop.models.MoleculeModel()
		self.model.load_state_dict(ckpt['state_dict'])
		self.model.to(self.device)
		self.model.eval()
		# Try to infer hidden dim H
		try:
			# Chemprop exposes FFN/molecule encoder; fall back if not available
			H = self.model.encoder.features_dim
		except Exception:
			H = 300
		self.H = int(H)
		if self.freeze:
			for p in self.model.parameters():
				p.requires_grad = False

	@torch.no_grad()
	def encode(self, smiles_list):
		"""
		Returns torch.FloatTensor of shape [B, H]
		- If cache exists and contains all smiles, use it
		- Otherwise, lazy-load chemprop and encode missing entries
		"""
		B = len(smiles_list)
		if self.cache is not None and all((s in self.cache) for s in smiles_list):
			X = np.stack([self.cache[s] for s in smiles_list], axis=0).astype(np.float32)
			return torch.from_numpy(X).to(self.device)

		self._lazy_load_model()
		import chemprop
		from rdkit import Chem

		# Parse all SMILES once
		mols_all = [Chem.MolFromSmiles(s) for s in smiles_list]

		# Infer feature size from model args (0 if none)
		feat_size = 0
		margs = getattr(self.model, 'args', None)
		if margs is not None:
			try:
				feat_size = int(getattr(margs, 'features_size', 0) or 0)
			except Exception:
				feat_size = 0

		# Probe once to get embedding dim H (avoid wrong default like 300)
		H = self.H
		if H is None:
			vi0 = next((i for i, m in enumerate(mols_all) if m is not None), None)
			if vi0 is None:
				# no valid molecules at all → return zeros
				H = 300
				self.H = H
				Z = torch.zeros((len(smiles_list), H), device=self.device, dtype=torch.float32)
				if self.cache is not None:
					for s, v in zip(smiles_list, Z.detach().cpu().numpy()):
						self.cache[s] = v.astype(np.float32)
				return Z
			# one-sample probe
			batch0 = [[mols_all[vi0]]]
			features0 = [np.zeros((feat_size,), dtype=np.float32)]
			try:
				z0 = self.model.encoder(batch0, features_batch=features0)
			except Exception:
				out0 = self.model(batch0, features_batch=features0)
				z0 = self.model.last_hidden if hasattr(self.model, 'last_hidden') else out0
			H = int(z0.shape[1])
			self.H = H

		# Chunked encoding
		outs = []
		B = len(smiles_list)
		chunk = 256
		for st in range(0, B, chunk):
			sub_idx = list(range(st, min(B, st + chunk)))
			sub_mols = [mols_all[i] for i in sub_idx]
			vi = [i for i, m in enumerate(sub_mols) if m is not None]

			if len(vi) == 0:
				outs.append(torch.zeros((len(sub_idx), H), device=self.device, dtype=torch.float32))
				continue

			# chemprop 1.x expects List[List[Mol]] and a features list (even if empty)
			batch_rdmols = [[sub_mols[i]] for i in vi]
			features_batch = [np.zeros((feat_size,), dtype=np.float32) for _ in vi]

			# Forward; keep dim consistent with probed H
			try:
				z_valid = self.model.encoder(batch_rdmols, features_batch=features_batch)  # [N_valid, H]
			except Exception:
				try:
					out = self.model(batch_rdmols, features_batch=features_batch)
					z_valid = self.model.last_hidden if hasattr(self.model, 'last_hidden') else out
				except Exception:
					z_valid = torch.zeros((len(vi), H), device=self.device, dtype=torch.float32)

			# If encoder unexpectedly returns different dim, coerce to H
			if z_valid.ndim != 2 or z_valid.shape[1] != H:
				if z_valid.ndim == 2:
					# pad/truncate to H
					if z_valid.shape[1] > H:
						z_valid = z_valid[:, :H]
					else:
						pad = torch.zeros((z_valid.shape[0], H - z_valid.shape[1]), device=z_valid.device, dtype=z_valid.dtype)
						z_valid = torch.cat([z_valid, pad], dim=1)
				else:
					z_valid = torch.zeros((len(vi), H), device=self.device, dtype=torch.float32)

			# Scatter back to full positions of this chunk
			z_full = torch.zeros((len(sub_idx), H), device=self.device, dtype=torch.float32)
			z_full[vi] = z_valid.detach()
			outs.append(z_full)

		Z = torch.cat(outs, dim=0)
		if self.cache is not None:
			for s, v in zip(smiles_list, Z.detach().cpu().numpy()):
				self.cache[s] = v.astype(np.float32)
		return Z

	@torch.no_grad()
	def precompute_and_save(self, smiles_list, out_npz: str):
		"""Precompute embeddings for a smiles list and save to .npz for fast training."""
		Z = self.encode(smiles_list).detach().cpu().numpy()
		np.savez_compressed(out_npz, smiles=np.array(smiles_list, dtype=object), emb=Z)