# scripts/smiles_features.py
import os
import json
import numpy as np
# Compat for RDKit on NumPy>=1.20
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'):   np.int = int
if not hasattr(np, 'bool'):  np.bool = bool

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.info'); RDLogger.DisableLog('rdApp.warning'); RDLogger.DisableLog('rdApp.error')
from typing import Iterable, Dict, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

DESC_NAMES = [name for name, _ in Descriptors._descList]
DESC_CALC = MolecularDescriptorCalculator(DESC_NAMES)

def _calc_one(smiles: str, nbits: int = 1024, radius: int = 2) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
	mol = Chem.MolFromSmiles(smiles)
	if mol is None:
		return None
	try:
		desc_vals = list(DESC_CALC.CalcDescriptors(mol))
	except Exception:
		desc_vals = [np.nan] * len(DESC_NAMES)
	# sanitize to finite range before casting
	desc_arr = np.asarray(desc_vals, dtype=np.float64)
	desc_arr = np.nan_to_num(desc_arr, nan=0.0, posinf=1e6, neginf=-1e6)
	desc_arr = np.clip(desc_arr, -1e9, 1e9).astype(np.float32)

	morgan = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits, useChirality=False).ToList()
	maccs  = MACCSkeys.GenMACCSKeys(mol).ToList()
	return (
		desc_arr,
		np.asarray(morgan, dtype=np.int8),
		np.asarray(maccs,  dtype=np.int8)
	)

def build_npz_from_smiles(unique_smiles: Iterable[str], out_path: str,
						  nbits: int = 1024, radius: int = 2, log_every: int = 1000) -> str:
	"""
	Build a compressed NPZ cache with SMILES features.
	Keys:
	- 'SMILES': array[str]
	- 'descnames': array[str]
	- 'desc': float32 [N, D_desc]
	- 'morgan': int8 [N, nbits]
	- 'maccs': int8 [N, 167]
	Returns the NPZ path.
	"""
	unique_smiles = [str(s).strip() for s in unique_smiles if str(s).strip()]
	unique_smiles = sorted(list(set(unique_smiles)))
	descnames = DESC_NAMES

	valid_smiles = []
	desc_rows, morgan_rows, maccs_rows = [], [], []

	for i, s in enumerate(unique_smiles):
		res = _calc_one(s, nbits=nbits, radius=radius)
		if res is None:
			continue
		d, mg, mc = res
		valid_smiles.append(s)
		desc_rows.append(d)
		morgan_rows.append(mg)
		maccs_rows.append(mc)
		if (i + 1) % log_every == 0:
			print(f"[smiles_features] processed {i+1}/{len(unique_smiles)}")

	if len(valid_smiles) == 0:
		raise RuntimeError("No valid SMILES found to build features.")

	desc = np.stack(desc_rows, axis=0).astype(np.float32)
	morgan = np.stack(morgan_rows, axis=0).astype(np.int8)
	maccs = np.stack(maccs_rows, axis=0).astype(np.int8)

	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	np.savez_compressed(
		out_path,
		SMILES=np.array(valid_smiles, dtype=object),
		descnames=np.array(descnames, dtype=object),
		desc=desc,
		morgan=morgan,
		maccs=maccs
	)
	print(f"[smiles_features] saved: {out_path}  (N={len(valid_smiles)}, desc={desc.shape[1]}, morgan={morgan.shape[1]}, maccs={maccs.shape[1]})")
	return out_path

def load_smiles_features_npz(npz_path: str) -> Dict[str, Dict[str, np.ndarray]]:
	"""
	Load NPZ and return a mapping: smiles -> {'desc':..., 'morgan':..., 'maccs':...}
	"""
	data = np.load(npz_path, allow_pickle=True)
	smiles = data['SMILES'].tolist()
	desc = data['desc']
	morgan = data['morgan']
	maccs = data['maccs']
	sm2feat = {}
	for i, s in enumerate(smiles):
		sm2feat[s] = {'desc': desc[i], 'morgan': morgan[i], 'maccs': maccs[i]}
	return sm2feat