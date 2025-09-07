# scripts/precompute_mpnn_cache.py
import os
import glob
import argparse
import pandas as pd
from chemprop_encoder import ChempropEncoder


def resolve_ckpt(path: str) -> str:
	# If path is a directory, try common filenames
	if os.path.isdir(path):
		for name in ['best.pt', 'model.pt', 'checkpoint.pt']:
			p = os.path.join(path, name)
			if os.path.exists(p):
				return p
		# fallback: any .pt inside
		cands = sorted(glob.glob(os.path.join(path, '*.pt')))
		if len(cands) > 0:
			return cands[0]
		return ''  # not found
	# If path is a file, return if exists
	return path if os.path.exists(path) else ''

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--split', type=str, default='all_amine_split_for_paper', help='name under data/crossval_splits')
	parser.add_argument('--k', type=int, default=5, help='number of CV folds to scan for SMILES')
	parser.add_argument('--ckpt', type=str, required=True, help='Chemprop checkpoint file or directory')
	parser.add_argument('--device', type=str, default='cuda', choices=['cuda','cpu'])
	parser.add_argument('--out', type=str, default=None, help='output npz path')
	args = parser.parse_args()

	base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
	cv_dir = os.path.join(base, 'data', 'crossval_splits', args.split)

	# Collect unique SMILES from all folds
	smiles = set()
	for i in range(args.k):
		for name in ['train.csv','valid.csv','test.csv']:
			fp = os.path.join(cv_dir, f'cv_{i}', name)
			if os.path.exists(fp):
				df = pd.read_csv(fp)
				if 'smiles' in df.columns:
					smiles.update(df['smiles'].astype(str).tolist())
	smiles = sorted(list(smiles))
	if len(smiles) == 0:
		raise RuntimeError(f'No SMILES found under {cv_dir}. Check split name and files.')

	# Resolve checkpoint
	ckpt_path = resolve_ckpt(args.ckpt)
	if ckpt_path == '':
		raise FileNotFoundError(f'Cannot find a valid Chemprop checkpoint from --ckpt="{args.ckpt}". '
			f'Pass a real file or a directory containing best.pt/model.pt.')

	# Output path
	out_npz = args.out or os.path.join(base, 'data', 'smiles_mpnn.npz')

	# Encode and save
	enc = ChempropEncoder(ckpt_path=ckpt_path, device=args.device, cache_npz=None, freeze=True)
	enc.precompute_and_save(smiles, out_npz)
	print(f"Saved: {out_npz}  size: {len(smiles)}  ckpt: {ckpt_path}")