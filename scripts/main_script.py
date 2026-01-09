import numpy as np 
import os
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_curve, roc_auc_score, average_precision_score
# from train_multitask import train_multitask_model, get_base_args, optimize_hyperparameters, train_hyperparam_optimized_model
from train_multitask import train_multitask_model, get_base_args, train_hyperparam_optimized_model
from predict_multitask_from_json import predict_multitask_from_json, get_base_predict_args, predict_multitask_from_json_cv
from rdkit import Chem
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt
import scipy.stats
import json
import sys
import random
import chemprop

def _normalize_metadata_missing(all_df, col_type_map, fill_numeric_with_zero=True):
	# Normalize metadata missing tokens and coerce numeric-like metadata to numeric (optionally fill NaN with 0.0)
	# - Turn common missing tokens to NaN for non-numeric metadata
	# - For numeric-like metadata, coerce to numeric and optionally fill NaN with 0.0
	import numpy as np
	missing_tokens = {'', 'na', 'NA', 'Na', 'N/A', 'none', 'None'}

	meta_cols = [c for c, (t, _) in col_type_map.items() if t == 'Metadata' and c in all_df.columns]
	for c in meta_cols:
		v_num = pd.to_numeric(all_df[c], errors = 'coerce')
		if v_num.notna().any():
			all_df[c] = v_num.fillna(0.0) if fill_numeric_with_zero else v_num
		else:
			s = all_df[c].astype(str).str.strip()
			all_df.loc[s.isin(missing_tokens), c] = np.nan
	return all_df

def _filter_invalid_smiles(df):
	# Filter out rows with invalid SMILES strings
	if 'smiles' not in df.columns:
		return df
	s = df['smiles'].astype(str).str.strip()
	ok = s.apply(lambda x: Chem.MolFromSmiles(x) is not None)
	bad = int((~ok).sum())
	if bad:
		print(f"[split] drop invalid SMILES: {bad}")
	return df.loc[ok].reset_index(drop=True)

def attach_y_task_to_col_type(col_type_df: pd.DataFrame, all_df: pd.DataFrame) -> pd.DataFrame:
	"""
	Adds Y_task column to col_type_df based on all_df contents.
	- For Type == 'Y_val':
	  - If the column is binary {0,1} (after numeric coercion), mark as classification
	  - Else mark as regression
	- Force raw text label 'Delivery_target' to Metadata (never a Y target)
	"""
	col_type_df = col_type_df.copy()
	if 'Y_task' not in col_type_df.columns:
		col_type_df['Y_task'] = ''

	# Force raw Delivery_target to Metadata
	if 'Delivery_target' in col_type_df['Column_name'].values:
		idx = col_type_df['Column_name'] == 'Delivery_target'
		col_type_df.loc[idx, 'Type'] = 'Metadata'
		col_type_df.loc[idx, 'Y_task'] = ''

	# Assign Y_task for Y_val columns
	y_mask = col_type_df['Type'] == 'Y_val'
	for i, row in col_type_df[y_mask].iterrows():
		col = row['Column_name']
		if col in all_df.columns:
			ser = pd.to_numeric(all_df[col], errors='coerce')
			u = set(ser.dropna().unique().tolist())
			if len(u) > 0 and u.issubset({0.0, 1.0, 0, 1}):
				col_type_df.at[i, 'Y_task'] = 'classification'
			else:
				col_type_df.at[i, 'Y_task'] = 'regression'
		else:
			# Keep empty; column not present in data
			col_type_df.at[i, 'Y_task'] = col_type_df.at[i, 'Y_task'] or ''

	# Drop exact duplicate rows by Column_name keeping the first occurrence
	col_type_df = col_type_df.drop_duplicates(subset=['Column_name'], keep='first').reset_index(drop=True)
	return col_type_df


def write_target_roles(col_type_df: pd.DataFrame, args_dir: str) -> None:
	"""
	Writes args_files/target_roles.json with regression and classification target lists.
	"""
	os.makedirs(args_dir, exist_ok=True)
	mask_y = (col_type_df['Type'] == 'Y_val')
	reg_targets = sorted(col_type_df.loc[mask_y & (col_type_df['Y_task'] == 'regression'), 'Column_name'].astype(str).tolist())
	clf_targets = sorted(col_type_df.loc[mask_y & (col_type_df['Y_task'] == 'classification'), 'Column_name'].astype(str).tolist())
	target_roles = {"regression_targets": reg_targets, "classification_targets": clf_targets}
	with open(os.path.join(args_dir, 'target_roles.json'), 'w', encoding='utf-8') as f:
		json.dump(target_roles, f, indent=2, ensure_ascii=False)

def merge_datasets(experiment_list, path_to_folders = '../data/data_files_to_merge', write_path = '../data'):
	# Merge all experiment datasets into unified all_data.csv and col_type.csv
	# Each experiment folder should contain:
	#   - main_data.csv: SMILES and activity measurements
	#   - formulations.csv: lipid composition ratios
	#   - individual_metadata.csv: per-sample metadata (optional)
	#   - experiment_metadata.csv is read from parent folder
	
	all_df = pd.DataFrame({})
	col_type = {'Column_name':[], 'Type':[]}
	y_task_col = []  # parallel to col_type to store Y task type
	
	experiment_df = pd.read_csv(path_to_folders + '/experiment_metadata.csv')
	if experiment_list is None:
		experiment_list = list(experiment_df.Experiment_ID)
		print('Will merge experiments:', experiment_list)
	
	helper_mol_weights = pd.read_csv(path_to_folders + '/Component_molecular_weights.csv')

	for folder in experiment_list:
		print('Merging:', folder)
		try:
			main_temp = pd.read_csv(path_to_folders + '/' + folder + '/main_data.csv')
		except Exception:
			continue
		
		data_n = len(main_temp)
		formulation_temp = pd.read_csv(path_to_folders + '/' + folder + '/formulations.csv')
		try:
			individual_temp = pd.read_csv(path_to_folders + '/' + folder + '/individual_metadata.csv')
		except Exception:
			individual_temp = pd.DataFrame({})
		
		if len(formulation_temp) == 1:
			formulation_temp = pd.concat([formulation_temp]*data_n, ignore_index=True)
		elif len(formulation_temp) != data_n:
			raise ValueError(f'For experiment {folder}: formulations rows {len(formulation_temp)} != main_data rows {data_n}')
		
		# Convert mass ratios to molar ratios if needed
		mass_ratio_variables = ['Cationic_Lipid_Mass_Ratio','Phospholipid_Mass_Ratio','Cholesterol_Mass_Ratio','PEG_Lipid_Mass_Ratio']
		molar_ratio_variables = ['Cationic_Lipid_Mol_Ratio','Phospholipid_Mol_Ratio','Cholesterol_Mol_Ratio','PEG_Lipid_Mol_Ratio']
		mass_count = sum(c in mass_ratio_variables for c in formulation_temp.columns)
		molar_count = sum(c in molar_ratio_variables for c in formulation_temp.columns)
		if mass_count>0 and molar_count>0:
			raise ValueError(f'For experiment {folder}: mixed mass & molar ratios.')
		elif mass_count<4 and molar_count<4:
			raise ValueError(f'For experiment {folder}: incomplete formulation (mass={mass_count}, molar={molar_count})')
		elif mass_count == 4:
			cat_lip_mol_fracs, phos_mol_fracs, chol_mol_fracs, peg_lip_mol_fracs = [], [], [], []
			for i in range(len(formulation_temp)):
				phos_id = formulation_temp['Helper_lipid_ID'][i]
				ion_lipid_mol = Chem.MolFromSmiles(main_temp['smiles'][i])
				ion_lipid_mw = Descriptors.MolWt(ion_lipid_mol)
				phospholipid_mw = float(helper_mol_weights.loc[0, phos_id])
				cholesterol_mw  = float(helper_mol_weights.loc[0, 'Cholesterol'])
				PEG_lipid_mw    = float(helper_mol_weights.loc[0, 'C14-PEG2000'])
				ion_moles  = formulation_temp['Cationic_Lipid_Mass_Ratio'][i]/ion_lipid_mw
				phos_moles = formulation_temp['Phospholipid_Mass_Ratio'][i]/phospholipid_mw
				chol_moles = formulation_temp['Cholesterol_Mass_Ratio'][i]/cholesterol_mw
				peg_moles  = formulation_temp['PEG_Lipid_Mass_Ratio'][i]/PEG_lipid_mw
				mol_sum = ion_moles + phos_moles + chol_moles + peg_moles
				cat_lip_mol_fracs.append(float(ion_moles/mol_sum*100.0))
				phos_mol_fracs.append(float(phos_moles/mol_sum*100.0))
				chol_mol_fracs.append(float(chol_moles/mol_sum*100.0))
				peg_lip_mol_fracs.append(float(peg_moles/mol_sum*100.0))
			
			# Ensure list lengths match in case of iteration issues
			def _ensure_len(lst, n):
				if len(lst) == n:
					return lst
				val = (lst[0] if len(lst) > 0 else np.nan)
				return [val] * n
			
			n_rows = len(formulation_temp)
			cat_lip_mol_fracs = _ensure_len(cat_lip_mol_fracs, n_rows)
			phos_mol_fracs    = _ensure_len(phos_mol_fracs,    n_rows)
			chol_mol_fracs    = _ensure_len(chol_mol_fracs,    n_rows)
			peg_lip_mol_fracs = _ensure_len(peg_lip_mol_fracs, n_rows)
			
			formulation_temp['Cationic_Lipid_Mol_Ratio'] = cat_lip_mol_fracs
			formulation_temp['Phospholipid_Mol_Ratio']   = phos_mol_fracs
			formulation_temp['Cholesterol_Mol_Ratio']    = chol_mol_fracs
			formulation_temp['PEG_Lipid_Mol_Ratio']      = peg_lip_mol_fracs
			
			if len(individual_temp) != data_n:
				raise ValueError(f'For experiment {folder}: individual_metadata rows {len(individual_temp)} != main_data rows {data_n}')
		
		# Build per-row experiment metadata and drop duplicates
		experiment_temp = experiment_df[experiment_df.Experiment_ID == folder]
		experiment_temp = pd.concat([experiment_temp]*data_n, ignore_index=True).reset_index(drop=True)
		drop_cols = set(experiment_temp.columns) & (set(main_temp.columns) | set(formulation_temp.columns) | set(individual_temp.columns))
		experiment_temp = experiment_temp.drop(columns=list(drop_cols), errors='ignore')
		
		# Concatenate all parts
		folder_df = pd.concat([main_temp, formulation_temp, individual_temp], axis=1).reset_index(drop=True)
		folder_df = pd.concat([folder_df, experiment_temp], axis=1)
		
		# Mark data source per experiment folder (align with LNP_ML_test)
		folder_df['Source'] = ('internal' if folder == 'Chinese_Academy_of_Sciences' else 'external')
		
		# Merge duplicate columns inside the current folder_df via backfilling first non-null
		if folder_df.columns.duplicated().any():
			dup_names = folder_df.columns[folder_df.columns.duplicated()].unique()
			for name in dup_names:
				same = [c for c in folder_df.columns if c == name]
				merged = folder_df[same].bfill(axis=1).iloc[:, 0]
				folder_df = folder_df.drop(columns=same)
				folder_df[name] = merged
		
		# Ensure sample weights
		if 'Sample_weight' not in folder_df.columns and 'Experiment_weight' in folder_df.columns:
			folder_df['Sample_weight'] = [float(folder_df.Experiment_weight[i]) for i, _ in enumerate(folder_df.smiles)]
		elif 'Sample_weight' not in folder_df.columns:
			folder_df['Sample_weight'] = 1.0
		
		# Also merge duplicate columns accumulated in all_df
		if all_df.columns.duplicated().any():
			dup_names = all_df.columns[all_df.columns.duplicated()].unique()
			for name in dup_names:
				same = [c for c in all_df.columns if c == name]
				merged = all_df[same].bfill(axis=1).iloc[:, 0]
				all_df = all_df.drop(columns=same)
				all_df[name] = merged
		
		all_df = pd.concat([all_df, folder_df], ignore_index=True)

	# Make the column type dict
	extra_x_variables = ['Cationic_Lipid_Mol_Ratio','Phospholipid_Mol_Ratio','Cholesterol_Mol_Ratio','PEG_Lipid_Mol_Ratio','Cationic_Lipid_to_mRNA_weight_ratio']
	# ADD HELPER LIPID ID
	# extra_x_categorical = ['Delivery_target','Helper_lipid_ID','Route_of_administration','Batch_or_individual_or_barcoded','screen_id']
	extra_x_categorical = ['Delivery_target','Helper_lipid_ID','Route_of_administration','Batch_or_individual_or_barcoded','Cargo_type','Model_type']
	# extra_x_categorical = ['Delivery_target', 'Helper_lipid_ID', 'Route_of_administration', 
	# 					'Batch_or_individual_or_barcoded', 'Cargo_type', 'Model_type',
	# 					'Purity', 'Mix_type', 'Target_or_delivered_gene', 'Value_name']
	# Make the column type dict
	# extra_x_variables = ['Cationic_Lipid_Mol_Ratio','Phospholipid_Mol_Ratio','Cholesterol_Mol_Ratio','PEG_Lipid_Mol_Ratio','Cationic_Lipid_to_mRNA_weight_ratio']
	# extra_x_categorical = ['Delivery_target', 'Helper_lipid_ID', 'Route_of_administration', 
	# 				   'Batch_or_individual_or_barcoded', 'Cargo_type', 'Model_type',
	# 				   'Purity', 'Mix_type', 'Target_or_delivered_gene', 'Value_name']

	# Normalize common tokens to reduce category fragmentation
	all_df = all_df.replace('im','intramuscular')
	all_df = all_df.replace('iv','intravenous')
	all_df = all_df.replace('a549','lung_epithelium')
	all_df = all_df.replace('bdmc','macrophage')
	all_df = all_df.replace('bmdm','dendritic_cell')
	all_df = all_df.replace('hela','generic_cell')
	all_df = all_df.replace('hek','generic_cell')
	all_df = all_df.replace('igrov1','generic_cell')
	if 'Model_type' in all_df.columns:
		all_df['Model_type'] = all_df['Model_type'].replace({'muscle':'Mouse','mouse':'Mouse','mice':'Mouse'})
	# Compatibility: rename Cargo -> Cargo_type if needed (align with LNP_ML_test)
	if 'Cargo' in all_df.columns and 'Cargo_type' not in all_df.columns:
		all_df = all_df.rename(columns = {'Cargo':'Cargo_type'})

	# Apply roles file (path adjust if needed)
	roles_path = '../data/internal_column_roles.csv'
	col_type_map = {}  # name -> (Type, Y_task)
	try:
		roles_df = pd.read_csv(roles_path)
		roles_df.columns = [c.strip().lower() for c in roles_df.columns]
		roles_df.rename(columns={'column_name':'Column_name','role':'Role','task':'Y_task'}, inplace=True)
		if 'Y_task' not in roles_df.columns:
			roles_df['Y_task'] = ''
		for _, r in roles_df.iterrows():
			name = str(r['Column_name'])
			role = str(r['Role']).strip().upper()
			y_task = (str(r['Y_task']).strip().lower() if pd.notna(r['Y_task']) else '')
			if role == 'Y':
				y_task = y_task if y_task in ['regression','classification'] else 'regression'
				col_type_map[name] = ('Y_val', y_task)
				if name not in all_df.columns:
					all_df[name] = np.nan  # ensure column exists for masking
			elif role == 'X':
				col_type_map[name] = ('X_val','')
			else:
				col_type_map[name] = ('Metadata','')
	except Exception as e:
		print('Warning: roles file not applied:', e)
	
	# Normalize metadata/missing values per roles (align with LNP_ML_test)
	all_df = _normalize_metadata_missing(all_df, col_type_map, fill_numeric_with_zero=True)
	
	# Turn common missing tokens to NaN across all object columns
	_missing_tokens = {'', 'na', 'NA', 'Na', 'N/A', 'none', 'None'}
	for c in all_df.columns:
		if all_df[c].dtype == object:
			s = all_df[c].astype(str).str.strip()
			all_df.loc[s.isin(_missing_tokens), c] = np.nan
	
	# One-hot for selected categorical X
	# extra_x_variables = ['Cationic_Lipid_Mol_Ratio','Phospholipid_Mol_Ratio','Cholesterol_Mol_Ratio','PEG_Lipid_Mol_Ratio','Cationic_Lipid_to_mRNA_weight_ratio']
	# extra_x_categorical = ['Delivery_target', 'Helper_lipid_ID', 'Route_of_administration', 
	# 				   'Batch_or_individual_or_barcoded', 'Cargo_type', 'Model_type',
	# 				   'Purity', 'Mix_type', 'Target_or_delivered_gene', 'Value_name']
	for x_cat in extra_x_categorical:
		if x_cat in all_df.columns:
			if x_cat in col_type_map and col_type_map[x_cat][0] == 'Y_val':
				continue
			dummies = pd.get_dummies(all_df[x_cat], prefix=x_cat)
			
			dummies.columns = [col.replace('.', '_').replace('~', 'to').replace(' ', '') 
							for col in dummies.columns]
			
			dummies = dummies.loc[:, ~dummies.columns.isin(all_df.columns)]
			all_df = pd.concat([all_df, dummies], axis=1)
	
	# Merge any remaining duplicate columns
	if all_df.columns.duplicated().any():
		dup_names = all_df.columns[all_df.columns.duplicated()].unique()
		for name in dup_names:
			same = [c for c in all_df.columns if c == name]
			merged = all_df[same].bfill(axis=1).iloc[:, 0]
			all_df = all_df.drop(columns=same)
			all_df[name] = merged
	
	# One-hot classification Y targets (string labels -> multiple 0/1 target columns)
	# This aligns with LNP_ML_test behavior for multi-class classification
	from pandas.api.types import is_object_dtype
	class_y_cols = [name for name, (typ, ytask) in col_type_map.items()
					if typ == 'Y_val' and ytask == 'classification' and name in all_df.columns]

	for col in class_y_cols:
		col_obj = all_df[col]
		if isinstance(col_obj, pd.DataFrame):
			merged = col_obj.bfill(axis=1).iloc[:, 0]
			all_df.drop(columns=[c for c in all_df.columns if c == col], inplace=True)
			all_df[col] = merged
			col_obj = all_df[col]
		
		if is_object_dtype(col_obj):
			dummies = pd.get_dummies(col_obj, prefix=col).astype(float)
			
			dummies.columns = [col.replace('.', '_').replace('~', 'to').replace(' ', '').replace('<', 'lt').replace('>', 'gt').replace('=', 'eq') 
							for col in dummies.columns]
			
			all_df = pd.concat([all_df.drop(columns=[col]), dummies], axis=1)
			
			for dcol in dummies.columns:
				col_type_map[dcol] = ('Y_val', 'classification')
			col_type_map.pop(col, None)
	
	# Build col_type table (Type + Y_task)
	for c in all_df.columns:
		if c == 'Sample_weight':
			col_type['Column_name'].append(c); col_type['Type'].append('Sample_weight'); y_task_col.append('')
		elif c in col_type_map:
			t, ytask = col_type_map[c]
			col_type['Column_name'].append(c); col_type['Type'].append(t); y_task_col.append(ytask)
		elif c in extra_x_variables:
			col_type['Column_name'].append(c); col_type['Type'].append('X_val'); y_task_col.append('')
		else:
			col_type['Column_name'].append(c); col_type['Type'].append('Metadata'); y_task_col.append('')
	
	col_type_df = pd.DataFrame(col_type)
	col_type_df['Y_task'] = y_task_col

	# Normalize quantified_delivery (both external and internal data)
	if 'quantified_delivery' in all_df.columns:
		all_df['split_name_for_normalization'] = np.nan
		
		# External rows: use quantified_delivery as unnormalized_delivery
		ext_mask = (all_df['Source'] != 'internal') if 'Source' in all_df.columns else pd.Series([True]*len(all_df))
		if ext_mask.any():
			sn_ext, norm_ext = generate_normalized_data(all_df.loc[ext_mask].copy())
			all_df.loc[ext_mask, 'split_name_for_normalization'] = sn_ext
			all_df.loc[ext_mask, 'unnormalized_delivery'] = pd.to_numeric(all_df.loc[ext_mask, 'quantified_delivery'], errors='coerce')
			all_df.loc[ext_mask, 'quantified_delivery'] = norm_ext
		
		# Internal rows: use quantified_total_luminescence as unnormalized_delivery, then normalize
		int_mask = (~ext_mask) if isinstance(ext_mask, pd.Series) else pd.Series([False]*len(all_df))
		if int_mask.any():
			# Check if quantified_total_luminescence exists for internal data
			if 'quantified_total_luminescence' in all_df.columns:
				# Fill unnormalized_delivery with quantified_total_luminescence for internal data
				all_df.loc[int_mask, 'unnormalized_delivery'] = pd.to_numeric(
					all_df.loc[int_mask, 'quantified_total_luminescence'], errors='coerce'
				)
				
				# Temporarily set quantified_delivery to unnormalized_delivery for normalization
				all_df.loc[int_mask, 'quantified_delivery'] = all_df.loc[int_mask, 'unnormalized_delivery']
				
				# Normalize internal data using the same method as external data
				sn_int, norm_int = generate_normalized_data(all_df.loc[int_mask].copy())
				all_df.loc[int_mask, 'split_name_for_normalization'] = sn_int
				all_df.loc[int_mask, 'quantified_delivery'] = norm_int
			else:
				# Fallback: keep original behavior if quantified_total_luminescence doesn't exist
				all_df.loc[int_mask, 'unnormalized_delivery'] = np.nan
				all_df.loc[int_mask, 'split_name_for_normalization'] = 'internal'
	
	# if raw Delivery_target missing, rebuild from one-hot or fill NA
	if 'Delivery_target' not in all_df.columns:
		dt_oh = [c for c in all_df.columns if c.startswith('Delivery_target_')]
		if len(dt_oh) > 0:
			all_df['Delivery_target'] = all_df[dt_oh].idxmax(axis=1).str.replace('Delivery_target_', '', n=1)
		else:
			all_df['Delivery_target'] = np.nan
	
	# Ensure key targets are numeric (align with LNP_ML_test)
	for col in ['quantified_total_luminescence', 'quantified_delivery']:
		if col in all_df.columns:
			all_df[col] = pd.to_numeric(all_df[col], errors='coerce')
	
	# Convert booleans to numeric for robustness
	all_df = all_df.replace({True:1.0, False:0.0})

	if 'Source' in all_df.columns:
		internal_mask = all_df['Source'] == 'internal'
		
		y_cols = [c for c, (t, _) in col_type_map.items() if t == 'Y_val']
		
		for col in y_cols:
			if col in all_df.columns:
				nan_count_before = all_df.loc[internal_mask, col].isna().sum()
				if nan_count_before > 0:
					all_df.loc[internal_mask & all_df[col].isna(), col] = 0.0
					print(f"Filled {nan_count_before} NaN -> 0 for internal data in Y column: {col}")
					
	all_df.to_csv(write_path + '/all_data.csv', index=False, na_rep='NaN')
	col_type_df.to_csv(write_path + '/col_type.csv', index=False)
	print('Merged to:', write_path + '/all_data.csv')
	print('Column roles to:', write_path + '/col_type.csv')
	
	args_dir = os.path.join(write_path, 'args_files')
	os.makedirs(args_dir, exist_ok=True)
	
	mask_y = (col_type_df['Type'] == 'Y_val')
	reg_targets = sorted(col_type_df.loc[mask_y & (col_type_df['Y_task'] == 'regression'), 'Column_name'].tolist())
	clf_targets = sorted(col_type_df.loc[mask_y & (col_type_df['Y_task'] == 'classification'), 'Column_name'].tolist())
	
	target_roles = {
		"regression_targets": reg_targets,
		"classification_targets": clf_targets
	}
	with open(os.path.join(args_dir, 'target_roles.json'), 'w', encoding='utf-8') as f:
		json.dump(target_roles, f, indent=2, ensure_ascii=False)
	print('Target roles to:', os.path.join(args_dir, 'target_roles.json'))


def split_df_by_col_type(df, col_types):
	"""
	Split dataframe into Y (targets), X (features), W (weights), and M (metadata)
	
	Args:
		df: Input dataframe
		col_types: DataFrame with columns 'Column_name' and 'Type'
	
	Returns:
		y, x, w, m: DataFrames for targets, features, weights, and metadata
	"""
	# Get column types
	y_cols = list(col_types.Column_name[col_types.Type == 'Y_val'])
	x_cols = list(col_types.Column_name[col_types.Type == 'X_val'])
	w_col = list(col_types.Column_name[col_types.Type == 'Sample_weight'])
	
	# Ensure PDI columns are in Y if they exist
	pdi_cols = [col for col in df.columns if col.startswith('PDI_')]
	for pdi_col in pdi_cols:
		if pdi_col not in y_cols and pdi_col in df.columns:
			y_cols.append(pdi_col)
			if pdi_col in x_cols:
				x_cols.remove(pdi_col)
	
	# Ensure size is in Y if it exists
	if 'size' in df.columns and 'size' not in y_cols:
		y_cols.append('size')
		if 'size' in x_cols:
			x_cols.remove('size')
	
	# Get available columns
	y_cols = [c for c in y_cols if c in df.columns]
	x_cols = [c for c in x_cols if c in df.columns]
	
	# Add smiles to Y if present
	if 'smiles' in df.columns and 'smiles' not in y_cols:
		y_cols = ['smiles'] + y_cols
	
	# Split dataframes
	y_df = df[y_cols] if len(y_cols) > 0 else pd.DataFrame()
	x_df = df[x_cols] if len(x_cols) > 0 else pd.DataFrame()
	
	# Handle weights
	if len(w_col) > 0 and w_col[0] in df.columns:
		w_df = df[[w_col[0]]]
	else:
		w_df = pd.DataFrame({'Sample_weight': [1.0] * len(df)})
	
	# Metadata is everything else
	m_cols = [c for c in df.columns if c not in y_cols and c not in x_cols and c not in w_col]
	m_df = df[m_cols] if len(m_cols) > 0 else pd.DataFrame()
	
	# Convert to numeric and fill NaN appropriately, but preserve smiles as string
	if 'smiles' in y_df.columns:
		smiles_col = y_df['smiles'].copy()
		y_df_numeric = y_df.drop(columns=['smiles']).apply(pd.to_numeric, errors='coerce')
		y_df = pd.concat([smiles_col, y_df_numeric], axis=1)
	else:
		y_df = y_df.apply(pd.to_numeric, errors='coerce')
	x_df = x_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)  # Fill 0 for X (features)
	w_df = w_df.apply(pd.to_numeric, errors='coerce').fillna(1.0)  # Fill 1 for weights
	
	return y_df, x_df, w_df, m_df

# def do_all_splits(path_to_splits = 'Data/Multitask_data/All_datasets/Split_specs'):
	all_csvs = os.listdir(path_to_splits)
	for csv in all_csvs:
		if csv.endswith('.csv'):
			specified_dataset_split(csv)

def train_valid_test_split(vals, train_frac, valid_frac, test_frac, random_state = 42):
	# only works for list inputs
	if train_frac + valid_frac + test_frac > 99:
		train_frac = float(train_frac)/100
		valid_frac = float(valid_frac)/100
		test_frac = float(test_frac)/100
	if abs(train_frac + valid_frac + test_frac-1)>0.01:
		raise ValueError('Sum of train, valid, test fractions is not 1! It\'s: ',train_frac + valid_frac + test_frac)
	if test_frac>0 and test_frac < 1:
		train, test = train_test_split(vals, test_size = test_frac, random_state = random_state)
	elif test_frac == 1:
		test = vals
		train = []
	else:
		train = vals
		test = []
	if valid_frac > 0 and valid_frac < 1:
		train, valid = train_test_split(train, test_size = valid_frac/(train_frac+valid_frac), random_state = random_state*2)
	elif valid_frac == 0:
		valid = []
	else:
		valid = train
		train = []
	return train, valid, test


def split_for_cv(vals,cv_fold, held_out_fraction):
	# randomly splits vals into cv_fold groups, plus held_out_fraction of vals are completely held out. So for example split_for_cv(vals,5,0.1) will hold out 10% of data and randomly put 18% into each of 5 folds
	random.shuffle(vals)
	held_out_vals = vals[:int(held_out_fraction*len(vals))]
	cv_vals = vals[int(held_out_fraction*len(vals)):]
	return [cv_vals[i::cv_fold] for i in range(cv_fold)],held_out_vals

def nested_split_for_cv(vals,cv_fold):
	# Returns nested_cv_vals: nested_cv_vals[i] has 
	random.shuffle(vals)
	initial_split = split_for_cv_for_nested(vals, cv_fold)
	nested_cv_vals = [([],initial_split[i]) for i in range(cv_fold)]
	for i in range(cv_fold):
		to_split = []
		for j in range(cv_fold):
			if j != i:
				to_split = to_split + initial_split[j]
		training_splits = split_for_cv_for_nested(to_split,cv_fold)
		for k in range(cv_fold):
			interior_split = []
			for l in range(cv_fold):
				if k != l:
					interior_split = interior_split + training_splits[l]
			nested_cv_vals[i][0].append((interior_split,training_splits[k]))
	return nested_cv_vals

def split_for_cv_for_nested(vals, cv_fold):
	random.shuffle(vals)
	return [vals[int(i*(len(vals)/cv_fold)):int((i+1)*(len(vals)/cv_fold))] for i in range(cv_fold)]

def specified_nested_cv_split(split_spec_fname, path_to_folders = '../data', is_morgan = False, cv_fold = 5, min_unique_vals = 2.0, pred_split_variables = ['Experiment_ID','Library_ID','Delivery_target','Route_of_administration']):
	# Splits the dataset according to the specifications in split_spec_fname
	# cv_fold: self-explanatory
	# ultra_held_out_fraction: if you want to hold a dataset out from even the cross-validation datasets this is the way to do it
	# This generates a NESTED split: for each of the cv_fold folds, there is a held-out test set and the training set. The training set is then split cv_fold different times into training and validation sets.
	# So, there are cv_fold^2 total splits of (training, validation, test)
	# Also adds a new row, "Experiment_grouping_ID". The rows sharing a grouping ID can be compared between each other since they share (by default) an experiment ID, library ID, delivery target, and route of administration

	all_df = pd.read_csv(path_to_folders + '/all_data.csv')
	pred_split_names = []
	for index, row in all_df.iterrows():
		pred_split_name = ''
		for vbl in pred_split_variables:
			pred_split_name = pred_split_name + row[vbl] + '_'
		pred_split_names.append(pred_split_name[:-1])
	all_df['Experiment_grouping_ID'] = pred_split_names



	split_df = pd.read_csv(path_to_folders+'/crossval_split_specs/'+split_spec_fname)
	split_path = path_to_folders + '/nested_crossval_splits/' + split_spec_fname[:-4]
	if is_morgan:
		split_path = split_path + '_morgan'
	for i in range(cv_fold):
		for j in range(cv_fold):
			path_if_none(split_path+'/test_cv_'+str(i)+'/valid_cv_'+str(j))

	perma_train = pd.DataFrame({})
	ultra_held_out = pd.DataFrame({})
	# nested_cv_vals = [([],initial_split[i]) for i in range(cv_fold)]
	nested_cv_splits = [[[[pd.DataFrame({}),pd.DataFrame({})] for _ in range(cv_fold)],pd.DataFrame({})] for _ in range(cv_fold)]
	# sub_cv_splits = 

	for index, row in split_df.iterrows():
		dtypes = row['Data_types_for_component'].split(',')
		vals = row['Values'].split(',')
		df_to_concat = all_df
		for i, dtype in enumerate(dtypes):
			df_to_concat = df_to_concat[df_to_concat[dtype.strip()]==vals[i].strip()].reset_index(drop = True)
		values_to_split = df_to_concat[row['Data_type_for_split']]
		unique_values_to_split = list(set(values_to_split))
		# print(row)
		if row['Train_or_split'].lower() == 'train' or len(unique_values_to_split)<min_unique_vals*cv_fold:
			perma_train = pd.concat([perma_train, df_to_concat])
		elif row['Train_or_split'].lower() == 'split':
			nested_cv_split_values = nested_split_for_cv(unique_values_to_split, cv_fold)
			# print('Type: ',type(to_concat))
			# print('Ultra held out type: ',type(ultra_held_out))
			for i in range(cv_fold):
				testvals = nested_cv_split_values[i][1]
				for j in range(cv_fold):
					trainvals = nested_cv_split_values[i][0][j][0]
					validvals = nested_cv_split_values[i][0][j][1]
					nested_cv_splits[i][0][j][0] = pd.concat([nested_cv_splits[i][0][j][0], df_to_concat[df_to_concat[row['Data_type_for_split']].isin(trainvals)]])
					nested_cv_splits[i][0][j][1] = pd.concat([nested_cv_splits[i][0][j][1], df_to_concat[df_to_concat[row['Data_type_for_split']].isin(validvals)]])
				nested_cv_splits[i][1] = pd.concat([nested_cv_splits[i][1], df_to_concat[df_to_concat[row['Data_type_for_split']].isin(testvals)]])

	col_types = pd.read_csv(path_to_folders + '/col_type.csv')
	col_types.loc[len(col_types.index)] = ['Experiment_grouping_ID','Metadata']


	for i in range(cv_fold):
		test_df = nested_cv_splits[i][1]
		# print(test_df.columns)
		y,x,w,m = split_df_by_col_type(test_df,col_types)
		yxwm_to_csvs(y,x,w,m,split_path+'/test_cv_'+str(i),'test')

		for j in range(cv_fold):
			train_df = nested_cv_splits[i][0][j][0]
			train_df = pd.concat([perma_train,train_df])
			y,x,w,m = split_df_by_col_type(train_df,col_types)
			yxwm_to_csvs(y,x,w,m,split_path+'/test_cv_'+str(i)+'/valid_cv_'+str(j),'train')

			valid_df = nested_cv_splits[i][0][j][1]
			y,x,w,m = split_df_by_col_type(valid_df,col_types)
			yxwm_to_csvs(y,x,w,m,split_path+'/test_cv_'+str(i)+'/valid_cv_'+str(j),'valid')

		# valid_df = cv_splits[(i+1)%cv_fold]
		# train_inds = list(range(cv_fold))
		# train_inds.remove(i)
		# train_inds.remove((i+1)%cv_fold)
		# train_df = pd.concat([perma_train]+[cv_splits[k] for k in train_inds])

		
		# y,x,w,m = split_df_by_col_type(valid_df,col_types)
		# yxwm_to_csvs(y,x,w,m,split_path+'/cv_'+str(i),'valid')
		# y,x,w,m = split_df_by_col_type(train_df,col_types)
		# yxwm_to_csvs(y,x,w,m,split_path+'/cv_'+str(i),'train')

def specified_cv_split(split_spec_fname, path_to_folders = '../data', is_morgan = False, cv_fold = 5, ultra_held_out_fraction = -1.0, min_unique_vals = 2.0, test_is_valid = False):
	# Splits the dataset according to the specifications in split_spec_fname
	# cv_fold: self-explanatory
	# ultra_held_out_fraction: if you want to hold a dataset out from even the cross-validation datasets this is the way to do it
	# test_is_valid: if true, then does the split where the test set is just the validation set, so that maximum data can be reserved for training set (this is for doing in siico screening)
	all_df = pd.read_csv(path_to_folders + '/all_data.csv', low_memory=False)
	split_df = pd.read_csv(path_to_folders+'/crossval_split_specs/'+split_spec_fname)
	split_path = path_to_folders + '/crossval_splits/' + split_spec_fname[:-4]
	if ultra_held_out_fraction>-0.5:
		split_path = split_path + '_with_ultra_held_out'
	if is_morgan:
		split_path = split_path + '_morgan'
	if test_is_valid:
		split_path = split_path + '_for_in_silico_screen'
	if ultra_held_out_fraction>-0.5:
		path_if_none(split_path + '/ultra_held_out')
	for i in range(cv_fold):
		path_if_none(split_path+'/cv_'+str(i))

	perma_train = pd.DataFrame({})
	ultra_held_out = pd.DataFrame({})
	cv_splits = [pd.DataFrame({}) for _ in range(cv_fold)]

	for index, row in split_df.iterrows():
		dtypes = row['Data_types_for_component'].split(',')
		vals = row['Values'].split(',')
		df_to_concat = all_df
		for i, dtype in enumerate(dtypes):
			df_to_concat = df_to_concat[df_to_concat[dtype.strip()]==vals[i].strip()].reset_index(drop = True)
		values_to_split = df_to_concat[row['Data_type_for_split']]
		unique_values_to_split = list(set(values_to_split))
		# print(row)
		if row['Train_or_split'].lower() == 'train' or len(unique_values_to_split)<min_unique_vals*cv_fold:
			perma_train = pd.concat([perma_train, df_to_concat])
		elif row['Train_or_split'].lower() == 'split':
			cv_split_values, ultra_held_out_values = split_for_cv(unique_values_to_split, cv_fold, ultra_held_out_fraction)
			to_concat = df_to_concat[df_to_concat[row['Data_type_for_split']].isin(ultra_held_out_values)]
			# print('Type: ',type(to_concat))
			# print('Ultra held out type: ',type(ultra_held_out))
			ultra_held_out = pd.concat([ultra_held_out, to_concat])
			for i, val in enumerate(cv_split_values):
				cv_splits[i] = pd.concat([cv_splits[i], df_to_concat[df_to_concat[row['Data_type_for_split']].isin(val)]])

	# Build classification Y set once (outside the loop)
	col_types = pd.read_csv(path_to_folders + '/col_type.csv')
	if 'Y_task' in col_types.columns:
		classification_y_cols = list(col_types.loc[(col_types['Type'] == 'Y_val') & (col_types['Y_task'] == 'classification'), 'Column_name'].astype(str))
	else:
		all_df = pd.read_csv(path_to_folders + '/all_data.csv')
		y_cols_all = list(col_types.Column_name[col_types.Type == 'Y_val'])
		classification_y_cols = []
		for c in y_cols_all:
			if c.lower() == 'smiles':
				continue
			if c in all_df.columns:
				u = set(pd.to_numeric(all_df[c], errors='coerce').dropna().unique().tolist())
				if len(u) > 0 and u.issubset({0.0, 1.0, 0, 1}):
					classification_y_cols.append(c)

	# Now move the dfs to datafiles
	if ultra_held_out_fraction>-0.5:
		y,x,w,m = split_df_by_col_type(ultra_held_out,col_types)
		yxwm_to_csvs(y,x,w,m,split_path+'/ultra_held_out','test')

	for i in range(cv_fold):
		test_df = cv_splits[i]
		train_inds = list(range(cv_fold))
		train_inds.remove(i)
		if test_is_valid:
			valid_df = cv_splits[i]
		else:
			valid_df = cv_splits[(i+1)%cv_fold]
			train_inds.remove((i+1)%cv_fold)
		train_df = pd.concat([perma_train]+[cv_splits[k] for k in train_inds])

		y,x,w,m = split_df_by_col_type(test_df,col_types)
		yxwm_to_csvs(y,x,w,m,split_path+'/cv_'+str(i),'test')
		# classification-only sets (keep smiles + classification_y_cols present in y)
		y_clf = y[['smiles'] + [c for c in classification_y_cols if c in y.columns]] if 'smiles' in y.columns else y[[c for c in classification_y_cols if c in y.columns]]
		yxwm_to_csvs(y_clf,x,w,m,split_path+'/cv_'+str(i),'test_clf')

		y,x,w,m = split_df_by_col_type(valid_df,col_types)
		yxwm_to_csvs(y,x,w,m,split_path+'/cv_'+str(i),'valid')
		y_clf = y[['smiles'] + [c for c in classification_y_cols if c in y.columns]] if 'smiles' in y.columns else y[[c for c in classification_y_cols if c in y.columns]]
		yxwm_to_csvs(y_clf,x,w,m,split_path+'/cv_'+str(i),'valid_clf')

		y,x,w,m = split_df_by_col_type(train_df,col_types)
		yxwm_to_csvs(y,x,w,m,split_path+'/cv_'+str(i),'train')
		y_clf = y[['smiles'] + [c for c in classification_y_cols if c in y.columns]] if 'smiles' in y.columns else y[[c for c in classification_y_cols if c in y.columns]]
		yxwm_to_csvs(y_clf,x,w,m,split_path+'/cv_'+str(i),'train_clf')

def yxwm_to_csvs(y, x, w, m, path,settype):
	# y is y values
	# x is x values
	# w is weights
	# m is metadata
	# set_type is either train, valid, or test
	# Coerce Y to numeric (except smiles) to avoid string->float errors
	if 'smiles' in y.columns:
		y = y.assign(**{c: pd.to_numeric(y[c], errors='coerce') for c in y.columns if c!='smiles'})
	else:
		y = y.apply(pd.to_numeric, errors='coerce')
	# Ensure X has no smiles, drop base categoricals if one-hot present, then numeric
	if 'smiles' in x.columns:
		x = x.drop(columns = ['smiles'])
	# drop base categorical columns when corresponding one-hots exist
	base_cats = ['Helper_lipid_ID','Delivery_target','Route_of_administration','Batch_or_individual_or_barcoded','Cargo_type','Model_type']
	for base in base_cats:
		if base in x.columns and any(col.startswith(base + '_') for col in x.columns):
			x = x.drop(columns=[base])
	x = x.apply(pd.to_numeric, errors='coerce').fillna(0.0)
	y.to_csv(path+'/'+settype+'.csv', index = False)
	x.to_csv(path + '/' + settype + '_extra_x.csv', index = False)
	# Ensure weights file has no header and no NaNs (Chemprop expects plain floats per line)
	w = w.fillna(1.0)
	w.to_csv(path + '/' + settype + '_weights.csv', index = False, header = False)
	m.to_csv(path + '/' + settype + '_metadata.csv', index = False)


# # def specified_dataset_split(split_spec_fname, path_to_folders = '../data', is_morgan = False):
# 	# 3 columns: Data_type, Value, Split_type
# 	# Splits the dataset according the the split specifications
# 	all_df = pd.read_csv(path_to_folders + '/all_data.csv')
# 	split_df = pd.read_csv(path_to_folders + '/Split_specs/' + split_spec_fname)
# 	split_path = path_to_folders + '/Splits/' + split_spec_fname[:-4]
# 	if is_morgan:
# 		split_path = split_path + '_morgan'
# 	path_if_none(split_path)
# 	train_df = pd.DataFrame({})
# 	valid_df = pd.DataFrame({})
# 	test_df = pd.DataFrame({})
# 	for index,row in split_df.iterrows():
# 		print(row)
# 		dtypes = row['Data_types_for_component'].split(',')
# 		vals = row['Values'].split(',')
# 		df_to_concat = all_df
# 		for i, dtype in enumerate(dtypes):
# 			print(len(df_to_concat))
# 			df_to_concat = df_to_concat[df_to_concat[dtype.strip()]==vals[i].strip()].reset_index(drop = True)
# 		print(len(df_to_concat))

# 		values_to_split = df_to_concat[row['Data_type_for_split']]
# 		unique_values_to_split = list(set(values_to_split))
# 		train_frac = float(row['Percent_train'])/100
# 		valid_frac = float(row['Percent_valid'])/100
# 		test_frac = float(row['Percent_test'])/100
# 		train_unique, valid_unique, test_unique = train_valid_test_split(unique_values_to_split,train_frac, valid_frac, test_frac)
		
# 		train_df = pd.concat([train_df,df_to_concat[df_to_concat[row['Data_type_for_split']].isin(train_unique)]])
# 		valid_df = pd.concat([valid_df,df_to_concat[df_to_concat[row['Data_type_for_split']].isin(valid_unique)]])
# 		test_df = pd.concat([test_df,df_to_concat[df_to_concat[row['Data_type_for_split']].isin(test_unique)]])
# 	train_test_valid_dfs_to_csv(split_path, train_df, valid_df, test_df, path_to_folders)

# # def all_randomly_split_dataset(path_to_folders = 'Data/Multitask_data/All_datasets'):
# 	all_df = pd.read_csv(path_to_folders + '/all_data.csv')
# 	train_df, test_df = train_test_split(all_df, test_size = 0.2, random_state = 42)
# 	train_df, valid_df = train_test_split(train_df, test_size = 0.25, random_state = 27)
# 	newpath = path_to_folders + '/Splits/Fully_random_splits'
# 	if not os.path.exists(newpath):
# 		os.makedirs(newpath)
# 	train_test_valid_dfs_to_csv(newpath, train_df, valid_df, test_df, path_to_folders)


def train_test_valid_dfs_to_csv(path_to_splits, train_df, valid_df, test_df, path_to_col_types):
	# Sends the training, validation, and test dataframes to csv as determined by the column types
	col_types = pd.read_csv(path_to_col_types + '/col_type.csv')

	y_vals,x_vals,weights,metadata_cols = split_df_by_col_type(train_df,col_types)
	y_vals_v,x_vals_v,weights_v,metadata_cols_v = split_df_by_col_type(valid_df,col_types)
	for col in y_vals.columns:
		if col != 'smiles':
			if np.isnan(np.nanmax(y_vals[col])):
				print('Deleting column ',col,' from training and validation sets due to lack of values in the training set')
				y_vals = y_vals.drop(columns = [col])
				y_vals_v = y_vals_v.drop(columns = [col])
			elif np.isnan(np.nanmax(y_vals_v[col])):
				print('Deleting column ',col,' from training and validation sets due to lack of values in the validation set')
				y_vals = y_vals.drop(columns = [col])
				y_vals_v = y_vals_v.drop(columns = [col])

	settype = 'train'
	y_vals.to_csv(path_to_splits + '/' + settype + '.csv', index = False)
	x_vals.to_csv(path_to_splits + '/' + settype + '_extra_x.csv', index = False)
	# Ensure no header and no NaNs for weights
	weights = weights.fillna(1.0)
	weights.to_csv(path_to_splits + '/' + settype + '_weights.csv', index = False, header = False)
	metadata_cols.to_csv(path_to_splits + '/' + settype + '_metadata.csv', index = False)

	settype = 'valid'
	y_vals_v.to_csv(path_to_splits + '/' + settype + '.csv', index = False)
	x_vals_v.to_csv(path_to_splits + '/' + settype + '_extra_x.csv', index = False)
	# Ensure no header and no NaNs for weights
	weights_v = weights_v.fillna(1.0)
	weights_v.to_csv(path_to_splits + '/' + settype + '_weights.csv', index = False, header = False)
	metadata_cols_v.to_csv(path_to_splits + '/' + settype + '_metadata.csv', index = False)

	y_vals,x_vals,weights,metadata_cols = split_df_by_col_type(test_df,col_types)
	settype = 'test'
	y_vals.to_csv(path_to_splits + '/' + settype + '.csv', index = False)
	x_vals.to_csv(path_to_splits + '/' + settype + '_extra_x.csv', index = False)
	# Ensure no header and no NaNs for weights
	weights = weights.fillna(1.0)
	weights.to_csv(path_to_splits + '/' + settype + '_weights.csv', index = False, header = False)
	metadata_cols.to_csv(path_to_splits + '/' + settype + '_metadata.csv', index = False)


def path_if_none(newpath):
	if not os.path.exists(newpath):
		os.makedirs(newpath)

# def run_training(path_to_folders = 'Data/Multitask_data/All_datasets/Splits', epochs = 40):
# 	train_multitask_model(get_base_args(),path_to_folders, epochs = epochs)

def run_ensemble_training(path_to_folders, ensemble_size = 5, epochs = 40, generator = None):
	for i in range(ensemble_size):
		train_multitask_model(get_base_args(), path_to_folders, epochs = epochs, generator = generator)
		os.rename(path_to_folders+'/trained_model',path_to_folders + '/trained_model_'+str(i))

def run_optimized_ensemble_training(path_to_folders, ensemble_size = 5, epochs = 40, generator = None, path_to_hyperparameters = '../data/args_files'):
	# Runs training according to the hyperparameter-optimized configurations identified in path_to_hyperparameters (or just path_to_folders if path_to_hyperparameters is not specified)
	opt_hyper = json.load(open(path_to_hyperparameters + '/optimized_configs.json','r'))
	print(opt_hyper)
	for i in range(ensemble_size):
		train_hyperparam_optimized_model(get_base_args(), path_to_folders, opt_hyper['depth'], opt_hyper['dropout'], opt_hyper['ffn_num_layers'], opt_hyper['hidden_size'], epochs = epochs, generator = generator)
		os.rename(path_to_folders+'/trained_model',path_to_folders + '/trained_model_'+str(i))

def run_all_trainings(path_to_splits = '../data'):
	# Do all trainings listed in Split_specs
	all_csvs = os.listdir(path_to_splits+'/Split_specs')
	for csv in all_csvs:
		if csv.endswith('.csv'):
			path_to_folders = path_to_splits + '/Splits/'+csv[:-4]
			if not os.path.isdir(path_to_folders+'/trained_model'):
				# print('haven\'t yet trained: ',csv)
				run_training(path_to_folders = path_to_folders)
			else:
				print('already trained ',csv)

# def combine_predictions(splits,combo_name, path_to_folders = 'Data/Multitask_data/All_datasets/Splits'):
# 	savepath = path_to_folders + '/Prediction_combos/'+combo_name
# 	path_if_none(savepath)
# 	all_df = {}
# 	for i,split in enumerate(splits):
# 		pred_df = pd.read_csv(path_to_folders +'/' + split + '/Predicted_vs_actual_in_silico.csv')
# 		# print(pred_df.smiles[:10])
# 		if i == 0:
# 			all_df['smiles'] = [smiles for smiles in pred_df['smiles']]
# 		# print(all_df['smiles'][:10])
# 		preds = pred_df['Avg_pred_quantified_delivery']
# 		mean = np.mean(preds)
# 		std = np.std(preds)
# 		all_df[split] = [(v - mean)/std for v in preds]
# 	all_avgs = []
# 	all_stds = []
# 	all_df = pd.DataFrame(all_df)
# 	print(all_df.head(10))
# 	print('now about to do a thing')
# 	for i, row in all_df.iterrows():
# 		all_avgs.append(np.mean([row[split] for split in splits]))
# 		all_stds.append(np.std([row[split] for split in splits]))
# 	all_df['Avg_pred'] = all_avgs
# 	all_df['Std_pred'] = all_stds
# 	all_df['Confidence'] = [1/val for val in all_df['Std_pred']]
# 	print(all_df.head(10))
# 	all_df.to_csv(savepath + '/predictions.csv', index = False)
# 	top_100 = np.argpartition(np.array(all_df.Avg_pred),-100)[-100:]
# 	top_100_df = all_df.loc[list(top_100),:]
# 	print('head of top 100: ')
# 	print(top_100_df.head(10))
# 	top_100_df.to_csv(savepath + '/top_100.csv',index = False)

# 	preds_for_pareto = all_df[['Avg_pred','Std_pred']].to_numpy()
# 	is_efficient = is_pareto_efficient(preds_for_pareto,return_mask = True)
# 	efficient_subset = all_df[is_efficient]

# 	plt.figure()
# 	plt.scatter(all_df.Avg_pred, all_df.Std_pred, color = 'gray')
# 	plt.scatter(efficient_subset.Avg_pred, efficient_subset.Std_pred, color = 'black')
# 	plt.xlabel('Average prediction')
# 	plt.ylabel('Standard deviation of predictions')
# 	# plt.legend(loc = 'lower right')
# 	plt.savefig(savepath + '/stdev_Pareto_frontier.png')
# 	plt.close()
# 	efficient_subset.to_csv(savepath + '/stdev_Pareto_frontier.csv', index = False)

# 	preds_for_pareto = all_df[['Avg_pred','Confidence']].to_numpy()
# 	is_efficient = is_pareto_efficient(preds_for_pareto,return_mask = True)
# 	efficient_subset = all_df[is_efficient]

# 	plt.figure()
# 	plt.scatter(all_df.Avg_pred, all_df.Std_pred, color = 'gray')
# 	plt.scatter(efficient_subset.Avg_pred, efficient_subset.Std_pred, color = 'black')
# 	plt.xlabel('Average prediction')
# 	plt.ylabel('Confidence of predictions')
# 	# plt.legend(loc = 'lower right')
# 	plt.savefig(savepath + '/confidence_Pareto_frontier.png')
# 	plt.close()
# 	efficient_subset.to_csv(savepath + '/confidence_Pareto_frontier.csv', index = False)

# 	for i in range(len(splits)):
# 		for j in range(i+1,len(splits)):
# 			plt.figure()
# 			plt.scatter(all_df[splits[i]], all_df[splits[j]],color = 'black')
# 			plt.xlabel(splits[i]+' prediction')
# 			plt.ylabel(splits[j]+' prediction')
# 			plt.savefig(savepath+'/'+splits[i]+'_vs_'+splits[j]+'.png')
# 			plt.close()


def ensemble_predict(path_to_folders = '../data/splits', ensemble_size = 5, predictions_done = [], path_to_new_test = '',standardize_predictions = True):
	# Makes predictions based on the ensemble model
	if path_to_new_test == '':
		path_to_data_folders = path_to_folders
		addition = ''
	else:
		addition = '_'+path_to_new_test
		path_to_data_folders = path_to_folders + '/in_silico_screens/'+path_to_new_test
	all_predictions = pd.read_csv(path_to_data_folders + '/test.csv')
	pred_names = list(all_predictions.columns)
	pred_names.remove('smiles')
	metadata = pd.read_csv(path_to_data_folders +'/test_metadata.csv')
	all_predictions = pd.concat([metadata, all_predictions], axis = 1)
	for i in range(ensemble_size):
		# try:
		# 	current_predictions = pd.read_csv(path_to_folders+'/trained_model_'+str(i)+'/Predictions/test_predictions'+addition+'.csv')
		# except:
		# if not i in predictions_done:
		# os.rename(path_to_folders + '/trained_model_'+str(i),path_to_folders+'/trained_model')
		make_predictions(path_to_folders, path_to_new_test = path_to_new_test, ensemble_number = i)
		# os.rename(path_to_folders+'/trained_model',path_to_folders + '/trained_model_'+str(i))
		current_predictions = pd.read_csv(path_to_folders+'/trained_model_'+str(i)+'/Predictions/test_predictions'+addition+'.csv')
		
		current_predictions.drop(columns = ['smiles'], inplace = True)
		for col in current_predictions.columns:
			if standardize_predictions:
				preds_to_standardize = current_predictions[col]
				std = np.std(preds_to_standardize)
				mean = np.mean(preds_to_standardize)
				current_predictions[col] = [(val-mean)/std for val in current_predictions[col]]
			current_predictions.rename(columns = {col:('m'+str(i)+'_pred_'+col)}, inplace = True)
		all_predictions = pd.concat([all_predictions, current_predictions], axis = 1)
	avg_pred = [[] for _ in pred_names]
	stdev_pred = [[] for _ in pred_names]
	# (root squared error)
	rse = [[] for _ in pred_names]
	# all_predictions.to_csv(path_to_folders+'/Predicted_vs_actual'+path_to_new_test+'.csv', index = False)
	for index, row in all_predictions.iterrows():
		for i,pname in enumerate(pred_names):
			all_preds = [row['m'+str(k)+'_pred_'+pname] for k in range(ensemble_size)]
			avg_pred[i].append(np.mean(all_preds))
			stdev_pred[i].append(np.std(all_preds, ddof = 1))
			if path_to_new_test=='':
				rse[i].append(np.sqrt((row[pname]-np.mean(all_preds))**2))
	for i, pname in enumerate(pred_names):
		all_predictions['Avg_pred_'+pname] = avg_pred[i]
		all_predictions['Std_pred_'+pname] = stdev_pred[i]
		if path_to_new_test == '':
			all_predictions['RSE_'+pname] = rse[i]
	all_predictions.to_csv(path_to_folders+'/Predicted_vs_actual'+path_to_new_test+'.csv', index = False)

def predict_each_test_set_cv(split, ensemble_size = 5, predictions_done = [], path_to_new_test = '',standardize_predictions = True):
	# Makes predictions on each test set in a cross-validation-split system
	# Not used for screening a new library, used for predicting on the test set of the existing dataset
	for i in range(ensemble_size):
		# try:
		# 	current_predictions = pd.read_csv(path_to_folders+'/trained_model_'+str(i)+'/Predictions/test_predictions'+addition+'.csv')
		# except:
		# if not i in predictions_done:
		# os.rename(path_to_folders + '/trained_model_'+str(i),path_to_folders+'/trained_model')
		output = pd.read_csv(path_to_folders+'/cv_'+str(i)+'/test.csv')
		metadata = pd.read_csv(path_to_folders+'/cv_'+str(i)+'/test_metadata.csv')
		output = pd.concat([metadata, output], axis = 1)
		try:
			output = pd.read_csv(path_to_folders+'/cv_'+str(i)+'/Predicted_vs_actual.csv')
		except:
			try:
				current_predictions = pd.read_csv(path_to_folders+'/cv_'+str(i)+'/trained_model/Predictions/test_predictions.csv')
			except:
				make_predictions_cv(path_to_folders, path_to_new_test = '', ensemble_number = i)
			# os.rename(path_to_folders+'/trained_model',path_to_folders + '/trained_model_'+str(i))
				current_predictions = pd.read_csv(path_to_folders+'/cv_'+str(i)+'/trained_model/Predictions/test_predictions.csv')
			
			current_predictions.drop(columns = ['smiles'], inplace = True)
			for col in current_predictions.columns:
				if standardize_predictions:
					preds_to_standardize = current_predictions[col]
					std = np.std(preds_to_standardize)
					mean = np.mean(preds_to_standardize)
					current_predictions[col] = [(val-mean)/std for val in current_predictions[col]]
				current_predictions.rename(columns = {col:('cv_'+str(i)+'_pred_'+col)}, inplace = True)
			output = pd.concat([output, current_predictions], axis = 1)
			output.to_csv(path_to_folders+'/cv_'+str(i)+'/Predicted_vs_actual.csv', index = False)

def make_pred_vs_actual(split_folder, ensemble_size = 5, predictions_done = [], path_to_new_test = '',standardize_predictions = True):
	# Makes predictions on each test set in a cross-validation-split system
	for cv in range(ensemble_size):
		data_dir = '../data/crossval_splits/'+split_folder+'/cv_'+str(cv)
		results_dir = '../results/crossval_splits/'+split_folder+'/cv_'+str(cv)
		path_if_none(results_dir)

		output = pd.read_csv(data_dir+'/test.csv')
		metadata = pd.read_csv(data_dir+'/test_metadata.csv')
		output = pd.concat([metadata, output], axis = 1)
		preds_out = results_dir+'/predicted_vs_actual.csv'
		
		if os.path.exists(preds_out):
			os.remove(preds_out)

		arguments = [
			'--test_path',data_dir+'/test.csv',
			'--features_path',data_dir+'/test_extra_x.csv',
			'--checkpoint_dir', data_dir,
			'--preds_path',data_dir+'/preds.csv'
		]
		if 'morgan' in split_folder:
			arguments = arguments + ['--features_generator','morgan_count']
		args = chemprop.args.PredictArgs().parse_args(arguments)
		chemprop.train.make_predictions(args=args)
		current_predictions = pd.read_csv(data_dir+'/preds.csv')
		current_predictions.drop(columns = ['smiles'], inplace = True)

		raw_current_predictions = current_predictions.copy()

		for col in current_predictions.columns:
			if standardize_predictions:
				preds_to_standardize = current_predictions[col]
				std = np.std(preds_to_standardize)
				mean = np.mean(preds_to_standardize)
				current_predictions[col] = [(val-mean)/std for val in current_predictions[col]]
			current_predictions.rename(columns = {col:('cv_'+str(cv)+'_pred_'+col)}, inplace = True)
		output = pd.concat([output, current_predictions], axis = 1)

		clf_dir = data_dir + '_clf'
		test_clf_csv = data_dir + '/test_clf.csv'
		raw_clf_predictions = None
		if os.path.isdir(clf_dir) and os.path.exists(test_clf_csv):
			arguments = [
				'--test_path', test_clf_csv,
				'--features_path', data_dir+'/test_extra_x.csv',
				'--checkpoint_dir', clf_dir,
				'--preds_path', data_dir+'/preds_clf.csv'
			]
			if 'morgan' in split_folder:
				arguments = arguments + ['--features_generator','morgan_count']
			args = chemprop.args.PredictArgs().parse_args(arguments)
			chemprop.train.make_predictions(args=args)
			clf_predictions = pd.read_csv(data_dir+'/preds_clf.csv')
			clf_predictions.drop(columns = ['smiles'], inplace = True)
			raw_clf_predictions = clf_predictions.copy()
			for col in clf_predictions.columns:
				if standardize_predictions:
					preds_to_standardize = clf_predictions[col]
					std = np.std(preds_to_standardize)
					mean = np.mean(preds_to_standardize)
					clf_predictions[col] = [(val-mean)/std for val in clf_predictions[col]]
				clf_predictions.rename(columns = {col:('cv_'+str(cv)+'_pred_'+col)}, inplace = True)
			output = pd.concat([output, clf_predictions], axis = 1)

		# ==================== UNIFIED POST-PROCESSING ====================
		# Convert Biodistribution predictions to Delivery_target classifications
		# This ensures consistent comparison between Chemprop and Attention models
		# =================================================================
		
		from biodistribution_to_target import apply_postprocessing_to_predictions
		from postprocessing_config import get_organ_thresholds
		
		# Use unified threshold configuration
		organ_thresholds = get_organ_thresholds('default')
		
		# Apply post-processing with cv-specific prefix
		output = apply_postprocessing_to_predictions(
			output,
			organ_thresholds=organ_thresholds,
			prediction_prefix=f'cv_{cv}_pred_',
			inplace=True
		)
		
		# ====================== END POST-PROCESSING ======================

		output.to_csv(preds_out, index = False)

		try:
			from sklearn.metrics import roc_auc_score, average_precision_score

			# Load target roles to distinguish regression vs classification
			roles_path = '../data/args_files/target_roles.json'
			try:
				with open(roles_path, 'r', encoding='utf-8') as f:
					roles = json.load(f)
				reg_target_names = set(roles.get('regression_targets', []))
				clf_target_names = set(roles.get('classification_targets', []))
			except Exception:
				reg_target_names = set()
				clf_target_names = set()

			# Generate test_scores.csv for regression only
			test_csv = data_dir + '/test.csv'
			if os.path.exists(test_csv):
				te = pd.read_csv(test_csv)
				rmse_rows = []
				for c in [x for x in te.columns if x.lower() != 'smiles']:
					# Only process regression targets
					if c not in reg_target_names:
						continue
					
					y_true = pd.to_numeric(te[c], errors='coerce').to_numpy()
					if c in raw_current_predictions.columns:
						y_pred = pd.to_numeric(raw_current_predictions[c], errors='coerce').to_numpy()
					else:
						y_pred = np.full(len(te), np.nan, dtype=float)

					m = np.isfinite(y_true) & np.isfinite(y_pred)
					if m.sum() == 0:
						rmse = float('nan')
					else:
						rmse = float(np.sqrt(np.mean((y_true[m] - y_pred[m])**2)))
					rmse_rows.append([c, rmse, 0.0, rmse])
				if len(rmse_rows) > 0:
					pd.DataFrame(rmse_rows, columns=['Task','Mean rmse','Standard deviation rmse','Fold 0 rmse']) \
						.to_csv(results_dir + '/test_scores.csv', index=False)

			# Generate test_scores_clf.csv for classification only
			test_clf_csv = data_dir + '/test_clf.csv'
			if os.path.exists(test_clf_csv) and raw_clf_predictions is not None:
				te = pd.read_csv(test_clf_csv)
				auc_rows = []
				
				for c in [x for x in te.columns if x.lower() != 'smiles']:
					# Only process classification targets
					if c not in clf_target_names:
						continue
					
					y_true = pd.to_numeric(te[c], errors='coerce').to_numpy()
					if c in raw_clf_predictions.columns:
						y_pred_raw = pd.to_numeric(raw_clf_predictions[c], errors='coerce').to_numpy()
					else:
						y_pred_raw = np.full(len(te), np.nan, dtype=float)
					
					m = np.isfinite(y_true) & np.isfinite(y_pred_raw)
					
					auc = pr = float('nan')
					
					if m.sum() > 0 and len(np.unique(y_true[m])) >= 2:
						try:
							y_pred = y_pred_raw[m]
							y_true_subset = y_true[m]
							auc = float(roc_auc_score(y_true_subset, y_pred))
							pr = float(average_precision_score(y_true_subset, y_pred))
						except Exception as e:
							pass
					
					auc_rows.append([c, auc, pr])
				
				if len(auc_rows) > 0:
					pd.DataFrame(auc_rows, columns=['Task', 'AUC', 'PR_AUC']) \
						.to_csv(results_dir + '/test_scores_clf.csv', index=False)
		except Exception as e:
			print(f"Warning: Could not generate test scores for cv_{cv}: {e}")

	if '_with_ultra_held_out' in split_folder:
		results_dir = '../results/crossval_splits/'+split_folder+'/ultra_held_out'
		path_if_none(results_dir)
		uho_dir = '../data/crossval_splits/'+split_folder+'/ultra_held_out'
		output = pd.read_csv(uho_dir+'/test.csv')
		metadata = pd.read_csv(uho_dir+'/test_metadata.csv')
		output = pd.concat([metadata, output], axis = 1)
		for cv in range(ensemble_size):
			model_dir = '../data/crossval_splits/'+split_folder+'/cv_'+str(cv)
			try:
				current_predictions = pd.read_csv(results_dir+'/preds_cv_'+str(cv)+'.csv')
			except:
				arguments = [
					'--test_path',uho_dir+'/test.csv',
					'--features_path',uho_dir+'/test_extra_x.csv',
					'--checkpoint_dir', model_dir,
					'--preds_path',results_dir+'/preds_cv_'+str(cv)+'.csv'
				]
				if 'morgan' in split_folder:
					arguments = arguments + ['--features_generator','morgan_count']
				args = chemprop.args.PredictArgs().parse_args(arguments)
				preds = chemprop.train.make_predictions(args=args)
				current_predictions = pd.read_csv(results_dir+'/preds_cv_'+str(cv)+'.csv')
			current_predictions.drop(columns = ['smiles'], inplace = True)
			for col in current_predictions.columns:
				if standardize_predictions:
					preds_to_standardize = current_predictions[col]
					std = np.std(preds_to_standardize)
					mean = np.mean(preds_to_standardize)
					current_predictions[col] = [(val-mean)/std for val in current_predictions[col]]
				current_predictions.rename(columns = {col:('cv_'+str(cv)+'_pred_'+col)}, inplace = True)
			output = pd.concat([output, current_predictions], axis = 1)
		pred_cols = [col for col in output.columns if '_pred_' in col]
		output['Avg_pred_quantified_delivery'] = output[pred_cols].mean(axis = 1)
		output.to_csv(results_dir+'/predicted_vs_actual.csv',index = False)





def ensemble_predict_cv(path_to_folders = '../data/crossval_splits', ensemble_size = 5, predictions_done = [], path_to_new_test = '',standardize_predictions = True):
	# Makes predictions on a new test set path_to_new_test (i.e. perform a screen on data stored in /in_silico_screen_results)
	# with ensemble model from cross-validation
	# i.e. this does the in silico screen of a new thing
	if not path_to_new_test == '':
		addition = '_'+path_to_new_test
		path_to_data_folders = path_to_folders + '/in_silico_screens/'+path_to_new_test
		path_if_none(path_to_folders+'/in_silico_screen_results')
		all_predictions_fname = path_to_folders+'/in_silico_screen_results/'+path_to_new_test+'.csv'
		all_predictions = pd.read_csv(path_to_data_folders + '/test.csv')
		pred_names = list(all_predictions.columns)
		pred_names.remove('smiles')
		metadata = pd.read_csv(path_to_data_folders +'/test_metadata.csv')
		all_predictions = pd.concat([metadata, all_predictions], axis = 1)
	for i in range(ensemble_size):
		# try:
		# 	current_predictions = pd.read_csv(path_to_folders+'/trained_model_'+str(i)+'/Predictions/test_predictions'+addition+'.csv')
		# except:
		# if not i in predictions_done:
		# os.rename(path_to_folders + '/trained_model_'+str(i),path_to_folders+'/trained_model')
		# print('HERE!!!!')
		try:
			current_predictions = pd.read_csv(path_to_folders+'/cv_'+str(i)+'/trained_model/Predictions/test_predictions'+addition+'.csv')
		except:
			make_predictions_cv(path_to_folders, path_to_new_test = path_to_new_test, ensemble_number = i)
		# os.rename(path_to_folders+'/trained_model',path_to_folders + '/trained_model_'+str(i))
		current_predictions = pd.read_csv(path_to_folders+'/cv_'+str(i)+'/trained_model/Predictions/test_predictions'+addition+'.csv')
		
		current_predictions.drop(columns = ['smiles'], inplace = True)
		for col in current_predictions.columns:
			if standardize_predictions:
				preds_to_standardize = current_predictions[col]
				std = np.std(preds_to_standardize)
				mean = np.mean(preds_to_standardize)
				current_predictions[col] = [(val-mean)/std for val in current_predictions[col]]
			current_predictions.rename(columns = {col:('m'+str(i)+'_pred_'+col)}, inplace = True)
		all_predictions = pd.concat([all_predictions, current_predictions], axis = 1)
	avg_pred = [[] for _ in pred_names]
	stdev_pred = [[] for _ in pred_names]
	# (root squared error)
	rse = [[] for _ in pred_names]
	# all_predictions.to_csv(path_to_folders+'/Predicted_vs_actual'+path_to_new_test+'.csv', index = False)
	for index, row in all_predictions.iterrows():
		for i,pname in enumerate(pred_names):
			all_preds = [row['m'+str(k)+'_pred_'+pname] for k in range(ensemble_size)]
			avg_pred[i].append(np.mean(all_preds))
			stdev_pred[i].append(np.std(all_preds, ddof = 1))
			if path_to_new_test=='':
				rse[i].append(np.sqrt((row[pname]-np.mean(all_preds))**2))
	for i, pname in enumerate(pred_names):
		all_predictions['Avg_pred_'+pname] = avg_pred[i]
		all_predictions['Std_pred_'+pname] = stdev_pred[i]
		if path_to_new_test == '':
			all_predictions['RSE_'+pname] = rse[i]
	all_predictions.to_csv(all_predictions_fname, index = False)

def make_predictions_cv(path_to_folders = '../data/crossval_splits', path_to_new_test = '', ensemble_number = -1):
	# Make predictions
	predict_folder = path_to_folders + '/trained_model/Predictions'
	if ensemble_number>-0.5:
		predict_folder = path_to_folders +'/cv_'+str(ensemble_number)+ '/trained_model/Predictions'
	path_if_none(predict_folder)
	predict_multitask_from_json_cv(get_base_predict_args(),model_path = path_to_folders, path_to_new_test = path_to_new_test, ensemble_number = ensemble_number)


def analyze_new_lipid_predictions(split_name, addition = '_in_silico',path_to_preds = '../data'):
	preds_vs_actual = pd.read_csv(path_to_preds + '/Splits/'+split_name+'/Predicted_vs_actual'+addition+'.csv')
	analyzed_path = path_to_preds+'/Splits/'+split_name+'/in_silico_screen_results'
	preds_vs_actual['Confidence'] = [1/val for val in preds_vs_actual['Std_pred_quantified_delivery']]
	path_if_none(analyzed_path)
	preds_for_pareto = preds_vs_actual[['Avg_pred_quantified_delivery','Std_pred_quantified_delivery']].to_numpy()
	print('Dimensions: ',preds_for_pareto.shape)
	is_efficient = is_pareto_efficient(preds_for_pareto,return_mask = True)
	efficient_subset = preds_vs_actual[is_efficient]
	# plt.figure()
	# plt.scatter(preds_vs_actual.Avg_pred_quantified_delivery, preds_vs_actual.Std_pred_quantified_delivery, color = 'gray')
	# plt.scatter(efficient_subset.Avg_pred_quantified_delivery, efficient_subset.Std_pred_quantified_delivery, color = 'black')
	# plt.xlabel('Average prediction')
	# plt.ylabel('Standard deviation of predictions')
	# # plt.legend(loc = 'lower right')
	# plt.savefig(analyzed_path + '/stdev_Pareto_frontier.png')
	# plt.close()
	efficient_subset.to_csv(analyzed_path + '/stdev_Pareto_frontier.csv', index = False)

	preds_for_pareto = preds_vs_actual[['Avg_pred_quantified_delivery','Confidence']].to_numpy()
	print('Dimensions: ',preds_for_pareto.shape)
	is_efficient = is_pareto_efficient(preds_for_pareto,return_mask = True)
	efficient_subset = preds_vs_actual[is_efficient]
	# plt.figure()
	# plt.scatter(preds_vs_actual.Avg_pred_quantified_delivery, preds_vs_actual.Std_pred_quantified_delivery, color = 'gray')
	# plt.scatter(efficient_subset.Avg_pred_quantified_delivery, efficient_subset.Std_pred_quantified_delivery, color = 'black')
	# plt.xlabel('Average prediction')
	# plt.ylabel('Standard deviation of predictions')
	# # plt.legend(loc = 'lower right')
	# plt.savefig(analyzed_path + '/confidence_Pareto_frontier.png')
	# plt.close()
	efficient_subset.to_csv(analyzed_path + '/confidence_Pareto_frontier.csv', index = False)

	top_50 = np.argpartition(np.array(preds_vs_actual.Avg_pred_quantified_delivery),-50)[-50:]
	print(list(top_50))
	top_50_df = preds_vs_actual.loc[list(top_50),:]
	top_50_df.to_csv(analyzed_path + '/top_50.csv',index = False)

def generate_normalized_data(all_df, split_variables=None):
	# Group-wise z-score with robust handling of missing group columns
	if split_variables is None:
		split_variables = ['Experiment_ID','Library_ID','Delivery_target','Model_type','Route_of_administration']
	
	# Keep only the split columns that actually exist; if none exist, normalize globally
	present = [c for c in split_variables if c in all_df.columns]
	if len(present) == 0:
		split_names = ['__all__'] * len(all_df)
	else:
		# Build group key as joined string; fill NaN to avoid "nan" propagation
		key_df = all_df[present].astype(str).fillna('NA')
		split_names = key_df.apply(lambda r: '_'.join(r.values.tolist()), axis=1).tolist()
	
	# If target column doesn't exist, return keys and NaNs
	if 'quantified_delivery' not in all_df.columns:
		return split_names, [np.nan] * len(all_df)
	
	# Ensure numeric
	qd = pd.to_numeric(all_df['quantified_delivery'], errors='coerce')
	
	# Compute mean/std per group with NaN-safety and zero-variance guard
	norm_dict = {}
	for key in set(split_names):
		mask = [k == key for k in split_names]
		sub = qd[mask]
		mu = np.nanmean(sub)
		sd = np.nanstd(sub)
		if not np.isfinite(mu): mu = 0.0
		if (not np.isfinite(sd)) or sd == 0.0: sd = 1.0
		norm_dict[key] = (float(mu), float(sd))
	
	norm_delivery = []
	for i, val in enumerate(qd):
		mu, sd = norm_dict[split_names[i]]
		norm_delivery.append((float(val) - mu) / sd if pd.notna(val) else np.nan)
	
	return split_names, norm_delivery

def is_pareto_efficient(costs, return_mask = True):
	"""
	Find the pareto-efficient points
	:param costs: An (n_points, n_costs) array
	:param return_mask: True to return a mask
	:return: An array of indices of pareto-efficient points.
		If return_mask is True, this will be an (n_points, ) boolean array
		Otherwise it will be a (n_efficient_points, ) integer array of indices.
	"""
	is_efficient = np.arange(costs.shape[0])
	n_points = costs.shape[0]
	next_point_index = 0  # Next index in the is_efficient array to search for
	while next_point_index<len(costs):
		nondominated_point_mask = np.any(costs>costs[next_point_index], axis=1)
		nondominated_point_mask[next_point_index] = True
		is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
		costs = costs[nondominated_point_mask]
		next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
	if return_mask:
		is_efficient_mask = np.zeros(n_points, dtype = bool)
		is_efficient_mask[is_efficient] = True
		return is_efficient_mask
	else:
		return is_efficient

def analyze_predictions(split_name,pred_split_variables = ['Experiment_ID','Library_ID','Delivery_target','Route_of_administration'], path_to_preds = 'Data/Multitask_data/All_datasets'):
	preds_vs_actual = pd.read_csv(path_to_preds + '/Splits/'+split_name+'/Predicted_vs_actual.csv')
	pred_split_names = []
	for index, row in preds_vs_actual.iterrows():
		pred_split_name = ''
		for vbl in pred_split_variables:
			pred_split_name = pred_split_name + row[vbl] + '_'
		pred_split_names.append(pred_split_name[:-1])
	preds_vs_actual['Prediction_split_name'] = pred_split_names
	unique_pred_split_names = set(pred_split_names)
	cols = preds_vs_actual.columns
	data_types = []
	for col in cols:
		if col.startswith('Avg_pred'):
			data_types.append(col[9:])

	summary_table = pd.DataFrame({})
	all_names = []
	all_dtypes = []
	all_ns = []
	all_pearson = []
	all_pearson_p_val = []
	all_kendall = []
	all_spearman = []
	all_rmse = []
	all_error_pearson = []
	all_error_pearson_p_val = []
	all_aucs = []
	all_goals = []

	for pred_split_name in unique_pred_split_names:
		path_if_none(path_to_preds+'/Splits/'+split_name+'/Results/'+pred_split_name)
		data_subset = preds_vs_actual[preds_vs_actual['Prediction_split_name']==pred_split_name].reset_index(drop=True)
		value_names = set(list(data_subset.Value_name))
		if len(value_names)>1:
			raise Exception('Multiple types of measurement in the same prediction split: split ',pred_split_name,' has value names ',value_names,'. Try adding more pred split variables.')
		else:
			value_name = [val_name for val_name in value_names][0]
		kept_dtypes = []
		for dtype in data_types:
			keep = False
			for val in data_subset[dtype]:
				if not np.isnan(val):
					keep = True
			if keep:
				analyzed_path = path_to_preds+'/Splits/'+split_name+'/Results/'+pred_split_name+'/'+dtype
				path_if_none(analyzed_path)
				# print(data_subset['Goal'])
				goal = data_subset['Goal'][0]
				all_goals.append(goal)
				kept_dtypes.append(dtype)
				analyzed_data = pd.DataFrame({'smiles':data_subset.smiles})
				actual = data_subset[dtype]
				pred = data_subset['Avg_pred_'+dtype]
				std_pred = data_subset['Std_pred_'+dtype]
				rse = data_subset['RSE_'+dtype]
				analyzed_data[dtype] = actual
				analyzed_data['Avg_pred_'+dtype] = pred
				analyzed_data['Std_pred_'+dtype] = std_pred
				analyzed_data['RSE_pred_'+dtype] = rse
				residuals = [actual[blah]-pred[blah] for blah in range(len(pred))]
				analyzed_data['Residual'] = residuals
				pearson = scipy.stats.pearsonr(actual, pred)
				spearman, pval = scipy.stats.spearmanr(actual, pred)
				kendall, pval = scipy.stats.kendalltau(actual, pred)
				rmse = np.sqrt(mean_squared_error(actual, pred))
				error_pearson = scipy.stats.pearsonr(std_pred,rse)
				all_names.append(pred_split_name)
				all_dtypes.append(dtype)
				all_pearson.append(pearson[0])
				all_pearson_p_val.append(pearson[1])
				all_kendall.append(kendall)
				all_spearman.append(spearman)
				all_rmse.append(rmse)
				all_error_pearson.append(error_pearson[0])
				all_error_pearson_p_val.append(error_pearson[1])
				all_ns.append(len(pred))

				# measure ROCs
				sorted_actual = sorted(actual)
				ranks = [float(sorted_actual.index(v))/len(actual) for v in actual]
				if goal == 'min':
					classification = [1*(rank<0.1) for rank in ranks]
					pred_for_class = [-v for v in pred]
				elif goal == 'max':
					classification = [1*(rank>0.9) for rank in ranks]
					pred_for_class = [v for v in pred]
				fpr, tpr, thresholds = roc_curve(classification,pred_for_class)
				# print(classification)
				try:
					auc_score = roc_auc_score(classification, pred_for_class)
				except:
					auc_score = np.nan
				all_aucs.append(auc_score)
				analyzed_data['Is_10th_percentile_hit_'+dtype] = classification


				# plt.figure()
				# plt.plot(fpr, tpr, color = 'black', label = 'ROC curve 10th percentile (area = %0.2f)' % auc_score)
				# plt.plot([0,1],[0,1],color = 'blue',linestyle = '--')
				# plt.xlim([0.0,1.0])
				# plt.ylim([0.0,1.05])
				# plt.xlabel('False positive rate')
				# plt.ylabel('True positive rate')
				# plt.legend(loc = 'lower right')
				# plt.savefig(analyzed_path + '/roc_curve.png')
				# plt.close()
				# plt.figure()
				# plt.scatter(pred,actual,color = 'black')
				# plt.plot(np.unique(pred),np.poly1d(np.polyfit(pred, actual, 1))(np.unique(pred)))
				# plt.xlabel('Predicted '+value_name)
				# plt.ylabel('Experimental '+value_name)
				# plt.savefig(analyzed_path+'/pred_vs_actual.png')
				# plt.close()
				# plt.figure()
				# plt.scatter(std_pred,residuals,color = 'black')
				# plt.plot(np.unique(std_pred),np.poly1d(np.polyfit(std_pred, residuals, 1))(np.unique(std_pred)))
				# plt.xlabel('Residual (Actual-Predicted) '+value_name)
				# plt.ylabel('Ensemble model uncertainty '+value_name)
				# plt.savefig(analyzed_path+'/residual_vs_stdev.png')
				# plt.close()
				# plt.figure()
				# plt.scatter(std_pred,rse,color = 'black')
				# plt.plot(np.unique(std_pred),np.poly1d(np.polyfit(std_pred, rse, 1))(np.unique(std_pred)))
				# plt.xlabel('Ensemble model uncertainty')
				# plt.ylabel('Root quared error')
				# plt.savefig(analyzed_path+'/std_vs_rse.png')
				# plt.close()
				analyzed_data.to_csv(analyzed_path+'/pred_vs_actual_data.csv', index = False)
	summary_table['Analysis'] = all_names
	summary_table['Measurement_type'] = all_dtypes
	summary_table['n'] = all_ns
	summary_table['Goal'] = all_goals
	summary_table['pearson_rho'] = all_pearson
	summary_table['pearson_rho_p_val'] = all_pearson_p_val
	summary_table['kendall_tau'] = all_kendall
	summary_table['spearman_r'] = all_spearman
	summary_table['rmse'] = all_rmse
	summary_table['error_pearson'] = all_error_pearson
	summary_table['error_pearson_p_val'] = all_error_pearson_p_val
	summary_table['AUC 10th percentile'] = all_aucs
	summary_table['Value_cutoff'] = ['n/a' for _ in all_aucs]
	summary_table.to_csv(path_to_preds+'/Splits/'+split_name+'/Results/Performance_summary.csv', index = False)
			
def run_optimized_cv_training(path_to_folders, ensemble_size = 5, epochs = 40, generator = None, path_to_hyperparameters = None):
	opt_hyper = json.load(open(path_to_hyperparameters + '/optimized_configs.json','r'))
	print(opt_hyper)
	for i in range(ensemble_size):
		train_hyperparam_optimized_model(get_base_args(), path_to_folders+'/cv_'+str(i), opt_hyper['depth'], opt_hyper['dropout'], opt_hyper['ffn_num_layers'], opt_hyper['hidden_size'], epochs = epochs, generator = generator)
		# os.rename(path_to_folders+'/trained_model',path_to_folders + '/trained_model_'+str(i))

def analyze_predictions_cv(split_name,pred_split_variables = ['Experiment_ID','Library_ID','Delivery_target','Route_of_administration'], path_to_preds = '../results/crossval_splits/', ensemble_number = 5, min_values_for_analysis = 10):
	summary_table = pd.DataFrame({})
	all_names = {}
	# all_dtypes = {}
	all_ns = {}
	all_pearson = {}
	all_pearson_p_val = {}
	all_kendall = {}
	all_spearman = {}
	all_rmse = {}
	all_unique = []
	for i in range(ensemble_number):
		preds_vs_actual = pd.read_csv(path_to_preds+split_name+'/cv_'+str(i)+'/predicted_vs_actual.csv')
		# rebuild Delivery_target if missing from one-hot
		if 'Delivery_target' not in preds_vs_actual.columns:
			dt_oh = [c for c in preds_vs_actual.columns if c.startswith('Delivery_target_')]
			if len(dt_oh) > 0:
				preds_vs_actual['Delivery_target'] = preds_vs_actual[dt_oh].idxmax(axis=1).str.replace('Delivery_target_', '', 1)
		# only use vars that actually exist; prepend Value_name if present
		use_vars = [v for v in pred_split_variables if v in preds_vs_actual.columns]
		if 'Value_name' in preds_vs_actual.columns and 'Value_name' not in use_vars:
			use_vars = ['Value_name'] + use_vars
		if len(use_vars) > 0:
			pred_split_names = preds_vs_actual[use_vars].astype(str).agg('_'.join, axis=1).tolist()
		else:
			pred_split_names = ['__all__'] * len(preds_vs_actual)
		all_unique = all_unique + list(set(pred_split_names))
	unique_pred_split_names = set(all_unique)
	for un in unique_pred_split_names:
		# all_names[un] = []
		# all_dtype,s[un] = []
		all_ns[un] = []
		all_pearson[un] = []
		all_pearson_p_val[un] = []
		all_kendall[un] = []
		all_spearman[un] = []
		all_rmse[un] = []
	for i in range(ensemble_number):
		preds_vs_actual = pd.read_csv(path_to_preds+split_name+'/cv_'+str(i)+'/predicted_vs_actual.csv')
		# rebuild Delivery_target if missing from one-hot
		if 'Delivery_target' not in preds_vs_actual.columns:
			dt_oh = [c for c in preds_vs_actual.columns if c.startswith('Delivery_target_')]
			if len(dt_oh) > 0:
				preds_vs_actual['Delivery_target'] = preds_vs_actual[dt_oh].idxmax(axis=1).str.replace('Delivery_target_', '', 1)
		use_vars = [v for v in pred_split_variables if v in preds_vs_actual.columns]
		if 'Value_name' in preds_vs_actual.columns and 'Value_name' not in use_vars:
			use_vars = ['Value_name'] + use_vars
		if len(use_vars) > 0:
			preds_vs_actual['Prediction_split_name'] = preds_vs_actual[use_vars].astype(str).agg('_'.join, axis=1)
		else:
			preds_vs_actual['Prediction_split_name'] = '__all__'
		cols = preds_vs_actual.columns
		data_types = []
		for col in cols:
			if col[:3]=='cv_':
				data_types.append(col)
		for pred_split_name in unique_pred_split_names:
			path_if_none(path_to_preds+split_name+'/cv_'+str(i)+'/results')
			data_subset = preds_vs_actual[preds_vs_actual['Prediction_split_name']==pred_split_name].reset_index(drop=True)
			value_names = set(list(data_subset['Value_name'])) if 'Value_name' in data_subset.columns else {'__n/a__'}
			for value_name in value_names:
				sub = data_subset if value_name == '__n/a__' else data_subset[data_subset['Value_name'] == value_name].reset_index(drop=True)
				if len(sub) == 0:
					continue
				analyzed_path = path_to_preds+split_name+'/cv_'+str(i)+'/results/'+pred_split_name
				if value_name != '__n/a__':
					analyzed_path = analyzed_path + '/' + str(value_name)
				path_if_none(analyzed_path)
				for c in data_types:
					if not c.startswith(f'cv_{i}_pred_'):
						continue
					task = c.replace(f'cv_{i}_pred_', '', 1)
					if task not in sub.columns:
						continue
					actual = pd.to_numeric(sub[task], errors='coerce')
					pred = pd.to_numeric(sub[c], errors='coerce')
					mask = np.isfinite(actual) & np.isfinite(pred)
					n = int(mask.sum())
					all_ns[pred_split_name] = all_ns[pred_split_name] + [n]
					if n >= min_values_for_analysis:
						try:
							pearson = scipy.stats.pearsonr(actual[mask], pred[mask])
						except Exception:
							pearson = (float('nan'), float('nan'))
						try:
							spearman, pval_s = scipy.stats.spearmanr(actual[mask], pred[mask])
						except Exception:
							spearman, pval_s = float('nan'), float('nan')
						try:
							kendall, pval_k = scipy.stats.kendalltau(actual[mask], pred[mask])
						except Exception:
							kendall, pval_k = float('nan'), float('nan')
						try:
							rmse = float(np.sqrt(mean_squared_error(actual[mask], pred[mask])))
						except Exception:
							rmse = float('nan')
						all_pearson[pred_split_name] = all_pearson[pred_split_name] + [float(pearson[0])]
						all_pearson_p_val[pred_split_name] = all_pearson_p_val[pred_split_name] + [float(pearson[1])]
						all_kendall[pred_split_name] = all_kendall[pred_split_name] + [float(kendall)]
						all_spearman[pred_split_name] = all_spearman[pred_split_name] + [float(spearman)]
						all_rmse[pred_split_name] = all_rmse[pred_split_name] + [rmse]
						# plt.figure()
						# plt.scatter(pred,actual,color = 'black')
						# # Use finite mask for fitting; skip if not enough variance
						# x = pred[mask].to_numpy()
						# y = actual[mask].to_numpy()
						# if len(x) >= 2 and np.isfinite(x).all() and np.isfinite(y).all() and np.std(x) > 0:
						# 	xs = np.linspace(np.min(x), np.max(x), 50)
						# 	coef = np.polyfit(x, y, 1)
						# 	plt.plot(xs, np.poly1d(coef)(xs))
						# plt.xlabel('Predicted '+str(value_name))
						# plt.ylabel('Experimental '+str(value_name))
						# plt.savefig(analyzed_path+'/pred_vs_actual.png')
						# plt.close()
					else:
						all_pearson[pred_split_name] = all_pearson[pred_split_name] + [float('nan')]
						all_pearson_p_val[pred_split_name] = all_pearson_p_val[pred_split_name] + [float('nan')]
						all_kendall[pred_split_name] = all_kendall[pred_split_name] + [float('nan')]
						all_spearman[pred_split_name] = all_spearman[pred_split_name] + [float('nan')]
						all_rmse[pred_split_name] = all_rmse[pred_split_name] + [float('nan')]
	crossval_results_path = path_to_preds+split_name+'/crossval_performance'
	path_if_none(crossval_results_path)
	def _pad_dict(d):
		maxn = max((len(v) for v in d.values()), default=0)
		out = {}
		for k, v in d.items():
			vv = list(v)
			if len(vv) < maxn:
				vv = vv + [np.nan] * (maxn - len(vv))
			out[k] = vv
		return out
	if len(all_ns) > 0:
		pd.DataFrame(_pad_dict(all_ns)).to_csv(crossval_results_path + '/n_vals.csv', index=True)
	if len(all_pearson) > 0:
		pd.DataFrame(_pad_dict(all_pearson)).to_csv(crossval_results_path + '/pearson.csv', index=True)
	if len(all_pearson_p_val) > 0:
		pd.DataFrame(_pad_dict(all_pearson_p_val)).to_csv(crossval_results_path + '/pearson_p_val.csv', index=True)
	if len(all_kendall) > 0:
		pd.DataFrame(_pad_dict(all_kendall)).to_csv(crossval_results_path + '/kendall.csv', index=True)
	if len(all_spearman) > 0:
		pd.DataFrame(_pad_dict(all_spearman)).to_csv(crossval_results_path + '/spearman.csv', index=True)
	if len(all_rmse) > 0:
		pd.DataFrame(_pad_dict(all_rmse)).to_csv(crossval_results_path + '/rmse.csv', index=True)

	# Now analyze the ultra-held-out set
	try:
		preds_vs_actual = pd.read_csv(path_to_preds+split_name+'/ultra_held_out/predicted_vs_actual.csv')
		# summary_table = pd.DataFrame({})
		names = []
		# all_dtypes = {}
		ns = []
		pearsons = []
		pearson_p_vals = []
		kendalls = []
		spearmans = []
		rmses = []
		split_names = []

		all_unique = []
			
		pred_split_names = []
		for index, row in preds_vs_actual.iterrows():
			pred_split_name = ''
			for vbl in pred_split_variables:
				pred_split_name = pred_split_name + row[vbl] + '_'
			pred_split_names.append(pred_split_name[:-1])
		all_unique = all_unique + list(set(pred_split_names))
		unique_pred_split_names = set(all_unique)
		preds_vs_actual['Prediction_split_name'] = pred_split_names
		# unique_pred_split_names = set(pred_split_names)
		cols = preds_vs_actual.columns
		data_types = []
		for col in cols:
			if col.startswith('Avg_pred_'):
				data_types.append(col)
			
		# all_error_pearson = {}
		# all_error_pearson_p_val = {}
		# all_aucs = []
		# all_goals = []

		for pred_split_name in unique_pred_split_names:
			# path_if_none(path_to_preds+split_name+'/ultra_held_out/results')
			split_names.append(pred_split_name)
			data_subset = preds_vs_actual[preds_vs_actual['Prediction_split_name']==pred_split_name].reset_index(drop=True)
			value_names = set(list(data_subset.Value_name))
			if len(value_names)>1:
				raise Exception('Multiple types of measurement in the same prediction split: split ',pred_split_name,' has value names ',value_names,'. Try adding more pred split variables.')
			elif len(value_names)==0:
				value_name = 'Empty, ignore!'
			else:
				value_name = [val_name for val_name in value_names][0]
			kept_dtypes = []
			for dtype in data_types:
				analyzed_path = path_to_preds+split_name+'/ultra_held_out/individual_dataset_results/'+pred_split_name
				path_if_none(analyzed_path)
				kept_dtypes.append(dtype)
				analyzed_data = pd.DataFrame({'smiles':data_subset.smiles})
				analyzed_data['quantified_delivery'] = data_subset['quantified_delivery']
				analyzed_data['Avg_pred_quantified_delivery'] = data_subset['Avg_pred_quantified_delivery']
				actual = data_subset['quantified_delivery']
				pred = data_subset['Avg_pred_quantified_delivery']

				pearson = scipy.stats.pearsonr(actual, pred)
				spearman, pval = scipy.stats.spearmanr(actual, pred)
				kendall, pval = scipy.stats.kendalltau(actual, pred)

				rmse = np.sqrt(mean_squared_error(actual, pred))

				rmses.append(rmse)
				pearsons.append(pearson[0])
				pearson_p_vals.append(pearson[1])
				kendalls.append(kendall)
				spearmans.append(spearman)
				ns.append(len(pred))

				# plt.figure()
				# plt.scatter(pred,actual,color = 'black')
				# plt.plot(np.unique(pred),np.poly1d(np.polyfit(pred, actual, 1))(np.unique(pred)))
				# plt.xlabel('Predicted '+value_name)
				# plt.ylabel('Experimental '+value_name)
				# plt.savefig(analyzed_path+'/pred_vs_actual.png')
				# plt.close()

				analyzed_data.to_csv(analyzed_path+'/pred_vs_actual_data.csv', index = False)
		uho_results_path = path_to_preds+split_name+'/ultra_held_out'
		path_if_none(uho_results_path)
		uho_results = pd.DataFrame({})
		uho_results['dataset_ID'] = split_names
		uho_results['n'] = ns
		uho_results['pearson'] = pearsons
		uho_results['pearson_p_val'] = pearson_p_vals
		uho_results['kendall'] = kendalls
		uho_results['spearman'] = spearmans
		uho_results['rmse'] = rmses


		uho_results.to_csv(uho_results_path+'/ultra_held_out_results.csv', index = False)
	except:
		pass



def make_predictions(path_to_folders = '../data/Splits', path_to_new_test = '', ensemble_number = -1):
	predict_folder = path_to_folders + '/trained_model/Predictions'
	if ensemble_number>-0.5:
		predict_folder = path_to_folders + '/trained_model_'+str(ensemble_number)+'/Predictions'
	path_if_none(predict_folder)
	predict_multitask_from_json(get_base_predict_args(),model_path = path_to_folders, path_to_new_test = path_to_new_test, ensemble_number = ensemble_number)

def make_all_predictions(path_to_splits = '../data'):
	all_csvs = os.listdir(path_to_splits+'/Split_specs')
	for csv in all_csvs:
		if csv.endswith('.csv'):
			path_to_folders = path_to_splits + '/Splits/'+csv[:-4]
			if not os.path.isdir(path_to_folders+'/trained_model'):
				print('haven\'t yet trained: ',csv[:-4])
				# run_training(path_to_folders = path_to_folders)
			else:
				print('Doing predictions for: ',csv[:-4])
				make_predictions(path_to_folders = path_to_folders)

def hyperparam_optimize_split(split, niters = 20):
	generator = None
	wo_in_silico = split.replace('_for_in_silico_screen','')
	if wo_in_silico.endswith('_morgan'):
		generator = ['morgan_count']
		specified_dataset_split(wo_in_silico[:-7]+'.csv',is_morgan = True)
	else:
		specified_dataset_split(wo_in_silico+'.csv')
	optimize_hyperparameters(get_base_args(), path_to_splits = 'Data/Multitask_data/All_datasets/Splits/'+split,epochs = 50, num_iters = niters, generator = generator)
	run_optimized_ensemble_training('Data/Multitask_data/All_datasets/Splits/'+split, ensemble_size = 5, epochs = 50, generator = generator)

# def analyze_predictions(split_folder, base_path = 'Data/Multitask_data/All_datasets'):

# merge_datasets(None)

# merge_datasets(['A549_form_screen','Whitehead_siRNA','LM_3CR','RM_BL_AG_carbonate'])

def check_cv_split(split_folder):
	# Verify cross-validation split integrity
	# Check for SMILES uniqueness within each fold and overlap between folds
	split_path = '../data/crossval_splits/' + split_folder
	if not os.path.exists(split_path):
		print(f"Error: split directory not found: {split_path}")
		return
	
	# Detect number of folds
	cv_dirs = [d for d in os.listdir(split_path) if d.startswith('cv_') and os.path.isdir(os.path.join(split_path, d))]
	if len(cv_dirs) == 0:
		print(f"Error: no cv_* directories found in {split_path}")
		return
	cv_nums = sorted([int(d.split('_')[1]) for d in cv_dirs])
	num_folds = len(cv_nums)
	print(f"Found {num_folds} folds in {split_folder}")
	print("")
	
	all_test_smiles = []
	all_train_smiles = []
	all_valid_smiles = []
	
	for cv in cv_nums:
		cv_dir = os.path.join(split_path, f'cv_{cv}')
		train_file = os.path.join(cv_dir, 'train.csv')
		valid_file = os.path.join(cv_dir, 'valid.csv')
		test_file = os.path.join(cv_dir, 'test.csv')
		
		if not all(os.path.exists(f) for f in [train_file, valid_file, test_file]):
			print(f"Warning: fold {cv} missing train/valid/test files")
			continue
		
		train_df = pd.read_csv(train_file)
		valid_df = pd.read_csv(valid_file)
		test_df = pd.read_csv(test_file)
		
		train_smiles = set(train_df['smiles'].tolist())
		valid_smiles = set(valid_df['smiles'].tolist())
		test_smiles = set(test_df['smiles'].tolist())
		
		# Check within-fold uniqueness
		train_dups = len(train_df) - len(train_smiles)
		valid_dups = len(valid_df) - len(valid_smiles)
		test_dups = len(test_df) - len(test_smiles)
		
		print(f"Fold {cv}:")
		print(f"  Train: {len(train_df)} samples, {len(train_smiles)} unique SMILES, {train_dups} duplicates")
		print(f"  Valid: {len(valid_df)} samples, {len(valid_smiles)} unique SMILES, {valid_dups} duplicates")
		print(f"  Test:  {len(test_df)} samples, {len(test_smiles)} unique SMILES, {test_dups} duplicates")
		
		# Check within-fold overlap
		train_valid_overlap = train_smiles & valid_smiles
		train_test_overlap = train_smiles & test_smiles
		valid_test_overlap = valid_smiles & test_smiles
		
		if train_valid_overlap:
			print(f"  WARNING: {len(train_valid_overlap)} SMILES overlap between train and valid")
		if train_test_overlap:
			print(f"  WARNING: {len(train_test_overlap)} SMILES overlap between train and test")
		if valid_test_overlap:
			print(f"  WARNING: {len(valid_test_overlap)} SMILES overlap between valid and test")
		
		all_test_smiles.append(test_smiles)
		all_train_smiles.append(train_smiles)
		all_valid_smiles.append(valid_smiles)
		print("")
	
	# Check test set overlap between folds
	print("Cross-fold test set overlap:")
	for i in range(num_folds):
		for j in range(i+1, num_folds):
			overlap = all_test_smiles[i] & all_test_smiles[j]
			if overlap:
				print(f"  Fold {cv_nums[i]} test and Fold {cv_nums[j]} test: {len(overlap)} shared SMILES")
	
	# Check total coverage
	all_test_union = set().union(*all_test_smiles)
	print("")
	print(f"Total unique SMILES across all test sets: {len(all_test_union)}")
	
	# Check against original data
	all_data_file = '../data/all_data.csv'
	if os.path.exists(all_data_file):
		all_data = pd.read_csv(all_data_file, low_memory=False)
		all_data_smiles = set(all_data['smiles'].dropna().tolist())
		print(f"Total unique SMILES in all_data.csv: {len(all_data_smiles)}")
		missing = all_data_smiles - all_test_union
		extra = all_test_union - all_data_smiles
		if missing:
			print(f"  {len(missing)} SMILES in all_data.csv not covered by any test set")
		if extra:
			print(f"  {len(extra)} SMILES in test sets not found in all_data.csv")
		if not missing and not extra:
			print("  All SMILES perfectly covered")

def main(argv):
	# args = sys.argv[1:]
	task_type = argv[1]
	if task_type == 'train':
		split_folder = argv[2]
		epochs = 50
		cv_num = 5
		for i,arg in enumerate(argv):
			if arg.replace('–', '-') == '--epochs' and i+1 < len(argv):
				epochs = int(argv[i+1])
		for cv in range(cv_num):
			split_dir = '../data/crossval_splits/'+split_folder+'/cv_'+str(cv)
			# ---- regression ----
			reg_train = os.path.join(split_dir, 'train.csv')
			if os.path.exists(reg_train):
				reg_header = pd.read_csv(reg_train, nrows=0).columns.tolist()
				# Load target roles
				roles_path = os.path.join('../data', 'args_files', 'target_roles.json')
				try:
					roles = json.load(open(roles_path, 'r', encoding='utf-8'))
					role_reg = roles.get('regression_targets', [])
					reg_targets = [c for c in role_reg if c in reg_header and c.lower() != 'smiles']
				except Exception:
					# Fallback: take all non-smiles columns
					reg_targets = [c for c in reg_header if c.lower() != 'smiles']

				if len(reg_targets) > 0:
					arguments = [
						'--epochs', str(epochs),
						'--save_dir', split_dir,
						'--seed','42',
						'--dataset_type','regression',
						'--data_path', os.path.join(split_dir,'train.csv'),
						'--features_path', os.path.join(split_dir,'train_extra_x.csv'),
						'--separate_val_path', os.path.join(split_dir,'valid.csv'),
						'--separate_val_features_path', os.path.join(split_dir,'valid_extra_x.csv'),
						'--separate_test_path', os.path.join(split_dir,'test.csv'),
						'--separate_test_features_path', os.path.join(split_dir,'test_extra_x.csv'),
						'--config_path','../data/args_files/optimized_configs.json',
						'--loss_function','mse','--metric','rmse'
					]
					# Use explicit target columns from roles
					arguments += ['--target_columns'] + reg_targets
					if 'morgan' in split_folder:
						arguments += ['--features_generator','morgan_count']
					args = chemprop.args.TrainArgs().parse_args(arguments)
					mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)
				else:
					print(f"[train][cv{cv}] no regression targets, skip.")
			else:
				print(f"[train][cv{cv}] train.csv not found, skip regression.")

			# ---- classification ----
			clf_train = os.path.join(split_dir, 'train_clf.csv')
			if os.path.exists(clf_train):
				clf_header = pd.read_csv(clf_train, nrows=0).columns.tolist()
				roles_path = os.path.join('../data', 'args_files', 'target_roles.json')
				try:
					roles = json.load(open(roles_path, 'r', encoding='utf-8'))
					role_clf = roles.get('classification_targets', [])
					clf_targets = [c for c in role_clf if c in clf_header and c.lower() != 'smiles']
				except Exception:
					clf_targets = [c for c in clf_header if c.lower() != 'smiles']

				if len(clf_targets) > 0:
					clf_dir = '../data/crossval_splits/'+split_folder+'/cv_'+str(cv)+'_clf'
					arguments = [
						'--epochs', str(epochs),
						'--save_dir', clf_dir,
						'--seed','42',
						'--dataset_type','classification',
						'--data_path', os.path.join(split_dir,'train_clf.csv'),
						'--features_path', os.path.join(split_dir,'train_extra_x.csv'),
						'--separate_val_path', os.path.join(split_dir,'valid_clf.csv'),
						'--separate_val_features_path', os.path.join(split_dir,'valid_extra_x.csv'),
						'--separate_test_path', os.path.join(split_dir,'test_clf.csv'),
						'--separate_test_features_path', os.path.join(split_dir,'test_extra_x.csv'),
						'--config_path','../data/args_files/optimized_configs.json',
						'--loss_function','binary_cross_entropy','--metric','auc'
					]
					arguments += ['--target_columns'] + clf_targets
					if 'morgan' in split_folder:
						arguments += ['--features_generator','morgan_count']
					args = chemprop.args.TrainArgs().parse_args(arguments)
					mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)
				else:
					print(f"[train][cv{cv}] no classification targets, skip.")
			else:
				print(f"[train][cv{cv}] train_clf.csv not found, skip classification.")	

	elif task_type == 'predict':
		cv_num = 5
		split_model_folder = '../data/crossval_splits/'+argv[2]
		screen_name = argv[3]
		# READ THE METADATA FILE TO A DF, THEN TAG ON THE PREDICTIONS TO GENERATE A COMPLETE PREDICTIONS FILE
		all_df = pd.read_csv('../data/libraries/'+screen_name+'/'+screen_name+'_metadata.csv')
		for cv in range(cv_num):
			# results_dir = '../results/crossval_splits/'+split_model_folder+'cv_'+str(cv)
			arguments = [
				'--test_path','../data/libraries/'+screen_name+'/'+screen_name+'.csv',
				'--features_path','../data/libraries/'+screen_name+'/'+screen_name+'_extra_x.csv',
				'--checkpoint_dir', split_model_folder+'/cv_'+str(cv),
				'--preds_path','../results/screen_results/'+argv[2]+'_preds'+'/'+screen_name+'/cv_'+str(cv)+'_preds.csv'
			]
			if 'morgan' in split_model_folder:
					arguments = arguments + ['--features_generator','morgan_count']
			args = chemprop.args.PredictArgs().parse_args(arguments)
			preds = chemprop.train.make_predictions(args=args)
			new_df = pd.read_csv('../results/screen_results/'+argv[2]+'_preds'+'/'+screen_name+'/cv_'+str(cv)+'_preds.csv')
			all_df['smiles'] = new_df.smiles
			all_df['cv_'+str(cv)+'_pred_delivery'] = new_df.quantified_delivery	
		all_df['avg_pred_delivery'] = all_df[['cv_'+str(cv)+'_pred_delivery' for cv in range(cv_num)]].mean(axis=1)
		all_df.to_csv('../results/screen_results/'+argv[2]+'_preds'+'/'+screen_name+'/pred_file.csv', index = False)
	elif task_type == 'hyperparam_optimize':
		split_folder = argv[2]
		cv = 0
		data_dir = '../data/crossval_splits/'+split_folder+'/cv_'+str(cv)
		arguments = [
			'--data_path',data_dir+'/train.csv',
			'--features_path', data_dir+'/train_extra_x.csv',
			'--separate_val_path', data_dir+'/valid.csv',
			'--separate_val_features_path', data_dir+'/valid_extra_x.csv',
			'--separate_test_path', data_dir+'/test.csv',
			'--separate_test_features_path', data_dir+'/test_extra_x.csv',
			'--dataset_type', 'regression',
			'--num_iters', '5',
			'--config_save_path','../results/'+split_folder+'/hyp_cv_0.json',
			'--epochs', '5'
		]
		args = chemprop.args.HyperoptArgs().parse_args(arguments)
		chemprop.hyperparameter_optimization.hyperopt(args)
	elif task_type == 'analyze':
		# output.to_csv(path_to_folders+'/cv_'+str(i)+'/Predicted_vs_actual.csv', index = False)
		split = argv[2]
		make_pred_vs_actual(split, predictions_done = [], ensemble_size = 5)
		_aggregate_cv_scores(split, ensemble_number = 5)
	elif task_type == 'merge_datasets':
		merge_datasets(None)
	elif task_type == 'split':
		split = argv[2]
		ultra_held_out = float(argv[3])
		is_morgan = False
		in_silico_screen = False
		if len(argv)>4:
			if argv[4]=='morgan':
				is_morgan = True
				if len(argv)>5 and argv[5]=='in_silico_screen_split':
					in_silico_screen = True
			elif argv[4]=='in_silico_screen_split':
				in_silico_screen = True
		specified_cv_split(split,ultra_held_out_fraction = ultra_held_out, is_morgan = is_morgan, test_is_valid = in_silico_screen)

	elif task_type == 'check':
		if len(argv) < 3:
			print("Usage: python main_script.py check <split_folder>")
			return
		split_folder = argv[2]
		check_cv_split(split_folder)

def _aggregate_cv_scores(split_folder, ensemble_number=5):
	# Aggregate per-fold test scores into cross-fold mean/std for regression and classification
	base_results = os.path.join('../results', 'crossval_splits', split_folder)
	out_dir = os.path.join(base_results, 'crossval_performance')
	os.makedirs(out_dir, exist_ok=True)

	# Regression aggregation
	reg_scores = {}
	for cv in range(ensemble_number):
		fp = os.path.join(base_results, f'cv_{cv}', 'test_scores.csv')
		if not os.path.exists(fp):
			continue
		df = pd.read_csv(fp)
		for _, row in df.iterrows():
			task = row['Task']
			val = row.get('Fold 0 rmse', np.nan)
			reg_scores.setdefault(task, []).append(val)

	reg_rows = []
	for task, vals in reg_scores.items():
		arr = np.array(vals, dtype=float)
		mean_rmse = float(np.nanmean(arr)) if arr.size else float('nan')
		std_rmse = float(np.nanstd(arr)) if arr.size else float('nan')
		reg_rows.append([task, mean_rmse, std_rmse, len(vals)])

	if reg_rows:
		pd.DataFrame(reg_rows, columns=['Task', 'Mean rmse', 'Standard deviation rmse', 'N_folds']) \
		  .to_csv(os.path.join(out_dir, 'test_scores_reg_agg.csv'), index=False)
		print(f"Regression scores saved to: {os.path.join(out_dir, 'test_scores_reg_agg.csv')}")

	# Classification aggregation (AUC / PR_AUC)
	auc_map = {}
	pr_map = {}
	for cv in range(ensemble_number):
		fp = os.path.join(base_results, f'cv_{cv}', 'test_scores_clf.csv')
		if not os.path.exists(fp):
			continue
		df = pd.read_csv(fp)
		for _, row in df.iterrows():
			task = row['Task']
			auc = row.get('AUC', np.nan)
			pr  = row.get('PR_AUC', np.nan)
			auc_map.setdefault(task, []).append(auc)
			pr_map.setdefault(task, []).append(pr)

	clf_rows = []
	tasks = sorted(set(list(auc_map.keys()) + list(pr_map.keys())))
	for task in tasks:
		auc_vals = auc_map.get(task, [])
		pr_vals = pr_map.get(task, [])
		auc_arr = np.array(auc_vals, dtype=float)
		pr_arr = np.array(pr_vals, dtype=float)
		mean_auc = float(np.nanmean(auc_arr)) if auc_arr.size else float('nan')
		std_auc = float(np.nanstd(auc_arr)) if auc_arr.size else float('nan')
		mean_pr = float(np.nanmean(pr_arr)) if pr_arr.size else float('nan')
		std_pr = float(np.nanstd(pr_arr)) if pr_arr.size else float('nan')
		clf_rows.append([task, mean_auc, std_auc, mean_pr, std_pr, len(auc_vals)])

	if clf_rows:
		pd.DataFrame(clf_rows, columns=['Task', 'Mean AUC', 'Std AUC', 'Mean PR_AUC', 'Std PR_AUC', 'N_folds']) \
		  .to_csv(os.path.join(out_dir, 'test_scores_clf_agg.csv'), index=False)
		print(f"Classification scores saved to: {os.path.join(out_dir, 'test_scores_clf_agg.csv')}")

if __name__ == '__main__':
	main(sys.argv)
