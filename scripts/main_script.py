import numpy as np 
import os
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_curve, roc_auc_score
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
from smiles_features import build_npz_from_smiles
from pandas.api.types import is_object_dtype
from train_multitask import train_multitask_model, get_base_args, train_hyperparam_optimized_model
from predict_multitask_from_json import predict_multitask_from_json, get_base_predict_args, predict_multitask_from_json_cv
from attention_fusion import train_cv as train_attn_cv, predict_cv as predict_attn_cv
from tqdm import tqdm
# path helpers
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
def _p(*parts):
	return os.path.join(ROOT_DIR, *parts)

def _filter_invalid_smiles(df):
	if 'smiles' not in df.columns:
		return df
	s = df['smiles'].astype(str).str.strip()
	ok = s.apply(lambda x: Chem.MolFromSmiles(x) is not None)
	bad = int((~ok).sum())
	if bad:
		print(f"[split] drop invalid SMILES: {bad}")
	return df.loc[ok].reset_index(drop=True)

def _normalize_metadata_missing(all_df, col_type_map, fill_numeric_with_zero=True):
	import numpy as np
	from pandas.api.types import is_numeric_dtype
	missing_tokens = {'', 'na', 'NA', 'Na', 'N/A', 'none', 'None'}

	meta_cols = [c for c, (t, _) in col_type_map.items() if t == 'Metadata' and c in all_df.columns]
	for c in meta_cols:
		v_num = pd.to_numeric(all_df[c], errors='coerce')
		if v_num.notna().any():
			all_df[c] = v_num.fillna(0.0) if fill_numeric_with_zero else v_num
		else:
			s = all_df[c].astype(str).str.strip()
			all_df.loc[s.isin(missing_tokens), c] = np.nan
	return all_df

def merge_datasets(experiment_list, path_to_folders = '../data/data_files_to_merge', write_path = '../data'):
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

		experiment_temp = experiment_df[experiment_df.Experiment_ID == folder]
		experiment_temp = pd.concat([experiment_temp]*data_n, ignore_index=True).reset_index(drop=True)
		drop_cols = set(experiment_temp.columns) & (set(main_temp.columns) | set(formulation_temp.columns) | set(individual_temp.columns))
		experiment_temp = experiment_temp.drop(columns=list(drop_cols), errors='ignore')

		folder_df = pd.concat([main_temp, formulation_temp, individual_temp], axis=1).reset_index(drop=True)
		folder_df = pd.concat([folder_df, experiment_temp], axis=1)

		if folder_df.columns.duplicated().any():
			dup_names = folder_df.columns[folder_df.columns.duplicated()].unique()
			for name in dup_names:
				same = [c for c in folder_df.columns if c == name]
				merged = folder_df[same].bfill(axis=1).iloc[:, 0]
				folder_df = folder_df.drop(columns=same)
				folder_df[name] = merged
		if 'Sample_weight' not in folder_df.columns and 'Experiment_weight' in folder_df.columns:
			folder_df['Sample_weight'] = [float(folder_df.Experiment_weight[i]) for i, _ in enumerate(folder_df.smiles)]
		elif 'Sample_weight' not in folder_df.columns:
			folder_df['Sample_weight'] = 1.0

		all_df = pd.concat([all_df, folder_df], ignore_index=True)

	# Basic normalization of string categories
	normalize_map = {'im':'intramuscular','iv':'intravenous','a549':'lung_epithelium','bdmc':'macrophage','bmdm':'dendritic_cell','hela':'generic_cell','hek':'generic_cell','igrov1':'generic_cell'}
	all_df = all_df.replace(normalize_map)
	if 'Model_type' in all_df.columns:
		all_df['Model_type'] = all_df['Model_type'].replace({'muscle':'Mouse','mouse':'Mouse','mice':'Mouse'})
	if 'Cargo' in all_df.columns and 'Cargo_type' not in all_df.columns:
		all_df = all_df.rename(columns={'Cargo':'Cargo_type'})

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

	all_df = _normalize_metadata_missing(all_df, col_type_map, fill_numeric_with_zero=True)

	_missing_tokens = {'', 'na', 'NA', 'Na', 'N/A', 'none', 'None'}
	for c in all_df.columns:
		if all_df[c].dtype == object:
			s = all_df[c].astype(str).str.strip()
			all_df.loc[s.isin(_missing_tokens), c] = np.nan

	# One-hot for selected categorical X
	extra_x_variables = ['Cationic_Lipid_Mol_Ratio','Phospholipid_Mol_Ratio','Cholesterol_Mol_Ratio','PEG_Lipid_Mol_Ratio','Cationic_Lipid_to_mRNA_weight_ratio']
	extra_x_categorical = ['Delivery_target','Helper_lipid_ID','Route_of_administration','Batch_or_individual_or_barcoded','Cargo_type','Model_type']
	for x_cat in extra_x_categorical:
		if x_cat in all_df.columns:
			if x_cat in col_type_map and col_type_map[x_cat][0] == 'Y_val':
				continue
			dummies = pd.get_dummies(all_df[x_cat], prefix=x_cat)
			dummies = dummies.loc[:, ~dummies.columns.isin(all_df.columns)]
			all_df = pd.concat([all_df, dummies], axis=1)

	if all_df.columns.duplicated().any():
		dup_names = all_df.columns[all_df.columns.duplicated()].unique()
		for name in dup_names:
			same = [c for c in all_df.columns if c == name]
			merged = all_df[same].bfill(axis=1).iloc[:, 0]
			all_df = all_df.drop(columns=same)
			all_df[name] = merged
		
	# One-hot classification Y targets (string labels -> multiple 0/1 target columns)
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

	# Normalize quantified_delivery if present
	if 'quantified_delivery' in all_df.columns:
		norm_split_names, norm_del = generate_normalized_data(all_df)
		all_df['split_name_for_normalization'] = norm_split_names
		all_df.rename(columns={'quantified_delivery':'unnormalized_delivery'}, inplace=True)
		all_df['quantified_delivery'] = norm_del

	# if raw Delivery_target missing, rebuild from one-hot or fill NA
	if 'Delivery_target' not in all_df.columns:
		dt_oh = [c for c in all_df.columns if c.startswith('Delivery_target_')]
		if len(dt_oh) > 0:
			all_df['Delivery_target'] = all_df[dt_oh].idxmax(axis=1).str.replace('Delivery_target_', '', n=1)
		else:
			all_df['Delivery_target'] = np.nan

	for col in ['quantified_total_luminescence', 'quantified_delivery']:
		if col in all_df.columns:
			all_df[col] = pd.to_numeric(all_df[col], errors='coerce')

	all_df = all_df.replace({True:1.0, False:0.0})
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

def _split_y_by_task(y_df, col_types):
	# Split Y into regression/classification sets (both keep 'smiles')
	if 'Y_task' in col_types.columns:
		task = {r['Column_name']: (str(r['Y_task']).lower() if (pd.notna(r['Y_task']) and r['Type']=='Y_val') else '')
				for _, r in col_types.iterrows()}
	else:
		task = {c: 'regression' for c in y_df.columns if c != 'smiles'}
	reg_cols = [c for c, t in task.items() if t == 'regression']
	clf_cols = [c for c, t in task.items() if t == 'classification']

	y_reg = y_df[['smiles'] + [c for c in y_df.columns if c in reg_cols]] if len(reg_cols)>0 else y_df[['smiles']]
	y_clf = y_df[['smiles'] + [c for c in y_df.columns if c in clf_cols]] if len(clf_cols)>0 else y_df[['smiles']]
	return y_reg, y_clf, reg_cols, clf_cols

def _keep_non_all_nan(y_tr, y_va, cols):
	keep = []
	for c in cols:
		if c in y_tr.columns:
			tr_vals = y_tr[c].dropna()
			tr_has = len(tr_vals) > 0 and not tr_vals.astype(str).str.strip().eq('').all()
			
			if tr_has:
				keep.append(c)
			else:
				print(f"Dropping Y (no valid values in train): {c}")
		else:
			print(f"Dropping Y (column missing): {c}")
	return keep

def _write_set(path, prefix, y, x, w, m):
	y.to_csv(f"{path}/{prefix}.csv", index=False)
	x.to_csv(f"{path}/{prefix}_extra_x.csv", index=False)
	# write weights: single column, numeric, no header
	if isinstance(w, pd.DataFrame):
		w_series = w.iloc[:, 0]
	else:
		w_series = w
	w_series = pd.to_numeric(w_series, errors='coerce').fillna(1.0).astype(float)
	w_series.to_csv(f"{path}/{prefix}_weights.csv", index=False, header=False)
	m.to_csv(f"{path}/{prefix}_metadata.csv", index=False)


def split_df_by_col_type(df, col_types):
	# Splits into 4 dataframes: y_vals, x_vals, sample_weights, metadata
	y_cols = list(col_types.Column_name[col_types.Type == 'Y_val'])
	x_cols = list(col_types.Column_name[col_types.Type == 'X_val'])
	weight_cols = list(col_types.Column_name[col_types.Type == 'Sample_weight'])
	metadata_cols = list(col_types.Column_name[col_types.Type.isin(['Metadata', 'X_val_categorical'])])

	y_df = df[y_cols].copy()
	if 'smiles' in df.columns and 'smiles' not in y_df.columns:
		y_df.insert(0, 'smiles', df['smiles'])
	y_num_cols = [c for c in y_df.columns if c.lower() != 'smiles']
	if len(y_num_cols) > 0:
		y_df[y_num_cols] = y_df[y_num_cols].apply(pd.to_numeric, errors='coerce')

	x_df = df[x_cols].copy()
	if 'smiles' in x_df.columns:
		x_df.drop(columns=['smiles'], inplace=True)
	x_df = x_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)

	w_df = df[weight_cols].copy() if len(weight_cols) > 0 else pd.DataFrame({'Sample_weight': [1.0] * len(df)})
	w_df = w_df.apply(pd.to_numeric, errors='coerce').fillna(1.0)

	m_df = df[metadata_cols].copy()

	return y_df, x_df, w_df, m_df

# # def do_all_splits(path_to_splits = 'Data/Multitask_data/All_datasets/Split_specs'):
# 	all_csvs = os.listdir(path_to_splits)
# 	for csv in all_csvs:
# 		if csv.endswith('.csv'):
# 			specified_dataset_split(csv)

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
	all_df = pd.read_csv(path_to_folders + '/all_data.csv')
	split_df = pd.read_csv(path_to_folders + '/crossval_split_specs/' + split_spec_fname)
	split_path = path_to_folders + '/crossval_splits/' + split_spec_fname[:-4]
	if ultra_held_out_fraction>-0.5: split_path += '_with_ultra_held_out'
	if is_morgan: split_path += '_morgan'
	if test_is_valid: split_path += '_for_in_silico_screen'
	if ultra_held_out_fraction>-0.5: path_if_none(split_path + '/ultra_held_out')
	for i in range(cv_fold): path_if_none(split_path + '/cv_' + str(i))

	perma_train = pd.DataFrame({}); ultra_held_out = pd.DataFrame({}); cv_splits = [pd.DataFrame({}) for _ in range(cv_fold)]
	for _, row in split_df.iterrows():
		dtypes = row['Data_types_for_component'].split(','); vals = row['Values'].split(',')
		df_to_concat = all_df.copy()
		for i, dtype in enumerate(dtypes):
			col = dtype.strip()
			if col not in df_to_concat.columns:
				print(f"[split] skip predicate: missing column {col}")
				continue
			df_to_concat = df_to_concat[df_to_concat[col] == vals[i].strip()].reset_index(drop=True)
		values_to_split = df_to_concat[row['Data_type_for_split']]
		uniq = list(set(values_to_split))
		if row['Train_or_split'].strip().lower() == 'train' or len(uniq) < min_unique_vals * cv_fold:
			perma_train = pd.concat([perma_train, df_to_concat])
		else:
			cv_vals, u_vals = split_for_cv(uniq, cv_fold, ultra_held_out_fraction)
			ultra_held_out = pd.concat([ultra_held_out, df_to_concat[df_to_concat[row['Data_type_for_split']].isin(u_vals)]])
			for i, val in enumerate(cv_vals):
				cv_splits[i] = pd.concat([cv_splits[i], df_to_concat[df_to_concat[row['Data_type_for_split']].isin(val)]])

	col_types = pd.read_csv(path_to_folders + '/col_type.csv')


	if ultra_held_out_fraction>-0.5 and len(ultra_held_out) > 0:
		y_u, x_u, w_u, m_u = split_df_by_col_type(ultra_held_out, col_types)
		yxwm_to_csvs(y_u, x_u, w_u, m_u, split_path + '/ultra_held_out', 'test')

	for i in range(cv_fold):
		test_df = cv_splits[i]
		train_ids = list(range(cv_fold)); train_ids.remove(i)
		if test_is_valid:
			valid_df = cv_splits[i]
		else:
			valid_df = cv_splits[(i+1)%cv_fold]; train_ids.remove((i+1)%cv_fold)
		train_df = pd.concat([perma_train] + [cv_splits[k] for k in train_ids])

		# after building train_df, valid_df, test_df
		train_df = _filter_invalid_smiles(train_df)
		valid_df = _filter_invalid_smiles(valid_df)
		test_df  = _filter_invalid_smiles(test_df)

		y_tr, x_tr, w_tr, m_tr = split_df_by_col_type(train_df, col_types)
		y_va, x_va, w_va, m_va = split_df_by_col_type(valid_df, col_types)
		y_te, x_te, w_te, m_te = split_df_by_col_type(test_df,  col_types)

		ytr_reg, ytr_clf, reg_cols, clf_cols = _split_y_by_task(y_tr, col_types)
		yva_reg, yva_clf, _, _              = _split_y_by_task(y_va, col_types)
		yte_reg, yte_clf, _, _              = _split_y_by_task(y_te, col_types)

		keep_reg = _keep_non_all_nan(ytr_reg, yva_reg, [c for c in ytr_reg.columns if c != 'smiles'])
		print(f"[split][cv{i}] keep_reg: {keep_reg}")
		keep_clf = _keep_non_all_nan(ytr_clf, yva_clf, [c for c in ytr_clf.columns if c != 'smiles'])

		ytr_reg = ytr_reg[['smiles'] + keep_reg] if len(keep_reg)>0 else ytr_reg[['smiles']]
		yva_reg = yva_reg[['smiles'] + keep_reg] if len(keep_reg)>0 else yva_reg[['smiles']]
		yte_reg = yte_reg[['smiles'] + keep_reg] if len(keep_reg)>0 else yte_reg[['smiles']]

		ytr_clf = ytr_clf[['smiles'] + keep_clf] if len(keep_clf)>0 else ytr_clf[['smiles']]
		yva_clf = yva_clf[['smiles'] + keep_clf] if len(keep_clf)>0 else yva_clf[['smiles']]
		yte_clf = yte_clf[['smiles'] + keep_clf] if len(keep_clf)>0 else yte_clf[['smiles']]

		base = split_path + '/cv_' + str(i)
		_write_set(base, 'train',     ytr_reg, x_tr, w_tr, m_tr)
		_write_set(base, 'valid',     yva_reg, x_va, w_va, m_va)
		_write_set(base, 'test',      yte_reg, x_te, w_te, m_te)
		_write_set(base, 'train_clf', ytr_clf, x_tr, w_tr, m_tr)
		_write_set(base, 'valid_clf', yva_clf, x_va, w_va, m_va)
		_write_set(base, 'test_clf',  yte_clf, x_te, w_te, m_te)


def yxwm_to_csvs(y, x, w, m, path,settype):
	# y is y values
	# x is x values
	# w is weights
	# m is metadata
	# set_type is either train, valid, or test
	y.to_csv(path+'/'+settype+'.csv', index = False)
	x.to_csv(path + '/' + settype + '_extra_x.csv', index = False)
	w.to_csv(path + '/' + settype + '_weights.csv', index = False)
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
	weights.to_csv(path_to_splits + '/' + settype + '_weights.csv', index = False)
	metadata_cols.to_csv(path_to_splits + '/' + settype + '_metadata.csv', index = False)

	settype = 'valid'
	y_vals_v.to_csv(path_to_splits + '/' + settype + '.csv', index = False)
	x_vals_v.to_csv(path_to_splits + '/' + settype + '_extra_x.csv', index = False)
	weights_v.to_csv(path_to_splits + '/' + settype + '_weights.csv', index = False)
	metadata_cols_v.to_csv(path_to_splits + '/' + settype + '_metadata.csv', index = False)

	y_vals,x_vals,weights,metadata_cols = split_df_by_col_type(test_df,col_types)
	settype = 'test'
	y_vals.to_csv(path_to_splits + '/' + settype + '.csv', index = False)
	x_vals.to_csv(path_to_splits + '/' + settype + '_extra_x.csv', index = False)
	weights.to_csv(path_to_splits + '/' + settype + '_weights.csv', index = False)
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
	# Not used for screening a new library, used for predicting on the test set of the existing dataset
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
		for col in current_predictions.columns:
			if standardize_predictions:
				preds_to_standardize = current_predictions[col]
				std = np.std(preds_to_standardize); mean = np.mean(preds_to_standardize)
				current_predictions[col] = [(val-mean)/std for val in current_predictions[col]]
			current_predictions.rename(columns = {col:('cv_'+str(cv)+'_pred_'+col)}, inplace = True)
		output = pd.concat([output, current_predictions], axis = 1)

		clf_dir = data_dir + '_clf'
		test_clf_csv = data_dir + '/test_clf.csv'
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
			for col in clf_predictions.columns:
				if standardize_predictions:
					preds_to_standardize = clf_predictions[col]
					std = np.std(preds_to_standardize); mean = np.mean(preds_to_standardize)
					clf_predictions[col] = [(val-mean)/std for val in clf_predictions[col]]
				clf_predictions.rename(columns = {col:('cv_'+str(cv)+'_pred_'+col)}, inplace = True)
			output = pd.concat([output, clf_predictions], axis = 1)
		else:
			print(f"[analyze][cv{cv}] classification checkpoint or test_clf.csv missing, skip classification.")

		output.to_csv(preds_out, index = False)

	if '_with_ultra_held_out' in split_folder:
		results_dir = '../results/crossval_splits/'+split_folder+'/ultra_held_out'
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
	plt.figure()
	plt.scatter(preds_vs_actual.Avg_pred_quantified_delivery, preds_vs_actual.Std_pred_quantified_delivery, color = 'gray')
	plt.scatter(efficient_subset.Avg_pred_quantified_delivery, efficient_subset.Std_pred_quantified_delivery, color = 'black')
	plt.xlabel('Average prediction')
	plt.ylabel('Standard deviation of predictions')
	# plt.legend(loc = 'lower right')
	plt.savefig(analyzed_path + '/stdev_Pareto_frontier.png')
	plt.close()
	efficient_subset.to_csv(analyzed_path + '/stdev_Pareto_frontier.csv', index = False)

	preds_for_pareto = preds_vs_actual[['Avg_pred_quantified_delivery','Confidence']].to_numpy()
	print('Dimensions: ',preds_for_pareto.shape)
	is_efficient = is_pareto_efficient(preds_for_pareto,return_mask = True)
	efficient_subset = preds_vs_actual[is_efficient]
	plt.figure()
	plt.scatter(preds_vs_actual.Avg_pred_quantified_delivery, preds_vs_actual.Std_pred_quantified_delivery, color = 'gray')
	plt.scatter(efficient_subset.Avg_pred_quantified_delivery, efficient_subset.Std_pred_quantified_delivery, color = 'black')
	plt.xlabel('Average prediction')
	plt.ylabel('Standard deviation of predictions')
	# plt.legend(loc = 'lower right')
	plt.savefig(analyzed_path + '/confidence_Pareto_frontier.png')
	plt.close()
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


				plt.figure()
				plt.plot(fpr, tpr, color = 'black', label = 'ROC curve 10th percentile (area = %0.2f)' % auc_score)
				plt.plot([0,1],[0,1],color = 'blue',linestyle = '--')
				plt.xlim([0.0,1.0])
				plt.ylim([0.0,1.05])
				plt.xlabel('False positive rate')
				plt.ylabel('True positive rate')
				plt.legend(loc = 'lower right')
				plt.savefig(analyzed_path + '/roc_curve.png')
				plt.close()
				plt.figure()
				plt.scatter(pred,actual,color = 'black')
				plt.plot(np.unique(pred),np.poly1d(np.polyfit(pred, actual, 1))(np.unique(pred)))
				plt.xlabel('Predicted '+value_name)
				plt.ylabel('Experimental '+value_name)
				plt.savefig(analyzed_path+'/pred_vs_actual.png')
				plt.close()
				plt.figure()
				plt.scatter(std_pred,residuals,color = 'black')
				plt.plot(np.unique(std_pred),np.poly1d(np.polyfit(std_pred, residuals, 1))(np.unique(std_pred)))
				plt.xlabel('Residual (Actual-Predicted) '+value_name)
				plt.ylabel('Ensemble model uncertainty '+value_name)
				plt.savefig(analyzed_path+'/residual_vs_stdev.png')
				plt.close()
				plt.figure()
				plt.scatter(std_pred,rse,color = 'black')
				plt.plot(np.unique(std_pred),np.poly1d(np.polyfit(std_pred, rse, 1))(np.unique(std_pred)))
				plt.xlabel('Ensemble model uncertainty')
				plt.ylabel('Root quared error')
				plt.savefig(analyzed_path+'/std_vs_rse.png')
				plt.close()
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
# 
			
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
		pred_split_names = []
		for index, row in preds_vs_actual.iterrows():
			pred_split_name = ''
			for vbl in pred_split_variables:
				pred_split_name = pred_split_name + row[vbl] + '_'
			pred_split_names.append(pred_split_name[:-1])
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
		pred_split_names = []
		for index, row in preds_vs_actual.iterrows():
			pred_split_name = ''
			for vbl in pred_split_variables:
				pred_split_name = pred_split_name + row[vbl] + '_'
			pred_split_names.append(pred_split_name[:-1])
		preds_vs_actual['Prediction_split_name'] = pred_split_names
		# unique_pred_split_names = set(pred_split_names)
		cols = preds_vs_actual.columns
		data_types = []
		for col in cols:
			if col[:3]=='cv_':
				data_types.append(col)
			
		# all_error_pearson = {}
		# all_error_pearson_p_val = {}
		# all_aucs = []
		# all_goals = []

		for pred_split_name in unique_pred_split_names:
			path_if_none(path_to_preds+split_name+'/cv_'+str(i)+'/results')
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
				# keep = False
				# for val in data_subset[dtype]:
				# 	if not np.isnan(val):
				# 		keep = True
				# if keep:
				analyzed_path = path_to_preds+split_name+'/cv_'+str(i)+'/results/'+pred_split_name+'/'+dtype
				path_if_none(analyzed_path)
				# print(data_subset['Goal'])
				# goal = data_subset['Goal'][0]
				# all_goals.append(goal)
				kept_dtypes.append(dtype)
				analyzed_data = pd.DataFrame({'smiles':data_subset.smiles})
				actual = data_subset['quantified_delivery']
				pred = data_subset['cv_'+str(i)+'_pred_quantified_delivery']
				# std_pred = data_subset['Std_pred_'+dtype]
				# rse = data_subset['RSE_'+dtype]
				# analyzed_data[dtype] = actual
				# analyzed_data['Prediction'] = pred
				# analyzed_data['Actual'] = actual
				# analyzed_data['RSE_pred_'+dtype] = rse
				# residuals = [actual[blah]-pred[blah] for blah in range(len(pred))]
				# analyzed_data['Residual'] = residuals
				if len(actual)>=min_values_for_analysis:
					pearson = scipy.stats.pearsonr(actual, pred)
					spearman, pval = scipy.stats.spearmanr(actual, pred)
					kendall, pval = scipy.stats.kendalltau(actual, pred)
					
					# error_pearson = scipy.stats.pearsonr(std_pred,rse)
					# all_names[pred_split_name].append(pred_split_name)
					# all_dtypes.append(dtype)
					rmse = np.sqrt(mean_squared_error(actual, pred))
					all_rmse[pred_split_name] = all_rmse[pred_split_name] + [rmse]
					
					all_pearson[pred_split_name] = all_pearson[pred_split_name] + [pearson[0]]
					all_pearson_p_val[pred_split_name] = all_pearson_p_val[pred_split_name] + [pearson[1]]
					all_kendall[pred_split_name] = all_kendall[pred_split_name] + [kendall]
					all_spearman[pred_split_name] = all_spearman[pred_split_name] + [spearman]
					# all_pearson_p_val[pred_split_name].append(pearson[1])
					# all_kendall[pred_split_name].append(kendall)
					# all_spearman[pred_split_name].append(spearman)
					plt.figure()
					plt.scatter(pred,actual,color = 'black')
					plt.plot(np.unique(pred),np.poly1d(np.polyfit(pred, actual, 1))(np.unique(pred)))
					plt.xlabel('Predicted '+value_name)
					plt.ylabel('Experimental '+value_name)
					plt.savefig(analyzed_path+'/pred_vs_actual.png')
					plt.close()
				else:
					all_rmse[pred_split_name] = all_rmse[pred_split_name] + [float('nan')]
					all_pearson[pred_split_name] = all_pearson[pred_split_name] + [float('nan')]
					all_pearson_p_val[pred_split_name] = all_pearson_p_val[pred_split_name] + [float('nan')]
					all_kendall[pred_split_name] = all_kendall[pred_split_name] + [float('nan')]
					all_spearman[pred_split_name] = all_spearman[pred_split_name] + [float('nan')]

				all_ns[pred_split_name] = all_ns[pred_split_name] + [len(pred)]

				analyzed_data.to_csv(analyzed_path+'/pred_vs_actual_data.csv', index = False)
	crossval_results_path = path_to_preds+split_name+'/crossval_performance'
	path_if_none(crossval_results_path)


	pd.DataFrame.from_dict(all_ns).to_csv(crossval_results_path+'/n_vals.csv', index = True)
	pd.DataFrame.from_dict(all_pearson).to_csv(crossval_results_path+'/pearson.csv', index = True)
	pd.DataFrame.from_dict(all_pearson_p_val).to_csv(crossval_results_path+'/pearson_p_val.csv', index = True)
	pd.DataFrame.from_dict(all_kendall).to_csv(crossval_results_path+'/kendall.csv', index = True)
	pd.DataFrame.from_dict(all_spearman).to_csv(crossval_results_path+'/spearman.csv', index = True)
	pd.DataFrame.from_dict(all_rmse).to_csv(crossval_results_path+'/rmse.csv', index = True)


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

				plt.figure()
				plt.scatter(pred,actual,color = 'black')
				plt.plot(np.unique(pred),np.poly1d(np.polyfit(pred, actual, 1))(np.unique(pred)))
				plt.xlabel('Predicted '+value_name)
				plt.ylabel('Experimental '+value_name)
				plt.savefig(analyzed_path+'/pred_vs_actual.png')
				plt.close()

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
def _cols_wo_smiles(cols):
	return [c for c in cols if c.lower() != 'smiles']

def _headers_match(a, b):
	return _cols_wo_smiles(list(a)) == _cols_wo_smiles(list(b))

def _check_regression_targets(split_dir):
	ok = True
	issues = []
	p_train = os.path.join(split_dir, 'train.csv')
	p_valid = os.path.join(split_dir, 'valid.csv')
	p_test  = os.path.join(split_dir, 'test.csv')
	if not (os.path.exists(p_train) and os.path.exists(p_valid) and os.path.exists(p_test)):
		return False, ['regression files missing']

	df_tr = pd.read_csv(p_train)
	df_va = pd.read_csv(p_valid)
	df_te = pd.read_csv(p_test)


	if not _headers_match(df_tr.columns, df_va.columns) or not _headers_match(df_tr.columns, df_te.columns):
		ok = False
		issues.append('regression header mismatch between train/valid/test')

	bad_non_numeric = []
	bad_all_nan = []
	for c in _cols_wo_smiles(df_tr.columns):
		s_tr = pd.to_numeric(df_tr[c], errors='coerce')
		if s_tr.isna().all():
			bad_all_nan.append(c)
		nan_rate = float(s_tr.isna().mean())
		if nan_rate > 0.5:
			bad_non_numeric.append((c, f'num_nan_in_train={nan_rate:.2f}'))

	if bad_all_nan:
		ok = False
		issues.append(f"regression all-NaN in train: {bad_all_nan}")
	if bad_non_numeric:
		issues.append(f"regression likely non-numeric (coerce->NaN): {bad_non_numeric}")

	return ok, issues

def _check_classification_targets(split_dir):
	p_train = os.path.join(split_dir, 'train_clf.csv')
	p_valid = os.path.join(split_dir, 'valid_clf.csv')
	p_test  = os.path.join(split_dir, 'test_clf.csv')
	if not (os.path.exists(p_train) and os.path.exists(p_valid) and os.path.exists(p_test)):
		return True, []

	ok = True
	issues = []

	df_tr = pd.read_csv(p_train)
	df_va = pd.read_csv(p_valid)
	df_te = pd.read_csv(p_test)

	if not _headers_match(df_tr.columns, df_va.columns) or not _headers_match(df_tr.columns, df_te.columns):
		ok = False
		issues.append('classification header mismatch between train/valid/test')

	non_binary = []
	for c in _cols_wo_smiles(df_tr.columns):
		u = set(pd.unique(pd.to_numeric(df_tr[c], errors='coerce').dropna()))
		if not u.issubset({0, 1}):
			non_binary.append((c, sorted(list(u))[:10]))

	if non_binary:
		ok = False
		issues.append(f"classification non-binary columns (need 0/1 or one-hot): {non_binary}")

	return ok, issues

def _preflight_check(split_folder, cv_num=5):
	all_ok = True
	for cv in range(cv_num):
		split_dir = '../data/crossval_splits/' + split_folder + '/cv_' + str(cv)
		if not os.path.isdir(split_dir):
			print(f"[check][cv{cv}] split dir missing: {split_dir}")
			all_ok = False
			continue

		ok_r, issues_r = _check_regression_targets(split_dir)
		ok_c, issues_c = _check_classification_targets(split_dir)

		if ok_r and ok_c:
			print(f"[check][cv{cv}] OK")
		else:
			all_ok = False
			if issues_r:
				print(f"[check][cv{cv}] regression:", issues_r)
			if issues_c:
				print(f"[check][cv{cv}] classification:", issues_c)
	return all_ok

def _collect_unique_smiles_from_all_data(path='../data/all_data.csv'):
	df = pd.read_csv(path, usecols=['smiles'])
	s = df['smiles'].dropna().astype(str).str.strip()
	return sorted(list(set(s[s != ''].tolist())))

def _collect_unique_smiles_from_split(split_folder, cv_num=5):
	smiles = []
	for cv in range(cv_num):
		base = '../data/crossval_splits/' + split_folder + '/cv_' + str(cv)
		for fname in ['train.csv','valid.csv','test.csv']:
			p = os.path.join(base, fname)
			if os.path.exists(p):
				df = pd.read_csv(p, usecols=['smiles'])
				s = df['smiles'].dropna().astype(str).str.strip()
				smiles += s[s!=''].tolist()
	return sorted(list(set(smiles)))

def main(argv):
	# minimal arg check
	if len(argv) < 2:
		print("Usage: script.py <task> [options]")
		return

	task_type = argv[1].strip().lower()

	if task_type == 'train':
		# Train both regression and classification heads when present
		if len(argv) < 3:
			raise ValueError("train requires: split_folder")
		split_folder = argv[2]
		epochs = 50
		cv_num = 5
		for i, arg in enumerate(argv):
			if arg.replace('–', '-') == '--epochs' and i + 1 < len(argv):
				epochs = int(argv[i + 1])

		def _add_repeat_flags(args_list, flag, values):
			for v in values:
				args_list += [flag, v]
			return args_list
		
		# put this helper near other helpers (top-level, before main)
		def _ensure_weights(split_dir, prefix):
			y_path = os.path.join(split_dir, f'{prefix}.csv')
			w_path = os.path.join(split_dir, f'{prefix}_weights.csv')
			if not os.path.exists(y_path):
				return
			n = len(pd.read_csv(y_path))
			pd.Series([1.0] * n, dtype=float).to_csv(w_path, index=False, header=False)


		# replace the for-cv training loop inside `if task_type == 'train':`
		for cv in range(cv_num):
			data_dir_base = _p('data', 'crossval_splits', split_folder, f'cv_{cv}')
			# ensure weights files are valid for all sets
			_ensure_weights(data_dir_base, 'train')
			_ensure_weights(data_dir_base, 'valid')
			_ensure_weights(data_dir_base, 'test')

			# ---- Regression head ----
			y_tr_path = os.path.join(data_dir_base, 'train.csv')
			if not os.path.exists(y_tr_path):
				print(f"[train][cv{cv}] train.csv not found, skip regression.")
			else:
				reg_header = pd.read_csv(y_tr_path, nrows=0).columns.tolist()

				roles_path = _p('data','args_files','target_roles.json')
				try:
					roles = json.load(open(roles_path, 'r', encoding='utf-8'))
					role_reg = roles.get('regression_targets', [])
					reg_targets = [c for c in role_reg if c in reg_header]
				except Exception:
					reg_targets = [c for c in reg_header if c.lower() != 'smiles']

				has_reg_targets = len(reg_targets) > 0

				if has_reg_targets:
					save_dir_reg = data_dir_base
					os.makedirs(save_dir_reg, exist_ok=True)
					arguments = [
						'--epochs', str(epochs),
						'--save_dir', save_dir_reg,
						'--seed', '42',
						'--dataset_type', 'regression',
						'--data_path', os.path.join(data_dir_base, 'train.csv'),
						'--separate_val_path', os.path.join(data_dir_base, 'valid.csv'),
						'--separate_test_path', os.path.join(data_dir_base, 'test.csv'),
						'--features_path', os.path.join(data_dir_base, 'train_extra_x.csv'),
						'--separate_val_features_path', os.path.join(data_dir_base, 'valid_extra_x.csv'),
						'--separate_test_features_path', os.path.join(data_dir_base, 'test_extra_x.csv'),
						'--config_path', _p('data','args_files','optimized_configs.json'),
						'--loss_function', 'mse', '--metric', 'rmse'
					]
					arguments += ['--target_columns'] + reg_targets
					if 'morgan' in split_folder:
						arguments += ['--features_generator', 'morgan_count']
					args = chemprop.args.TrainArgs().parse_args(arguments)
					mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)
				else:
					print(f"[train][cv{cv}] no regression targets, skip.")

			# ---- Classification head ----
			clf_train_path = os.path.join(data_dir_base, 'train_clf.csv')
			if os.path.exists(clf_train_path):
				clf_header = pd.read_csv(clf_train_path, nrows=0).columns.tolist()
				clf_targets = [c for c in clf_header if c.lower() != 'smiles']
				has_clf_targets = len(clf_targets) > 0

				if has_clf_targets:
					save_dir_clf = _p('data', 'crossval_splits', split_folder, f'cv_{cv}_clf')
					os.makedirs(save_dir_clf, exist_ok=True)
					arguments = [
						'--epochs', str(epochs),
						'--save_dir', save_dir_clf,
						'--seed', '42',
						'--dataset_type', 'classification',
						'--data_path', os.path.join(data_dir_base, 'train_clf.csv'),
						'--separate_val_path', os.path.join(data_dir_base, 'valid_clf.csv'),
						'--separate_test_path', os.path.join(data_dir_base, 'test_clf.csv'),
						'--features_path', os.path.join(data_dir_base, 'train_extra_x.csv'),
						'--separate_val_features_path', os.path.join(data_dir_base, 'valid_extra_x.csv'),
						'--separate_test_features_path', os.path.join(data_dir_base, 'test_extra_x.csv'),
						'--config_path', _p('data','args_files','optimized_configs.json'),
						'--loss_function', 'binary_cross_entropy', '--metric', 'auc'
					]
					arguments += ['--target_columns'] + clf_targets
					if 'morgan' in split_folder:
						arguments += ['--features_generator', 'morgan_count']
					args = chemprop.args.TrainArgs().parse_args(arguments)
					mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)
				else:
					print(f"[train][cv{cv}] no classification targets, skip.")
			else:
				print(f"[train][cv{cv}] train_clf.csv not found, skip classification.")

	elif task_type == 'predict':
		# Predict for a library, aggregate over CV for both heads when available
		if len(argv) < 4:
			raise ValueError("predict requires: split_folder screen_name")
		cv_num = 5
		split_folder = argv[2]
		split_model_folder = _p('data', 'crossval_splits', split_folder)
		screen_name = argv[3]

		lib_dir = _p('data', 'libraries', screen_name)
		test_path = os.path.join(lib_dir, screen_name + '.csv')
		test_feat_path = os.path.join(lib_dir, screen_name + '_extra_x.csv')
		meta_path = os.path.join(lib_dir, screen_name + '_metadata.csv')

		out_dir = _p('results', 'screen_results', split_folder + '_preds', screen_name)
		os.makedirs(out_dir, exist_ok=True)

		# start result df with metadata if exists, else from predictions later
		all_df = pd.read_csv(meta_path) if os.path.exists(meta_path) else None
		reg_cols_union = set()
		clf_cols_union = set()

		for cv in range(cv_num):
			# ---- Regression predictions ----
			reg_ckpt = os.path.join(split_model_folder, 'cv_' + str(cv))
			reg_preds_path = os.path.join(out_dir, f'cv_{cv}_preds_reg.csv')
			if os.path.isdir(reg_ckpt):
				arguments = [
					'--test_path', test_path,
					'--features_path', test_feat_path,
					'--checkpoint_dir', reg_ckpt,
					'--preds_path', reg_preds_path
				]
				if 'morgan' in split_folder:
					arguments += ['--features_generator', 'morgan_count']
				args = chemprop.args.PredictArgs().parse_args(arguments)
				chemprop.train.make_predictions(args=args)

				reg_df = pd.read_csv(reg_preds_path)
				if all_df is None:
					all_df = pd.DataFrame({})
				all_df['smiles'] = reg_df['smiles']
				for col in reg_df.columns:
					if col.lower() == 'smiles': continue
					all_df[f'cv_{cv}_pred_{col}'] = reg_df[col]
					reg_cols_union.add(col)
			else:
				print(f"[predict][cv{cv}] regression checkpoint not found, skip.")

			# ---- Classification predictions ----
			clf_ckpt = os.path.join(split_model_folder, 'cv_' + str(cv) + '_clf')
			clf_preds_path = os.path.join(out_dir, f'cv_{cv}_preds_clf.csv')
			if os.path.isdir(clf_ckpt):
				arguments = [
					'--test_path', test_path,
					'--features_path', test_feat_path,
					'--checkpoint_dir', clf_ckpt,
					'--preds_path', clf_preds_path
				]
				if 'morgan' in split_folder:
					arguments += ['--features_generator', 'morgan_count']
				args = chemprop.args.PredictArgs().parse_args(arguments)
				chemprop.train.make_predictions(args=args)

				clf_df = pd.read_csv(clf_preds_path)
				if all_df is None:
					all_df = pd.DataFrame({})
				all_df['smiles'] = clf_df['smiles']
				for col in clf_df.columns:
					if col.lower() == 'smiles': continue
					all_df[f'cv_{cv}_pred_{col}'] = clf_df[col]
					clf_cols_union.add(col)
			else:
				print(f"[predict][cv{cv}] classification checkpoint not found, skip.")

		# Aggregate mean across CV for all tasks
		if all_df is None:
			raise RuntimeError("No predictions produced.")

		# Regression averages
		for col in sorted(reg_cols_union):
			cols = [f'cv_{i}_pred_{col}' for i in range(cv_num) if f'cv_{i}_pred_{col}' in all_df.columns]
			if len(cols) > 0:
				all_df[f'avg_pred_{col}'] = all_df[cols].mean(axis=1)

		# Classification averages (mean of probabilities)
		for col in sorted(clf_cols_union):
			cols = [f'cv_{i}_pred_{col}' for i in range(cv_num) if f'cv_{i}_pred_{col}' in all_df.columns]
			if len(cols) > 0:
				all_df[f'avg_pred_{col}'] = all_df[cols].mean(axis=1)

		out_file = os.path.join(out_dir, 'pred_file.csv')
		all_df.to_csv(out_file, index=False)
		print("Saved predictions to:", out_file)

	elif task_type == 'check':
		if len(argv) < 3:
			raise ValueError("check requires: split_folder")
		split_folder = argv[2]
		cv_num = 5
		ok = _preflight_check(split_folder, cv_num=cv_num)
		if not ok:
			print("[check] Found issues. Please fix (e.g., rerun merge_datasets to one-hot numeric multiclass) and re-split.")
			# sys.exit(1)
		else:
			print("[check] All folds look good. You can start training.")

	elif task_type == 'hyperparam_optimize':
		# Simple single-fold hyperopt (cv_0)
		if len(argv) < 3:
			raise ValueError("hyperparam_optimize requires: split_folder")
		split_folder = argv[2]
		data_dir = '../data/crossval_splits/' + split_folder + '/cv_0'
		arguments = [
			'--data_path', os.path.join(data_dir, 'train.csv'),
			'--features_path', os.path.join(data_dir, 'train_extra_x.csv'),
			'--separate_val_path', os.path.join(data_dir, 'valid.csv'),
			'--separate_val_features_path', os.path.join(data_dir, 'valid_extra_x.csv'),
			'--separate_test_path', os.path.join(data_dir, 'test.csv'),
			'--separate_test_features_path', os.path.join(data_dir, 'test_extra_x.csv'),
			'--dataset_type', 'regression',
			'--num_iters', '5',
			'--config_save_path', '../results/' + split_folder + '/hyp_cv_0.json',
			'--epochs', '5'
		]
		os.makedirs('../results/' + split_folder, exist_ok=True)
		args = chemprop.args.HyperoptArgs().parse_args(arguments)
		chemprop.hyperparameter_optimization.hyperopt(args)

	elif task_type == 'analyze':
		# Use existing helpers to build predicted_vs_actual and CV metrics
		if len(argv) < 3:
			raise ValueError("analyze requires: split_folder")
		split = argv[2]
		make_pred_vs_actual(split, predictions_done=[], ensemble_size=5)
		analyze_predictions_cv(split)

	elif task_type == 'train_attention':
		if len(argv) < 3:
			raise ValueError("train_attention requires: split_folder [--epochs E] [--device cpu|cuda] [--config path]")
		split_folder = argv[2]
		epochs = 20; device = 'cpu'; cfg_path = _p('data','args_files','attention_config.json')
		for i, arg in enumerate(argv):
			if arg.replace('–','-') == '--epochs' and i+1 < len(argv):
				epochs = int(argv[i+1])
			if arg.replace('–','-') == '--device' and i+1 < len(argv):
				device = argv[i+1]
				if device.lower() == 'gpu': device = 'cuda'
			if arg.replace('–','-') == '--config' and i+1 < len(argv):
				cfg_path = argv[i+1]
		npz_path = _p('data', 'smiles_features.npz')
		for cv in tqdm(range(5), desc=f"Train attention | {split_folder}", leave=True):
			train_attn_cv(split_folder, cv_index=cv, npz_path=npz_path, epochs=epochs, d_model=128, device=device)
			out_csv = _p('results', 'crossval_splits', split_folder, f'cv_{cv}_attn', 'predicted_vs_actual.csv')
			os.makedirs(os.path.dirname(out_csv), exist_ok=True)
			predict_attn_cv(split_folder, cv_index=cv, npz_path=npz_path, out_csv=out_csv, device=device)

	elif task_type == 'predict_attention':
		if len(argv) < 3:
			raise ValueError("predict_attention requires: split_folder [--config path]")
		split_folder = argv[2]
		cfg_path = _p('data','args_files','attention_config.json')
		for i, arg in enumerate(argv):
			if arg.replace('–','-') == '--config' and i+1 < len(argv):
				cfg_path = argv[i+1]
		npz_path = _p('data', 'smiles_features.npz')
		for cv in range(5):
			out_csv = _p('results', 'crossval_splits', split_folder, f'cv_{cv}_attn', 'predicted_vs_actual.csv')
			os.makedirs(os.path.dirname(out_csv), exist_ok=True)
			df = predict_attn_cv(split_folder, cv_index=cv, npz_path=npz_path, out_csv=out_csv, device='cpu')
			print(f"[predict_attention][cv{cv}] wrote: {out_csv}")

	elif task_type == 'split':
		# Create CV splits (will also write *_clf files if roles specify classification)
		if len(argv) < 4:
			raise ValueError("split requires: split_spec_fname ultra_held_out_fraction [morgan] [in_silico_screen_split]")
		split_spec = argv[2]
		ultra_held_out = float(argv[3])
		is_morgan = False
		in_silico_screen = False
		if len(argv) > 4:
			if argv[4] == 'morgan':
				is_morgan = True
				if len(argv) > 5 and argv[5] == 'in_silico_screen_split':
					in_silico_screen = True
			elif argv[4] == 'in_silico_screen_split':
				in_silico_screen = True
		specified_cv_split(split_spec, ultra_held_out_fraction=ultra_held_out, is_morgan=is_morgan, test_is_valid=in_silico_screen)

	elif task_type == 'build_smiles_features':
		# Usage:
		#   python scripts/main_script.py build_smiles_features all_data
		#   python scripts/main_script.py build_smiles_features split <split_folder>
		out_path = '../data/smiles_features.npz'
		if len(argv) >= 3 and argv[2] == 'all_data':
			smiles = _collect_unique_smiles_from_all_data('../data/all_data.csv')
		elif len(argv) >= 4 and argv[2] == 'split':
			smiles = _collect_unique_smiles_from_split(argv[3])
		else:
			raise ValueError("build_smiles_features requires: all_data  OR  split <split_folder>")
		print(f"[build_smiles_features] unique SMILES: {len(smiles)}")
		build_npz_from_smiles(smiles, out_path, nbits=1024, radius=2)

	else:
		print(f"Unknown task: {task_type}")

	# if task_type == 'new_hyperparam_optimize':
	# 	arguments = [
	# 		'--data_path','../data/crossval_splits/small_test_split/cv_0/train.csv',
	# 		'--features_path', '../data/crossval_splits/small_test_split/cv_0/train_extra_x.csv',
	# 		'--separate_val_path', '../data/crossval_splits/small_test_split/cv_0/valid.csv',
	# 		'--separate_val_features_path', '../data/crossval_splits/small_test_split/cv_0/valid_extra_x.csv',
	# 		'--separate_test_path','../data/crossval_splits/small_test_split/cv_0/test.csv',
	# 		'--separate_test_features_path','../data/crossval_splits/small_test_split/cv_0/test_extra_x.csv',
	# 		'--dataset_type', 'regression',
	# 		'--num_iters', '5',
	# 		'--config_save_path','..results/hyp_cv_0.json',
	# 		'--epochs', '5'
	# 	]
	# 	args = chemprop.args.HyperoptArgs().parse_args(arguments)
	# 	chemprop.hyperparameter_optimization.hyperopt(args)

	# if task_type == 'new_train':
	# 	arguments = [
	# 		'--epochs','15',
	# 		'--save_dir','../data/crossval_splits/small_test_split/cv_0',
	# 		'--seed','42',
	# 		'--dataset_type','regression',
	# 		'--data_path','../data/crossval_splits/small_test_split/cv_0/train.csv',
	# 		'--features_path', '../data/crossval_splits/small_test_split/cv_0/train_extra_x.csv',
	# 		'--separate_val_path', '../data/crossval_splits/small_test_split/cv_0/valid.csv',
	# 		'--separate_val_features_path', '../data/crossval_splits/small_test_split/cv_0/valid_extra_x.csv',
	# 		'--separate_test_path','../data/crossval_splits/small_test_split/cv_0/test.csv',
	# 		'--separate_test_features_path','../data/crossval_splits/small_test_split/cv_0/test_extra_x.csv',
	# 		'--data_weights_path','../data/crossval_splits/small_test_split/cv_0/train_weights.csv',
	# 		'--config_path','../data/args_files/optimized_configs.json',
	# 		'--loss_function','mse','--metric','rmse'
	# 	]
	# 	args = chemprop.args.TrainArgs().parse_args(arguments)
	# 	mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)
	# if task_type == 'new_predict':
	# 	arguments = [
	# 		'--test_path','../data/crossval_splits/small_test_split/cv_0/test.csv',
	# 		'--features_path','../data/crossval_splits/small_test_split/cv_0/test_extra_x.csv',
	# 		'--checkpoint_dir', '../data/crossval_splits/small_test_split/cv_0',
	# 		'--preds_path','../results/crossval_splits/small_test_split/cv_0/preds.csv'
	# 	]
	# 	args = chemprop.args.PredictArgs().parse_args(arguments)
	# 	preds = chemprop.train.make_predictions(args=args)	
	# elif task_type == 'hyperparam_optimize':
	# 	split_list = argv[2]
	# 	# arg is the name of a split list file
	# 	split_df = pd.read_csv('Data/Multitask_data/All_datasets/Split_lists/'+split_list+'.csv')
	# 	for split in split_df['split']:
	# 		# print('starting split: ',split)
	# 		hyperparam_optimize_split(split)
	# elif task_type == 'train_optimized_cv_already_split':
	# 	to_split = argv[2]
	# 	generator = None
	# 	if to_split.endswith('_morgan'):
	# 		generator = ['morgan']
	# 	run_optimized_cv_training('../data/crossval_splits/'+to_split, epochs = 10, path_to_hyperparameters = '../data/args_files', generator = generator)
	# elif task_type == 'specified_cv_split':
	# 	split = argv[2]
	# 	ultra_held_out = float(argv[3])
	# 	is_morgan = False
	# 	in_silico_screen = False
	# 	if len(argv)>4:
	# 		if argv[4]=='morgan':
	# 			is_morgan = True
	# 			if len(argv)>5 and argv[5]=='in_silico_screen_split':
	# 				in_silico_screen = True
	# 		elif argv[4]=='in_silico_screen_split':
	# 			in_silico_screen = True
	# 	specified_cv_split(split,ultra_held_out_fraction = ultra_held_out, is_morgan = is_morgan, test_is_valid = in_silico_screen)
	# elif task_type == 'specified_nested_cv_split':
	# 	split = argv[2]
	# 	is_morgan = False
	# 	if len(argv)>3:
	# 		if argv[3]=='morgan':
	# 			is_morgan = True
	# 	specified_nested_cv_split(split, is_morgan = is_morgan)
	# elif task_type == 'analyze_cv':
	# 	# output.to_csv(path_to_folders+'/cv_'+str(i)+'/Predicted_vs_actual.csv', index = False)
	# 	split = argv[2]
	# 	predict_each_test_set_cv(path_to_folders =  'Data/Multitask_data/All_datasets/crossval_splits/'+split, predictions_done = [], ensemble_size = 5)
	# 	analyze_predictions_cv(split)


	# elif task_type == 'ensemble_screen_cv':
	# 	split = argv[2]
	# 	in_silico_folders = argv[3:]
	# 	for folder in in_silico_folders:
	# 		ensemble_predict_cv(path_to_folders = 'Data/Multitask_data/All_datasets/crossval_splits/'+split, ensemble_size = 5, path_to_new_test = folder)
	# elif task_type == 'analyze':
	# 	split = argv[2]
	# 	ensemble_predict(path_to_folders =  'Data/Multitask_data/All_datasets/Splits/'+split, predictions_done = [], ensemble_size = 5)
	# 	analyze_predictions(split)
	# elif task_type == 'train_optimized_from_to_already_split':
	# 	from_split = argv[2]
	# 	to_split = argv[3]
	# 	run_optimized_ensemble_training('Data/Multitask_data/All_datasets/Splits/'+to_split,ensemble_size = 5, epochs = 50, path_to_hyperparameters = 'Data/Multitask_data/All_datasets/Splits/'+from_split)
	# elif task_type == 'train_optimized_from_to':
	# 	from_split = argv[2]
	# 	to_split = argv[3]
	# 	if to_split.endswith('_morgan'):
	# 		specified_dataset_split(to_split[:-7] + '.csv', is_morgan = True)
	# 	else:
	# 		specified_dataset_split(to_split + '.csv')
	# 	run_optimized_ensemble_training('Data/Multitask_data/All_datasets/Splits/'+to_split,ensemble_size = 5, epochs = 50, path_to_hyperparameters = 'Data/Multitask_data/All_datasets/Splits/'+from_split)
	# elif task_type == 'analyze_new_library':
	# 	split = argv[2]
	# 	ensemble_predict(path_to_folders = 'Data/Multitask_data/All_datasets/Splits/'+split, predictions_done = [], ensemble_size = 5, addition = '_in_silico')
	# 	analyze_new_lipid_predictions(split)
	# elif task_type == 'ensemble_screen':
	# 	split = argv[2]
	# 	in_silico_folders = argv[3:]
	# 	for folder in in_silico_folders:
	# 		ensemble_predict(path_to_folders = 'Data/Multitask_data/All_datasets/Splits/'+split, ensemble_size = 5, path_to_new_test = folder)
	# elif task_type == 'combine_library_analyses':
	# 	combo_name = argv[2]
	# 	splits = argv[3:]
	# 	combine_predictions(splits, combo_name)
	# 

if __name__ == '__main__':
	main(sys.argv)
