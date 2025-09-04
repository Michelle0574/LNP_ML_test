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

def merge_datasets(experiment_list, path_to_folders = '../data/data_files_to_merge', write_path = '../data'):
	# Each folder contains the following files: 
	# main_data.csv: a csv file with columns: 'smiles', which should contain the SMILES of the ionizable lipid, the activity measurements for that measurement
	# If the same ionizable lipid is measured multiple times (i.e. for different properties, or transfection in vitro and in vivo) make separate rows, one for each measurement
	# formulations.csv: a csv file with columns:
		# Cationic_Lipid_Mol_Ratio
		# Phospholipid_Mol_Ratio
		# Cholesterol_Mol_Ratio
		# PEG_Lipid_mol_ratio
		# Cationic_Lipid_to_mRNA_weight_ratio
		# Helper_lipid_ID
		# If the dataset contains only 1 formulation in it: still provide the formulations data thing but with only one row; the model will copy it
		# Otherwise match the row to the data in formulations.csv
	# individual_metadata.csv: metadata that contains as many rows as main_data, each row is certain metadata for each lipid
		# For example, could contain the identity (SMILES) of the amine to be used in training/test splits, or contain a dosage if the dataset includes varying dosage
		# Either includes a column called "Sample_weight" with weight for each sample (each ROW, that is; weight for a kind of experiment will be determined separately)
			# alternatively, default sample weight of 1
	# experiment_metadata.csv: contains metadata about particular dataset. This includes:
		# Experiment_ID: each experiment will be given a unique ID.
		# There will be two ROWS and any number of columns

	# Based on these files, Merge_datasets will merge all the datasets into one dataset. In particular, it will output 2 files:
		# all_merged.csv: each row  will contain all the data for a measurement (SMILES, info on dose/formulation/etc, metadata, sample weights, activity value)
		# col_type.csv: two columns, column name and type. Four types: Y_val, X_val, X_val_cat (categorical X value), Metadata, Sample_weight

	# Some metadata columns that should be held consistent, in terms of names:
		# Purity ("Pure" or "Crude")
		# ng_dose (for the dose, duh)
		# Sample_weight
		# Amine_SMILES
		# Tail_SMILES
		# Library_ID
		# Experimenter_ID
		# Experiment_ID
		# Cargo (siRNA, DNA, mRNA, RNP are probably the relevant 4 options)
		# Model_type (either the cell type or the name of the animal (probably "mouse"))


	all_df = pd.DataFrame({})
	col_type = {'Column_name':[],'Type':[]}
	experiment_df = pd.read_csv(path_to_folders + '/experiment_metadata.csv')
	if experiment_list == None:
		experiment_list = list(experiment_df.Experiment_ID)
		print(experiment_list)
	y_val_cols = []
	helper_mol_weights = pd.read_csv(path_to_folders + '/Component_molecular_weights.csv')

	for folder in experiment_list:
		print(folder)
		contin = False
		try:
			main_temp = pd.read_csv(path_to_folders + '/' + folder + '/main_data.csv')
			contin = True
		except:
			pass
		if contin:
			y_val_cols = y_val_cols + list(main_temp.columns)
			for col in main_temp.columns:
				if 'Unnamed' in col:
					print('\n\n\nTHERE IS A BS UNNAMED COLUMN IN FOLDER: ',folder,'\n\n')
			data_n = len(main_temp)
			formulation_temp = pd.read_csv(path_to_folders + '/' + folder + '/formulations.csv')

			try:
				individual_temp = pd.read_csv(path_to_folders + '/' + folder + '/individual_metadata.csv')
			except:
				individual_temp = pd.DataFrame({})
			if len(formulation_temp) == 1:
				formulation_temp = pd.concat([formulation_temp]*data_n,ignore_index = True)
			elif len(formulation_temp) != data_n:
				print(len(formulation_temp))
				to_raise = 'For experiment ID: ',folder,': Length of formulation file (', str(len(formulation_temp))#, ') doesn\'t match length of main datafile (',str(data_n),')'
				raise ValueError(to_raise)

			# Change formulations from mass to molar ratio
			form_cols = formulation_temp.columns
			mass_ratio_variables = ['Cationic_Lipid_Mass_Ratio','Phospholipid_Mass_Ratio','Cholesterol_Mass_Ratio','PEG_Lipid_Mass_Ratio']
			molar_ratio_variables = ['Cationic_Lipid_Mol_Ratio','Phospholipid_Mol_Ratio','Cholesterol_Mol_Ratio','PEG_Lipid_Mol_Ratio']
			mass_count = 0
			molar_count = 0
			for col in form_cols:
				if col in mass_ratio_variables:
					mass_count += 1
				elif col in molar_ratio_variables:
					molar_count += 1
			if mass_count>0 and molar_count>0:
				raise ValueError('For experiment ID: ',folder,': Formulation information includes both mass and molar ratios.')
			elif mass_count<4 and molar_count<4:
				raise ValueError('For experiment ID: ',folder,': Formulation not completely specified, mass count: ',mass_count,', molar count: ',molar_count)
			elif mass_count == 4:
				cat_lip_mol_fracs = []
				phos_mol_fracs = []
				chol_mol_fracs = []
				peg_lip_mol_fracs = []
				# Change mass ratios to weight ratios
				for i in range(len(formulation_temp)):
					phos_id = formulation_temp['Helper_lipid_ID'][i]
					ion_lipid_mol = Chem.MolFromSmiles(main_temp['smiles'][i])
					ion_lipid_mol_weight = Chem.Descriptors.MolWt(ion_lipid_mol)
					phospholipid_mol_weight = helper_mol_weights[phos_id][0]
					cholesterol_mol_weight = helper_mol_weights['Cholesterol']
					PEG_lipid_mol_weight = helper_mol_weights['C14-PEG2000']
					ion_lipid_moles = formulation_temp['Cationic_Lipid_Mass_Ratio'][i]/ion_lipid_mol_weight
					phospholipid_moles = formulation_temp['Phospholipid_Mass_Ratio'][i]/phospholipid_mol_weight
					cholesterol_moles = formulation_temp['Cholesterol_Mass_Ratio'][i]/cholesterol_mol_weight
					PEG_lipid_moles = formulation_temp['PEG_Lipid_Mass_Ratio'][i]/PEG_lipid_mol_weight
					mol_sum = ion_lipid_moles+phospholipid_moles+cholesterol_moles+PEG_lipid_moles
					cat_lip_mol_fracs.append(float(ion_lipid_moles/mol_sum*100))
					phos_mol_fracs.append(float(phospholipid_moles/mol_sum*100))
					chol_mol_fracs.append(float(cholesterol_moles/mol_sum*100))
					peg_lip_mol_fracs.append(float(PEG_lipid_moles/mol_sum*100))
				formulation_temp['Cationic_Lipid_Mol_Ratio'] = cat_lip_mol_fracs
				formulation_temp['Phospholipid_Mol_Ratio'] = phos_mol_fracs
				formulation_temp['Cholesterol_Mol_Ratio'] = chol_mol_fracs
				formulation_temp['PEG_Lipid_Mol_Ratio'] = peg_lip_mol_fracs

		
			if len(individual_temp) != data_n:
				print(len(individual_temp))
				raise ValueError('For experiment ID: ',folder,': Length of individual metadata file  (',len(individual_temp), ') doesn\'t match length of main datafile (',data_n,')')
			experiment_temp = experiment_df[experiment_df.Experiment_ID == folder]
			# print(folder)
			# print(experiment_temp)
			experiment_temp = pd.concat([experiment_temp]*data_n, ignore_index = True).reset_index(drop = True)
			to_drop = []
			for col in experiment_temp.columns:
				if col in individual_temp.columns:
					print('Column ',col,' in experiment ID ',folder,'is being provided for each individual lipid.')
					to_drop.append(col)
			experiment_temp = experiment_temp.drop(columns = to_drop)
			# print(folder)
			# print(experiment_temp.columns)
			# print(main_temp.columns)
			# print(formulation_temp.columns)
			# print(individual_temp.columns)
			folder_df = pd.concat([main_temp, formulation_temp, individual_temp], axis = 1).reset_index(drop = True)
			folder_df = pd.concat([folder_df, experiment_temp], axis = 1)
			# print(folder_df.columns)
			if 'Sample_weight' not in folder_df.columns:
				# print(folder)
				# folder_df['Sample_weight'] = [float(folder_df.Experiment_weight[i])/list(folder_df.smiles).count(smile) for i,smile in enumerate(folder_df.smiles)]
				folder_df['Sample_weight'] = [float(folder_df.Experiment_weight[i]) for i,smile in enumerate(folder_df.smiles)]

				# print(folder_df.Experiment_weight)
				# print(folder_df['Sample_weight'])
			# print (folder_df.columns)
			all_df = pd.concat([all_df,folder_df], ignore_index = True)

	# Make the column type dict
	extra_x_variables = ['Cationic_Lipid_Mol_Ratio','Phospholipid_Mol_Ratio','Cholesterol_Mol_Ratio','PEG_Lipid_Mol_Ratio','Cationic_Lipid_to_mRNA_weight_ratio']
	# ADD HELPER LIPID ID
	# extra_x_categorical = ['Delivery_target','Helper_lipid_ID','Route_of_administration','Batch_or_individual_or_barcoded','screen_id']
	extra_x_categorical = ['Delivery_target','Helper_lipid_ID','Route_of_administration','Batch_or_individual_or_barcoded','Cargo_type','Model_type']

	# Make changes:
	all_df = all_df.replace('im','intramuscular')
	all_df = all_df.replace('iv','intravenous')
	all_df = all_df.replace('a549','lung_epithelium')
	all_df = all_df.replace('bdmc','macrophage')
	all_df = all_df.replace('bmdm','dendritic_cell')
	all_df = all_df.replace('hela','generic_cell')
	all_df = all_df.replace('hek','generic_cell')
	all_df = all_df.replace('igrov1','generic_cell')
	all_df = all_df.replace({'Model_type':'muscle'},'Mouse')

	# other_x_vals = ['Target_organ']
	# form_variables.append('Helper_lipid_ID')

	for x_cat in extra_x_categorical:
		dummies = pd.get_dummies(all_df[x_cat], prefix = x_cat)
		print(dummies.columns)
		all_df = pd.concat([all_df, dummies], axis = 1)
		extra_x_variables = extra_x_variables + list(dummies.columns)

	for column in all_df.columns:
		col_type['Column_name'].append(column)
		if column in y_val_cols:
			col_type['Type'].append('Y_val')
		elif column in extra_x_variables:
			col_type['Type'].append('X_val')
		elif column in extra_x_categorical:
			col_type['Type'].append('Metadata')
		elif column == 'Sample_weight':
			col_type['Type'].append('Sample_weight')
		else:
			col_type['Type'].append('Metadata')

	col_type_df = pd.DataFrame(col_type)
	# print(col_type_df)
	norm_split_names, norm_del = generate_normalized_data(all_df)
	all_df['split_name_for_normalization'] = norm_split_names
	all_df.rename(columns = {'quantified_delivery':'unnormalized_delivery'}, inplace = True)
	all_df['quantified_delivery'] = norm_del
	# all_df = all_df.applymap(lambda x: x.lower() if isinstance(x, str) else x)  # Convert all strings to lower case
	all_df = all_df.replace({True: 1.0, False: 0.0})
	all_df.to_csv(write_path + '/all_data.csv', index = False)
	col_type_df.to_csv(write_path + '/col_type.csv', index = False)


def split_df_by_col_type(df,col_types):
	# Splits into 4 dataframes: y_vals, x_vals, sample_weights, metadata
	y_vals_cols = col_types.Column_name[col_types.Type == 'Y_val']
	x_vals_cols = col_types.Column_name[col_types.Type == 'X_val']
	# print(x_vals_cols)
	xvals_df = df[x_vals_cols]
	# print('SUCCESSFUL!!!')
	weight_cols = col_types.Column_name[col_types.Type == 'Sample_weight']
	metadata_cols = col_types.Column_name[col_types.Type.isin(['Metadata','X_val_categorical'])]
	return df[y_vals_cols],xvals_df,df[weight_cols],df[metadata_cols]

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
	all_df = pd.read_csv(path_to_folders + '/all_data.csv')
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

	col_types = pd.read_csv(path_to_folders + '/col_type.csv')

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
		y,x,w,m = split_df_by_col_type(valid_df,col_types)
		yxwm_to_csvs(y,x,w,m,split_path+'/cv_'+str(i),'valid')
		y,x,w,m = split_df_by_col_type(train_df,col_types)
		yxwm_to_csvs(y,x,w,m,split_path+'/cv_'+str(i),'train')

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
		try:
			output = pd.read_csv(results_dir+'/predicted_vs_actual.csv')
		except:
			try:
				current_predictions = pd.read_csv(data_dir+'/preds.csv')
			except:
				arguments = [
					'--test_path',data_dir+'/test.csv',
					'--features_path',data_dir+'/test_extra_x.csv',
					'--checkpoint_dir', data_dir,
					'--preds_path',data_dir+'/preds.csv'
				]
				if 'morgan' in split_folder:
					arguments = arguments + ['--features_generator','morgan_count']
				args = chemprop.args.PredictArgs().parse_args(arguments)
				preds = chemprop.train.make_predictions(args=args)	
			# os.rename(path_to_folders+'/trained_model',path_to_folders + '/trained_model_'+str(i))
				current_predictions = pd.read_csv(data_dir+'/preds.csv')
			
			current_predictions.drop(columns = ['smiles'], inplace = True)
			for col in current_predictions.columns:
				if standardize_predictions:
					preds_to_standardize = current_predictions[col]
					std = np.std(preds_to_standardize)
					mean = np.mean(preds_to_standardize)
					current_predictions[col] = [(val-mean)/std for val in current_predictions[col]]
				current_predictions.rename(columns = {col:('cv_'+str(cv)+'_pred_'+col)}, inplace = True)
			output = pd.concat([output, current_predictions], axis = 1)
			output.to_csv(results_dir+'/predicted_vs_actual.csv', index = False)
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

def generate_normalized_data(all_df, split_variables = ['Experiment_ID','Library_ID','Delivery_target','Model_type','Route_of_administration']):
	split_names = []
	norm_dict = {}
	for index, row in all_df.iterrows():
		split_name = ''
		for vbl in split_variables:
			# print(row[vbl])
			# print(vbl)
			split_name = split_name + str(row[vbl])+'_'
		split_names.append(split_name[:-1])
	unique_split_names = set(split_names)
	for split_name in unique_split_names:
		data_subset = all_df[[spl==split_name for spl in split_names]]
		norm_dict[split_name] = (np.mean(data_subset['quantified_delivery']), np.std(data_subset['quantified_delivery']))
	norm_delivery = []
	for i, row in all_df.iterrows():
		val = row['quantified_delivery']
		split = split_names[i]
		stdev = norm_dict[split][1]
		mean = norm_dict[split][0]
		norm_delivery.append((float(val)-mean)/stdev)
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


def main(argv):
	# args = sys.argv[1:]
	task_type = argv[1]
	if task_type == 'train':
		split_folder = argv[2]
		epochs = 50
		cv_num = 5
		for i,arg in enumerate(argv):
			if arg.replace('–', '-') == '--epochs':
				epochs = argv[i+1]
				# print('this many epochs: ',str(epochs))
		# exit()
		for cv in range(cv_num):
			split_dir = '../data/crossval_splits/'+split_folder+'/cv_'+str(cv)
			arguments = [
				'--epochs',str(epochs),
				'--save_dir',split_dir,
				'--seed','42',
				'--dataset_type','regression',
				'--data_path',split_dir+'/train.csv',
				'--features_path', split_dir+'/train_extra_x.csv',
				'--separate_val_path', split_dir+'/valid.csv',
				'--separate_val_features_path', split_dir+'/valid_extra_x.csv',
				'--separate_test_path',split_dir+'/test.csv',
				'--separate_test_features_path',split_dir+'/test_extra_x.csv',
				'--data_weights_path',split_dir+'/train_weights.csv',
				'--config_path','../data/args_files/optimized_configs.json',
				'--loss_function','mse','--metric','rmse'
			]
			if 'morgan' in split_folder:
				arguments += ['--features_generator','morgan_count']
			args = chemprop.args.TrainArgs().parse_args(arguments)
			mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)
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
		data_dir = '../data/crossval_splits/'+split_folder+'/cv_'+str(cv)
		arguments = [
			'--data_path',data_dir+'/train.csv',
			'--features_path', data_dir+'/train_extra_x.csv',
			'--separate_val_path', data_dir+'/valid.csv',
			'--separate_val_features_path', data_dir+'/valid_extra_x.csv',
			'--separate_test_path',data_dir+'/test.csv',
			'--separate_test_features_path',data_dir+'/test_extra_x.csv',
			'--dataset_type', 'regression',
			'--num_iters', '5',
			'--config_save_path','..results/'+split_folder+'/hyp_cv_0.json',
			'--epochs', '5'
		]
		args = chemprop.args.HyperoptArgs().parse_args(arguments)
		chemprop.hyperparameter_optimization.hyperopt(args)
	elif task_type == 'analyze':
		# output.to_csv(path_to_folders+'/cv_'+str(i)+'/Predicted_vs_actual.csv', index = False)
		split = argv[2]
		make_pred_vs_actual(split, predictions_done = [], ensemble_size = 5)
		analyze_predictions_cv(split)
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
