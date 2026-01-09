import logging
import numpy as np
import pandas as pd
import os
from typing import List, Dict
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import (
    Mol,
    AllChem,
    MACCSkeys,
    Descriptors
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_morgan(mol: Mol, radius: int = 2, nBits: int = 1024) -> List[int]:
    # Morgan fingerprint (also known as ECFP)
    # Output: [nBits] binary vector
    return AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=radius,
        nBits=nBits,
        useChirality=False
    ).ToList()


def get_maccs(mol: Mol) -> List[int]:
    # MACCS keys: 167-bit structural key descriptors
    # Output: [167] binary vector
    return MACCSkeys.GenMACCSKeys(mol).ToList()


def get_rdkit_descriptors(mol: Mol) -> List[float]:
    # RDKit molecular descriptors (200+ physicochemical properties)
    # Output: [~208] float vector (number may vary by RDKit version)
    desc_dict = Descriptors.CalcMolDescriptors(mol)
    return list(desc_dict.values())


def load_unique_smiles(csv_path: str) -> np.ndarray:
    # Load and extract unique SMILES from all_data.csv
    logger.info(f"Loading SMILES from {csv_path}...")
    df = pd.read_csv(csv_path, usecols=["smiles"], low_memory=False)
    
    # Clean: remove NaN, empty strings, strip whitespace
    cleaned = df["smiles"].dropna()
    cleaned = cleaned[cleaned.map(lambda x: isinstance(x, str) and x.strip() != "")].map(str.strip)
    
    smiles_list = np.unique(cleaned.to_numpy())
    logger.info(f"Found {len(smiles_list)} unique SMILES")
    
    return smiles_list


def compute_features(smiles_list: np.ndarray) -> Dict[str, np.ndarray]:
    # Compute all three types of RDKit features
    # Returns:
    #   - morgan: [N, 1024] binary
    #   - maccs: [N, 167] binary
    #   - rdkit_feats: [N, ~208] float
    
    logger.info("Computing RDKit features...")
    morgan_list, maccs_list, rdkit_feats_list = [], [], []
    failed_smiles = []
    
    for smiles in tqdm(smiles_list, desc="Processing SMILES"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Failed to parse SMILES: {smiles}")
            failed_smiles.append(smiles)
            # Append zero vectors for failed molecules
            morgan_list.append([0] * 1024)
            maccs_list.append([0] * 167)
            rdkit_feats_list.append([0.0] * len(Descriptors._descList))
            continue
        
        try:
            morgan_list.append(get_morgan(mol))
            maccs_list.append(get_maccs(mol))
            rdkit_feats_list.append(get_rdkit_descriptors(mol))
        except Exception as e:
            logger.error(f"Error computing features for {smiles}: {e}")
            failed_smiles.append(smiles)
            morgan_list.append([0] * 1024)
            maccs_list.append([0] * 167)
            rdkit_feats_list.append([0.0] * len(Descriptors._descList))
    
    if failed_smiles:
        logger.warning(f"Failed to compute features for {len(failed_smiles)} SMILES")
    
    # Get descriptor names
    desc_names = [x[0] for x in Descriptors._descList]
    
    return {
        "SMILES": smiles_list,
        "morgan": np.array(morgan_list, dtype=np.int8),
        "maccs": np.array(maccs_list, dtype=np.int8),
        "rdkit_feats": np.array(rdkit_feats_list, dtype=np.float32),
        "rdkit_desc_names": np.array(desc_names)
    }


def main():
    # Paths
    data_csv = "../data/all_data.csv"
    output_npz = "../data/rdkit_features.npz"
    
    # Check if input exists
    if not os.path.exists(data_csv):
        raise FileNotFoundError(f"Input file not found: {data_csv}")
    
    # Step 1: Load unique SMILES
    smiles_list = load_unique_smiles(data_csv)
    
    # Step 2: Compute features
    features = compute_features(smiles_list)
    
    # Step 3: Save to npz
    logger.info(f"Saving features to {output_npz}...")
    np.savez_compressed(output_npz, **features)
    
    # Step 4: Print summary
    logger.info("\n" + "="*60)
    logger.info("Feature computation completed!")
    logger.info("="*60)
    logger.info(f"Total SMILES: {len(features['SMILES'])}")
    logger.info(f"Morgan fingerprint shape: {features['morgan'].shape}")
    logger.info(f"MACCS keys shape: {features['maccs'].shape}")
    logger.info(f"RDKit descriptors shape: {features['rdkit_feats'].shape}")
    logger.info(f"Number of RDKit descriptors: {len(features['rdkit_desc_names'])}")
    logger.info("="*60)
    
    # Step 5: Verify by loading
    logger.info("\nVerifying saved file...")
    loaded = np.load(output_npz, allow_pickle=True)
    logger.info(f"Loaded keys: {list(loaded.keys())}")
    logger.info(f"Sample SMILES (first 3): {loaded['SMILES'][:3]}")
    logger.info(f"Sample Morgan (first SMILES, first 10 bits): {loaded['morgan'][0, :10]}")
    logger.info(f"Sample MACCS (first SMILES, first 10 bits): {loaded['maccs'][0, :10]}")
    logger.info(f"Sample RDKit descriptors (first SMILES, first 5): {loaded['rdkit_feats'][0, :5]}")


if __name__ == "__main__":
    main()