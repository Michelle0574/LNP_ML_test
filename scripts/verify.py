import pandas as pd
import os

split_folder = 'by_source_smiles'  # 或 'by_source_smiles'
base_path = f'../data/crossval_splits/{split_folder}'

print(f"检查 {split_folder} 的分层效果\n" + "="*60)

for cv in range(5):  # 5 folds
    print(f"\n📁 CV Fold {cv}:")
    for split_type in ['train', 'valid', 'test']:
        meta_path = f'{base_path}/cv_{cv}/{split_type}_metadata.csv'
        if os.path.exists(meta_path):
            meta = pd.read_csv(meta_path)
            source_counts = meta['Source'].value_counts()
            total = len(meta)
            print(f"  {split_type:6s}: {total:5d} 样本 | ", end="")
            for src in ['external', 'internal']:
                if src in source_counts.index:
                    count = source_counts[src]
                    pct = count/total*100
                    print(f"{src}: {count:4d} ({pct:5.1f}%) ", end="")
            print()

# 检查 ultra held-out（如果有）
uho_path = f'{base_path}/ultra_held_out/test_metadata.csv'
if os.path.exists(uho_path):
    print(f"\n📁 Ultra Held-Out:")
    meta = pd.read_csv(uho_path)
    source_counts = meta['Source'].value_counts()
    total = len(meta)
    print(f"  test  : {total:5d} 样本 | ", end="")
    for src in ['external', 'internal']:
        if src in source_counts.index:
            count = source_counts[src]
            pct = count/total*100
            print(f"{src}: {count:4d} ({pct:5.1f}%) ", end="")
    print()