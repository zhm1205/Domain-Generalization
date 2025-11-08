import os
import pandas as pd
from pathlib import Path
import yaml

# 读取配置
filepath = Path(__file__).resolve().parent.parent
config = yaml.safe_load(open(filepath.joinpath("./configs/mimic_cxr.yaml")))

raw_data_folder = config["raw_data_folder"]

def analyze_reid_split():
    print("Loading MIMIC-CXR data...")
    
    # 读取官方数据文件
    split_df = pd.read_csv(f'{raw_data_folder}/mimic-cxr-2.0.0-split.csv')
    metadata_df = pd.read_csv(f'{raw_data_folder}/mimic-cxr-2.0.0-metadata.csv')
    
    print("Split DataFrame shape:", split_df.shape)
    print("Metadata DataFrame shape:", metadata_df.shape)
    print(metadata_df.ViewPosition.value_counts())

    # 由于两个DataFrame都有subject_id和study_id，我们需要验证它们是否一致
    full_df = split_df.merge(metadata_df, on='dicom_id', how='left')
    
    # 检查subject_id和study_id是否一致
    subject_id_match = (full_df['subject_id_x'] == full_df['subject_id_y']).all()
    study_id_match = (full_df['study_id_x'] == full_df['study_id_y']).all()
    
    print(f"subject_id columns match: {subject_id_match}")
    print(f"study_id columns match: {study_id_match}")
    
    if subject_id_match and study_id_match:
        # 如果一致，删除重复列
        full_df = full_df.drop(['subject_id_y', 'study_id_y'], axis=1)
        full_df = full_df.rename(columns={'subject_id_x': 'subject_id', 'study_id_x': 'study_id'})
        print("✓ Duplicate columns removed")
    else:
        print("✗ Warning: subject_id or study_id columns don't match!")
        return
    
    print(f"\nTotal images: {len(full_df)}")
    print(f"Split distribution:")
    print(full_df['split'].value_counts())
    print()
    
    # 分析每个split中的subject分布
    for split_name in ['train', 'validate', 'test']:
        split_data = full_df[full_df['split'] == split_name]
        subjects = split_data['subject_id'].unique()
        
        print(f"=== {split_name.upper()} SET ===")
        print(f"Images: {len(split_data)}")
        print(f"Unique subjects: {len(subjects)}")
        
        # 分析每个subject的图像数量分布
        subject_counts = split_data['subject_id'].value_counts()
        print(f"Images per subject - Min: {subject_counts.min()}, Max: {subject_counts.max()}, Mean: {subject_counts.mean():.2f}")
        
        # ReID需要每个身份至少2张图像
        valid_subjects = subject_counts[subject_counts >= 2]
        print(f"Subjects with ≥2 images: {len(valid_subjects)}")
        print(f"Valid images for ReID: {valid_subjects.sum()}")
        
        # 分析分布
        print(f"Subject distribution:")
        print(f"  2 images: {len(subject_counts[subject_counts == 2])}")
        print(f"  3-5 images: {len(subject_counts[(subject_counts >= 3) & (subject_counts <= 5)])}")
        print(f"  6-10 images: {len(subject_counts[(subject_counts >= 6) & (subject_counts <= 10)])}")
        print(f"  >10 images: {len(subject_counts[subject_counts > 10])}")
        print()
    
    # 检查是否有subject跨split的情况
    train_subjects = set(full_df[full_df['split'] == 'train']['subject_id'].unique())
    val_subjects = set(full_df[full_df['split'] == 'validate']['subject_id'].unique())
    test_subjects = set(full_df[full_df['split'] == 'test']['subject_id'].unique())
    
    print("=== SPLIT INTEGRITY CHECK ===")
    print(f"Train-Val overlap: {len(train_subjects & val_subjects)}")
    print(f"Train-Test overlap: {len(train_subjects & test_subjects)}")
    print(f"Val-Test overlap: {len(val_subjects & test_subjects)}")
    
    if len(train_subjects & test_subjects) == 0:
        print("✓ No subject leakage between train and test!")
    else:
        print("✗ Warning: Subject leakage detected!")
    
    # 分析view position分布（对ReID可能有用）
    print("\n=== VIEW POSITION ANALYSIS ===")
    view_dist = full_df.groupby(['split', 'ViewPosition']).size().unstack(fill_value=0)
    print(view_dist)

if __name__ == "__main__":
    analyze_reid_split()