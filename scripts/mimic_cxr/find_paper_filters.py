import os
import pandas as pd
from pathlib import Path
import yaml

# Read configuration
filepath = Path(__file__).resolve().parent.parent
config = yaml.safe_load(open(filepath.joinpath("./configs/scripts/mimic_cxr.yaml")))

raw_data_folder = config["raw_data_folder"]

def find_paper_matching_filters():
    """Try different filters to match the paper's 111,333 images"""
    print("=== Finding Filters to Match Paper's 111,333 Images ===")
    
    # Load data
    split_df = pd.read_csv(f'{raw_data_folder}/mimic-cxr-2.0.0-split.csv')
    metadata_df = pd.read_csv(f'{raw_data_folder}/mimic-cxr-2.0.0-metadata.csv')
    
    # Merge data
    full_df = split_df.merge(metadata_df, on='dicom_id', how='left')
    
    # Handle duplicate columns
    if (full_df['subject_id_x'] == full_df['subject_id_y']).all():
        full_df = full_df.drop(['subject_id_y', 'study_id_y'], axis=1)
        full_df = full_df.rename(columns={'subject_id_x': 'subject_id', 'study_id_x': 'study_id'})
    
    print(f"Total images in dataset: {len(full_df)}")
    print()
    
    # Strategy 1: Only frontal views (PA and AP)
    print("=== Strategy 1: Only Frontal Views (PA + AP) ===")
    frontal_df = full_df[full_df['ViewPosition'].isin(['PA', 'AP'])].copy()
    print(f"Frontal view images: {len(frontal_df)}")
    
    # Apply ReID filter (>=2 images per patient)
    patient_counts_frontal = frontal_df['subject_id'].value_counts()
    valid_patients_frontal = patient_counts_frontal[patient_counts_frontal >= 2].index
    reid_frontal = frontal_df[frontal_df['subject_id'].isin(valid_patients_frontal)]
    
    print(f"ReID-eligible frontal images: {len(reid_frontal)}")
    print(f"Difference from paper: {len(reid_frontal) - 111333}")
    print()
    
    # Strategy 2: Only PA views
    print("=== Strategy 2: Only PA Views ===")
    pa_df = full_df[full_df['ViewPosition'] == 'PA'].copy()
    print(f"PA view images: {len(pa_df)}")
    
    patient_counts_pa = pa_df['subject_id'].value_counts()
    valid_patients_pa = patient_counts_pa[patient_counts_pa >= 2].index
    reid_pa = pa_df[pa_df['subject_id'].isin(valid_patients_pa)]
    
    print(f"ReID-eligible PA images: {len(reid_pa)}")
    print(f"Difference from paper: {len(reid_pa) - 111333}")
    print()
    
    # Strategy 3: Train set only with all views
    print("=== Strategy 3: Train Set Only (All Views) ===")
    train_df = full_df[full_df['split'] == 'train'].copy()
    print(f"Train set images: {len(train_df)}")
    
    patient_counts_train = train_df['subject_id'].value_counts()
    valid_patients_train = patient_counts_train[patient_counts_train >= 2].index
    reid_train = train_df[train_df['subject_id'].isin(valid_patients_train)]
    
    print(f"ReID-eligible train images: {len(reid_train)}")
    print(f"Difference from paper: {len(reid_train) - 111333}")
    print()
    
    # Strategy 4: Train set only with frontal views
    print("=== Strategy 4: Train Set + Frontal Views ===")
    train_frontal_df = full_df[(full_df['split'] == 'train') & 
                               (full_df['ViewPosition'].isin(['PA', 'AP']))].copy()
    print(f"Train frontal images: {len(train_frontal_df)}")
    
    patient_counts_train_frontal = train_frontal_df['subject_id'].value_counts()
    valid_patients_train_frontal = patient_counts_train_frontal[patient_counts_train_frontal >= 2].index
    reid_train_frontal = train_frontal_df[train_frontal_df['subject_id'].isin(valid_patients_train_frontal)]
    
    print(f"ReID-eligible train frontal images: {len(reid_train_frontal)}")
    print(f"Difference from paper: {len(reid_train_frontal) - 111333}")
    print()
    
    # Strategy 5: Different patient threshold
    print("=== Strategy 5: Different Patient Thresholds ===")
    for min_images in [3, 4, 5]:
        patient_counts_all = full_df['subject_id'].value_counts()
        valid_patients_thresh = patient_counts_all[patient_counts_all >= min_images].index
        reid_thresh = full_df[full_df['subject_id'].isin(valid_patients_thresh)]
        
        print(f"ReID with ≥{min_images} images per patient: {len(reid_thresh)} images")
        print(f"  Difference from paper: {len(reid_thresh) - 111333}")
    
    print()
    
    # Strategy 6: Combined filters to get close to 111,333
    print("=== Strategy 6: Finding Best Combination ===")
    
    # Try frontal views with higher patient threshold
    for min_images in [2, 3, 4, 5, 6]:
        frontal_df = full_df[full_df['ViewPosition'].isin(['PA', 'AP'])].copy()
        patient_counts = frontal_df['subject_id'].value_counts()
        valid_patients = patient_counts[patient_counts >= min_images].index
        reid_filtered = frontal_df[frontal_df['subject_id'].isin(valid_patients)]
        
        difference = abs(len(reid_filtered) - 111333)
        print(f"Frontal + ≥{min_images} images/patient: {len(reid_filtered)} images (diff: {difference})")
        
        if difference < 5000:  # Close match
            print(f"  ★ CLOSE MATCH with frontal views + ≥{min_images} images per patient!")
    
    # Check if using only certain splits helps
    print()
    print("=== Strategy 7: Different Split Combinations ===")
    
    # Train + Val only
    train_val_df = full_df[full_df['split'].isin(['train', 'validate'])].copy()
    patient_counts_tv = train_val_df['subject_id'].value_counts()
    valid_patients_tv = patient_counts_tv[patient_counts_tv >= 2].index
    reid_train_val = train_val_df[train_val_df['subject_id'].isin(valid_patients_tv)]
    
    print(f"Train + Validate sets: {len(reid_train_val)} images")
    print(f"  Difference from paper: {len(reid_train_val) - 111333}")

if __name__ == "__main__":
    find_paper_matching_filters()
