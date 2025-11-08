import os
import pandas as pd
from pathlib import Path
import yaml
import numpy as np
from collections import defaultdict
from PIL import Image
import shutil
from tqdm import tqdm
from typing import Optional, Tuple, Dict, Any
import math
import random

def load_and_merge_data(config):
    """Load and merge MIMIC-CXR data"""
    print("Loading MIMIC-CXR data...")
    
    raw_data_folder = config['raw_data_folder']
    
    # Read official data files
    split_df = pd.read_csv(f'{raw_data_folder}/mimic-cxr-2.0.0-split.csv')
    metadata_df = pd.read_csv(f'{raw_data_folder}/mimic-cxr-2.0.0-metadata.csv')
    chexpert_df = pd.read_csv(f'{raw_data_folder}/mimic-cxr-2.0.0-chexpert.csv')
    
    print(f"Split data: {split_df.shape}")
    print(f"Metadata: {metadata_df.shape}")
    print(f"MIMIC-CXR labels: {chexpert_df.shape}")
    
    # Merge data
    # First merge split and metadata (on dicom_id)
    full_df = split_df.merge(metadata_df, on='dicom_id', how='left')
    
    # Check and handle duplicate columns
    subject_id_match = (full_df['subject_id_x'] == full_df['subject_id_y']).all()
    study_id_match = (full_df['study_id_x'] == full_df['study_id_y']).all()
    
    if subject_id_match and study_id_match:
        full_df = full_df.drop(['subject_id_y', 'study_id_y'], axis=1)
        full_df = full_df.rename(columns={'subject_id_x': 'subject_id', 'study_id_x': 'study_id'})
    else:
        raise ValueError("Subject ID or Study ID mismatch between split and metadata files!")
    
    # Merge CheXpert labels (on subject_id and study_id)
    full_df = full_df.merge(chexpert_df, on=['subject_id', 'study_id'], how='left')
    
    print(f"Final merged data: {full_df.shape}")
    return full_df

def generate_image_paths(df, config):
    """Generate raw and processed image paths.
    """
    raw_data_folder = config['raw_data_folder']
    processed_dir = config.get('shared_images_dir', os.path.join(config['output_folder'], 'images'))
    os.makedirs(processed_dir, exist_ok=True) 

    def get_raw_path(row):
        subject_id = row['subject_id']
        study_id = row['study_id']
        dicom_id = row['dicom_id']
        folder_prefix = f"p{str(subject_id)[:2]}"
        return os.path.join(
            raw_data_folder,
            folder_prefix,
            f"p{subject_id}",
            f"s{study_id}",
            f"{dicom_id}.jpg"
        )

    def get_processed_path(row):
        return os.path.join(processed_dir, f"{row['dicom_id']}.png")

    print("Generating image paths (raw + processed)...")
    df['image_path'] = df.apply(get_raw_path, axis=1)
    df['processed_image_path'] = df.apply(get_processed_path, axis=1)

    sample_paths = df['image_path'].sample(min(100, len(df))).tolist()
    existing_count = sum(1 for p in sample_paths if os.path.exists(p))
    print(f"Raw path validation: {existing_count}/{len(sample_paths)} sample raw paths exist")
    if existing_count == 0:
        print("Warning: No sample RAW paths found! Please check the path generation logic.")

    return df


def resize_and_save_image(src_path, dst_path, config):
    """Resize image with aspect ratio preservation and center padding, save as PNG"""
    try:
        target_size = tuple(config['resize'])
        image_config = config.get('image_processing', {})
        bg_color = image_config.get('background_color', 0)
        compression_level = image_config.get('compression_level', 3)
        
        # Load image and convert to grayscale for medical images
        image = Image.open(src_path).convert('L')
        
        # Calculate scaling factor to maintain aspect ratio
        img_width, img_height = image.size
        target_width, target_height = target_size
        
        scale = min(target_width / img_width, target_height / img_height)
        
        # Resize image
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create new image with target size and black background
        final_image = Image.new('L', target_size, bg_color)
        
        # Calculate position to center the image
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        
        # Paste resized image onto the center
        final_image.paste(resized_image, (x_offset, y_offset))
        
        # Save as PNG with compression
        final_image.save(dst_path, 'PNG', compress_level=compression_level)
        
        return True, None
        
    except Exception as e:
        return False, str(e)

def preprocess_images(df, output_folder, config):
    """Preprocess images: resize and save to target folder"""
    target_size = tuple(config['resize'])
    print(f"\n=== Preprocessing Images to {target_size} ===")
    
    # Create processed images storage directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Statistics
    success_count = 0
    error_count = 0
    error_files = []
    
    print("Processing images...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Resizing images"):
        src_path = row['image_path']
        
        # Generate processed file path (always PNG for medical images)
        dicom_id = row['dicom_id']
        processed_path = os.path.join(output_folder, f"{dicom_id}.png")
        
        # Skip if file already exists
        if os.path.exists(processed_path):
            success_count += 1
            continue
            
        # Check if source file exists
        if not os.path.exists(src_path):
            error_count += 1
            error_files.append(f"Source not found: {src_path}")
            continue
        
        # Process image
        success, error_msg = resize_and_save_image(src_path, processed_path, config)
        
        if success:
            success_count += 1
        else:
            error_count += 1
            error_files.append(f"{src_path}: {error_msg}")
    
    print(f"Image preprocessing complete:")
    print(f"  Successfully processed: {success_count}")
    print(f"  Errors: {error_count}")
    
    if error_count > 0 and error_count <= 10:
        print("  Sample errors:")
        for error in error_files[:5]:
            print(f"    {error}")
    elif error_count > 10:
        print(f"  Too many errors ({error_count}), check the first few:")
        for error in error_files[:3]:
            print(f"    {error}")
    
    return df

def apply_reid_filtering_by_study(df):
    """Apply ReID filtering: frontal views + patients with ≥2 studies"""
    print("\n=== Applying ReID Filtering ===")
    
    frontal_views =['PA', 'AP', 'AP AXIAL', 'AP LLD', 'PA LLD','PA RLD','AP RLD']
    min_studies = 2
    
    original_count = len(df)
    original_patients = df['subject_id'].nunique()
    
    # Step 1: Keep only frontal views
    df_frontal = df[df['ViewPosition'].isin(frontal_views)].copy()
    
    frontal_count = len(df_frontal)
    frontal_patients = df_frontal['subject_id'].nunique()
    
    print(f"Step 1 - Frontal views only ({frontal_views}):")
    print(f"  Images: {original_count} -> {frontal_count}")
    print(f"  Patients: {original_patients} -> {frontal_patients}")
    
    # Step 2: Keep only patients with ≥min_studies studies
    patient_study_counts = df_frontal.groupby('subject_id')['study_id'].nunique()
    
    print(f"\nStep 2 - Patient study distribution:")
    print(f"  1 study: {len(patient_study_counts[patient_study_counts == 1])} patients")
    print(f"  2 studies: {len(patient_study_counts[patient_study_counts == 2])} patients")
    print(f"  3+ studies: {len(patient_study_counts[patient_study_counts >= 3])} patients")
    
    # Filter patients with ≥min_studies studies
    valid_patients = patient_study_counts[patient_study_counts >= min_studies].index
    df_reid = df_frontal[df_frontal['subject_id'].isin(valid_patients)].copy()
    
    reid_count = len(df_reid)
    reid_patients = len(valid_patients)
    
    print(f"\nReID filtering result (≥{min_studies} studies per patient):")
    print(f"  Images: {frontal_count} -> {reid_count}")
    print(f"  Patients: {frontal_patients} -> {reid_patients}")
    print(f"  Avg images per patient: {reid_count / reid_patients:.2f}")
    
    # Detailed split analysis
    print(f"\n=== Detailed Split Analysis ===")
    for split_name in ['train', 'validate', 'test']:
        split_data = df_reid[df_reid['split'] == split_name]
        split_patients = split_data['subject_id'].nunique()
        split_studies = split_data.groupby('subject_id')['study_id'].nunique().sum()
        avg_studies_per_patient = split_data.groupby('subject_id')['study_id'].nunique().mean()
        avg_images_per_patient = len(split_data) / split_patients if split_patients > 0 else 0
        
        print(f"{split_name.upper()}:")
        print(f"  Images: {len(split_data)}")
        print(f"  Patients: {split_patients}")
        print(f"  Studies: {split_studies}")
        print(f"  Avg studies per patient: {avg_studies_per_patient:.2f}")
        print(f"  Avg images per patient: {avg_images_per_patient:.2f}")
        print()
    
    return df_reid

def apply_reid_filtering_by_patient(df, config=None):
    """
    Apply ReID filtering following Macpherson et al.:
      1) Keep frontal views only (default: ['PA', 'AP'])
      2) Keep patients with at least N images (default: 2; count by unique dicom_id)

    Args:
        df (pd.DataFrame): merged MIMIC-CXR dataframe with columns:
            ['subject_id', 'study_id', 'dicom_id', 'ViewPosition', 'split', ...]
        config (dict, optional):
            - frontal_views: list[str], default ['PA', 'AP']
            - min_images_per_patient: int, default 2

    Returns:
        pd.DataFrame: filtered dataframe for ReID
    """
    print("\n=== Applying ReID Filtering (frontal + ≥ images/patient) ===")

    # Read parameters
    cfg = config or {}
    frontal_views = cfg.get('frontal_views', ['PA', 'AP', 'AP AXIAL', 'AP LLD',"AP RLD", 'PA LLD','PA RLD'])
    min_images = int(cfg.get('min_images_per_patient', 2))

    # Basic stats before filtering
    original_count = len(df)
    original_patients = df['subject_id'].nunique()

    # Step 1: Keep only frontal views
    df_frontal = df[df['ViewPosition'].isin(frontal_views)].copy()
    frontal_count = len(df_frontal)
    frontal_patients = df_frontal['subject_id'].nunique()

    print(f"Step 1 - Frontal views only {frontal_views}:")
    print(f"  Images:   {original_count} -> {frontal_count}")
    print(f"  Patients: {original_patients} -> {frontal_patients}")

    # Step 2: Keep only patients with ≥ min_images images (count by unique dicom_id)
    patient_img_counts = df_frontal.groupby('subject_id')['dicom_id'].nunique()

    # Distribution overview
    n_1img = int((patient_img_counts == 1).sum())
    n_2img = int((patient_img_counts == 2).sum())
    n_3plus = int((patient_img_counts >= 3).sum())
    print("\nStep 2 - Patient image-count distribution (frontal only):")
    print(f"  1 image:     {n_1img} patients")
    print(f"  2 images:    {n_2img} patients")
    print(f"  3+ images:   {n_3plus} patients")

    valid_patients = patient_img_counts[patient_img_counts >= min_images].index
    df_reid = df_frontal[df_frontal['subject_id'].isin(valid_patients)].copy()

    reid_count = len(df_reid)
    reid_patients = df_reid['subject_id'].nunique()
    avg_imgs_per_patient = (reid_count / reid_patients) if reid_patients > 0 else 0.0

    print(f"\nReID filtering result (≥{min_images} images per patient):")
    print(f"  Images:   {frontal_count} -> {reid_count}")
    print(f"  Patients: {frontal_patients} -> {reid_patients}")
    print(f"  Avg images per patient: {avg_imgs_per_patient:.2f}")

    # Detailed split analysis (optional but helpful)
    print("\n=== Detailed Split Analysis ===")
    for split_name in ['train', 'validate', 'test']:
        split_data = df_reid[df_reid['split'] == split_name]
        split_images = len(split_data)
        split_patients = split_data['subject_id'].nunique()

        # Averages per patient within the split
        if split_patients > 0:
            avg_imgs_pp = split_data.groupby('subject_id')['dicom_id'].nunique().mean()
            avg_studies_pp = split_data.groupby('subject_id')['study_id'].nunique().mean()
        else:
            avg_imgs_pp = 0.0
            avg_studies_pp = 0.0

        print(f"{split_name.upper()}:")
        print(f"  Images:                 {split_images}")
        print(f"  Patients:               {split_patients}")
        print(f"  Avg images / patient:   {avg_imgs_pp:.2f}")
        print(f"  Avg studies / patient:  {avg_studies_pp:.2f}\n")

    return df_reid

def reassign_splits(df, split_strategy='official', config=None):
    """
    Reassign train/val/test splits
    
    Args:
        df: DataFrame with ReID data
        split_strategy: 'official' (use original) or 'rebalanced' (rebalance splits)
        config: Configuration dict with split ratios
    """
    print(f"\n=== Split Strategy: {split_strategy} ===")
    
    if split_strategy == 'official':
        print("Using official MIMIC-CXR splits")
        return df
    
    elif split_strategy == 'rebalanced':
        print("Rebalancing splits for better test set size...")
        
        # Get target ratios from config or use defaults
        if config:
            target_train = config.get('target_train_ratio', 0.75)
            target_val = config.get('target_val_ratio', 0.10)
            target_test = config.get('target_test_ratio', 0.15)
        else:
            target_train, target_val, target_test = 0.75, 0.10, 0.15
        
        # Ensure ratios sum to 1.0
        total_ratio = target_train + target_val + target_test
        if abs(total_ratio - 1.0) > 0.01:
            print(f"Warning: Split ratios sum to {total_ratio:.3f}, normalizing...")
            target_train /= total_ratio
            target_val /= total_ratio
            target_test /= total_ratio
        
        # Current split distribution
        original_splits = df['split'].value_counts()
        print("Original split distribution:")
        for split_name, count in original_splits.items():
            patients = df[df['split'] == split_name]['subject_id'].nunique()
            print(f"  {split_name}: {count} images, {patients} patients")
        
        # Get unique patients (ensure no patient appears in multiple splits)
        all_patients = df['subject_id'].unique()
        
        import random
        random.seed(42)  # For reproducibility
        patient_list = list(all_patients)
        random.shuffle(patient_list)
        
        # Calculate target patient counts
        total_patients = len(patient_list)
        n_train = int(total_patients * target_train)
        n_val = int(total_patients * target_val)
        n_test = total_patients - n_train - n_val  # Remaining goes to test
        
        # Assign patients to splits
        train_patients = patient_list[:n_train]
        val_patients = patient_list[n_train:n_train + n_val]
        test_patients = patient_list[n_train + n_val:]
        
        # Create mapping
        patient_split_map = {}
        for p in train_patients:
            patient_split_map[p] = 'train'
        for p in val_patients:
            patient_split_map[p] = 'validate'
        for p in test_patients:
            patient_split_map[p] = 'test'
        
        # Apply new splits
        df_rebalanced = df.copy()
        df_rebalanced['split'] = df_rebalanced['subject_id'].map(patient_split_map)
        
        # Print new distribution
        new_splits = df_rebalanced['split'].value_counts()
        print(f"\nRebalanced split distribution (target ratios: {target_train:.1%}/{target_val:.1%}/{target_test:.1%}):")
        for split_name in ['train', 'validate', 'test']:
            if split_name in new_splits:
                count = new_splits[split_name]
                patients = df_rebalanced[df_rebalanced['split'] == split_name]['subject_id'].nunique()
                pct = count / len(df_rebalanced) * 100
                print(f"  {split_name}: {count} images ({pct:.1f}%), {patients} patients")
        
        return df_rebalanced
    
    else:
        raise ValueError(f"Unknown split_strategy: {split_strategy}")



def build_patient_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-patient stats: n_images, n_studies."""
    stats = df.groupby('subject_id').agg(
        n_images=('dicom_id', 'nunique'),
        n_studies=('study_id', 'nunique')
    ).reset_index()
    return stats

def _bucket_label_by_n_images(n: int) -> str:
    # [2], [3], [4-5], [6-10], >10
    if n <= 2: return "2"
    if n == 3: return "3"
    if 4 <= n <= 5: return "4-5"
    if 6 <= n <= 10: return "6-10"
    return ">10"

# ===== NEW: studies bucketing =====
def _bucket_label_by_n_studies(n: int) -> str:
    # [2], [3], [4-5], [6-10], >10
    if n <= 2: return "2"
    if n == 3: return "3"
    if 4 <= n <= 5: return "4-5"
    if 6 <= n <= 10: return "6-10"
    return ">10"

def make_buckets(stats: pd.DataFrame,
                 bucket_by: str = "n_images",
                 mode: str = "fixed",
                 fixed_edges: list[int] = [2,3,5,10]) -> pd.Series:
    """
    Tag each patient with a bucket.
    Currently: fixed long-tail buckets for either n_images or n_studies.
    """
    if mode != "fixed":
        # 你的需求里只用 fixed；如需 quantile 可后续扩展
        pass
    key = bucket_by
    arr = stats[key].astype(int)
    if bucket_by == "n_images":
        return arr.map(_bucket_label_by_n_images)
    elif bucket_by == "n_studies":
        return arr.map(_bucket_label_by_n_studies)
    else:
        raise ValueError(f"Unsupported bucket_by={bucket_by}, choose 'n_images' or 'n_studies'")


def _cap_rows_for_patient(pdf: pd.DataFrame,
                          per_study_images: Optional[int],
                          per_patient_images: Optional[int],
                          rng: np.random.Generator) -> pd.DataFrame:
    """Apply optional caps; deterministic via rng."""
    rows = pdf
    if per_study_images is not None:
        parts = []
        for sid, grp in rows.groupby('study_id'):
            if len(grp) > per_study_images:
                parts.append(grp.sample(n=per_study_images, random_state=rng.integers(0, 2**31-1)))
            else:
                parts.append(grp)
        rows = pd.concat(parts, ignore_index=True)
    if per_patient_images is not None and len(rows) > per_patient_images:
        rows = rows.sample(n=per_patient_images, random_state=rng.integers(0, 2**31-1))
    return rows

def select_patient_subset_by_images(
    df: pd.DataFrame,
    fraction: float = 0.10,
    tolerance: float = 0.02,
    seed: int = 42,
    bucket_by: str = "n_images",         # 现在也支持 "n_studies"
    bucket_mode: str = "fixed",
    fixed_edges: list[int] = [2,3,5,10],
    min_images_per_patient: int = 2,
    min_studies_per_patient: Optional[int] = None,  # NEW: 严格过滤 n_studies>=K（如 3）
    prefer_min_studies: Optional[int] = None,       # 可保留，通常在设置严格过滤后可置 None
    cap_per_patient_images: Optional[int] = None,
    cap_per_study_images: Optional[int] = None,
) -> pd.DataFrame:
    """
    Stratified-by-bucket quota selection (images quota per bucket).
    - 可按 n_studies 进行严格过滤与分层，避免子集清一色 study=2。
    """
    assert 0 < fraction < 1, "fraction should be in (0,1)"
    total_images = len(df)
    target_total = int(round(total_images * fraction))
    tol_abs = int(math.floor(total_images * tolerance))
    bucket_order = ["2", "3", "4-5", "6-10", ">10"]
    rng = np.random.default_rng(seed)

    # ---- per-patient stats & eligibility ----
    stats = build_patient_stats(df)  # -> subject_id, n_images, n_studies
    stats = stats[stats['n_images'] >= min_images_per_patient].copy()
    if min_studies_per_patient is not None:
        stats = stats[stats['n_studies'] >= int(min_studies_per_patient)].copy()
    if len(stats) == 0:
        raise RuntimeError("No patients left after min_* filtering.")

    # bucketing by either n_images or n_studies
    stats['bucket'] = make_buckets(stats, bucket_by=bucket_by, mode=bucket_mode, fixed_edges=fixed_edges)
    pid2bucket = stats.set_index('subject_id')['bucket'].to_dict()

    work = df[df['subject_id'].isin(stats['subject_id'])].copy()
    work['_bucket'] = work['subject_id'].map(pid2bucket)

    # ---- image mass per bucket in FULL (to compute image quotas) ----
    mass_full = work.groupby('_bucket')['dicom_id'].nunique().reindex(bucket_order, fill_value=0)
    total_mass_full = int(mass_full.sum())
    if total_mass_full == 0:
        raise RuntimeError("Unexpected zero mass after filtering.")

    # quota: 该 bucket 目标“图像数”
    quota = {b: int(round(mass_full[b] * fraction)) for b in bucket_order}

    # ---- patient order in each bucket ----
    bucket2patients: Dict[str, list[int]] = {}
    for b in bucket_order:
        sub = stats[stats['bucket'] == b]
        if prefer_min_studies is not None:
            hi = sub[sub['n_studies'] >= prefer_min_studies]['subject_id'].tolist()
            lo = sub[sub['n_studies'] <  prefer_min_studies]['subject_id'].tolist()
            rng.shuffle(hi); rng.shuffle(lo)
            pts = hi + lo
        else:
            pts = sub['subject_id'].tolist()
            rng.shuffle(pts)
        bucket2patients[b] = pts

    selected_pids: set[int] = set()
    selected_rows: list[pd.DataFrame] = []
    selected_seq: list[Tuple[int, int, str]] = []
    bucket_selected_images = {b: 0 for b in bucket_order}

    def _add_patient(pid: int, b: str) -> Optional[int]:
        nonlocal selected_rows
        pdf = work[work['subject_id'] == pid]
        capped = _cap_rows_for_patient(pdf, cap_per_study_images, cap_per_patient_images, rng)
        eff = len(capped)
        if eff < min_images_per_patient:
            return None
        selected_pids.add(pid)
        selected_rows.append(capped)
        selected_seq.append((pid, eff, b))
        bucket_selected_images[b] += eff
        return eff

    # 1) primary pass: 填满各 bucket 的 quota（± 按质量分配的容差）
    def _bucket_tol(b: str) -> int:
        if total_mass_full == 0: return 1
        return max(1, int(round(tol_abs * (mass_full[b] / total_mass_full))))

    for b in bucket_order:
        if quota[b] <= 0:
            continue
        btol = _bucket_tol(b)
        for pid in bucket2patients[b]:
            if pid in selected_pids:
                continue
            if bucket_selected_images[b] >= quota[b] + btol:
                break
            _add_patient(pid, b)

    current_total = sum(len(x) for x in selected_rows)

    # 2) deficit fill: 若总量未达标，继续从“缺口最大”的桶补充
    if current_total < target_total - tol_abs:
        cursors = {b: 0 for b in bucket_order}
        bucket_lists = {b: [pid for pid in bucket2patients[b] if pid not in selected_pids] for b in bucket_order}
        while current_total < target_total - tol_abs:
            deficits = {b: max(0, quota[b] - bucket_selected_images[b]) for b in bucket_order}
            if sum(deficits.values()) == 0:
                pick_order = sorted(bucket_order, key=lambda x: mass_full[x], reverse=True)
            else:
                pick_order = sorted(bucket_order, key=lambda x: deficits[x], reverse=True)
            picked = False
            for b in pick_order:
                lst = bucket_lists[b]
                while cursors[b] < len(lst) and lst[cursors[b]] in selected_pids:
                    cursors[b] += 1
                if cursors[b] >= len(lst):
                    continue
                pid = lst[cursors[b]]; cursors[b] += 1
                eff = _add_patient(pid, b)
                if eff is not None:
                    current_total += eff
                    picked = True
                    break
            if not picked:
                break

    # 3) overshoot clamp（如严重超额，回退一个最合适的患者）
    if current_total > target_total + tol_abs and len(selected_seq) >= 1:
        overshoot = current_total - target_total
        best_idx, best_gap = None, overshoot
        for idx, (_, eff, b) in enumerate(reversed(selected_seq[-10:]), 1):
            gap = abs((current_total - eff) - target_total)
            if gap < best_gap:
                best_gap, best_idx = gap, len(selected_seq) - idx
        if best_idx is not None:
            pid_to_remove, _, _ = selected_seq.pop(best_idx)
            keep_pids = {pid for pid, _, _ in selected_seq}
            selected_rows = []
            bucket_selected_images = {b: 0 for b in bucket_order}
            for pid, _, b in selected_seq:
                pdf = work[work['subject_id'] == pid]
                capped = _cap_rows_for_patient(pdf, cap_per_study_images, cap_per_patient_images, rng)
                selected_rows.append(capped)
                bucket_selected_images[b] += len(capped)

    subset = pd.concat(selected_rows, ignore_index=True)

    # ---- sanity ----
    # 每位患者至少 min_images_per_patient 张
    per_pat_counts = subset.groupby('subject_id')['dicom_id'].nunique()
    if per_pat_counts.min() < min_images_per_patient:
        raise RuntimeError("Found a patient with < min_images_per_patient in subset.")
    # 若设定了 min_studies_per_patient，也检查
    if min_studies_per_patient is not None:
        per_pat_st = subset.groupby('subject_id')['study_id'].nunique()
        if per_pat_st.min() < int(min_studies_per_patient):
            raise RuntimeError("Found a patient with < min_studies_per_patient in subset.")

    # 报告：各 bucket 的图像占比（全量 vs 子集）
    subset_mass = subset.groupby('_bucket')['dicom_id'].nunique().reindex(bucket_order, fill_value=0)
    print("\n[Subset] bucket image mass (FULL -> SUBSET) by", bucket_by)
    for b in bucket_order:
        full_b = int(mass_full[b]); sub_b = int(subset_mass[b])
        share_full = full_b / total_mass_full if total_mass_full else 0.0
        share_sub = sub_b / max(1, subset.shape[0])
        print(f"  {b:>4}: {full_b:>6} ({share_full:5.1%}) -> {sub_b:>6} ({share_sub:5.1%})")

    avg_imgs_per_patient = float(subset.shape[0]) / max(1, subset['subject_id'].nunique())
    print(f"[Subset] total images={subset.shape[0]} (target ~{target_total}, tol ±{tol_abs}); "
          f"patients={subset['subject_id'].nunique()}; "
          f"avg imgs/patient={avg_imgs_per_patient:.2f}")

    return subset


# REPLACE the old assign_splits_greedy with this stratified-by-bucket version
def assign_splits_greedy(
    df_subset: pd.DataFrame,
    ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 42,
    bucket_by: str = "n_studies",   # 建议与子集阶段一致：n_studies 或 n_images
    bucket_mode: str = "fixed",
    fixed_edges: list[int] = [2,3,5,10],
    prefer_min_studies: Optional[int] = None,
    tolerance: float = 0.02,
) -> pd.DataFrame:
    """
    Strict stratified assignment by bucket:
      - For each bucket b, compute image targets per split: target[b, split] ≈ mass_b * ratios[split]
      - Within each bucket, assign patients to the split with the largest **bucket-level** deficit
      - Fallback to global deficit only if the bucket-level targets are all met (due to rounding)
    """
    assert abs(sum(ratios) - 1.0) < 1e-6
    splits = ['train', 'validate', 'test']
    total_images = len(df_subset)
    global_targets = {s: int(round(total_images * r)) for s, r in zip(splits, ratios)}
    tol_abs = int(math.floor(total_images * tolerance))
    rng = np.random.default_rng(seed)

    # ---- per-patient stats on the SUBSET, and buckets ----
    sstats = build_patient_stats(df_subset)  # columns: subject_id, n_images, n_studies
    sstats['bucket'] = make_buckets(sstats, bucket_by=bucket_by, mode=bucket_mode, fixed_edges=fixed_edges)
    pid2bucket = sstats.set_index('subject_id')['bucket'].to_dict()

    # tag rows with bucket
    df_subset = df_subset.copy()
    df_subset['_bucket'] = df_subset['subject_id'].map(pid2bucket)

    # mass per bucket (image counts in subset)
    bucket_order = ["2", "3", "4-5", "6-10", ">10"]
    mass_b = df_subset.groupby('_bucket')['dicom_id'].nunique().reindex(bucket_order, fill_value=0).to_dict()

    # helper: allocate integer targets for a bucket using largest-remainder (Hamilton) method
    def _alloc_targets(total_mass: int, ratios_vec: Tuple[float, float, float]) -> Dict[str, int]:
        raw = [total_mass * r for r in ratios_vec]
        floors = [int(math.floor(x)) for x in raw]
        remain = total_mass - sum(floors)
        # assign the remaining to the largest fractional parts
        fracs = [(x - f, i) for i, (x, f) in enumerate(zip(raw, floors))]
        fracs.sort(reverse=True)
        alloc = floors[:]
        for k in range(remain):
            alloc[fracs[k][1]] += 1
        return {s: alloc[i] for i, s in enumerate(splits)}

    # per-bucket split targets & counters
    bucket_targets: Dict[str, Dict[str, int]] = {b: _alloc_targets(mass_b[b], ratios) for b in bucket_order}
    bucket_counts:  Dict[str, Dict[str, int]] = {b: {s: 0 for s in splits} for b in bucket_order}

    # global counters
    global_counts = {s: 0 for s in splits}

    # precompute patient rows/size and lists per bucket
    pid2n = df_subset.groupby('subject_id').size().to_dict()

    bucket2pids: Dict[str, list] = {}
    for b in bucket_order:
        pids_b = sstats[sstats['bucket'] == b]['subject_id'].tolist()
        if prefer_min_studies is not None:
            hi = sstats[(sstats['bucket'] == b) & (sstats['n_studies'] >= prefer_min_studies)]['subject_id'].tolist()
            lo = [p for p in pids_b if p not in set(hi)]
            rng.shuffle(hi); rng.shuffle(lo)
            pids_b = hi + lo
        else:
            rng.shuffle(pids_b)
        bucket2pids[b] = pids_b

    # assignment map
    pid2split: Dict[int, str] = {}

    # ---- assign per bucket by bucket-level deficits ----
    for b in bucket_order:
        if mass_b[b] == 0:
            continue
        for pid in bucket2pids[b]:
            nimg = pid2n.get(pid, 0)
            if nimg <= 0:
                continue
            # pick split with largest **bucket** deficit
            deficits_b = {s: bucket_targets[b][s] - bucket_counts[b][s] for s in splits}
            if max(deficits_b.values()) > 0:
                # within bucket, fill the largest bucket deficit
                s_pick = max(deficits_b, key=lambda s: deficits_b[s])
            else:
                # bucket already meets targets (rounding), fallback to global deficit
                deficits_g = {s: global_targets[s] - global_counts[s] for s in splits}
                s_pick = max(deficits_g, key=lambda s: deficits_g[s])

            pid2split[pid] = s_pick
            bucket_counts[b][s_pick] += nimg
            global_counts[s_pick]    += nimg

    # materialize split assignments
    df_subset['split_subset'] = df_subset['subject_id'].map(pid2split)

    # sanity: patient-disjoint
    sets = [set(df_subset[df_subset['split_subset']==s]['subject_id'].unique()) for s in splits]
    assert sets[0].isdisjoint(sets[1]) and sets[0].isdisjoint(sets[2]) and sets[1].isdisjoint(sets[2]), \
        "Patient leakage across subset splits!"

    # diagnostics
    cur_counts = df_subset['split_subset'].value_counts().to_dict()
    print(f"\n[Subset-Resplit] Image totals (target ~{ratios}): "
          f"train={cur_counts.get('train',0)}, "
          f"validate={cur_counts.get('validate',0)}, "
          f"test={cur_counts.get('test',0)}; total={total_images}")

    # optional: bucket-level check summary
    print("[Subset-Resplit] Per-bucket targets vs assigned (images):")
    for b in bucket_order:
        if mass_b[b] == 0: 
            continue
        tgt = bucket_targets[b]; cnt = bucket_counts[b]
        print(f"  bucket {b:>4}  target  (t/v/t): "
              f"{tgt['train']:>6}/{tgt['validate']:>6}/{tgt['test']:>6} | "
              f"assigned: {cnt['train']:>6}/{cnt['validate']:>6}/{cnt['test']:>6}")

    return df_subset


def save_subset_resplit_csvs(df_subset: pd.DataFrame, output_folder: str, tag: str) -> None:
    os.makedirs(output_folder, exist_ok=True)
    for split_name, fname in [('train', f"train_{tag}.csv"),
                              ('validate', f"val_{tag}.csv"),
                              ('test', f"test_{tag}.csv")]:
        part = df_subset[df_subset['split_subset'] == split_name]
        part.to_csv(os.path.join(output_folder, fname), index=False)
    print(f"[Subset-Resplit] Saved CSVs with tag '{tag}' to {output_folder}")

def dump_subset_meta(output_folder: str, tag: str, meta: Dict[str, Any]) -> None:
    path = os.path.join(output_folder, f"subset_meta_{tag}.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True)
    print(f"[Subset-Resplit] Saved meta: {path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate MIMIC-CXR datasets')
    parser.add_argument('--config', type=str, default='../configs/mimic_cxr.yaml',
                        help='Config file path')
    parser.add_argument('--resize', action='store_true',
                        help='Skip image resizing (only generate CSV files)')
    parser.add_argument('--split-strategy', type=str, choices=['official', 'rebalanced'], 
                        default='rebalanced', help='Split strategy: official (use original) or rebalanced (larger test set)')
    parser.add_argument('--filter-strategy', type=str, choices=['study', 'patient'], 
                    default='patient', help='Filter strategy: study or patient')
    parser.add_argument('--make-subset-resplit', action='store_true',
                        help='Create a ~fraction patient-level subset and re-split into 70/10/20 (image-level) without altering original outputs.')
    parser.add_argument('--subset-fraction', type=float, default=None,
                        help='Target overall image fraction for the subset (e.g., 0.10). Overrides config if set.')
    parser.add_argument('--subset-tag', type=str, default='sub10_p70_10_20',
                        help='Suffix tag for subset CSVs, e.g., train_{tag}.csv / val_{tag}.csv / test_{tag}.csv')

    args = parser.parse_args()
    do_resize = args.resize
    
    # Load config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"=== MIMIC-CXR Dataset Generation ===")
    print(f"Config: {args.config}")
    print(f"Split strategy: {args.split_strategy}")
    print(f"Resize images: {do_resize}")
    print()
    
    # Load and merge data
    df = load_and_merge_data(config)
    
    # Generate image paths
    df = generate_image_paths(df, config)

    if do_resize:
        # Process images
        output_folder = config['output_folder']
        output_images_dir = config.get('shared_images_dir', os.path.join(output_folder, 'images'))
        os.makedirs(output_images_dir, exist_ok=True)
        
        target_size = tuple(config['resize'])
        print(f"Processing {len(df)} images to size {target_size}...")
        preprocess_images(df, output_images_dir, config)
        print(f"Images saved to: {output_images_dir}")
    else:
        print("Skipping image resize (do_resize=False)")

    df['image_path'] = df['processed_image_path']
    df = df.drop(columns=['processed_image_path'])

    output_folder = config['output_folder']
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate ReID dataset first (this will be the base for both datasets)
    if args.filter_strategy == 'patient':
        df_reid = apply_reid_filtering_by_patient(df, config)
    else:
        df_reid = apply_reid_filtering_by_study(df)
    df_reid = reassign_splits(df_reid, args.split_strategy, config)

    train_df = df_reid[df_reid['split'] == 'train']
    val_df = df_reid[df_reid['split'] == 'validate']
    test_df = df_reid[df_reid['split'] == 'test']

    train_df.to_csv(os.path.join(output_folder, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_folder, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_folder, 'test.csv'), index=False)

    print("\n=== Dataset generation completed! ===")
    print("🔗 Both datasets use IDENTICAL data for Unlearnable Example consistency!")
    
    # Final summary
    final_splits = df_reid['split'].value_counts()
    print(f"\nFinal dataset summary:")
    print(f"  Total images: {len(df_reid):,}")
    print(f"  Total patients: {df_reid['subject_id'].nunique():,}")
    print(f"  Split distribution:")
    for split_name in ['train', 'validate', 'test']:
        if split_name in final_splits:
            count = final_splits[split_name]
            patients = df_reid[df_reid['split'] == split_name]['subject_id'].nunique()
            pct = count / len(df_reid) * 100
            print(f"    {split_name}: {count:,} images ({pct:.1f}%), {patients:,} patients")

    subset_cfg = config.get('subset_resplit')
    if isinstance(subset_cfg, dict) and subset_cfg.get('enabled', False):
        print("\n=== Subset-Resplit (config-driven) ===")

        tag = subset_cfg.get('tag', 'sub10_p70_10_20')
        fraction = float(subset_cfg.get('fraction', 0.10))
        ratios = tuple(subset_cfg.get('ratios', [0.7, 0.1, 0.2]))
        seed = int(subset_cfg.get('seed', 42))
        bucket_by = subset_cfg.get('bucket_by', 'n_images')
        bucket_mode = subset_cfg.get('bucket_mode', 'fixed')
        fixed_edges = subset_cfg.get('fixed_edges', [2, 3, 5, 10])
        min_images_per_patient = int(subset_cfg.get('min_images_per_patient', 2))
        prefer_min_studies = subset_cfg.get('prefer_min_studies', 2)
        prefer_min_studies = None if prefer_min_studies is None else int(prefer_min_studies)
        tolerance = float(subset_cfg.get('tolerance', 0.02))

        cap_cfg = subset_cfg.get('cap', {}) or {}
        cap_per_patient_images = cap_cfg.get('per_patient_images', None)
        cap_per_patient_images = None if cap_per_patient_images is None else int(cap_per_patient_images)
        cap_per_study_images = cap_cfg.get('per_study_images', None)
        cap_per_study_images = None if cap_per_study_images is None else int(cap_per_study_images)

        # Stage 1: 选出“≈10% 图像”的患者级子集
        min_studies_per_patient = subset_cfg.get('min_studies_per_patient', None)
        min_studies_per_patient = None if min_studies_per_patient is None else int(min_studies_per_patient)

        sub_df = select_patient_subset_by_images(
            df=df_reid,
            fraction=fraction,
            tolerance=tolerance,
            seed=seed,
            bucket_by=bucket_by,
            bucket_mode=bucket_mode,
            fixed_edges=fixed_edges,
            min_images_per_patient=min_images_per_patient,
            min_studies_per_patient=min_studies_per_patient,   # <-- NEW: 严格过滤
            prefer_min_studies=prefer_min_studies,
            cap_per_patient_images=cap_per_patient_images,
            cap_per_study_images=cap_per_study_images,
        )


        tot_all = len(df_reid); tot_sub = len(sub_df)
        print(f"[Subset] total={tot_all} -> subset={tot_sub} "
            f"({tot_sub / max(1, tot_all):.2%}, target={fraction:.0%})")

        # Stage 2: 子集内部做 70/10/20（按“图像数”逼近），仍然 patient-disjoint
        sub_df = assign_splits_greedy(
            df_subset=sub_df,
            ratios=ratios,
            seed=seed,
            bucket_by=bucket_by,
            bucket_mode=bucket_mode,
            fixed_edges=fixed_edges,
            prefer_min_studies=prefer_min_studies,
            tolerance=tolerance,
        )

        # 额外保存一套CSV：train_{tag}.csv / val_{tag}.csv / test_{tag}.csv
        save_subset_resplit_csvs(sub_df, output_folder, tag)

        # 记录元信息，便于复现/论文附录
        meta = {
            'fraction': fraction,
            'ratios': list(ratios),
            'seed': seed,
            'bucket_by': bucket_by,
            'bucket_mode': bucket_mode,
            'fixed_edges': fixed_edges,
            'min_images_per_patient': min_images_per_patient,
            'prefer_min_studies': prefer_min_studies,
            'tolerance': tolerance,
            'cap': {
                'per_patient_images': cap_per_patient_images,
                'per_study_images': cap_per_study_images,
            },
            'subset_images': int(tot_sub),
            'full_images': int(tot_all),
            'subset_share': float(tot_sub / max(1, tot_all)),
            'split_counts': sub_df['split_subset'].value_counts().to_dict(),
            'patients_per_split': {
                s: int(sub_df[sub_df['split_subset'] == s]['subject_id'].nunique())
                for s in ['train', 'validate', 'test']
            },
        }
        dump_subset_meta(output_folder, tag, meta)
        # Final summary
        final_splits = sub_df['split_subset'].value_counts()
        print(f"\nFinal dataset summary:")
        print(f"  Total images: {len(sub_df):,}")
        print(f"  Total patients: {sub_df['subject_id'].nunique():,}")
        print(f"  Split distribution:")
        for split_name in ['train', 'validate', 'test']:
            if split_name in final_splits:
                count = final_splits[split_name]
                patients = sub_df[sub_df['split_subset'] == split_name]['subject_id'].nunique()
                pct = count / len(sub_df) * 100
                print(f"    {split_name}: {count:,} images ({pct:.1f}%), {patients:,} patients")
        
        # ===== NEW: study-level summary =====
        total_studies = sub_df['study_id'].nunique()
        print(f"  Total studies: {total_studies:,}")

        for split_name in ['train', 'validate', 'test']:
            split_data = sub_df[sub_df['split_subset'] == split_name]
            split_studies = split_data['study_id'].nunique()
            split_patients = split_data['subject_id'].nunique()
            avg_studies_pp = (split_data.groupby('subject_id')['study_id'].nunique().mean()
                            if split_patients > 0 else 0.0)
            print(f"    {split_name}: {split_studies:,} studies; "
                f"avg studies/patient: {avg_studies_pp:.2f}")

        print("=== Subset-Resplit: done ===")

if __name__ == "__main__":
    main()
