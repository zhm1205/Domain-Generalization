import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, List


def sample_patients_from_csv(
    file_path: str,
    k: int,
    seed: int = 42,
    output_csv: Optional[str] = None,
    save_patient_ids: bool = False,
    patient_ids_path: Optional[str] = None,
) -> None:
    """
    随机采样 k 个患者（以 subject_id 为单位），并将这些患者的所有图像行
    保存为一个新的 CSV 文件（列结构与原始 CSV 相同）。采样可通过 seed 复现。

    Args:
        file_path: 原始 train.csv 路径，需包含 'subject_id' 列。
        k: 采样患者数。
        seed: 随机种子，保证结果可复现。
        output_csv: 采样后保存的 CSV 路径；若为空，则默认与原文件同目录，
                    命名为 `<原名>_sampled_k{k}_seed{seed}.csv`。
        save_patient_ids: 是否另存被采样的 subject_id 列表（每行一个）。
        patient_ids_path: subject_id 列表保存路径；若为空，则默认与 output_csv 同目录，
                          命名为 `<原名>_sampled_patient_ids_k{k}_seed{seed}.txt`。
    """
    print(f"Reading data from: {file_path}")

    patients_to_images: Dict[str, List[dict]] = defaultdict(list)
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            if not fieldnames or 'subject_id' not in fieldnames:
                print("Error: The CSV file must contain a 'subject_id' column.")
                return
            for row in reader:
                patients_to_images[row['subject_id']].append(row)
    except FileNotFoundError:
        print(f"Error: The file was not found at '{file_path}'")
        return

    all_patient_ids = sorted(patients_to_images.keys())  # 排序确保跨环境一致
    total_patients = len(all_patient_ids)
    if k <= 0:
        print(f"Error: k must be positive; got k={k}.")
        return
    if k > total_patients:
        print(f"Error: You requested to sample {k} patients, but only {total_patients} are available.")
        return

    rng = random.Random(seed)  # 局部 RNG，避免影响全局随机态
    sampled_patient_ids = rng.sample(all_patient_ids, k)

    # 汇总这些患者的所有图像行
    sampled_rows: List[dict] = []
    for pid in sampled_patient_ids:
        sampled_rows.extend(patients_to_images[pid])

    # 决定输出路径
    in_path = Path(file_path)
    if output_csv is None:
        output_csv_path = in_path.with_name(f"{in_path.stem}_sampled_k{k}_seed{seed}{in_path.suffix}")
    else:
        output_csv_path = Path(output_csv)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 写出采样后的 CSV
    fieldnames = list(sampled_rows[0].keys()) if sampled_rows else None
    if not fieldnames:
        print("Error: No rows selected, cannot write CSV.")
        return

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as wf:
        writer = csv.DictWriter(wf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sampled_rows)

    # 可选：保存被采样的 subject_id 列表
    if save_patient_ids:
        if patient_ids_path is None:
            patient_ids_path = output_csv_path.with_name(
                f"{output_csv_path.stem.replace('.csv','')}_patient_ids_k{k}_seed{seed}.txt"
            )
        with open(patient_ids_path, 'w', encoding='utf-8') as pf:
            for pid in sampled_patient_ids:
                pf.write(f"{pid}\n")

    # 统计信息
    total_sampled_images = len(sampled_rows)
    print("\n--- Sampling Statistics ---")
    print(f"Total unique patients in the file: {total_patients}")
    print(f"Number of patients sampled (k):   {k}")
    print("-" * 30)
    print(f"Successfully sampled {len(sampled_patient_ids)} patients.")
    print(f"These patients correspond to a total of {total_sampled_images} images.")
    print(f"Saved CSV to: {output_csv_path}")
    if save_patient_ids:
        print(f"Saved subject_id list to: {patient_ids_path}")
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample K patients from a MIMIC-CXR train.csv file, save to a new CSV, and show statistics."
    )

    train_df_path = "/home/dengzhipeng/data/project/reid_ue/mimic_cxr/train.csv"

    parser.add_argument('-k', '--num_patients', type=int, default=1000,
                        help="The number of patients to randomly sample.")
    parser.add_argument('--file_path', type=str, default=train_df_path,
                        help="Path to the train.csv file.")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducible sampling.")
    parser.add_argument('--output_csv', type=str, default=None,
                        help="Path to save the sampled CSV. Defaults to <input>_sampled_k{K}_seed{seed}.csv")
    parser.add_argument('--save_patient_ids', action='store_true',
                        help="Also save the sampled subject_id list (one per line).")
    parser.add_argument('--patient_ids_path', type=str, default=None,
                        help="Path to save the subject_id list (txt). If omitted, an auto name is used.")

    args = parser.parse_args()

    sample_patients_from_csv(
        file_path=args.file_path,
        k=args.num_patients,
        seed=args.seed,
        output_csv=args.output_csv,
        save_patient_ids=args.save_patient_ids,
        patient_ids_path=args.patient_ids_path,
    )
