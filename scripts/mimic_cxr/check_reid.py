import os
import pandas as pd
from pathlib import Path
import yaml
import re

# 读取配置
filepath = Path(__file__).resolve().parent.parent
config = yaml.safe_load(open(filepath.joinpath("./configs/mimic_cxr.yaml")))

raw_data_folder = config["raw_data_folder"]

FRONTAL_TOKENS = {"AP", "PA"}
LATERAL_REGEX = r"\b(LAT|LATERAL|LL|RL|LAO|RAO|LPO|RPO|SWIMMERS|XTABLE)\b"
AP_REGEX = r"\bAP\b"
PA_REGEX = r"\bPA\b"

def prepare_full_df(split_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Merge and normalize columns: subject_id, study_id, dicom_id, ViewPosition, split, PPSD."""
    df = split_df.merge(metadata_df, on="dicom_id", how="left", validate="one_to_one")

    # unify subject_id / study_id
    for key in ["subject_id", "study_id"]:
        kx, ky = f"{key}_x", f"{key}_y"
        if key not in df.columns:
            if kx in df.columns and ky in df.columns:
                if not (df[kx].astype(str).fillna("") == df[ky].astype(str).fillna("")).all():
                    raise ValueError(f"{key} mismatch between split and metadata")
                df = df.drop(columns=[ky]).rename(columns={kx: key})
            elif kx in df.columns:
                df = df.rename(columns={kx: key})
            elif ky in df.columns:
                df = df.rename(columns={ky: key})
            else:
                raise KeyError(f"Missing {key} after merge")

    # normalize ViewPosition + PPSD
    df["ViewPosition"] = df["ViewPosition"].astype(object)  # keep NaN
    if "PerformedProcedureStepDescription" not in df.columns:
        raise KeyError("metadata.csv 缺少列 'PerformedProcedureStepDescription'")

    df["PPSD_NORM"] = df["PerformedProcedureStepDescription"].astype(str).str.upper()
    return df

def infer_view_from_ppsd(ppsd: str):
    """Return ('AP'/'PA'/None, reason_str) based on PPSD alone, conservatively."""
    if not isinstance(ppsd, str):
        return None, "ppsd_missing"

    # lateral present? then do NOT impute
    if re.search(LATERAL_REGEX, ppsd, flags=re.IGNORECASE):
        return None, "ppsd_contains_lateral"

    has_pa = re.search(PA_REGEX, ppsd) is not None
    has_ap = re.search(AP_REGEX, ppsd) is not None

    if has_pa and not has_ap:
        return "PA", "ppsd_pa_only"
    if has_ap and not has_pa:
        return "AP", "ppsd_ap_only"
    if has_ap and has_pa:
        return None, "ppsd_ap_and_pa_ambiguous"
    return None, "ppsd_no_ap_pa"

def make_nan_viewposition_imputation_files(
    split_csv_path: str,
    metadata_csv_path: str,
    out_dir: str = "./",
    out_prefix: str = "mimic_cxr_viewposition"
):
    os.makedirs(out_dir, exist_ok=True)

    split_df = pd.read_csv(split_csv_path)
    metadata_df = pd.read_csv(metadata_csv_path)
    full = prepare_full_df(split_df, metadata_df)

    # 仅挑出 ViewPosition 为 NaN/空/None 的行
    vp_is_nan = full["ViewPosition"].isna() | (full["ViewPosition"].astype(str).str.strip().isin(["", "NONE", "NaN", "nan"]))
    nan_df = full.loc[vp_is_nan, ["dicom_id","subject_id","study_id","split","ViewPosition","PPSD_NORM"]].drop_duplicates(subset=["dicom_id"]).copy()

    if nan_df.empty:
        print("没有 ViewPosition 为 NaN/空 的样本。")
        return None, None

    # 基于 PPSD 做保守推断
    inferred = []
    for _, row in nan_df.iterrows():
        v, reason = infer_view_from_ppsd(row["PPSD_NORM"])
        inferred.append((row["dicom_id"], row["subject_id"], row["study_id"], row["split"], row["ViewPosition"], row["PPSD_NORM"], v, reason))

    out_cols = ["dicom_id","subject_id","study_id","split","original_ViewPosition","PPSD_NORM","inferred_ViewPosition","inferred_reason"]
    out_df = pd.DataFrame(inferred, columns=out_cols)

    # 导出：完整候选清单（含原因，供人工标注）
    full_path = os.path.join(out_dir, f"{out_prefix}_nan_candidates_full.csv")
    out_df.to_csv(full_path, index=False)

    # 导出：仅包含成功推断的映射（用于 join）
    map_df = out_df[out_df["inferred_ViewPosition"].isin(FRONTAL_TOKENS)][["dicom_id","inferred_ViewPosition"]].drop_duplicates(subset=["dicom_id"])
    map_path = os.path.join(out_dir, f"{out_prefix}_imputed_mapping.csv")
    map_df.to_csv(map_path, index=False)

    # 简要统计
    print(f"NaN/empty ViewPosition 总数: {len(nan_df):,}")
    print("推断结果分布：")
    print(out_df["inferred_reason"].value_counts().to_string())
    print(f"已自动补成 AP/PA 的条目数: {len(map_df):,}")
    print(f"已保存：\n  - 完整候选: {full_path}\n  - 可直接 join 的映射: {map_path}")

    return full_path, map_path

split_csv = f"{raw_data_folder}/mimic-cxr-2.0.0-split.csv"
meta_csv  = f"{raw_data_folder}/mimic-cxr-2.0.0-metadata.csv"

full_csv, mapping_csv = make_nan_viewposition_imputation_files(
    split_csv_path=split_csv,
    metadata_csv_path=meta_csv,
    out_dir="./",                                 # 你想保存的目录
    out_prefix="mimic_cxr_viewposition"          # 文件名前缀
)

orig = pd.read_csv(meta_csv)
imp  = pd.read_csv(mapping_csv)  # 只包含成功推断的 dicom_id → AP/PA

merged = orig.merge(imp, on="dicom_id", how="left")
merged["ViewPosition_filled"] = merged["inferred_ViewPosition"].fillna(merged["ViewPosition"])
merged.to_csv("./metadata_with_viewposition_filled.csv", index=False)
print("已生成: ./metadata_with_viewposition_filled.csv（优先使用 ViewPosition_filled）")
