import pandas as pd
from pathlib import Path
import yaml

# 读取配置
filepath = Path(__file__).resolve().parent.parent
config = yaml.safe_load(open(filepath.joinpath("./configs/scripts/mimic_cxr.yaml")))
raw_data_folder = config["raw_data_folder"]

# 读取数据
split_df = pd.read_csv(f"{raw_data_folder}/mimic-cxr-2.0.0-split.csv")
meta_df  = pd.read_csv(f"{raw_data_folder}/mimic-cxr-2.0.0-metadata.csv")

# 三键合并，确保唯一性
cols = ["subject_id", "study_id", "dicom_id"]
df = split_df.merge(meta_df, on=cols, how="left")

# 查看所有视角
all_views = df["ViewPosition"].value_counts()
print("\n=== 全部 ViewPosition 统计 ===")
print(all_views)

# 定义不同视角策略
view_sets = {
    "ALL": None,  # 不筛
    "PA_only": {"PA"},
    "AP_only": {"AP"},
    "PA+AP": {"PA", "AP"}
}

def filter_and_count(df, allowed_views=None):
    # 视角过滤
    if allowed_views is not None:
        dff = df[df["ViewPosition"].isin(allowed_views)].copy()
    else:
        dff = df.copy()

    # 原始数量
    orig_count = len(dff)

    # 每 study 取 1 张（PA 优先）
    priority = dff["ViewPosition"].map({"PA": 0, "AP": 1}).fillna(99)
    dff = dff.sort_values(["subject_id", "study_id", priority.name]).drop_duplicates(
        ["subject_id", "study_id"], keep="first"
    )
    per_study_count = len(dff)

    # 只保留 ≥2 study 的患者
    study_counts = dff.groupby("subject_id")["study_id"].nunique()
    valid_subjects = study_counts[study_counts >= 2].index
    dff_reid = dff[dff["subject_id"].isin(valid_subjects)].copy()
    reid_count = len(dff_reid)

    # split 分布
    split_stats = {}
    for split_name in ["train", "validate", "test"]:
        split_data = dff_reid[dff_reid["split"] == split_name]
        split_stats[split_name] = {
            "images": len(split_data),
            "patients": split_data["subject_id"].nunique()
        }

    return orig_count, per_study_count, reid_count, split_stats

print("\n=== 各视角条件统计 ===")
for view_name, allowed in view_sets.items():
    orig, per_study, reid, split_stats = filter_and_count(df, allowed_views=allowed)
    print(f"\n[{view_name}]")
    print(f"  原始数量: {orig}")
    print(f"  每 study 取1张后: {per_study}")
    print(f"  ≥2 study/患者 后: {reid}")
    for split_name, stats in split_stats.items():
        print(f"    {split_name}: images={stats['images']}, patients={stats['patients']}")
