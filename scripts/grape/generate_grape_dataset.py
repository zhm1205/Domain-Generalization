#!/usr/bin/env python3
"""
GRAPE data preprocessor — aligns with your MIMIC‑CXR pipeline (single master CSV + splits).

• Identity: patient‑level (subject_id), consistent with npj Digital Medicine practice
• Cross‑study: filter by different study_id (Visit Number) during evaluation (not enforced here)
• Input images: ROI only (RGB) → resized PNG
• Masks: rasterized from JSON polygons (0=bg, 1=OD, 2=OC)
• Single master CSV: grape_all.csv  (+ train/val/test CSVs)

Config (YAML) example:
---------------------------------
raw_data_folder: /path/to/GRAPE_raw
excel_path: /path/to/VF and clinical information.xlsx
output_folder: /path/to/grape_processed

paths:
  roi_dir: "ROI images"
  ann_dir: "Annotated Images"      # for optional consistency check
  json_dir: "json"
  out_roi_dir: "roi"               # inside output_folder
  out_masks_dir: "masks"           # inside output_folder

resize:
  roi: [256, 256]

image_processing:
  background_color: [0, 0, 0]
  compression_level: 3

reid:
  min_studies_per_subject: 2        # keep subjects with ≥N Visit Numbers

splits:
  train: 0.75
  val: 0.10
  test: 0.15
  seed: 42

excel:
  baseline_sheet: "Baseline"
  followup_sheet: "Follow-up"
  # If VF columns cannot be auto-detected, list them explicitly:
  # vf_columns: ["VF1", "VF2", ..., "VF61"]

checks:
  overlay_consistency: true         # compare (Annotated-ROI) with JSON→mask
  overlay_iou_threshold: 0.90
  report_csv: "mask_consistency_report.csv"
---------------------------------

Usage:
  python grape_prepare.py --config /path/to/grape.yaml
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm

# ---------------
# Utils
# ---------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _to_tuple_wh(x) -> Tuple[int, int]:
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return int(x[0]), int(x[1])
    raise ValueError("resize.roi must be [W, H]")


def _read_yaml(path: Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _list_files(root: Path, subdir: str, exts=(".png", ".jpg", ".jpeg", ".bmp")) -> Dict[str, Path]:
    base = (root / subdir).expanduser()
    if not base.exists():
        return {}
    out = {}
    for p in base.rglob("*"):
        if p.suffix.lower() in exts:
            out[p.stem] = p
    return out


def _list_json(root: Path, subdir: str) -> Dict[str, Path]:
    base = (root / subdir).expanduser()
    if not base.exists():
        return {}
    out = {}
    for p in base.rglob("*.json"):
        out[p.stem] = p
    return out


def _resize_pad_rgb(img: Image.Image, target_wh: Tuple[int, int], bg=(0, 0, 0)) -> Tuple[Image.Image, float, int, int, int, int]:
    tw, th = target_wh
    w, h = img.size
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img2 = img.resize((nw, nh), resample=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (tw, th), tuple(map(int, bg)))
    x0, y0 = (tw - nw) // 2, (th - nh) // 2
    canvas.paste(img2, (x0, y0))
    return canvas, float(scale), int(x0), int(y0), int(w), int(h)


def _polygon_mask(size_wh: Tuple[int, int], polygons: List[List[Tuple[float, float]]], value: int) -> Image.Image:
    mask = Image.new("L", (size_wh[0], size_wh[1]), 0)
    drw = ImageDraw.Draw(mask)
    for poly in polygons:
        if len(poly) >= 3:
            drw.polygon([tuple(map(float, pt)) for pt in poly], outline=value, fill=value)
    return mask


# ---------------
# Excel parsing
# ---------------

VF_COUNT = 61


# NOTE: we now assume exact column names from the user; no fuzzy matching needed.
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df  # keep columns as-is


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def _auto_detect_vf_columns(df: pd.DataFrame, explicit: Optional[List[str]] = None) -> List[str]:
    if explicit:
        return [c for c in explicit if c in df.columns]
    # Heuristics: match like vf*, td*, or take last 61 numeric columns
    vf_like = [c for c in df.columns if re.match(r"^(vf|td)[\s_\-]*\d+", c)]
    if len(vf_like) >= VF_COUNT:
        return vf_like[:VF_COUNT]
    # fallback: take last 61 columns
    if df.shape[1] >= VF_COUNT:
        return list(df.columns[-VF_COUNT:])
    raise RuntimeError("Cannot detect 61 VF columns; set excel.vf_columns in config.")


def load_excel_build_long_table(cfg: dict) -> pd.DataFrame:
    """
    读取 GRAPE 的 Baseline / Follow-up 两个 sheet，并将多级表头扁平化为单级列名后再取列。
    扁平规则（与截图匹配）：
      - ('Progression Status','PLR2'|'PLR3'|'MD') -> 'PLR2' / 'PLR3' / 'MD'
      - ('OCT RNFL thickness','Mean'|'S'|'N'|'I'|'T') -> 'OCT RNFL thickness Mean' 等
      - ('VF', 0..60) -> 'VF_00'..'VF_60'
      - 其它：子列为空 -> 顶层名；否则 'Top Sub'
    """
    import re
    excel_path = Path(cfg["excel_path"]).expanduser()
    bs_name = cfg.get("excel", {}).get("baseline_sheet", "Baseline")
    fu_name = cfg.get("excel", {}).get("followup_sheet", "Follow-up")
    explicit_vf = cfg.get("excel", {}).get("vf_columns")

    # ---------- helpers ----------
    def _read_sheet(name: str) -> pd.DataFrame:
        """优先尝试多级表头(header=[0,1])，若不成立则退回单级(header=0)。"""
        try:
            df = pd.read_excel(excel_path, sheet_name=name, header=[0, 1])
            if not isinstance(df.columns, pd.MultiIndex):
                return pd.read_excel(excel_path, sheet_name=name)
            # 如果二级全空也退回单级；但通常本数据集中因 VF/Progression 存在二级，不会触发
            if all((c[1] is None) or (str(c[1]).strip() == "") for c in df.columns):
                return pd.read_excel(excel_path, sheet_name=name)
            return df
        except Exception:
            return pd.read_excel(excel_path, sheet_name=name)

    def _norm_space(s: str) -> str:
        """统一空白字符（含 NBSP）为普通空格，并压缩多空格。"""
        if s is None:
            return ""
        s = str(s).replace("\xa0", " ")
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    def _sub_is_empty(sub: str) -> bool:
        """判断二级列是否“空”（NaN/空串/'Unnamed:*'）。"""
        if sub is None:
            return True
        t = _norm_space(sub).lower()
        return (t == "" or t == "nan" or t.startswith("unnamed:"))

    def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
        """将 MultiIndex 列扁平为单级列名；若本来就是单级，原样返回。"""
        if not isinstance(df.columns, pd.MultiIndex):
            # 同样做一次空白规范
            df2 = df.copy()
            df2.columns = [_norm_space(c) for c in df2.columns]
            return df2

        new_cols = []
        for top, sub in df.columns:
            top_s = _norm_space(top)
            sub_s_raw = "" if _sub_is_empty(sub) else _norm_space(sub)

            # 特例 1：Progression Status 组 -> 直接用子列名（PLR2/PLR3/MD）
            if top_s.lower() == "progression status" and sub_s_raw:
                name = sub_s_raw

            # 特例 2：OCT RNFL thickness 组 -> "OCT RNFL thickness {sub}"
            elif top_s.lower().startswith("oct rnfl thickness") and sub_s_raw:
                name = f"OCT RNFL thickness {sub_s_raw}"

            # 特例 3：VF 组 -> "VF_{xx}"
            elif top_s.upper() == "VF":
                ss = sub_s_raw
                try:
                    idx = int(float(ss))
                    name = f"VF_{idx:02d}"
                except Exception:
                    name = f"VF_{ss if ss!='' else 'NA'}"

            # 子列为空 -> 顶层名
            elif sub_s_raw == "":
                name = top_s

            # 其它情况 -> "Top Sub"
            else:
                name = f"{top_s} {sub_s_raw}".strip()

            new_cols.append(name)

        out = df.copy()
        out.columns = new_cols
        return out

    def _num_series(df: pd.DataFrame, col: str, *, fill=None, as_int=False):
        """将指定列转为数值 Series；缺失时返回与 df 等长的填充值 Series。"""
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
        else:
            s = pd.Series(np.nan, index=df.index, dtype=float)
        if fill is not None:
            s = s.fillna(fill)
        if as_int:
            s = s.astype(int)
        return s

    def _series(df: pd.DataFrame, col: str):
        """取列为 Series；缺失时返回 NaN Series（与 df 行数一致）。"""
        return df[col] if col in df.columns else pd.Series(np.nan, index=df.index)

    def _pick_vf_cols_flat(df: pd.DataFrame) -> list[str]:
        """在已扁平的列名中选择 61 个 VF 列。优先匹配 'VF_XX'；否则退回最后 61 列；若配置给了列名则严格按配置。"""
        if explicit_vf:
            missing = [c for c in explicit_vf if c not in df.columns]
            if missing:
                raise KeyError(f"VF columns missing: {missing}")
            return explicit_vf

        vf_cols = [c for c in df.columns if c.startswith("VF_")]
        if len(vf_cols) >= 61:
            def _vf_key(x: str):
                m = re.search(r"(\d+)$", x)
                return int(m.group(1)) if m else 10**9
            vf_cols = sorted(vf_cols, key=_vf_key)[:61]
            return vf_cols

        # 退化：最后 61 列
        if df.shape[1] < 61:
            raise RuntimeError("Sheet has fewer than 61 columns; cannot auto-pick 61 VF columns")
        return list(df.columns[-61:])

    # ---------- read & flatten ----------
    base_df_raw = _read_sheet(bs_name)
    foll_df_raw = _read_sheet(fu_name)
    base_df = _flatten_columns(base_df_raw)
    foll_df = _flatten_columns(foll_df_raw)

    # ---------- sanity of required columns ----------
    # 这里用规范化过的“单级列名”，避免 NBSP/Unnamed 干扰
    for need in ["Subject Number", "Laterality", "Corresponding CFP"]:
        if need not in base_df.columns:
            raise KeyError(f"Baseline 缺少必需列：{need}")
    for need in ["Subject Number", "Laterality", "Visit Number", "Corresponding CFP"]:
        if need not in foll_df.columns:
            raise KeyError(f"Follow-up 缺少必需列：{need}")

    # ---------- pick VF columns (flattened) ----------
    vf_b = _pick_vf_cols_flat(base_df)
    vf_f = _pick_vf_cols_flat(foll_df)

    # ---------- Baseline ----------
    base_use = pd.DataFrame({
        "subject_id": base_df["Subject Number"].astype(str),
        "laterality": base_df["Laterality"].astype(str),
        "study_id": pd.Series(0, index=base_df.index, dtype=int),       # baseline 访问号=0
        "time_years": pd.Series(0.0, index=base_df.index, dtype=float),
        "cfp_name": base_df["Corresponding CFP"].astype(str),
        # 兼容 Resolusion/Resolution 两种拼法
        "device": _series(base_df, "Acquisition Device"),
        "resolution": _series(base_df, "Resolusion") if "Resolusion" in base_df.columns else _series(base_df, "Resolution"),
    })

    # Progression Status（子列）
    base_use["plr2"]    = _num_series(base_df, "PLR2", fill=0, as_int=True)
    base_use["plr3"]    = _num_series(base_df, "PLR3", fill=0, as_int=True)
    base_use["md_prog"] = _num_series(base_df, "MD",   fill=0, as_int=True)
    # 主列（若存在就保留；有的表没有）
    base_use["progression_status"] = _series(base_df, "Progression Status")

    # Category of Glaucoma
    base_use["category_of_glaucoma"] = _series(base_df, "Category of Glaucoma")

    # OCT RNFL thickness
    for sub in ["Mean", "S", "N", "I", "T"]:
        col = f"OCT RNFL thickness {sub}"
        if col in base_df.columns:
            base_use[col] = pd.to_numeric(base_df[col], errors="coerce")

    # VF（61 列）
    for i, c in enumerate(vf_b, start=1):
        base_use[f"vf_{i:02d}"] = pd.to_numeric(base_df[c], errors="coerce").fillna(-1)

    # ---------- Follow-up ----------
    fu_use = pd.DataFrame({
        "subject_id": foll_df["Subject Number"].astype(str),
        "laterality": foll_df["Laterality"].astype(str),
        "study_id": pd.to_numeric(foll_df["Visit Number"], errors="coerce").fillna(0).astype(int),
        "time_years": pd.to_numeric(_series(foll_df, "Interval Years"), errors="coerce"),
        "cfp_name": foll_df["Corresponding CFP"].astype(str),
        "device": _series(foll_df, "Acquisition Device"),
        "resolution": _series(foll_df, "Resolusion") if "Resolusion" in foll_df.columns else _series(foll_df, "Resolution"),
    })
    # 通常仅 Baseline 提供，Follow-up 用 NaN 占位，保证字段对齐
    fu_use["progression_status"]   = pd.Series(np.nan, index=foll_df.index)
    fu_use["category_of_glaucoma"] = pd.Series(np.nan, index=foll_df.index)

    # VF（61 列）
    for i, c in enumerate(vf_f, start=1):
        fu_use[f"vf_{i:02d}"] = pd.to_numeric(foll_df[c], errors="coerce").fillna(-1)

    # ---------- concat & clean ----------
    df = pd.concat([base_use, fu_use], ignore_index=True)
    df["laterality"] = df["laterality"].str.upper().str.strip()
    df = df.drop_duplicates(subset=["subject_id", "laterality", "study_id"]).reset_index(drop=True)
    return df

# ---------------
# Path mapping & preprocessing
# ---------------


def build_file_indices(raw_root: Path, cfg_paths: dict) -> Tuple[Dict[str, Path], Dict[str, Path], Dict[str, Path]]:
    # Index by stem for flexible lookup
    roi_idx = _list_files(raw_root, cfg_paths.get("roi_dir", "ROI images"))
    ann_idx = _list_files(raw_root, cfg_paths.get("ann_dir", "Annotated Images"))
    js_idx = _list_json(raw_root, cfg_paths.get("json_dir", "json"))
    return roi_idx, ann_idx, js_idx


def attach_paths(df: pd.DataFrame, roi_idx: Dict[str, Path], ann_idx: Dict[str, Path], js_idx: Dict[str, Path]) -> pd.DataFrame:
    """Attach ROI/Annotated/JSON paths using the exact 'Corresponding CFP' filename (stem).
    We try:
      stem = Path(cfp_name).stem
      ROI/Annotated/JSON lookup by that stem; fallback heuristics (_roi/_ROI suffix etc.).
    """
    basenames, roi_paths, ann_paths, json_paths = [], [], [], []
    for r in df[["subject_id", "laterality", "study_id", "cfp_name"]].itertuples(index=False):
        raw_name = str(r.cfp_name)
        stem = Path(raw_name).stem
        cand_stems = [stem, stem.replace("CFP", "ROI"), f"{stem}_ROI", f"ROI_{stem}"]
        rp = next((str(roi_idx[s]) for s in cand_stems if s in roi_idx), "")
        ap = next((str(ann_idx[s]) for s in cand_stems if s in ann_idx), "")
        jp = next((str(js_idx[s]) for s in cand_stems if s in js_idx), "")
        basenames.append(stem)
        roi_paths.append(rp)
        ann_paths.append(ap)
        json_paths.append(jp)

    df = df.copy()
    df["basename"] = basenames
    df["roi_src_path"] = roi_paths
    df["ann_src_path"] = ann_paths
    df["json_src_path"] = json_paths

    # Simple diagnostics
    miss = (df["roi_src_path"] == "").sum()
    if miss:
        print(f"[WARN] ROI not found for {miss} rows. Check 'Corresponding CFP' naming vs ROI filenames.")
    return df

def preprocess_cfp_and_save(df: pd.DataFrame, out_cfp_dir: Path, cfp_size: Tuple[int,int], bg_rgb, compress_level: int) -> pd.DataFrame:
    _ensure_dir(out_cfp_dir)
    out_paths, dicom_ids, widths, heights, scales, x0s, y0s = [], [], [], [], [], [], []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Preprocess CFP"):
        src = row.cfp_src_path
        if not src or not Path(src).exists():
            out_paths.append(""); dicom_ids.append("")
            widths.append(np.nan); heights.append(np.nan)
            scales.append(np.nan); x0s.append(np.nan); y0s.append(np.nan)
            continue
        try:
            img = Image.open(src).convert("RGB")
            out_img, scale, x0, y0, ow, oh = _resize_pad_rgb(img, cfp_size, bg=bg_rgb)
            stem = Path(src).stem
            out_path = out_cfp_dir / f"{stem}.png"
            out_img.save(out_path, format="PNG", compress_level=int(compress_level))
            out_paths.append(str(out_path))
            dicom_ids.append(stem)
            widths.append(ow); heights.append(oh)
            scales.append(scale); x0s.append(x0); y0s.append(y0)
        except Exception as e:
            print(f"[WARN] CFP preprocess failed for {src}: {e}")
            out_paths.append(""); dicom_ids.append("")
            widths.append(np.nan); heights.append(np.nan)
            scales.append(np.nan); x0s.append(np.nan); y0s.append(np.nan)

    df = df.copy()
    df["image_path"] = out_paths     # ← 关键：image_path 改为cfp图
    df["dicom_id"]   = dicom_ids
    df["orig_width"] = widths
    df["orig_height"] = heights
    df["full_scale"] = scales
    df["full_x0"] = x0s
    df["full_y0"] = y0s
    return df

def drop_reused_cfp_per_subject(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df2 = df.sort_values(by=["subject_id", "study_id"]).drop_duplicates(
        subset=["subject_id", "dicom_id"], keep="first"
    ).reset_index(drop=True)
    print(f"Dedup: drop reused CFP per subject: {before} -> {len(df2)} rows "
          f"(removed {before - len(df2)})")
    return df2

def preprocess_roi_and_save(df: pd.DataFrame, out_roi_dir: Path, roi_size: Tuple[int, int], bg_rgb, compress_level: int) -> pd.DataFrame:
    _ensure_dir(out_roi_dir)
    image_paths, dicom_ids, widths, heights = [], [], [], []
    scales, x0s, y0s, ow_list, oh_list = [], [], [], [], []

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Preprocess ROI"):
        src = row.roi_src_path
        if not src or not Path(src).exists():
            image_paths.append("")
            dicom_ids.append("")
            widths.append(np.nan); heights.append(np.nan)
            scales.append(np.nan); x0s.append(np.nan); y0s.append(np.nan)
            ow_list.append(np.nan); oh_list.append(np.nan)
            continue
        try:
            img = Image.open(src).convert("RGB")
            out, scale, x0, y0, ow, oh = _resize_pad_rgb(img, roi_size, bg=bg_rgb)
            stem = Path(src).stem
            out_path = out_roi_dir / f"{stem}.png"
            out.save(out_path, format="PNG", compress_level=int(compress_level))
            image_paths.append(str(out_path))
            dicom_ids.append(stem)
            widths.append(ow); heights.append(oh)
            scales.append(scale); x0s.append(x0); y0s.append(y0)
            ow_list.append(ow); oh_list.append(oh)
        except Exception as e:
            print(f"[WARN] ROI preprocess failed for {src}: {e}")
            image_paths.append("")
            dicom_ids.append("")
            widths.append(np.nan); heights.append(np.nan)
            scales.append(np.nan); x0s.append(np.nan); y0s.append(np.nan)
            ow_list.append(np.nan); oh_list.append(np.nan)

    df = df.copy()
    df["image_path"] = image_paths
    df["dicom_id"] = dicom_ids
    df["orig_width"] = widths
    df["orig_height"] = heights
    df["roi_scale"] = scales
    df["roi_x0"] = x0s
    df["roi_y0"] = y0s
    return df


# ---------------
# JSON → mask (0 bg, 1 OD, 2 OC)
# ---------------

# === Visualization helpers (extracted for reuse during training) ===
# You can import these from this file or copy-paste into your training code.
DEFAULT_COLORS = {1: (0, 255, 0), 2: (255, 0, 0)}  # OD=green, OC=red

def mask_to_rgba(mask: np.ndarray, colors: Dict[int, Tuple[int, int, int]] = DEFAULT_COLORS, alpha: float = 0.45) -> Image.Image:
    """Convert a label mask (H,W uint8, 0/1/2) to an RGBA overlay image (filled regions).
    alpha in [0,1]. Background (0) is fully transparent.
    """
    if isinstance(mask, Image.Image):
        mask = np.array(mask.convert("L"), dtype=np.uint8)
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for cls_id, (r, g, b) in colors.items():
        m = (mask == cls_id)
        rgba[m] = (r, g, b, int(round(alpha * 255)))
    return Image.fromarray(rgba, mode="RGBA")

def overlay_on_roi(roi_img: Image.Image, mask: np.ndarray | Image.Image, colors: Dict[int, Tuple[int, int, int]] = DEFAULT_COLORS, alpha: float = 0.45) -> Image.Image:
    """Return ROI with a filled, semi-transparent overlay. Output is RGB PIL.Image."""
    roi = roi_img.convert("RGBA")
    ovl = mask_to_rgba(mask, colors=colors, alpha=alpha)
    out = Image.alpha_composite(roi, ovl).convert("RGB")
    return out

def save_overlay(roi_path: str | Path, mask_path: str | Path, out_path: str | Path, colors: Dict[int, Tuple[int, int, int]] = DEFAULT_COLORS, alpha: float = 0.45) -> None:
    """Convenience: save ROI+mask overlay PNG for quick visual QC."""
    if not roi_path or not mask_path:
        return
    rp, mp = Path(roi_path), Path(mask_path)
    if not (rp.exists() and mp.exists()):
        return
    roi = Image.open(rp).convert("RGB")
    mask = Image.open(mp).convert("L")
    out = overlay_on_roi(roi, mask, colors=colors, alpha=alpha)
    out.save(out_path)

# === End of visualization helpers ===

# ---------------
# JSON → mask (0 bg, 1 OD, 2 OC)
# ---------------

OD_KEYS = ["od", "disc", "opticdisc", "optic_disc", "optic disc"]
OC_KEYS = ["oc", "cup", "opticcup", "optic_cup", "optic cup"]


def _extract_polygons(json_obj: dict) -> Tuple[List[List[Tuple[float, float]]], List[List[Tuple[float, float]]]]:
    od_polys: List[List[Tuple[float, float]]] = []
    oc_polys: List[List[Tuple[float, float]]] = []

    # Case 1: {"od": [[x,y],...], "oc": [[x,y],...]}
    lower = {k.lower(): v for k, v in json_obj.items() if isinstance(k, str)}
    for k, v in lower.items():
        if any(key in k for key in OD_KEYS) and isinstance(v, (list, tuple)):
            od_polys.append([(float(a), float(b)) for a, b in v])
        if any(key in k for key in OC_KEYS) and isinstance(v, (list, tuple)):
            oc_polys.append([(float(a), float(b)) for a, b in v])

    # Case 2: shapes: [{label: 'OD', points: [[x,y],...]}]
    if not od_polys and not oc_polys and "shapes" in json_obj:
        for shp in json_obj["shapes"]:
            label = str(shp.get("label", "")).lower()
            pts = shp.get("points", [])
            if any(k in label for k in OD_KEYS):
                od_polys.append([(float(a), float(b)) for a, b in pts])
            if any(k in label for k in OC_KEYS):
                oc_polys.append([(float(a), float(b)) for a, b in pts])

    return od_polys, oc_polys


def _transform_polygons(polys: List[List[Tuple[float, float]]], scale: float, x0: float, y0: float) -> List[List[Tuple[float, float]]]:
    out = []
    for poly in polys:
        out.append([(float(x) * scale + x0, float(y) * scale + y0) for x, y in poly])
    return out


def rasterize_masks(df: pd.DataFrame, out_masks_dir: Path, roi_size: Tuple[int, int]) -> pd.DataFrame:
    _ensure_dir(out_masks_dir)
    mask_paths = []

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Rasterize masks"):
        src_json = row.json_src_path
        out_path = ""
        if src_json and Path(src_json).exists() and row.image_path:
            try:
                with open(src_json, "r", encoding="utf-8") as f:
                    js = json.load(f)
                od_polys, oc_polys = _extract_polygons(js)
                # Map JSON polygons from original ROI coords to resized canvas coords
                scale = float(row.roi_scale) if not pd.isna(row.roi_scale) else 1.0
                x0 = float(row.roi_x0) if not pd.isna(row.roi_x0) else 0.0
                y0 = float(row.roi_y0) if not pd.isna(row.roi_y0) else 0.0
                od_t = _transform_polygons(od_polys, scale, x0, y0)
                oc_t = _transform_polygons(oc_polys, scale, x0, y0)

                od = _polygon_mask(roi_size, od_t, value=1)
                oc = _polygon_mask(roi_size, oc_t, value=2)
                # Compose to single label map
                np_mask = np.array(od, dtype=np.uint8)
                np_oc = np.array(oc, dtype=np.uint8)
                np_mask[np_oc == 2] = 2
                out_img = Image.fromarray(np_mask, mode="L")
                stem = Path(row.image_path).stem
                out_path = str(out_masks_dir / f"{stem}.png")
                out_img.save(out_path)
            except Exception as e:
                print(f"[WARN] Mask rasterization failed for {src_json}: {e}")
                out_path = ""
        mask_paths.append(out_path)

    df = df.copy()
    df["mask_path"] = mask_paths
    return df


# ---------------
# Optional: Annotated vs JSON→mask consistency
# ---------------


def _extract_overlay_regions(roi_img: Image.Image, ann_img: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    """Very simple color-threshold approach to extract green (OD) and red (OC) overlays.
    Returns two uint8 binary arrays (H,W) for green-ish and red-ish regions.
    """
    A = np.asarray(ann_img.convert("RGB"), dtype=np.uint8)
    R = np.asarray(roi_img.convert("RGB"), dtype=np.uint8)
    D = A.astype(int) - R.astype(int)
    # Enhancing saturated colored strokes: look where annotated deviates strongly.
    diff = np.abs(D).sum(axis=2)
    strong = diff > 40  # heuristic

    # Color masks in annotated image (favor saturated hues)
    r, g, b = A[..., 0], A[..., 1], A[..., 2]
    is_red = (r > 160) & (r > g + 30) & (r > b + 30)
    is_green = (g > 160) & (g > r + 30) & (g > b + 30)

    red_mask = (strong & is_red).astype(np.uint8)
    green_mask = (strong & is_green).astype(np.uint8)

    # Slight dilation to connect strokes
    def dilate(bin_img: np.ndarray, k: int = 3, it: int = 2) -> np.ndarray:
        im = Image.fromarray((bin_img * 255).astype(np.uint8), mode="L")
        for _ in range(it):
            im = im.filter(ImageFilter.MaxFilter(k))
        return (np.array(im) > 0).astype(np.uint8)

    red_mask = dilate(red_mask)
    green_mask = dilate(green_mask)

    return green_mask, red_mask  # (OD approx, OC approx)


def _binary_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = (a > 0).astype(np.uint8)
    b = (b > 0).astype(np.uint8)
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter) / float(union) if union > 0 else 1.0


def consistency_check(df: pd.DataFrame, report_csv: Path, iou_thr: float = 0.9) -> None:
    records = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Overlay consistency"):
        if not row.ann_src_path or not row.image_path or not row.mask_path:
            continue
        ap, rp, mp = Path(row.ann_src_path), Path(row.image_path), Path(row.mask_path)
        if not (ap.exists() and rp.exists() and mp.exists()):
            continue
        try:
            ann = Image.open(ap).convert("RGB")
            roi = Image.open(rp).convert("RGB")
            mask = np.array(Image.open(mp).convert("L"), dtype=np.uint8)
            od_json = (mask == 1).astype(np.uint8)
            oc_json = (mask == 2).astype(np.uint8)
            od_ovl, oc_ovl = _extract_overlay_regions(roi, ann)
            iou_od = _binary_iou(od_json, od_ovl)
            iou_oc = _binary_iou(oc_json, oc_ovl)
            note = "OK"
            if (iou_od < iou_thr) or (iou_oc < iou_thr):
                note = "LOW_IOU"
            records.append({
                "dicom_id": Path(row.image_path).stem,
                "iou_od": round(iou_od, 4),
                "iou_oc": round(iou_oc, 4),
                "note": note,
            })
        except Exception as e:
            records.append({
                "dicom_id": Path(row.image_path).stem if row.image_path else "",
                "iou_od": np.nan,
                "iou_oc": np.nan,
                "note": f"ERROR: {e}",
            })
    if records:
        pd.DataFrame(records).to_csv(report_csv, index=False)
        low = [r for r in records if r.get("note") == "LOW_IOU"]
        print(f"Consistency report saved: {report_csv} | total={len(records)} | low_iou={len(low)}")
    else:
        print("No consistency records written (no pairs found).")


# ---------------
# Filtering & Splits (patient-level)
# ---------------


def drop_rows_without_images(df: pd.DataFrame, require_mask: bool = False) -> pd.DataFrame:
    """Drop rows that lack ROI image. If require_mask=True, also drop rows without mask.
    We check path non-empty AND file exists.
    """
    def _exists(p):
        try:
            return bool(p) and Path(p).exists()
        except Exception:
            return False
    has_img = df.get("image_path", "").astype(str).apply(_exists)
    if require_mask:
        has_msk = df.get("mask_path", "").astype(str).apply(_exists)
        keep = has_img & has_msk
    else:
        keep = has_img
    before = len(df)
    out = df[keep].reset_index(drop=True)
    print(f"Filter: drop rows without image{'/mask' if require_mask else ''}: {before} -> {len(out)}")
    return out


def filter_min_studies(df: pd.DataFrame, min_studies_per_subject: int = 2) -> pd.DataFrame:
    """Keep only subjects that have ≥ N DISTINCT study_id WITH images.
    This ensures cross-study positives exist for ReID.
    """
    # Count distinct study_id among rows that survived image filtering
    grp = df.groupby("subject_id")["study_id"].nunique()
    keep_subjects = grp[grp >= int(min_studies_per_subject)].index
    out = df[df["subject_id"].isin(keep_subjects)].reset_index(drop=True)
    print(
        f"Filter: subjects with ≥{min_studies_per_subject} image-bearing studies: "
        f"{df['subject_id'].nunique()} -> {out['subject_id'].nunique()} subjects"
    )
    return out


def assign_splits_by_subject(df: pd.DataFrame, ratios: dict, seed: int = 42) -> pd.DataFrame:
    train_r = float(ratios.get("train", 0.75))
    val_r = float(ratios.get("val", 0.10))
    test_r = float(ratios.get("test", 0.15))
    tot = train_r + val_r + test_r
    if abs(tot - 1.0) > 1e-6:
        train_r, val_r = train_r / tot, val_r / tot
        test_r = 1.0 - train_r - val_r
        print(f"[WARN] split ratios normalized to {train_r:.2f}/{val_r:.2f}/{test_r:.2f}")

    subjects = sorted(df["subject_id"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n = len(subjects)
    n_train = int(round(n * train_r))
    n_val = int(round(n * val_r))
    train_subj = set(subjects[:n_train])
    val_subj = set(subjects[n_train:n_train + n_val])
    test_subj = set(subjects[n_train + n_val:])

    def _assign(s):
        if s in train_subj:
            return "train"
        if s in val_subj:
            return "val"
        return "test"

    df = df.copy()
    df["split"] = df["subject_id"].map(_assign)

    for sp in ["train", "val", "test"]:
        sub = df[df["split"] == sp]
        print(f"{sp.upper():8s}: images={len(sub):6d} | subjects={sub['subject_id'].nunique():4d}")

    return df


# ---------------
# Main
# ---------------

def export_overlays(df: pd.DataFrame, out_dir: Path, alpha: float = 0.5) -> None:
    _ensure_dir(out_dir)
    cnt = 0
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Export overlays"):
        if not row.image_path or not row.mask_path:
            continue
        out_path = out_dir / f"{Path(row.image_path).stem}.png"
        try:
            save_overlay(row.image_path, row.mask_path, out_path, alpha=alpha)
            cnt += 1
        except Exception as e:
            print(f"[WARN] overlay failed for {row.image_path}: {e}")
    print(f"Overlays saved: {cnt}")

def save_splits_and_artifacts(
    df: pd.DataFrame,
    out_root: Path,
    *,
    sort_for_readability: bool = True,
) -> None:
    """
    将最终表保存为 all/train/val/test 三份 CSV，并做最小必要的列补齐。
    - 统一把 split 的 'validate' 当成 'val' 输出 val.csv
    - 导出前按 subject/laterality/时间/visit 排序，便于人工排查
    - study_id 输出为全局唯一字符串 'subject_study'
    """

    out_root.mkdir(parents=True, exist_ok=True)

    # —— 排序（便于排查）——
    df2 = df.copy()
    if sort_for_readability:
        # 尽量用数值时间、数值 study 排序
        sid_num = pd.to_numeric(df2.get("study_id", 0), errors="coerce").fillna(0).astype(int)
        ty = pd.to_numeric(df2.get("time_years", np.nan), errors="coerce")
        df2["_sid_num"] = sid_num
        df2["_t_years"] = ty.fillna(1e9)
        df2 = df2.sort_values(
            by=["subject_id", "laterality", "_sid_num", "_t_years", "dicom_id"],
            ascending=[True, True, True, True, True]
        ).drop(columns=["_sid_num", "_t_years"], errors="ignore")

    # —— study_id 输出为全局唯一字符串 —— 
    # （注意：这里只是导出时格式化；df2 内部字段保持同名便于下游读取）
    sid_str = pd.to_numeric(df2.get("study_id", 0), errors="coerce").fillna(0).astype(int).astype(str)
    df2["study_id"] = df2["subject_id"].astype(str) + "_" + sid_str

    # —— 需要导出的列 ——（缺了就补 NaN）
    vf_cols = [f"vf_{i:02d}" for i in range(1, 62)]
    out_cols = [
        "subject_id", "study_id", "dicom_id", "image_path", "mask_path",
        "laterality", "time_years", "split",
        "width", "height",           # 若前面没赋值，会在下面被补成 NaN
        "plr2", "plr3", "md_prog", "category_of_glaucoma",
    ] + vf_cols
    for c in out_cols:
        if c not in df2.columns:
            df2[c] = np.nan

    # —— all.csv —— 
    all_csv = out_root / "grape_all.csv"
    df2[out_cols].to_csv(all_csv, index=False)
    print(f"Saved: {all_csv} (rows={len(df2)})")

    # —— train / val / test ——（统一把 validate 视为 val）
    split_alias = df2["split"].astype(str).str.lower().replace({"validate": "val"})
    df2 = df2.assign(split_alias=split_alias)

    for sp in ["train", "val", "test"]:
        sub = df2[df2["split_alias"] == sp][out_cols]
        p = out_root / f"{sp}.csv"
        sub.to_csv(p, index=False)
        print(f"Saved: {p} (rows={len(sub)})")

def drop_duplicate_images(
    df: pd.DataFrame,
    prefer_mask: bool = True,
    prefer_baseline: bool = True,
) -> pd.DataFrame:
    """
    按“图像唯一性”去重，默认以 dicom_id（ROI 文件名 stem）为主键；
    如果没有 dicom_id 列，则退回 image_path。

    保留策略（按优先级排序）：
      1) 有 mask 的优先（prefer_mask=True）
      2) 基线 study_id==0 优先（prefer_baseline=True）
      3) 时间更早的优先（time_years 更小）
      4) study_id 更小优先
    """

    if "dicom_id" in df.columns and df["dicom_id"].notna().any():
        key = "dicom_id"
    else:
        key = "image_path"

    tmp = df.copy()

    # 排序辅助列
    has_mask = tmp.get("mask_path", pd.Series("", index=tmp.index)).astype(str).str.len() > 0
    tmp["_has_mask_rank"] = (~has_mask).astype(int) if prefer_mask else 0  # 有mask排前面
    sid_num = pd.to_numeric(tmp.get("study_id", 0), errors="coerce").fillna(0).astype(int)
    tmp["_is_baseline_rank"] = (sid_num != 0).astype(int) if prefer_baseline else 0  # baseline排前面
    ty = pd.to_numeric(tmp.get("time_years", np.nan), errors="coerce")
    tmp["_time_years_rank"] = ty.fillna(1e9)  # 时间越早越靠前
    tmp["_sid_rank"] = sid_num

    before = len(tmp)
    # 关键：让“更优先”的排在前面，然后 drop_duplicates(keep='first')
    tmp = tmp.sort_values(
        by=[key, "_has_mask_rank", "_is_baseline_rank", "_time_years_rank", "_sid_rank"],
        ascending=[True, True, True, True, True]
    )
    out = tmp.drop_duplicates(subset=[key], keep="first").drop(
        columns=["_has_mask_rank", "_is_baseline_rank", "_time_years_rank", "_sid_rank"], errors="ignore"
    )
    removed = before - len(out)

    # 诊断输出
    # 统计：同一个 dicom_id 是否来自多个 study_id
    if key == "dicom_id":
        multi_st = out.groupby("dicom_id")["study_id"].nunique()
        reused = int((multi_st > 1).sum())
        print(f"[Filter] dropped duplicate images by '{key}': {before} -> {len(out)} (removed {removed}). "
              f"Unique dicom_ids reused by multiple studies (kept one each): {reused}")
    else:
        print(f"[Filter] dropped duplicate images by '{key}': {before} -> {len(out)} (removed {removed}).")

    return out

    # def main():
    #     ap = argparse.ArgumentParser(description="GRAPE data preprocessor (patient-level, ROI only)")
    #     ap.add_argument("--config", type=str, default="../configs/grape.yaml", help="YAML config path")
    #     args = ap.parse_args()

    #     cfg = _read_yaml(Path(args.config))
    #     raw_root = Path(cfg["raw_data_folder"]).expanduser()
    #     out_root = Path(cfg["output_folder"]).expanduser()

    #     cfg_paths = cfg.get("paths", {})
    #     out_roi_dir = out_root / cfg_paths.get("out_roi_dir", "roi")
    #     out_masks_dir = out_root / cfg_paths.get("out_masks_dir", "masks")
    #     out_cfp_dir = out_root / cfg_paths.get("out_cfp_dir", "cfp")
    #     roi_w, roi_h = _to_tuple_wh(cfg.get("resize", {}).get("roi", [256, 256]))
        
    #     bg_rgb = tuple(cfg.get("image_processing", {}).get("background_color", [0, 0, 0]))
    #     comp = int(cfg.get("image_processing", {}).get("compression_level", 3))

    #     # 1) Read Excel → long table
    #     print("[1/7] Loading Excel & building table...")
    #     df = load_excel_build_long_table(cfg)
    #     print(f"   Loaded rows: {len(df)} | subjects={df['subject_id'].nunique()} | studies(unique per subj mean)={df.groupby('subject_id')['study_id'].nunique().mean():.2f}")

    #     # 2) Index raw files & attach paths
    #     print("[2/7] Indexing ROI/Annotated/JSON & attaching paths...")
    #     roi_idx, ann_idx, js_idx, cfp_idx = build_file_indices(raw_root, cfg_paths)
    #     df = attach_paths(df, roi_idx, ann_idx, js_idx, cfp_idx)

    #     # 3) Preprocess CFP → out cfp PNG + dicom_id
    #     print("[3/7] Preprocessing CFP images...")
    # df = preprocess_cfp_and_save(df, out_cfp_dir, (roi_w, roi_h), bg_rgb, comp)

    # # print("[3/7] Preprocessing ROI images...")
    # # df = preprocess_roi_and_save(df, out_roi_dir, (roi_w, roi_h), bg_rgb, comp)

    # # 4) Rasterize JSON → mask PNG
    # # print("[4/7] Rasterizing JSON to masks...")
    # # df = rasterize_masks(df, out_masks_dir, (roi_w, roi_h))
    # # 4.5) 必须先丢掉没有全图的记录
    # df = drop_rows_without_images(df, require_mask=False)

    # # 4.6) 丢掉“同 subject 重复 CFP”的行（防止图像内容被多次复用导致检索过易）
    # if bool(cfg.get("dedup", {}).get("drop_reused_cfp_per_subject", True)):
    #     df = drop_reused_cfp_per_subject(df)

    # # 5/6) 过滤 & 按 subject 划分
    # print("[5/7] Filtering subjects with enough studies...")
    # min_st = int(cfg.get("reid", {}).get("min_studies_per_subject", 2))
    # df = filter_min_studies(df, min_st)

    # print("[6/7] Assigning splits (patient-level)...")
    # df = assign_splits_by_subject(df, cfg.get("splits", {}), seed=int(cfg.get("splits", {}).get("seed", 42)))

    # # 7) 保存 CSV（把 image_path/dicom_id 写进去；mask_path 这次会是空）
    # print("[7/7] Saving CSVs...")
    # save_splits_and_artifacts(df, out_root)
    
def main():
    ap = argparse.ArgumentParser(description="GRAPE data preprocessor (patient-level, ROI only)")
    ap.add_argument("--config", type=str, default="../configs/grape.yaml", help="YAML config path")
    args = ap.parse_args()

    cfg = _read_yaml(Path(args.config))
    raw_root = Path(cfg["raw_data_folder"]).expanduser()
    out_root = Path(cfg["output_folder"]).expanduser()

    cfg_paths = cfg.get("paths", {})
    out_roi_dir = out_root / cfg_paths.get("out_roi_dir", "roi")
    out_masks_dir = out_root / cfg_paths.get("out_masks_dir", "masks")

    roi_w, roi_h = _to_tuple_wh(cfg.get("resize", {}).get("roi", [256, 256]))
    bg_rgb = tuple(cfg.get("image_processing", {}).get("background_color", [0, 0, 0]))
    comp = int(cfg.get("image_processing", {}).get("compression_level", 3))

    # 1) Read Excel → long table
    print("[1/7] Loading Excel & building table...")
    df = load_excel_build_long_table(cfg)
    print(f"   Loaded rows: {len(df)} | subjects={df['subject_id'].nunique()} | studies(unique per subj mean)={df.groupby('subject_id')['study_id'].nunique().mean():.2f}")

    # 2) Index raw files & attach paths
    print("[2/7] Indexing ROI/Annotated/JSON & attaching paths...")
    roi_idx, ann_idx, js_idx = build_file_indices(raw_root, cfg_paths)
    df = attach_paths(df, roi_idx, ann_idx, js_idx)

    # 3) Preprocess ROI → out roi PNG + dicom_id
    print("[3/7] Preprocessing ROI images...")
    df = preprocess_roi_and_save(df, out_roi_dir, (roi_w, roi_h), bg_rgb, comp)

    # 4) Rasterize JSON → mask PNG
    print("[4/7] Rasterizing JSON to masks...")
    df = rasterize_masks(df, out_masks_dir, (roi_w, roi_h))

    # 4.5) Drop rows without ROI (and optionally without mask if you want strict seg-only)
    require_mask = bool(cfg.get("filter", {}).get("require_mask", False))
    df = drop_rows_without_images(df, require_mask=require_mask)

    # 4.9) NEW: 按图像唯一性去重（同一 dicom_id 仅保留一条）
    df = drop_duplicate_images(df, prefer_mask=True, prefer_baseline=True)

    # 5) Filter: subjects with ≥N studies
    print("[5/7] Filtering subjects with enough studies...")
    min_st = int(cfg.get("reid", {}).get("min_studies_per_subject", 2))
    df = filter_min_studies(df, min_st)

    # 6) Assign splits (patient-level)
    print("[6/7] Assigning splits (patient-level)...")
    df = assign_splits_by_subject(df, cfg.get("splits", {}), seed=int(cfg.get("splits", {}).get("seed", 42)))

    # 7) Save CSVs
    print("[7/7] Saving CSVs...")
    save_splits_and_artifacts(df, out_root)

    # Optional: overlay consistency check
    chk = cfg.get("checks", {}).get("overlay_consistency", True)
    if chk:
        report_csv = out_root / cfg.get("checks", {}).get("report_csv", "mask_consistency_report.csv")
        thr = float(cfg.get("checks", {}).get("overlay_iou_threshold", 0.90))
        consistency_check(df, report_csv, iou_thr=thr)

    # Export overlays for quick visual QC (A: label mask for training already saved; B2: overlay)
    vis_cfg = cfg.get("vis", {})
    if bool(vis_cfg.get("enable_overlay", True)):
        out_overlay_dir = out_root / vis_cfg.get("overlay_dir", "vis/overlay")
        alpha = float(vis_cfg.get("alpha", 0.5))
        export_overlays(df, out_overlay_dir, alpha=alpha)

    print("\n=== Done. ===")

if __name__ == "__main__":
    main()
