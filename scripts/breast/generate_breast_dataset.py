#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse, json, hashlib
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import yaml

# pydicom
try:
    import pydicom
except ImportError:
    print("Please `pip install pydicom`", file=sys.stderr)
    raise

# ---------------------------
# Utils
# ---------------------------

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_yaml(path: str, obj: dict):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)

def save_json(path: str, obj: dict):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# ---------------------------
# DICOM index (stem -> full path)
# ---------------------------

def build_dicom_index(raw_root: str, cache_path: Optional[str] = None) -> Dict[str, str]:
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)
    index = {}
    dups = []
    for p in Path(raw_root).rglob("*.dcm"):
        stem = p.stem
        if stem in index and str(p) != index[stem]:
            dups.append(stem)
        index[stem] = str(p)
    if dups:
        print(f"[WARN] Duplicate DICOM stems ({len(dups)}). Keeping last occurrence. e.g., {dups[:3]}")
    if cache_path:
        ensure_dir(os.path.dirname(cache_path))
        with open(cache_path, "w") as f:
            json.dump(index, f)
    print(f"[Index] DICOM files indexed: {len(index)}")
    return index

# ---------------------------
# Parse from filename stem:
# <anon_patientid>_20990909_<L/R>_<CC/MLO>_<sequence>
# ---------------------------

def parse_from_stem(stem: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    parts = stem.split("_")
    lat = view = None
    seq = None
    if len(parts) >= 5:
        lat = parts[-3].upper()
        view = parts[-2].upper()
        try:
            seq = int(parts[-1])
        except Exception:
            seq = None
    return lat, view, seq

# ---------------------------
# Mask path finder
# ---------------------------

def find_mask_path(dicom_id: str, ann_dir: str) -> Optional[str]:
    cand = [
        f"{dicom_id}_mask.png",
        f"{dicom_id}.png",
        f"{dicom_id}_MASK.png",
        f"{dicom_id}.PNG",
    ]
    for name in cand:
        p = os.path.join(ann_dir, name)
        if os.path.exists(p):
            return p
    # fallback glob
    m = list(Path(ann_dir).glob(f"{dicom_id}*mask*.png"))
    if m:
        return str(m[0])
    return None

# ---------------------------
# DICOM -> uint8 'L' image (0..255)
# ---------------------------

def dicom_to_uint8(path: str) -> np.ndarray:
    d = pydicom.dcmread(path)
    arr = d.pixel_array.astype(np.float32)

    slope = float(getattr(d, "RescaleSlope", 1.0))
    intercept = float(getattr(d, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    lo, hi = np.percentile(arr, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(arr)), float(np.max(arr))
        if hi <= lo:
            hi = lo + 1.0

    arr = (arr - lo) / (hi - lo)
    arr = np.clip(arr, 0.0, 1.0)

    phot = str(getattr(d, "PhotometricInterpretation", "")).upper()
    if phot == "MONOCHROME1":  # invert
        arr = 1.0 - arr

    img8 = np.rint(arr * 255.0).astype(np.uint8)
    return img8

# ---------------------------
# Letterbox (keep ratio + center pad)
# ---------------------------

def compute_letterbox(h: int, w: int, target_h: int, target_w: int) -> Dict[str, int]:
    scale = min(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    return {"scale": scale, "new_w": new_w, "new_h": new_h, "x_off": x_off, "y_off": y_off}

def apply_letterbox_img(img: Image.Image, tx: Dict[str, int], bg=0, resample=Image.Resampling.LANCZOS) -> Image.Image:
    canvas = Image.new("L", (tx["new_w"] + 2*tx["x_off"], tx["new_h"] + 2*tx["y_off"]), color=bg)
    resized = img.resize((tx["new_w"], tx["new_h"]), resample=resample)
    canvas.paste(resized, (tx["x_off"], tx["y_off"]))
    return canvas

def apply_letterbox_mask(mask: Image.Image, tx: Dict[str, int]) -> Image.Image:
    canvas = Image.new("L", (tx["new_w"] + 2*tx["x_off"], tx["new_h"] + 2*tx["y_off"]), color=0)
    resized = mask.resize((tx["new_w"], tx["new_h"]), resample=Image.Resampling.NEAREST)
    canvas.paste(resized, (tx["x_off"], tx["y_off"]))
    return canvas

# ---------------------------
# Overlay
# ---------------------------

def mask_to_rgba_binary(mask01: np.ndarray, alpha: float = 0.45) -> Image.Image:
    h, w = mask01.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    m = (mask01 == 1)
    rgba[m] = (0, 255, 0, int(round(alpha * 255)))
    return Image.fromarray(rgba, mode="RGBA")

def make_overlay(base_gray: Image.Image, mask_bin: Image.Image, alpha: float) -> Image.Image:
    base_rgb = Image.merge("RGB", (base_gray, base_gray, base_gray)).convert("RGBA")
    m = np.array(mask_bin, dtype=np.uint8)
    m01 = (m > 0).astype(np.uint8)
    overlay = mask_to_rgba_binary(m01, alpha=alpha)
    return Image.alpha_composite(base_rgb, overlay)

# ---------------------------
# Mismatch analysis
# ---------------------------

def analyze_size_mismatch(img_h, img_w, msk_h, msk_w, tol=0.01):
    if img_h<=0 or img_w<=0 or msk_h<=0 or msk_w<=0:
        return False, None, None, False
    ar_img = img_w / img_h
    ar_msk = msk_w / msk_h
    same_ar = abs(ar_img - ar_msk) <= tol
    sh = img_h / msk_h
    sw = img_w / msk_w
    uniform = same_ar and (abs(sh - sw) <= tol)
    return uniform, sh, sw, same_ar

# ---------------------------
# Splitting (patient-disjoint, stratify by x_case)
# ---------------------------

def stratified_patient_split_on_all(df_all: pd.DataFrame, ratios=(0.7,0.1,0.2), seed=42):
    rng = np.random.default_rng(seed)
    patients = []
    for pid, g in df_all.groupby("subject_id"):
        pos = int((g["x_case"] == 1).any())
        n = len(g)
        patients.append((pid, pos, n))
    # 优先分配阳性和大样本
    patients.sort(key=lambda x: (-x[1], -x[2]))
    splits = ["train","validate","test"]
    total_pos = sum(p[1] for p in patients)
    total_pat = len(patients)
    targets_pat = {s: int(round(total_pat * r)) for s, r in zip(splits, ratios)}
    targets_pos = {s: int(round(total_pos * r)) for s, r in zip(splits, ratios)}
    assign, counts_pat, counts_pos = {}, {s:0 for s in splits}, {s:0 for s in splits}
    for pid, pos, n in patients:
        s_pick = max(splits, key=lambda s: (targets_pos[s]-counts_pos[s], targets_pat[s]-counts_pat[s]))
        assign[pid] = s_pick
        counts_pat[s_pick] += 1
        if pos: counts_pos[s_pick] += 1
    return assign

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser("CSAW: Build Seg/Cls(REID-ready)/REID datasets")
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config")
    ap.add_argument("--skip-images", action="store_true", help="Skip heavy image/mask processing (only rebuild CSVs)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    raw_root = cfg["raw_data_folder"]
    meta_csv = cfg["metadata_csv"]
    ann_dir  = cfg["annotations_dir"]
    out_root = cfg["output_folder"]

    t512  = tuple(cfg.get("resize_train", [512, 512]))
    t1024 = tuple(cfg.get("resize_archive", [1024, 1024]))
    ratios = tuple(cfg.get("split_ratios", [0.7, 0.1, 0.2]))
    seed = int(cfg.get("seed", 42))
    alpha = float(cfg.get("overlay_alpha", 0.45))
    comp = int(cfg.get("compression_level", 3))
    area_thr_1024 = int(cfg.get("mask_area_threshold_px_1024", 30))

    mismatch_policy = cfg.get("mismatch_policy", "log_and_skip")
    uniform_tol = float(cfg.get("uniform_scale_tol", 0.01))

    # Outputs
    images_512   = os.path.join(out_root, "images_512")
    images_1024  = os.path.join(out_root, "images_1024")
    masks_512    = os.path.join(out_root, "masks_bin_512")
    masks_1024   = os.path.join(out_root, "masks_bin_1024")
    overlays_512 = os.path.join(out_root, "overlays_512")
    csv_dir = os.path.join(out_root, "csvs")
    for d in [images_512, images_1024, masks_512, masks_1024, overlays_512, csv_dir]:
        ensure_dir(d)

    cache_idx = os.path.join(out_root, "cache", "dicom_index.json")
    dicom_index = build_dicom_index(raw_root, cache_path=cache_idx)

    # Load metadata
    df = pd.read_csv(meta_csv)
    if "anon_patientid" not in df.columns or "anon_filename" not in df.columns:
        raise RuntimeError("metadata_csv must contain columns: anon_patientid, anon_filename")

    # Core columns
    df["subject_id"] = df["anon_patientid"].astype(str)
    df["dicom_file"] = df["anon_filename"].astype(str)
    df["dicom_id"]   = df["dicom_file"].str.replace(".dcm", "", case=False, regex=False)

    # Laterality / ViewPosition
    if "imagelaterality" in df.columns:
        df["Laterality"] = df["imagelaterality"].astype(str).str.upper().map({"LEFT":"L","RIGHT":"R","L":"L","R":"R"})
    else:
        df["Laterality"] = df["dicom_id"].apply(lambda s: parse_from_stem(s)[0])

    if "viewposition" in df.columns:
        df["ViewPosition"] = df["viewposition"].astype(str).str.upper().map({"CC":"CC","MLO":"MLO"})
    else:
        df["ViewPosition"] = df["dicom_id"].apply(lambda s: parse_from_stem(s)[1])

    # study_id
    if "sequence" in df.columns:
        df["study_id"] = df["sequence"].astype("Int64")
    else:
        df["study_id"] = df["dicom_id"].apply(lambda s: parse_from_stem(s)[2]).astype("Int64")

    # x_case
    if "x_case" not in df.columns:
        df["x_case"] = 0
        print("[WARN] metadata_csv lacks x_case; default 0.")

    # Map to dicom path
    df["dicom_path"] = df["dicom_id"].map(lambda s: dicom_index.get(s, None))
    miss = df["dicom_path"].isna().sum()
    if miss > 0:
        print(f"[WARN] Missing DICOM paths for {miss} rows; they will be dropped.")
        df = df[~df["dicom_path"].isna()].copy()

    # Mask paths & existence (raw)
    df["mask_raw"] = df["dicom_id"].apply(lambda s: find_mask_path(s, ann_dir))
    df["has_mask_raw"] = df["mask_raw"].notna().astype(int)

    # Accumulators for per-image metadata written later
    meta_rows = []           # per dicom_id meta for CSV join
    mismatch_rows = []       # logs for size mismatches
    region_count = 0         # masks accepted as "region"
    overlay_done = 0

    # Process images & masks
    if not args.skip_images:
        print(f"\n=== Processing ALL DICOMs ({len(df)}) to 512 & 1024 ===")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="proc"):
            dicom_id   = row["dicom_id"]
            dicom_path = row["dicom_path"]
            lat        = str(row["Laterality"]) if pd.notna(row["Laterality"]) else None

            # Output paths
            out_img_512  = os.path.join(images_512,  f"{dicom_id}.png")
            out_img_1024 = os.path.join(images_1024, f"{dicom_id}.png")
            out_msk_512  = os.path.join(masks_512,   f"{dicom_id}.png")
            out_msk_1024 = os.path.join(masks_1024,  f"{dicom_id}.png")
            out_ovl_512  = os.path.join(overlays_512,f"{dicom_id}.png")

            # Load DICOM -> uint8
            try:
                img8 = dicom_to_uint8(dicom_path)  # (H,W)
            except Exception as e:
                print(f"[ERR] dicom_to_uint8 failed: {dicom_path} | {e}")
                continue

            H, W = img8.shape
            flip = 1 if (lat is not None and lat.upper()=="R") else 0
            if flip:
                img8 = np.fliplr(img8)

            img_pil = Image.fromarray(img8, mode="L")

            # Letterbox to 1024 & 512
            tx_1024 = compute_letterbox(H, W, t1024[1], t1024[0])
            tx_512  = compute_letterbox(H, W, t512[1],  t512[0])

            img_1024 = apply_letterbox_img(img_pil, tx_1024, bg=0)
            img_512  = apply_letterbox_img(img_pil, tx_512,  bg=0)

            # Save images (skip-if-exist)
            if not os.path.exists(out_img_1024): img_1024.save(out_img_1024, "PNG", compress_level=comp)
            if not os.path.exists(out_img_512):  img_512.save(out_img_512,  "PNG", compress_level=comp)

            # Default mask meta (no valid region)
            has_region_mask = 0
            mask_area_px_1024 = 0
            mask_area_ratio_1024 = 0.0

            # If mask exists: load -> flip -> size check -> letterbox -> area -> threshold -> (maybe save)
            if row["has_mask_raw"] == 1:
                mask_path = row["mask_raw"]
                try:
                    m = Image.open(mask_path)
                    if m.mode != "L":
                        m = m.convert("L")
                    mask_np = (np.array(m, dtype=np.uint8) > 0).astype(np.uint8)  # 0/255->0/1
                except Exception as e:
                    print(f"[ERR] load mask failed: {mask_path} | {e}")
                    mask_np = None

                if mask_np is not None:
                    hm, wm = mask_np.shape
                    if flip:
                        mask_np = np.fliplr(mask_np)

                    # size check against original image size
                    if (H, W) != (hm, wm):
                        uniform, sh, sw, same_ar = analyze_size_mismatch(H, W, hm, wm, tol=uniform_tol)
                        mismatch_rows.append({
                            "dicom_id": dicom_id,
                            "dicom_path": dicom_path,
                            "mask_path": mask_path,
                            "img_h": int(H), "img_w": int(W),
                            "msk_h": int(hm), "msk_w": int(wm),
                            "same_ar": bool(same_ar),
                            "uniform_scale": bool(uniform),
                            "scale_h": None if sh is None else float(sh),
                            "scale_w": None if sw is None else float(sw),
                        })
                        if mismatch_policy == "log_and_skip":
                            # do nothing more with mask
                            pass
                        elif mismatch_policy == "auto_scale_if_uniform" and uniform:
                            # resample mask to original image size with NEAREST
                            mtmp = Image.fromarray((mask_np*255).astype(np.uint8), mode="L").resize((W, H), resample=Image.Resampling.NEAREST)
                            mask_np = (np.array(mtmp, dtype=np.uint8) > 0).astype(np.uint8)
                        elif mismatch_policy == "force_resize":
                            mtmp = Image.fromarray((mask_np*255).astype(np.uint8), mode="L").resize((W, H), resample=Image.Resampling.NEAREST)
                            mask_np = (np.array(mtmp, dtype=np.uint8) > 0).astype(np.uint8)
                        else:
                            # treated as skip
                            mask_np = None

                    # if mask usable
                    if mask_np is not None:
                        m_pil = Image.fromarray((mask_np*255).astype(np.uint8), mode="L")

                        # Letterbox to 1024 first (阈值以 1024 为准)
                        msk_1024 = apply_letterbox_mask(m_pil, tx_1024)
                        mask_area_px_1024 = int(np.count_nonzero(np.array(msk_1024, dtype=np.uint8)))

                        if mask_area_px_1024 >= area_thr_1024:
                            has_region_mask = 1
                            mask_area_ratio_1024 = float(mask_area_px_1024) / float(t1024[0]*t1024[1])

                            # 保存 1024 & 512 的 mask
                            if not os.path.exists(out_msk_1024): msk_1024.save(out_msk_1024, "PNG", compress_level=comp)
                            msk_512 = apply_letterbox_mask(m_pil, tx_512)
                            if not os.path.exists(out_msk_512):  msk_512.save(out_msk_512,  "PNG", compress_level=comp)

                            # overlay (512)
                            if not os.path.exists(out_ovl_512):
                                ovl = make_overlay(img_512, msk_512, alpha=alpha)
                                ovl.save(out_ovl_512, "PNG", compress_level=comp)
                                overlay_done += 1

                            region_count += 1
                        else:
                            # 点：不保存 mask/overlay（按你的要求：分割不考虑点；诊断也算负）
                            pass

            # collect per-image meta
            meta_rows.append({
                "dicom_id": dicom_id,
                "image_path_512": out_img_512 if os.path.exists(out_img_512) else "",
                "image_path_1024": out_img_1024 if os.path.exists(out_img_1024) else "",
                "mask_bin_512": out_msk_512 if (has_region_mask and os.path.exists(out_msk_512)) else "",
                "mask_bin_1024": out_msk_1024 if (has_region_mask and os.path.exists(out_msk_1024)) else "",
                "overlay_512": out_ovl_512 if (has_region_mask and os.path.exists(out_ovl_512)) else "",
                "orig_h": int(H),
                "orig_w": int(W),
                "flip": int(flip),
                "has_mask_raw": int(row["has_mask_raw"]),
                "has_region_mask": int(has_region_mask),
                "mask_area_px_1024": int(mask_area_px_1024),
                "mask_area_ratio_1024": float(mask_area_ratio_1024),
            })

        print(f"[Done] Processed images: {len(df)} | region masks kept: {region_count} | overlays: {overlay_done}")

    # Join per-image meta back to df
    meta_df = pd.DataFrame(meta_rows) if meta_rows else pd.DataFrame(columns=[
        "dicom_id","image_path_512","image_path_1024","mask_bin_512","mask_bin_1024","overlay_512",
        "orig_h","orig_w","flip","has_mask_raw","has_region_mask","mask_area_px_1024","mask_area_ratio_1024"
    ])
    df = df.merge(meta_df, on="dicom_id", how="left")

    # Split mapping on ALL images (patient-disjoint + stratify by x_case)
    assign = stratified_patient_split_on_all(df, ratios=ratios, seed=seed)
    df["split"] = df["subject_id"].map(assign)

    # ---- Build REID CSV (all images) ----
    nstud = df.groupby("subject_id")["study_id"].nunique()
    df["n_studies"] = df["subject_id"].map(nstud)
    df["identity_side"] = df["subject_id"].astype(str) + "_" + df["Laterality"].astype(str)
    df["exam_key"] = df["subject_id"].astype(str) + "_" + df["study_id"].astype(str)

    reid_cols = [
        "subject_id","identity_side","study_id","exam_key","dicom_id","split",
        "image_path_512","image_path_1024","Laterality","ViewPosition",
        "orig_h","orig_w","flip","n_studies","x_case"
    ]
    for s, fname in [("train","reid_train.csv"),("validate","reid_val.csv"),("test","reid_test.csv")]:
        part = df[df["split"]==s][reid_cols]
        part.to_csv(os.path.join(csv_dir, fname), index=False)
        print(f"[CSV:REID] {s}: {len(part)} -> {fname}")

    # ---- Build Classification CSV (cls_visible based on region mask only) ----
    df["cls_visible"] = df["has_region_mask"].fillna(0).astype(int)  # 点/无mask 都为0
    cls_cols = [
        "subject_id","study_id","dicom_id","split",
        "image_path_512","image_path_1024","Laterality","ViewPosition",
        "cls_visible","mask_area_px_1024","mask_area_ratio_1024",
        "orig_h","orig_w","flip"
    ]
    for s, fname in [("train","cls_train.csv"),("validate","cls_val.csv"),("test","cls_test.csv")]:
        part = df[df["split"]==s][cls_cols]
        part.to_csv(os.path.join(csv_dir, fname), index=False)
        print(f"[CSV:CLS] {s}: {len(part)} -> {fname}")

    # ---- Build Segmentation CSV (only region masks) ----
    df_seg = df[df["has_region_mask"] == 1].copy()
    seg_cols = [
        "subject_id","study_id","dicom_id","split",
        "image_path_512","image_path_1024","mask_bin_512","mask_bin_1024","overlay_512",
        "Laterality","ViewPosition","mask_area_px_1024","mask_area_ratio_1024",
        "orig_h","orig_w","flip","dicom_path","mask_raw"
    ]
    for s, fname in [("train","seg_train.csv"),("validate","seg_val.csv"),("test","seg_test.csv")]:
        part = df_seg[df_seg["split"]==s][[c for c in seg_cols if c in df_seg.columns]]
        part.to_csv(os.path.join(csv_dir, fname), index=False)
        print(f"[CSV:SEG] {s}: {len(part)} -> {fname}")

    # ---- Stats & mismatch logs ----
    stats = {
        "total_images": int(len(df)),
        "patients_total": int(df["subject_id"].nunique()),
        "split_patients": {s: int(df[df["split"]==s]["subject_id"].nunique()) for s in ["train","validate","test"]},
        "split_images": {s: int((df["split"]==s).sum()) for s in ["train","validate","test"]},
        "region_masks_total": int((df["has_region_mask"]==1).sum()),
        "cls_visible_counts": {
            s: int(df[(df["split"]==s) & (df["cls_visible"]==1)].shape[0]) for s in ["train","validate","test"]
        },
        "class_weights_recommendation": {  # per split (neg:pos) suggestion
            s: {
                "pos": int(df[(df["split"]==s) & (df["cls_visible"]==1)].shape[0]),
                "neg": int(df[(df["split"]==s) & (df["cls_visible"]==0)].shape[0]),
            } for s in ["train","validate","test"]
        },
        "area_threshold_1024": int(area_thr_1024),
        "mismatch_policy": mismatch_policy,
    }
    save_yaml(os.path.join(csv_dir, "stats.yaml"), stats)
    print(f"[STATS] saved -> {os.path.join(csv_dir, 'stats.yaml')}")

    # mismatch csv
    if mismatch_rows:
        mm_csv = os.path.join(csv_dir, cfg.get("mismatch_log_csv", "mask_size_mismatches.csv"))
        pd.DataFrame(mismatch_rows).to_csv(mm_csv, index=False)
        print(f"[LOG] Saved size mismatches: {mm_csv} ({len(mismatch_rows)})")

    print("[DONE] CSAW: Seg / Cls(visible) / REID datasets are ready.")

if __name__ == "__main__":
    main()
