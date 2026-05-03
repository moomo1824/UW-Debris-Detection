# -*- coding: utf-8 -*-
"""model3.ipynb
"""

!pip -q install ultralytics torchvision pycocotools thop scikit-learn

import os, glob, yaml, shutil, json, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import box_iou

from ultralytics import YOLO

DATA_ROOT = "/kaggle/input/uw-garbage-debris-dataa/Underwater_garbage_debris_data"

CLASSES = [
    'Mask','can','cellphone','electronics','gbottle','glove','metal',
    'misc','net','pbag','pbottle','plastic','rod','sunglasses','tire'
]
NC = len(CLASSES)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

train_img_dir = f"{DATA_ROOT}/train/images"
train_lbl_dir = f"{DATA_ROOT}/train/labels"
val_img_dir   = f"{DATA_ROOT}/valid/images"
val_lbl_dir   = f"{DATA_ROOT}/valid/labels"
test_img_dir  = f"{DATA_ROOT}/test/images"
test_lbl_dir  = f"{DATA_ROOT}/test/labels"

def stems_in_dir(img_dir, exts=(".jpg",".jpeg",".png")):
    s=set()
    for e in exts:
        for p in glob.glob(os.path.join(img_dir, f"*{e}")):
            s.add(os.path.splitext(os.path.basename(p))[0])
    return s

def label_stems(lbl_dir):
    return set(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(lbl_dir, "*.txt")))

def class_counts_from_labels(lbl_dir, ncls):
    counts = np.zeros(ncls, dtype=np.int64)
    files = glob.glob(os.path.join(lbl_dir, "*.txt"))
    for lf in files:
        with open(lf, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                c = int(line.split()[0])
                if 0 <= c < ncls:
                    counts[c] += 1
    return counts, len(files)

print("Split consistency:")
for split, idir, ldir in [("train",train_img_dir,train_lbl_dir),("valid",val_img_dir,val_lbl_dir),("test",test_img_dir,test_lbl_dir)]:
    im = stems_in_dir(idir)
    lb = label_stems(ldir)
    print(split, "images:", len(im), "labels:", len(lb), "extra_labels:", len(lb-im), "missing_labels:", len(im-lb))

train_counts, ntrain = class_counts_from_labels(train_lbl_dir, NC)
val_counts,   nval   = class_counts_from_labels(val_lbl_dir,   NC)
test_counts,  ntest  = class_counts_from_labels(test_lbl_dir,  NC)

print("\nTrain label files:", ntrain, dict(zip(CLASSES, train_counts.tolist())))
print("Valid label files:", nval, dict(zip(CLASSES, val_counts.tolist())))
print("Test  label files:", ntest, dict(zip(CLASSES, test_counts.tolist())))

WORK_ROOT = "/kaggle/working/uw_cleaned_A"
os.makedirs(WORK_ROOT, exist_ok=True)

CLEAN_VALID_IMG = os.path.join(WORK_ROOT, "valid/images")
CLEAN_VALID_LBL = os.path.join(WORK_ROOT, "valid/labels")
os.makedirs(CLEAN_VALID_IMG, exist_ok=True)
os.makedirs(CLEAN_VALID_LBL, exist_ok=True)

val_img_stems = stems_in_dir(val_img_dir)
val_lbl_stems = label_stems(val_lbl_dir)

extra_lbl = sorted(list(val_lbl_stems - val_img_stems))
print("Extra valid label stems:", len(extra_lbl), extra_lbl[:10])

# copy valid images
for ext in ("*.jpg","*.jpeg","*.png"):
    for p in glob.glob(os.path.join(val_img_dir, ext)):
        shutil.copy2(p, os.path.join(CLEAN_VALID_IMG, os.path.basename(p)))

# copy only matching labels
kept = 0
for p in glob.glob(os.path.join(val_lbl_dir, "*.txt")):
    stem = os.path.splitext(os.path.basename(p))[0]
    if stem in val_img_stems:
        shutil.copy2(p, os.path.join(CLEAN_VALID_LBL, os.path.basename(p)))
        kept += 1

print("Clean valid images:", len(stems_in_dir(CLEAN_VALID_IMG)))
print("Clean valid labels:", len(glob.glob(os.path.join(CLEAN_VALID_LBL, "*.txt"))), "kept:", kept)

OVS_ROOT = os.path.join(WORK_ROOT, "train_oversampled")
OVS_IMG  = os.path.join(OVS_ROOT, "images")
OVS_LBL  = os.path.join(OVS_ROOT, "labels")
os.makedirs(OVS_IMG, exist_ok=True)
os.makedirs(OVS_LBL, exist_ok=True)

RARE_THRESH = 200
rare_classes = [i for i,cnt in enumerate(train_counts.tolist()) if cnt < RARE_THRESH]
print("Rare classes (idx):", rare_classes)
print("Rare classes (name,count):", [(CLASSES[i], int(train_counts[i])) for i in rare_classes])

train_imgs = sorted(
    glob.glob(os.path.join(train_img_dir, "*.jpg")) +
    glob.glob(os.path.join(train_img_dir, "*.jpeg")) +
    glob.glob(os.path.join(train_img_dir, "*.png"))
)

def read_label_classes(lbl_path):
    cls=set()
    if not os.path.exists(lbl_path):
        return cls
    with open(lbl_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            cls.add(int(line.split()[0]))
    return cls

# Map image -> classes
img_to_classes = {}
for ip in train_imgs:
    stem = os.path.splitext(os.path.basename(ip))[0]
    lp = os.path.join(train_lbl_dir, stem + ".txt")
    img_to_classes[ip] = read_label_classes(lp)

# candidate images containing any rare class
rare_set = set(rare_classes)
candidate = [ip for ip,cs in img_to_classes.items() if len(rare_set.intersection(cs)) > 0]
print("Images containing rare classes:", len(candidate), "/", len(train_imgs))

# Copy originals once
for ip in train_imgs:
    stem = os.path.splitext(os.path.basename(ip))[0]
    lp = os.path.join(train_lbl_dir, stem + ".txt")
    if os.path.exists(lp):
        shutil.copy2(ip, os.path.join(OVS_IMG, os.path.basename(ip)))
        shutil.copy2(lp, os.path.join(OVS_LBL, os.path.basename(lp)))


DUP_MULT = 1.0
n_dup = int(len(candidate) * DUP_MULT)
print("Planned duplicates:", n_dup)

random.seed(42)
choices = [random.choice(candidate) for _ in range(n_dup)]

for k, ip in enumerate(choices):
    stem = os.path.splitext(os.path.basename(ip))[0]
    lp = os.path.join(train_lbl_dir, stem + ".txt")
    ext = os.path.splitext(ip)[1]
    new_stem = f"{stem}__raredup{k:06d}"
    shutil.copy2(ip, os.path.join(OVS_IMG, new_stem + ext))
    shutil.copy2(lp, os.path.join(OVS_LBL, new_stem + ".txt"))

print("Oversampled train images:", len(glob.glob(os.path.join(OVS_IMG, "*"))))
print("Oversampled train labels:", len(glob.glob(os.path.join(OVS_LBL, "*.txt"))))

YAML_OVS = os.path.join(WORK_ROOT, "data_oversampled.yaml")

data_ovs = {
    "path": WORK_ROOT,
    "train": "train_oversampled/images",
    "val": CLEAN_VALID_IMG,
    "test": test_img_dir,
    "nc": NC,
    "names": CLASSES
}
with open(YAML_OVS, "w") as f:
    yaml.dump(data_ovs, f, sort_keys=False)

print("Wrote:", YAML_OVS)
!cat $YAML_OVS

IMG_SIZE = 640
TRAIN_LR0 = 2e-4
TRAIN_BATCH = 16
EPOCHS_MAIN = 60        # safe overnight on T4
CLOSE_MOSAIC = 10
PATIENCE = 15

PROJECT_DIR = "/kaggle/working/runs_cmp_A"
os.makedirs(PROJECT_DIR, exist_ok=True)

EXPORT_ROOT = "/kaggle/output/variant_exports"
os.makedirs(EXPORT_ROOT, exist_ok=True)

SUMMARY_CSV = "/kaggle/output/model_comparison_results.csv"

def _safe_copy(src, dst_dir):
    if src and os.path.isfile(src):
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))

def export_variant(variant_name, run_name, run_dir, ckpt_path, val_met, test_met, extra_fields=None):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(EXPORT_ROOT, f"{variant_name}__{run_name}__{ts}")
    os.makedirs(out_dir, exist_ok=True)

    # metrics json
    with open(os.path.join(out_dir, "val_metrics.json"), "w") as f:
        json.dump(val_met, f, indent=2)
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(test_met, f, indent=2)

    # copy weights and artifacts
    _safe_copy(ckpt_path, out_dir)
    weights_dir = os.path.join(run_dir, "weights")
    if os.path.isdir(weights_dir):
        shutil.copytree(weights_dir, os.path.join(out_dir, "weights"), dirs_exist_ok=True)

    for pat in ["results.csv","results.png","confusion_matrix*.png","*_curve.png","val_batch*.jpg","val_batch*.png","labels.jpg","labels.png"]:
        for p in glob.glob(os.path.join(run_dir, pat)):
            _safe_copy(p, out_dir)

    # append summary row
    row = {
        "variant": variant_name, "run": run_name, "run_dir": run_dir, "best_pt": ckpt_path,
        "val_map50": val_met["map50"], "val_map50_95": val_met["map50_95"],
        "val_precision": val_met["precision"], "val_recall": val_met["recall"], "val_f1": val_met["f1"],
        "val_tp": val_met.get("tp"), "val_fp": val_met.get("fp"), "val_fn": val_met.get("fn"), "val_mean_iou": val_met.get("mean_iou"),
        "test_map50": test_met["map50"], "test_map50_95": test_met["map50_95"],
        "test_precision": test_met["precision"], "test_recall": test_met["recall"], "test_f1": test_met["f1"],
        "test_tp": test_met.get("tp"), "test_fp": test_met.get("fp"), "test_fn": test_met.get("fn"), "test_mean_iou": test_met.get("mean_iou"),
    }
    if extra_fields:
        row.update(extra_fields)

    df_row = pd.DataFrame([row])
    if os.path.exists(SUMMARY_CSV):
        df_old = pd.read_csv(SUMMARY_CSV)
        df_all = pd.concat([df_old, df_row], ignore_index=True)
    else:
        df_all = df_row
    df_all.to_csv(SUMMARY_CSV, index=False)

    zip_path = shutil.make_archive(out_dir, 'zip', out_dir)
    print("Exported:", out_dir)
    print("Zipped   :", zip_path)
    print("Summary  :", SUMMARY_CSV)
    return out_dir, zip_path

def _scalar(x):
    try:
        return float(x)
    except Exception:
        return float(np.array(x).reshape(-1).mean())

def f1_from_pr(p, r):
    return (2*p*r) / (p + r + 1e-12)

def eval_ultralytics(ckpt, data_yaml, split="val", imgsz=640, save_plots=True):
    y = YOLO(ckpt)
    m = y.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        device=0 if device=="cuda" else "cpu",
        plots=save_plots,
        save_json=True,
        verbose=False
    )
    rd = dict(m.results_dict) if hasattr(m, "results_dict") else {}
    p = _scalar(rd.get("metrics/precision(B)", np.nan))
    r = _scalar(rd.get("metrics/recall(B)", np.nan))
    out = {
        "map50": _scalar(getattr(m.box, "map50", np.nan)),
        "map50_95": _scalar(getattr(m.box, "map", np.nan)),
        "precision": p,
        "recall": r,
        "f1": f1_from_pr(p, r),
    }
    return out

def yolo_xywhn_to_xyxy(boxes, w, h):
    # boxes: (N,4) normalized xywh
    x,y,bw,bh = boxes[:,0]*w, boxes[:,1]*h, boxes[:,2]*w, boxes[:,3]*h
    x1 = x - bw/2
    y1 = y - bh/2
    x2 = x + bw/2
    y2 = y + bh/2
    return np.stack([x1,y1,x2,y2], axis=1)

def load_yolo_labels(lbl_path, img_w, img_h):
    if not os.path.exists(lbl_path):
        return np.zeros((0,4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    boxes=[]
    cls=[]
    with open(lbl_path,"r") as f:
        for line in f:
            if not line.strip():
                continue
            parts=line.split()
            c=int(parts[0])
            xywh=list(map(float, parts[1:5]))
            cls.append(c)
            boxes.append(xywh)
    boxes=np.array(boxes, dtype=np.float32)
    cls=np.array(cls, dtype=np.int64)
    if len(boxes)==0:
        return np.zeros((0,4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    xyxy = yolo_xywhn_to_xyxy(boxes, img_w, img_h)
    return xyxy.astype(np.float32), cls

def compute_tp_fp_fn_mean_iou(ckpt, img_dir, lbl_dir, imgsz=640, conf=0.25, iou_thr=0.5, max_images=None):
    y = YOLO(ckpt)
    imgs = sorted(glob.glob(os.path.join(img_dir,"*.jpg")) + glob.glob(os.path.join(img_dir,"*.jpeg")) + glob.glob(os.path.join(img_dir,"*.png")))
    if max_images:
        imgs = imgs[:max_images]

    TP=0; FP=0; FN=0
    ious=[]
    for ip in imgs:
        im = Image.open(ip).convert("RGB")
        w,h = im.size
        stem = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(lbl_dir, stem + ".txt")
        gt_boxes, gt_cls = load_yolo_labels(lp, w, h)

        pred = y.predict(source=ip, imgsz=imgsz, conf=conf, device=0 if device=="cuda" else "cpu", verbose=False)[0]
        if pred.boxes is None or len(pred.boxes)==0:
            FN += len(gt_boxes)
            continue

        pb = pred.boxes.xyxy.cpu().numpy().astype(np.float32)
        pc = pred.boxes.cls.cpu().numpy().astype(np.int64)

        if len(gt_boxes)==0:
            FP += len(pb)
            continue

        iou = box_iou(torch.from_numpy(pb), torch.from_numpy(gt_boxes)).numpy()  # (Np,Ng)
        # Greedy match by IoU, class-aware
        used_g=set()
        for pi in range(len(pb)):
            # best gt index for this pred with same class
            valid = [gi for gi in range(len(gt_boxes)) if gi not in used_g and pc[pi]==gt_cls[gi]]
            if not valid:
                FP += 1
                continue
            gi_best = max(valid, key=lambda gi: iou[pi, gi])
            if iou[pi, gi_best] >= iou_thr:
                TP += 1
                used_g.add(gi_best)
                ious.append(float(iou[pi, gi_best]))
            else:
                FP += 1

        FN += (len(gt_boxes) - len(used_g))

    mean_iou = float(np.mean(ious)) if len(ious) else float("nan")
    return {"tp":TP, "fp":FP, "fn":FN, "mean_iou":mean_iou}

def save_test_prediction_images(ckpt, test_img_dir, out_dir, imgsz=640, conf=0.25, n=16):
    os.makedirs(out_dir, exist_ok=True)
    y = YOLO(ckpt)
    imgs = sorted(glob.glob(os.path.join(test_img_dir,"*.jpg")) + glob.glob(os.path.join(test_img_dir,"*.jpeg")) + glob.glob(os.path.join(test_img_dir,"*.png")))
    sample = imgs[:n]
    y.predict(source=sample, imgsz=imgsz, conf=conf, device=0 if device=="cuda" else "cpu", save=True, project=out_dir, name="pred", exist_ok=True, verbose=False)
    print("Saved predicted images under:", out_dir)

import torch.nn as nn

def nt_xent(z1, z2, temp=0.2):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temp
    sim.fill_diagonal_(-1e9)
    targets = torch.arange(B, device=z.device)
    targets = torch.cat([targets + B, targets], dim=0)
    return F.cross_entropy(sim, targets)

class ContrastiveLearningDataset(Dataset):
    def __init__(self, img_dir, imgsz=640):
        self.img_paths = sorted(
            glob.glob(os.path.join(img_dir, "*.jpg")) +
            glob.glob(os.path.join(img_dir, "*.jpeg")) +
            glob.glob(os.path.join(img_dir, "*.png"))
        )
        self.t = T.Compose([
            T.RandomResizedCrop(imgsz, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
    def __len__(self): return len(self.img_paths)
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        return self.t(img), self.t(img)

def find_detect_idx(ultra_model):
    for i, m in enumerate(ultra_model.model):
        if m.__class__.__name__.lower() == "detect":
            return i
    return None

def probe_p5_channels(det_model, detect_idx, imgsz=640):
    probe = {}
    def hook(module, inp, out): probe["out"] = out
    h = det_model.model[detect_idx - 1].register_forward_hook(hook)
    det_model.eval()
    with torch.no_grad():
        _ = det_model(torch.zeros(1, 3, imgsz, imgsz, device=device))
    h.remove()
    out = probe["out"]
    p5 = out[-1] if isinstance(out, (list, tuple)) else out
    return int(p5.shape[1])

def ssl_pretrain_yolo11s(img_dir_for_ssl, imgsz=640, ssl_epochs=8, ssl_batch=32, ssl_lr=3e-4, temp=0.2,
                         save_pt="/kaggle/working/yolo11s_ssl_only.pt"):
    y = YOLO("yolo11s.pt")
    det_model = y.model.to(device)
    detect_idx = find_detect_idx(det_model)
    assert detect_idx is not None, "Detect not found"

    C = probe_p5_channels(det_model, detect_idx, imgsz=imgsz)

    feat_holder = {}
    def hook_fn(module, inp, out):
        feat_holder["feat"] = out[-1] if isinstance(out, (list, tuple)) else out
    hh = det_model.model[detect_idx - 1].register_forward_hook(hook_fn)

    proj = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(C, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
    ).to(device)

    ds = ContrastiveLearningDataset(img_dir_for_ssl, imgsz=imgsz)
    dl = DataLoader(ds, batch_size=ssl_batch, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    opt = torch.optim.AdamW(list(det_model.model[:detect_idx].parameters()) + list(proj.parameters()), lr=ssl_lr)

    det_model.train(); proj.train()
    for ep in range(ssl_epochs):
        total = 0.0
        for x1, x2 in dl:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            _ = det_model(x1)
            z1 = F.normalize(proj(feat_holder["feat"]), dim=1)

            _ = det_model(x2)
            z2 = F.normalize(proj(feat_holder["feat"]), dim=1)

            loss = nt_xent(z1, z2, temp=temp)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())

        print(f"[SSL] ep {ep+1}/{ssl_epochs} loss {total/len(dl):.4f}")

    hh.remove()
    y.save(save_pt)
    print("Saved SSL checkpoint:", save_pt)
    return save_pt

SSL_ONLY_PT = ssl_pretrain_yolo11s(
    img_dir_for_ssl=train_img_dir,
    imgsz=IMG_SIZE,
    ssl_epochs=8,
    ssl_batch=32,
    ssl_lr=3e-4,
    temp=0.2,
    save_pt="/kaggle/working/yolo11s_ssl_only.pt"
)

run_name = f"C_ssl_lr{TRAIN_LR0}_b{TRAIN_BATCH}"
y = YOLO(SSL_ONLY_PT)

run_dir = os.path.join(PROJECT_DIR, run_name)
y.train(
    data=YAML_OVS,
    epochs=EPOCHS_MAIN,
    imgsz=IMG_SIZE,
    batch=TRAIN_BATCH,
    device=0 if device=="cuda" else "cpu",
    workers=2,
    close_mosaic=CLOSE_MOSAIC,
    optimizer="Adam",
    lr0=TRAIN_LR0,
    cos_lr=True,
    patience=PATIENCE,
    plots=True,
    project=PROJECT_DIR,
    name=run_name,
    exist_ok=True
)

best_pt = os.path.join(run_dir, "weights/best.pt")
last_pt = os.path.join(run_dir, "weights/last.pt")
ckpt = best_pt if os.path.exists(best_pt) else last_pt
print("Checkpoint:", ckpt)

val_met = eval_ultralytics(ckpt, YAML_OVS, split="val",  imgsz=IMG_SIZE, save_plots=True)
test_met = eval_ultralytics(ckpt, YAML_OVS, split="test", imgsz=IMG_SIZE, save_plots=True)

val_extra  = compute_tp_fp_fn_mean_iou(ckpt, CLEAN_VALID_IMG, CLEAN_VALID_LBL, imgsz=IMG_SIZE, conf=0.25, iou_thr=0.5)
test_extra = compute_tp_fp_fn_mean_iou(ckpt, test_img_dir, test_lbl_dir, imgsz=IMG_SIZE, conf=0.25, iou_thr=0.5)

val_met.update(val_extra)
test_met.update(test_extra)

pred_out = "/kaggle/output/test_predictions_C"
save_test_prediction_images(ckpt, test_img_dir, pred_out, imgsz=IMG_SIZE, conf=0.25, n=16)

print("VAL:", val_met)
print("TEST:", test_met)

export_variant("C_ssl", run_name, run_dir, ckpt, val_met, test_met)
