import argparse
import logging
import math
from functools import partial
from typing import Dict, List, Tuple
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import termcolor
import torch
import torch.nn.functional as F
import torch.utils
import torchvision.ops as ops
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import Dice
from pytorch_lightning.callbacks import EarlyStopping
from torchmetrics.classification import MulticlassJaccardIndex as IoU
from sam2rad.datasets.utils import unpad_and_resize_mask
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from sam2rad import (
    DATASETS,
    AverageMeter,
    CompositeLoss,
    DotDict,
    build_sam2rad,
    build_samrad,
    convert_to_semantic,
    dice_loss,
    focal_loss,
    get_dataloaders,
    overlay_contours,
)

from sam2rad.logging import setup_logging
torch.autograd.set_detect_anomaly(True)

setup_logging(output="training_logs", name="sam2rad")

logger = logging.getLogger("sam2rad")

torch.set_float32_matmul_precision("high")

parser = argparse.ArgumentParser(description="Train a segmentation model")
parser.add_argument("--config", type=str, help="Path to model config file")


import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from PIL import Image
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from sam2rad.datasets.utils import unpad_and_resize_mask
from sam2rad import overlay_contours

class SavePredictionsCallback(pl.Callback):
    """
    保存：
    1) 原有：训练/验证 overlay 网格图、验证阶段原分辨率 pred mask、指标
    2) 新增：提示网络的可视化
       - 框提示（pred_boxes）：<save_dir>/box_prompts/{train|val}/epoch_E/<name>_boxes.png
       - 掩码提示（interim_mask_output）：<save_dir>/mask_prompts/{train|val}/epoch_E/<name>_mask.png & _overlay.png
    """
    def __init__(
        self,
        save_dir="outputs_bias0_5",
        vis_batches=5,
        threshold=0.5,
        box_prompt_subdir="box_prompts",
        mask_prompt_subdir="mask_prompts",
    ):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.vis_batches = vis_batches
        self.threshold = threshold

        # 可视化存放目录
        self.box_dir = self.save_dir / box_prompt_subdir
        self.mask_dir = self.save_dir / mask_prompt_subdir

        # 收集用于网格可视化/提示可视化的批次
        self.train_vis_outputs: List[Dict] = []
        self.train_vis_filenames: List[List[str]] = []
        self.val_vis_outputs: List[Dict] = []
        self.val_vis_filenames: List[List[str]] = []

        # 聚合指标
        self.val_preds = []
        self.val_gts = []
        self.val_image_names = []

        # Binary metrics
        self.metric_iou = torchmetrics.JaccardIndex(task="binary")
        self.metric_f1 = torchmetrics.F1Score(task="binary")
        self.metric_precision = torchmetrics.Precision(task="binary")
        self.metric_recall = torchmetrics.Recall(task="binary")
        self.metric_accuracy = torchmetrics.Accuracy(task="binary")

        self.csv_path = self.save_dir / "metrics.csv"
        if not self.csv_path.exists():
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "miou", "f1", "precision", "recall", "accuracy"])

    # ---------- 训练阶段 ----------
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx < self.vis_batches:
            self.train_vis_outputs.append(outputs)
            self.train_vis_filenames.append(batch.get("filename", ["train_b{}_i{}".format(batch_idx, i) for i in range(outputs["images"].size(0))]))

    def on_train_epoch_end(self, trainer, pl_module):
        if not self.train_vis_outputs:
            return
        # 1) 网格图（原逻辑）
        self._make_overlay_grid(
            self.train_vis_outputs,
            out_path=self.save_dir / f"debug_train_epoch{trainer.current_epoch}.png",
        )
        # 2) 新增：提示可视化
        self._save_prompt_visuals(
            outputs_list=self.train_vis_outputs,
            filenames_list=self.train_vis_filenames,
            split="train",
            epoch=trainer.current_epoch,
        )
        # 清空
        self.train_vis_outputs.clear()
        self.train_vis_filenames.clear()

    # ---------- 验证阶段 ----------
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.val_preds.append(outputs["pred_masks"].detach().cpu())
        self.val_gts.append(outputs["target_masks"].detach().cpu())

        if batch_idx < self.vis_batches:
            self.val_vis_outputs.append(outputs)
            self.val_vis_filenames.append(batch.get("filename", [f"val_b{batch_idx}_i{i}" for i in range(outputs["images"].size(0))]))

        self.val_image_names.extend(batch["filename"])

    def on_validation_epoch_end(self, trainer, pl_module):
        # 指标
        preds = torch.cat(self.val_preds, dim=0)
        gts = torch.cat(self.val_gts, dim=0)
        preds_bin = (preds > self.threshold).int()
        gts_bin = (gts > 0).int()

        miou   = self.metric_iou(preds_bin, gts_bin).item()
        f1     = self.metric_f1(preds_bin, gts_bin).item()
        prec   = self.metric_precision(preds_bin, gts_bin).item()
        recall = self.metric_recall(preds_bin, gts_bin).item()
        acc    = self.metric_accuracy(preds_bin, gts_bin).item()

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                trainer.current_epoch,
                f"{miou * 100:.2f}",
                f"{f1   * 100:.2f}",
                f"{prec * 100:.2f}",
                f"{recall * 100:.2f}",
                f"{acc   * 100:.2f}"
            ])

        # 保存原分辨率预测掩码（原逻辑）
        mask_root = self.save_dir / "val_pred_masks" / f"epoch_{trainer.current_epoch}"
        mask_root.mkdir(parents=True, exist_ok=True)

        for mask, name in zip(preds_bin, self.val_image_names):
            filename = Path(name).stem + ".png"
            try:
                with Image.open(name) as img:
                    orig_w, orig_h = img.size
            except Exception as e:
                print(f"[SavePredictionsCallback] Warning: cannot open {name}: {e}")
                orig_h, orig_w = mask.shape[-2:]

            restored_mask = unpad_and_resize_mask(mask.unsqueeze(0), (orig_h, orig_w))
            mask_arr = (restored_mask.squeeze(0).numpy() * 255).astype(np.uint8)
            Image.fromarray(mask_arr).save(mask_root / filename)

        # 网格图（原逻辑）
        if self.val_vis_outputs:
            self._make_overlay_grid(
                self.val_vis_outputs,
                out_path=self.save_dir / f"debug_val_epoch{trainer.current_epoch}.png",
            )

        # 新增：提示可视化
        if self.val_vis_outputs:
            self._save_prompt_visuals(
                outputs_list=self.val_vis_outputs,
                filenames_list=self.val_vis_filenames,
                split="val",
                epoch=trainer.current_epoch,
            )

        # 清空缓存
        self.val_preds.clear()
        self.val_gts.clear()
        self.val_vis_outputs.clear()
        self.val_vis_filenames.clear()
        self.val_image_names.clear()

    # ---------- 新增：分别保存“框提示 & 掩码提示” ----------
    @torch.no_grad()
    def _save_prompt_visuals(self, outputs_list, filenames_list, split: str, epoch: int):
        box_root  = self.box_dir  / split / f"epoch_{epoch}"
        mask_root = self.mask_dir / split / f"epoch_{epoch}"
        box_root.mkdir(parents=True, exist_ok=True)
        mask_root.mkdir(parents=True, exist_ok=True)

        for o, filelist in zip(outputs_list, filenames_list):
            imgs = o["images"]                     # (B, H, W, 3) 或 (B, 3, H, W)
            pred_boxes = o.get("pred_boxes", None) # 期望 (B, 4) 或 (B, N, 4)
            interim = o.get("interim_mask_output", None)  # (B, 1, H', W') logits

            B = imgs.size(0)
            for j in range(B):
                # 1) 尝试读取原图（优先用原分辨率可视化）
                name = filelist[j] if (j < len(filelist)) else f"{split}_e{epoch}_i{j}"
                stem = Path(name).stem
                try:
                    with Image.open(name) as im:
                        bg = np.array(im.convert("RGB"))
                        H, W = bg.shape[:2]
                        got_orig = True
                except Exception:
                    # 回退：用 batch 里的可视化图像
                    img_np = imgs[j].detach().cpu().numpy()
                    if img_np.ndim == 3 and img_np.shape[0] in (1, 3):  # CHW -> HWC
                        img_np = np.transpose(img_np, (1, 2, 0))
                    bg = img_np.astype(np.uint8)
                    H, W = bg.shape[:2]
                    got_orig = False

                # 2) 可视化框提示（pred_boxes）
                if pred_boxes is not None:
                    pb = pred_boxes[j].detach().cpu()
                    if pb.ndim == 1:     # (4,)
                        pb = pb.unsqueeze(0)
                    elif pb.ndim > 2:    # 兜底 reshape
                        pb = pb.reshape(-1, 4)

                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.imshow(bg)
                    for b in pb:
                        x1, y1, x2, y2 = b.numpy().tolist()
                        # 假设是相对坐标 xyxy；若是像素坐标请删除下面两行
                        x1 *= W; x2 *= W; y1 *= H; y2 *= H
                        rect = Rectangle((x1, y1), max(0.0, x2 - x1), max(0.0, y2 - y1),
                                         linewidth=2, edgecolor='red', facecolor='none')
                        ax.add_patch(rect)
                    ax.set_axis_off()
                    fig.savefig(box_root / f"{stem}_boxes.png", bbox_inches="tight", pad_inches=0)
                    plt.close(fig)

                # 3) 可视化掩码提示（interim_mask_output）
                if interim is not None:
                    m = interim[j]
                    if m.ndim == 3 and m.size(0) == 1:  # (1, H', W')
                        m = m[0]
                    m = torch.sigmoid(m.detach().cpu())  # logits -> prob

                    # 还原到原图尺寸
                    if got_orig:
                        m_res = unpad_and_resize_mask(m.unsqueeze(0), (H, W)).squeeze(0)
                    else:
                        # 插值到当前 bg 大小
                        m_res = F.interpolate(m.unsqueeze(0).unsqueeze(0),
                                              size=(H, W), mode="bilinear",
                                              align_corners=False).squeeze()

                    m_bin = (m_res > self.threshold).to(torch.uint8).numpy() * 255

                    # 保存二值掩码
                    Image.fromarray(m_bin).save(mask_root / f"{stem}_mask.png")

                    # 保存叠加图
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.imshow(bg)
                    ax.imshow(m_res.numpy(), alpha=0.5)  # 不指定颜色，默认 colormap
                    ax.set_axis_off()
                    fig.savefig(mask_root / f"{stem}_overlay.png", bbox_inches="tight", pad_inches=0)
                    plt.close(fig)

    # ---------- 你原先的网格可视化（保留） ----------
    def _make_overlay_grid(self, outputs, out_path: Path):
        pred = torch.cat([o["pred_masks"] for o in outputs], dim=0)
        gt = torch.cat([o["target_masks"] for o in outputs], dim=0)
        imgs = torch.cat([o["images"] for o in outputs], dim=0)

        pred_boxes_all = []
        gt_boxes_all = []

        for o in outputs:
            b = o["pred_masks"].size(0)
            pred_boxes = o.get("pred_boxes", None)
            gt_boxes = o.get("gt_boxes", None)

            if pred_boxes is not None:
                pred_boxes_all.append(pred_boxes.detach().cpu())
            else:
                pred_boxes_all.append(torch.zeros((b, 4)))

            if gt_boxes is not None:
                gt_boxes_all.append(gt_boxes.detach().cpu())
            else:
                gt_boxes_all.append(torch.zeros((b, 4)))

        pred_boxes_all = torch.cat(pred_boxes_all, dim=0)
        gt_boxes_all = torch.cat(gt_boxes_all, dim=0)

        n = pred.size(0)
        fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
        axes = axes if n > 1 else axes[None, ...]

        for i, (p, g, img, pred_box, gt_box) in enumerate(zip(pred, gt, imgs, pred_boxes_all, gt_boxes_all)):
            img_np = img.cpu().numpy().astype(np.uint8)
            if img_np.ndim == 3 and img_np.shape[0] in (1, 3):  # CHW -> HWC（容错）
                img_np = np.transpose(img_np, (1, 2, 0))
            H, W = img_np.shape[:2]

            # 左：GT
            img_gt = overlay_contours(img_np, g.cpu().numpy().astype(np.uint8))
            axes[i, 0].imshow(img_gt)
            axes[i, 0].imshow(g.cpu(), alpha=0.2)
            axes[i, 0].set_title("GT")
            axes[i, 0].axis("off")

            x1, y1, x2, y2 = gt_box.numpy()
            x1 *= W; x2 *= W; y1 *= H; y2 *= H
            rect_gt = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='green', facecolor='none')
            axes[i, 0].add_patch(rect_gt)

            # 右：Pred
            img_pred = overlay_contours(img_np, p.cpu().numpy().astype(np.uint8))
            axes[i, 1].imshow(img_pred)
            axes[i, 1].imshow(p.cpu(), alpha=0.2)
            axes[i, 1].set_title("Pred")
            axes[i, 1].axis("off")

            x1, y1, x2, y2 = pred_box.numpy()
            x1 *= W; x2 *= W; y1 *= H; y2 *= H
            rect_pred = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
            axes[i, 1].add_patch(rect_pred)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()



# 9.4更改
# class SavePredictionsCallback(pl.Callback):
#     def __init__(self, save_dir="outputs_b_prompt可视化", vis_batches=5, threshold=0.5):
#         super().__init__()
#         self.save_dir = Path(save_dir)
#         self.vis_batches = vis_batches
#         self.threshold = threshold
#
#         self.train_vis_outputs = []
#         self.val_vis_outputs = []
#         self.val_preds = []
#         self.val_gts = []
#         self.val_image_names = []
#
#         # Binary metrics
#         self.metric_iou = torchmetrics.JaccardIndex(task="binary")
#         self.metric_f1 = torchmetrics.F1Score(task="binary")
#         self.metric_precision = torchmetrics.Precision(task="binary")
#         self.metric_recall = torchmetrics.Recall(task="binary")
#         self.metric_accuracy = torchmetrics.Accuracy(task="binary")
#
#         self.csv_path = self.save_dir / "metrics.csv"
#         if not self.csv_path.exists():
#             self.csv_path.parent.mkdir(parents=True, exist_ok=True)
#             with open(self.csv_path, "w", newline="") as f:
#                 writer = csv.writer(f)
#                 writer.writerow(["epoch", "miou", "f1", "precision", "recall", "accuracy"])
#
#     # ---------- 训练阶段可视化 ----------
#     def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
#         if batch_idx < self.vis_batches:
#             self.train_vis_outputs.append(outputs)
#
#     def on_train_epoch_end(self, trainer, pl_module):
#         if not self.train_vis_outputs:
#             return
#         self._make_overlay_grid(
#             self.train_vis_outputs,
#             out_path=self.save_dir / f"debug_train_epoch{trainer.current_epoch}.png",
#         )
#         self.train_vis_outputs.clear()
#
#     # ---------- 验证阶段 ----------
#     def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
#         self.val_preds.append(outputs["pred_masks"].detach().cpu())
#         self.val_gts.append(outputs["target_masks"].detach().cpu())
#
#         if batch_idx < self.vis_batches:
#             self.val_vis_outputs.append(outputs)
#
#         self.val_image_names.extend(batch["filename"])
#
#     def on_validation_epoch_end(self, trainer, pl_module):
#         # 聚合批次
#         preds = torch.cat(self.val_preds, dim=0)
#         gts = torch.cat(self.val_gts, dim=0)
#         preds_bin = (preds > self.threshold).int()
#         gts_bin = (gts > 0).int()
#
#         # 计算指标
#         miou   = self.metric_iou(preds_bin, gts_bin).item()
#         f1     = self.metric_f1(preds_bin, gts_bin).item()
#         prec   = self.metric_precision(preds_bin, gts_bin).item()
#         recall = self.metric_recall(preds_bin, gts_bin).item()
#         acc    = self.metric_accuracy(preds_bin, gts_bin).item()
#
#         with open(self.csv_path, "a", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow([
#                 trainer.current_epoch,
#                 f"{miou * 100:.2f}", "",
#                 f"{f1   * 100:.2f}", "",
#                 f"{prec * 100:.2f}", "",
#                 f"{recall * 100:.2f}", "",
#                 f"{acc   * 100:.2f}"
#             ])
#
#         # -------- 新增：保存为原始分辨率的掩码 --------
#         mask_root = self.save_dir / "val_pred_masks" / f"epoch_{trainer.current_epoch}"
#         mask_root.mkdir(parents=True, exist_ok=True)
#
#         for mask, name in zip(preds_bin, self.val_image_names):
#             filename = Path(name).stem + ".png"
#
#             # 读取原图尺寸
#             try:
#                 with Image.open(name) as img:
#                     orig_w, orig_h = img.size
#             except Exception as e:
#                 print(f"[SavePredictionsCallback] Warning: cannot open {name}: {e}")
#                 orig_h, orig_w = mask.shape[-2:]  # fallback 到当前掩码尺寸
#
#             # 使用 unpad_and_resize_mask 恢复原始大小
#             restored_mask = unpad_and_resize_mask(mask.unsqueeze(0), (orig_h, orig_w))  # (1, H, W)
#
#             # 转为 PIL 格式保存
#             mask_arr = (restored_mask.squeeze(0).numpy() * 255).astype(np.uint8)
#             mask_img = Image.fromarray(mask_arr)
#             mask_img.save(mask_root / filename)
#         # ------------------------------------------------
#
#         # 可视化 Overlay
#         if self.val_vis_outputs:
#             self._make_overlay_grid(
#                 self.val_vis_outputs,
#                 out_path=self.save_dir / f"debug_val_epoch{trainer.current_epoch}.png",
#             )
#
#         # 清空缓存
#         self.val_preds.clear()
#         self.val_gts.clear()
#         self.val_vis_outputs.clear()
#         self.val_image_names.clear()
#
#     def _make_overlay_grid(self, outputs, out_path: Path):
#         pred = torch.cat([o["pred_masks"] for o in outputs], dim=0)
#         gt = torch.cat([o["target_masks"] for o in outputs], dim=0)
#         imgs = torch.cat([o["images"] for o in outputs], dim=0)
#
#         pred_boxes_all = []
#         gt_boxes_all = []
#
#         for o in outputs:
#             b = o["pred_masks"].size(0)
#
#             pred_boxes = o.get("pred_boxes", None)
#             gt_boxes = o.get("gt_boxes", None)
#
#             if pred_boxes is not None:
#                 pred_boxes_all.append(pred_boxes.detach().cpu())
#             else:
#                 pred_boxes_all.append(torch.zeros((b, 4)))
#
#             if gt_boxes is not None:
#                 gt_boxes_all.append(gt_boxes.detach().cpu())
#             else:
#                 gt_boxes_all.append(torch.zeros((b, 4)))
#
#         pred_boxes_all = torch.cat(pred_boxes_all, dim=0)
#         gt_boxes_all = torch.cat(gt_boxes_all, dim=0)
#
#         n = pred.size(0)
#         fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
#         axes = axes if n > 1 else axes[None, ...]
#
#         for i, (p, g, img, pred_box, gt_box) in enumerate(zip(pred, gt, imgs, pred_boxes_all, gt_boxes_all)):
#             img_np = img.cpu().numpy().astype(np.uint8)
#             H, W = img_np.shape[:2]
#
#             # ---- 左图：GT ----
#             img_gt = overlay_contours(img_np, g.cpu().numpy().astype(np.uint8))
#             axes[i, 0].imshow(img_gt)
#             axes[i, 0].imshow(g.cpu(), alpha=0.2)
#             axes[i, 0].set_title("GT")
#             axes[i, 0].axis("off")
#
#             x1, y1, x2, y2 = gt_box.numpy()
#             x1 *= W
#             x2 *= W
#             y1 *= H
#             y2 *= H
#             rect_gt = Rectangle((x1, y1), x2 - x1, y2 - y1,
#                                 linewidth=2, edgecolor='green', facecolor='none')
#             axes[i, 0].add_patch(rect_gt)
#
#             # ---- 右图：Pred ----
#             img_pred = overlay_contours(img_np, p.cpu().numpy().astype(np.uint8))
#             axes[i, 1].imshow(img_pred)
#             axes[i, 1].imshow(p.cpu(), alpha=0.2)
#             axes[i, 1].set_title("Pred")
#             axes[i, 1].axis("off")
#
#             x1, y1, x2, y2 = pred_box.numpy()
#             x1 *= W
#             x2 *= W
#             y1 *= H
#             y2 *= H
#             # ✅ 预测框用红色
#             rect_pred = Rectangle((x1, y1), x2 - x1, y2 - y1,
#                                   linewidth=2, edgecolor='red', facecolor='none')
#             axes[i, 1].add_patch(rect_pred)
#
#         out_path.parent.mkdir(parents=True, exist_ok=True)
#         plt.savefig(out_path, bbox_inches="tight")
#         plt.close()

    # ---------- 内部函数 ----------
    # def _make_overlay_grid(self, outputs, out_path: Path):
    #     pred = torch.cat([o["pred_masks"] for o in outputs], dim=0)
    #     gt   = torch.cat([o["target_masks"] for o in outputs], dim=0)
    #     imgs = torch.cat([o["images"]      for o in outputs], dim=0)
    #
    #     n = pred.size(0)
    #     fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    #     axes = axes if n > 1 else axes[None, ...]
    #
    #     for i, (p, g, img) in enumerate(zip(pred, gt, imgs)):
    #         img_gt = overlay_contours(img.cpu().numpy().astype(np.uint8),
    #                                   g.cpu().numpy().astype(np.uint8))
    #         img_pred = overlay_contours(img.cpu().numpy().astype(np.uint8),
    #                                     p.cpu().numpy().astype(np.uint8))
    #
    #         axes[i, 0].imshow(img_gt)
    #         axes[i, 0].imshow(g.cpu(), alpha=0.2)
    #         axes[i, 0].set_title("GT")
    #         axes[i, 1].imshow(img_pred)
    #         axes[i, 1].imshow(p.cpu(), alpha=0.2)
    #         axes[i, 1].set_title("Pred")
    #         axes[i, 0].axis("off")
    #         axes[i, 1].axis("off")
    #
    #     out_path.parent.mkdir(parents=True, exist_ok=True)
    #     plt.savefig(out_path, bbox_inches="tight")
    #     plt.close()


# import os
# import csv
# from pathlib import Path
# import torch
# import torchmetrics
# import pytorch_lightning as pl
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# class SavePredictionsCallback(pl.Callback):
#     def __init__(self, save_dir="outputs", vis_batches=5, threshold=0.5):
#         super().__init__()
#         self.save_dir = Path(save_dir)
#         self.vis_batches = vis_batches
#         self.threshold = threshold
#
#         self.train_vis_outputs = []
#         self.val_vis_outputs = []
#         self.val_preds = []
#         self.val_gts = []
#         self.val_image_names = []  # ✅ 新增：图像名缓存
#
#         # Binary metrics
#         self.metric_iou = torchmetrics.JaccardIndex(task="binary")
#         self.metric_f1 = torchmetrics.F1Score(task="binary")
#         self.metric_precision = torchmetrics.Precision(task="binary")
#         self.metric_recall = torchmetrics.Recall(task="binary")
#         self.metric_accuracy = torchmetrics.Accuracy(task="binary")
#
#         self.csv_path = self.save_dir / "metrics.csv"
#         if not self.csv_path.exists():
#             self.csv_path.parent.mkdir(parents=True, exist_ok=True)
#             with open(self.csv_path, "w", newline="") as f:
#                 writer = csv.writer(f)
#                 writer.writerow(["epoch", "miou", "f1", "precision", "recall", "accuracy"])
#
#     def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
#         if batch_idx < self.vis_batches:
#             self.train_vis_outputs.append(outputs)
#
#     def on_train_epoch_end(self, trainer, pl_module):
#         if not self.train_vis_outputs:
#             return
#
#         self._make_overlay_grid(
#             self.train_vis_outputs,
#             out_path=self.save_dir / f"debug_train_epoch{trainer.current_epoch}.png",
#         )
#         self.train_vis_outputs.clear()
#
#     def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
#         self.val_preds.append(outputs["pred_masks"].detach().cpu())
#         self.val_gts.append(outputs["target_masks"].detach().cpu())
#
#         if batch_idx < self.vis_batches:
#             self.val_vis_outputs.append(outputs)
#
#         self.val_image_names.extend(batch["filename"])  # ✅ 新增：收集文件名
#
#     def on_validation_epoch_end(self, trainer, pl_module):
#         preds = torch.cat(self.val_preds, dim=0)
#         gts = torch.cat(self.val_gts, dim=0)
#         preds_bin = (preds > self.threshold).int()
#         gts_bin = (gts > 0).int()
#
#         miou = self.metric_iou(preds_bin, gts_bin).item()
#         f1 = self.metric_f1(preds_bin, gts_bin).item()
#         prec = self.metric_precision(preds_bin, gts_bin).item()
#         recall = self.metric_recall(preds_bin, gts_bin).item()
#         acc = self.metric_accuracy(preds_bin, gts_bin).item()
#
#         with open(self.csv_path, "a", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow([
#                 trainer.current_epoch,
#                 f"{miou * 100:.2f}", "",
#                 f"{f1 * 100:.2f}", "",
#                 f"{prec * 100:.2f}", "",
#                 f"{recall * 100:.2f}", "",
#                 f"{acc * 100:.2f}"
#             ])
#
#         # ✅ 使用原始图像名保存预测掩码
#         mask_root = self.save_dir / "val_pred_masks" / f"epoch_{trainer.current_epoch}"
#         mask_root.mkdir(parents=True, exist_ok=True)
#         for mask, name in zip(preds_bin, self.val_image_names):
#             filename = Path(name).stem + ".png"  # 去除路径和扩展名
#             mask_arr = (mask.squeeze(0).numpy() * 255).astype(np.uint8)
#             Image.fromarray(mask_arr).save(mask_root / filename)
#
#         # 保存 Overlay 可视化
#         if self.val_vis_outputs:
#             self._make_overlay_grid(
#                 self.val_vis_outputs,
#                 out_path=self.save_dir / f"debug_val_epoch{trainer.current_epoch}.png",
#             )
#
#         # ✅ 清空缓存
#         self.val_preds.clear()
#         self.val_gts.clear()
#         self.val_vis_outputs.clear()
#         self.val_image_names.clear()  # 清空文件名缓存
#
#     def _make_overlay_grid(self, outputs, out_path: Path):
#         pred = torch.cat([o["pred_masks"] for o in outputs], dim=0)
#         gt = torch.cat([o["target_masks"] for o in outputs], dim=0)
#         imgs = torch.cat([o["images"] for o in outputs], dim=0)
#
#         n = pred.size(0)
#         fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
#         axes = axes if n > 1 else axes[None, ...]
#
#         for i, (p, g, img) in enumerate(zip(pred, gt, imgs)):
#             img_gt = overlay_contours(
#                 img.cpu().numpy().astype(np.uint8),
#                 g.cpu().numpy().astype(np.uint8)
#             )
#             img_pred = overlay_contours(
#                 img.cpu().numpy().astype(np.uint8),
#                 p.cpu().numpy().astype(np.uint8)
#             )
#
#             axes[i, 0].imshow(img_gt)
#             axes[i, 0].imshow(g.cpu(), alpha=0.2)
#             axes[i, 0].set_title("GT")
#             axes[i, 1].imshow(img_pred)
#             axes[i, 1].imshow(p.cpu(), alpha=0.2)
#             axes[i, 1].set_title("Pred")
#             axes[i, 0].axis("off")
#             axes[i, 1].axis("off")
#
#         out_path.parent.mkdir(parents=True, exist_ok=True)
#         plt.savefig(out_path, bbox_inches="tight")
#         plt.close()

# class SavePredictionsCallback(pl.Callback):
#     def __init__(self, save_dir="outputs", vis_batches=5, threshold=0.5):
#         super().__init__()
#         self.save_dir = Path(save_dir)
#         self.vis_batches = vis_batches
#         self.threshold = threshold
#
#         self.train_vis_outputs = []
#         self.val_vis_outputs = []
#         self.val_preds = []
#         self.val_gts = []
#
#         # Binary metrics
#         self.metric_iou = torchmetrics.JaccardIndex(task="binary")
#         self.metric_f1 = torchmetrics.F1Score(task="binary")
#         self.metric_precision = torchmetrics.Precision(task="binary")
#         self.metric_recall = torchmetrics.Recall(task="binary")
#         self.metric_accuracy = torchmetrics.Accuracy(task="binary")
#
#         self.csv_path = self.save_dir / "metrics.csv"
#         if not self.csv_path.exists():
#             self.csv_path.parent.mkdir(parents=True, exist_ok=True)
#             with open(self.csv_path, "w", newline="") as f:
#                 writer = csv.writer(f)
#                 writer.writerow(["epoch", "miou", "f1", "precision", "recall", "accuracy"])
#
#
#     def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
#         if batch_idx < self.vis_batches:
#             self.train_vis_outputs.append(outputs)c
#
#     def on_train_epoch_end(self, trainer, pl_module):
#         if not self.train_vis_outputs:
#             return
#
#         self._make_overlay_grid(
#             self.train_vis_outputs,
#             out_path=self.save_dir / f"debug_train_epoch{trainer.current_epoch}.png",
#         )
#         self.train_vis_outputs.clear()
#
#     def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
#         self.val_preds.append(outputs["pred_masks"].detach().cpu())
#         self.val_gts.append(outputs["target_masks"].detach().cpu())
#
#         if batch_idx < self.vis_batches:
#             self.val_vis_outputs.append(outputs)
#
#     def on_validation_epoch_end(self, trainer, pl_module):
#         preds = torch.cat(self.val_preds, dim=0)
#         gts = torch.cat(self.val_gts, dim=0)
#
#         preds_bin = (preds > self.threshold).int()
#         gts_bin = (gts > 0).int()
#
#         miou = self.metric_iou(preds_bin, gts_bin).item()
#         f1 = self.metric_f1(preds_bin, gts_bin).item()
#         prec = self.metric_precision(preds_bin, gts_bin).item()
#         recall = self.metric_recall(preds_bin, gts_bin).item()
#         acc = self.metric_accuracy(preds_bin, gts_bin).item()
#
#         with open(self.csv_path, "a", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow([
#                 trainer.current_epoch,
#                 f"{miou * 100:.2f}", "",  # mIoU
#                 f"{f1 * 100:.2f}", "",  # F1
#                 f"{prec * 100:.2f}", "",  # Precision
#                 f"{recall * 100:.2f}", "",  # Recall
#                 f"{acc * 100:.2f}"  # Accuracy（末尾不加空）
#             ])
#
#         # 保存预测掩码
#         mask_root = self.save_dir / "val_pred_masks" / f"epoch_{trainer.current_epoch}"
#         mask_root.mkdir(parents=True, exist_ok=True)
#         for idx, mask in enumerate(preds_bin):
#             mask_arr = (mask.squeeze(0).numpy() * 255).astype(np.uint8)
#             Image.fromarray(mask_arr).save(mask_root / f"sample_{idx}.png")
#
#
#         # 保存 Overlay 可视化
#         if self.val_vis_outputs:
#             self._make_overlay_grid(
#                 self.val_vis_outputs,
#                 out_path=self.save_dir / f"debug_val_epoch{trainer.current_epoch}.png",
#             )
#
#         self.val_preds.clear()
#         self.val_gts.clear()
#         self.val_vis_outputs.clear()
#
#     def _make_overlay_grid(self, outputs, out_path: Path):
#         pred = torch.cat([o["pred_masks"] for o in outputs], dim=0)
#         gt = torch.cat([o["target_masks"] for o in outputs], dim=0)
#         imgs = torch.cat([o["images"] for o in outputs], dim=0)
#
#         n = pred.size(0)
#         fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
#         axes = axes if n > 1 else axes[None, ...]
#
#         for i, (p, g, img) in enumerate(zip(pred, gt, imgs)):
#             img_gt = overlay_contours(
#                 img.cpu().numpy().astype(np.uint8),
#                 g.cpu().numpy().astype(np.uint8)
#             )
#             img_pred = overlay_contours(
#                 img.cpu().numpy().astype(np.uint8),
#                 p.cpu().numpy().astype(np.uint8)
#             )
#
#             axes[i, 0].imshow(img_gt)
#             axes[i, 0].imshow(g.cpu(), alpha=0.2)
#             axes[i, 0].set_title("GT")
#             axes[i, 1].imshow(img_pred)
#             axes[i, 1].imshow(p.cpu(), alpha=0.2)
#             axes[i, 1].set_title("Pred")
#             axes[i, 0].axis("off")
#             axes[i, 1].axis("off")
#
#         out_path.parent.mkdir(parents=True, exist_ok=True)
#         plt.savefig(out_path, bbox_inches="tight")
#         plt.close()



#
# class SavePredictionsCallback(pl.Callback):
#     """
#     A PyTorch Lightning callback to save and visualize predictions during training and validation.
#     """
#
#     def __init__(self):
#         self.val_outputs = []
#         self.train_outputs = []
#
#     def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
#         if batch_idx < 5:
#             self.train_outputs.append(outputs)
#
#         return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
#
#     def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
#         if batch_idx < 5:
#             self.val_outputs.append(outputs)
#
#         return super().on_validation_batch_end(
#             trainer, pl_module, outputs, batch, batch_idx
#         )
#
#     def on_validation_epoch_end(self, trainer, pl_module):
#         outputs = self.val_outputs
#         pred = torch.cat([o["pred_masks"] for o in outputs], dim=0)
#         gt = torch.cat([o["target_masks"] for o in outputs], dim=0)
#         images = torch.cat([o["images"] for o in outputs], dim=0)
#         fig, axes = plt.subplots(pred.size(0), 2, figsize=(2 * 4, pred.size(0) * 4))
#         if pred.size(0) == 1:
#             axes = axes[None, ...]
#
#         for i, (p, g, img) in enumerate(zip(pred, gt, images)):
#             img_gt = overlay_contours(
#                 img.cpu().numpy().astype(np.uint8), g.cpu().numpy().astype(np.uint8)
#             )
#             img_pred = overlay_contours(
#                 img.cpu().numpy().astype(np.uint8), p.cpu().numpy().astype(np.uint8)
#             )
#             axes[i, 0].imshow(img_gt)
#             axes[i, 0].imshow(g.cpu(), alpha=0.2)
#             axes[i, 1].imshow(img_pred)
#             axes[i, 1].imshow(p.cpu(), alpha=0.2)
#
#         plt.savefig("debug_val_progress.png", bbox_inches="tight")
#         plt.close()
#
#         self.val_outputs.clear()
#
#         return super().on_validation_epoch_end(trainer, pl_module)
#
#     def on_train_epoch_end(self, trainer, pl_module):
#         outputs = self.train_outputs
#         if len(outputs) > 0:
#             pred = torch.cat([o["pred_masks"] for o in outputs], dim=0)
#             gt = torch.cat([o["target_masks"] for o in outputs], dim=0)
#             images = torch.cat([o["images"] for o in outputs], dim=0)
#
#             fig, axes = plt.subplots(pred.size(0), 2, figsize=(2 * 4, pred.size(0) * 4))
#             if pred.size(0) == 1:
#                 axes = axes[None, ...]
#
#             for i, (p, g, img) in enumerate(zip(pred, gt, images)):
#                 img_gt = overlay_contours(
#                     img.cpu().numpy().astype(np.uint8), g.cpu().numpy().astype(np.uint8)
#                 )
#                 img_pred = overlay_contours(
#                     img.cpu().numpy().astype(np.uint8), p.cpu().numpy().astype(np.uint8)
#                 )
#                 axes[i, 0].imshow(img_gt)
#                 axes[i, 0].imshow(g.cpu(), alpha=0.2)
#                 axes[i, 1].imshow(img_pred)
#                 axes[i, 1].imshow(p.cpu(), alpha=0.2)
#
#                 # Contours
#
#             plt.savefig("debug_train_progress.png", bbox_inches="tight")
#             plt.close()
#
#             self.train_outputs.clear()
#
#         return super().on_train_epoch_end(trainer, pl_module)


def build_model(config):
    if "sam2" in config.image_encoder:
        return build_sam2rad(config)

    return build_samrad(config)


class SegmentationModule(torch.nn.Module):
    """
    Combines segment anything with learnable prompts.
    """

    def __init__(
        self,
        cfg,
        prompts: Dict[str, torch.nn.Parameter],
    ):
        super(SegmentationModule, self).__init__()
        self.model = build_model(cfg)
        # Sometimes use manual prompts only (box, mask, etc.) so that the predicted prompts align with manual prompts.
        self.model.prompt_sampler.p[0] = 0.0  # Learned prompts
        # If box or mask prompt is used during training, the model can be prompted to correct a prediction by providing a box or mask prompt (human-in-the-loop)
        self.model.prompt_sampler.p[2] = 1.0 # Box
        self.model.prompt_sampler.p[3] = 0.0  # Mask

        self.dataset_names = list(prompts.keys())
        self.learnable_prompts = torch.nn.ParameterDict(prompts)

        self.num_classes = self.learnable_prompts[cfg.dataset.name].size(0)

    def forward(self, batch, dataset_index=0, inference=False):
        """Get the learnable prompts for the dataset and make predictions"""
        imgs = batch["images"]
        prompts = self.learnable_prompts[self.dataset_names[dataset_index]].to(
            imgs.device
        )  # (num_classes, num_tokens, 256)

        outputs = self.model(batch, prompts, inference=inference)
        return outputs

class Learner(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn,
        lr: List[float],
        pixel_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.pixel_mean = torch.tensor(pixel_mean).view(1, 3, 1, 1)
        self.pixel_std = torch.tensor(pixel_std).view(1, 3, 1, 1)

        self.label_smoothing = 0.1
        self.image_size = (
            self.model.model.prompt_sampler.prompt_encoder.input_image_size[0]
        )

        self.train_dice_metric = AverageMeter()
        self.train_iou_metric = AverageMeter()
        self.val_dice_metric = AverageMeter()
        self.val_iou_metric = AverageMeter()

        self.iou = IoU(
            num_classes=self.model.num_classes + 1,
            ignore_index=0,  # ignore background
            average="micro",
        )

        self.dice = Dice(
            num_classes=self.model.num_classes + 1,
            ignore_index=0,  # ignore background
            average="micro",
        )

    def on_validation_epoch_end(self) -> None:
        self.val_dice_metric.reset()
        self.val_iou_metric.reset()
        return super().on_validation_epoch_end()

    def on_train_epoch_end(self) -> None:
        self.train_dice_metric.reset()
        self.train_iou_metric.reset()
        return super().on_train_epoch_end()

    @staticmethod
    def _masks_to_boxes_norm(bin_masks: torch.Tensor) -> torch.Tensor:
        """
        bin_masks: (N,1,H,W) 0/1
        return: (N,4) [0,1] xyxy
        """
        assert bin_masks.ndim == 4 and bin_masks.size(1) == 1
        N, _, H, W = bin_masks.shape
        boxes = torch.zeros((N, 4), device=bin_masks.device, dtype=torch.float32)
        m = (bin_masks > 0).squeeze(1)  # (N,H,W)

        # 行/列是否有前景
        rows = m.any(dim=2)  # (N,H)
        cols = m.any(dim=1)  # (N,W)

        # 找到每个样本的 ymin/ymax, xmin/xmax
        def first_true(x):  # x: (N,L) bool -> (N,)
            idx = torch.argmax(x.int(), dim=1)
            has = x.any(dim=1)
            idx[~has] = 0
            return idx, has

        def last_true(x):
            # 翻转后找 first_true 再还原
            xr = torch.flip(x, dims=[1])
            idxr, has = first_true(xr)
            idx = (x.size(1) - 1) - idxr
            return idx, has

        y1, has_r1 = first_true(rows)
        y2, has_r2 = last_true(rows)
        x1, has_c1 = first_true(cols)
        x2, has_c2 = last_true(cols)
        valid = has_r1 & has_r2 & has_c1 & has_c2

        # 像素坐标 -> 归一化 [0,1]
        boxes[:, 0] = x1.float() / max(W, 1)
        boxes[:, 1] = y1.float() / max(H, 1)
        boxes[:, 2] = (x2.float() + 1e-6) / max(W, 1)
        boxes[:, 3] = (y2.float() + 1e-6) / max(H, 1)

        # 无效样本保持 0
        boxes[~valid] = 0
        return boxes


    @staticmethod
    def generalized_box_iou_loss(pred_boxes, target_boxes, ignore_boxes=None):
        """
        Generalized box iou loss.
        pred_boxes: (B, 4) x1, y1, x2, y2
        target_boxes: (B, 4) x1, y1, x2, y2
        """
        if ignore_boxes is None:
            ignore_boxes = torch.zeros_like(pred_boxes).bool()
        loss = ops.generalized_box_iou_loss(pred_boxes, target_boxes, reduction="none")
        loss = (loss * (1 - ignore_boxes)).sum() / (1 - ignore_boxes).sum()
        return loss

    @staticmethod
    def reshape_inputs(batch):
        batch["boxes"] = batch["boxes"].reshape(-1, 4)
        batch["boxes_normalized"] = batch["boxes_normalized"].reshape(-1, 4)
        batch["ignore"] = batch["ignore"].reshape(-1)
        lr_masks = batch["low_res_masks"]
        batch["low_res_masks"] = lr_masks.reshape(
            -1, 1, lr_masks.size(2), lr_masks.size(3)
        )
        masks = batch["masks"]
        batch["masks"] = masks.reshape(-1, 1, masks.size(2), masks.size(3))
        return batch

    def training_step(self, batch, batch_idx):
        b, c, h, w = batch["masks"].shape
        batch = self.reshape_inputs(batch)
        gt = batch["masks"].float()  # (B*C, 1, H, W)
        # === 极简验证：偏差框 vs 真GT框 的 IoU ===
        with torch.no_grad():
            true_boxes_norm = self._masks_to_boxes_norm(gt)              # [0,1] xyxy, 来自当前 batch 的真 GT
            biased_boxes_norm = batch["boxes_normalized"]                # 训练用的“偏差框”监督
            ious = ops.box_iou(biased_boxes_norm, true_boxes_norm).diag()  # 一一对应 IoU

            valid = (true_boxes_norm[:, 2] > true_boxes_norm[:, 0]) & (true_boxes_norm[:, 3] > true_boxes_norm[:, 1])
            iou_mean = (ious[valid].mean().item() if valid.any() else 0.0)
            rate_same = ((ious >= 0.999) & valid).float().mean().item()

            self.log("bias/iou_mean_train", iou_mean, on_step=False, on_epoch=True)
            self.log("bias/rate_same_ge_0.999_train", rate_same, on_step=False, on_epoch=True)

        outputs = self.model(batch)  # 训练路径 inference=False
        pred = outputs["pred"]

        # segmentation loss
        loss_seg = self.loss_fn(pred, gt)  # (B,)
        is_non_empty = (gt.sum(dim=(1, 2, 3)) > 10).float()
        loss_seg = (loss_seg * is_non_empty).sum() / (is_non_empty.sum() + 1e-6)

        # bbox loss
        loss_box = 0.0
        if outputs["pred_boxes"] is not None:
            pred_boxes = outputs["pred_boxes"]              # [0,1] xyxy
            target_boxes = batch["boxes_normalized"]        # [0,1] xyxy
            ignore_boxes = batch["ignore"].float()
            loss_box = self.generalized_box_iou_loss(pred_boxes, target_boxes, ignore_boxes)

        # object score head
        object_score_logits = torch.clip(
            outputs["object_score_logits"].view(-1), -10, 10
        )
        if self.label_smoothing > 0:
            target = is_non_empty * (1 - self.label_smoothing) + self.label_smoothing / 2
        else:
            target = is_non_empty
        loss_object = F.binary_cross_entropy_with_logits(object_score_logits, target)

        # interim mask aux loss
        interim_mask_loss = 0.0
        if outputs["interim_mask_output"] is not None:
            interim_mask_loss = ops.sigmoid_focal_loss(
                outputs["interim_mask_output"], gt, reduction="none", alpha=0.6, gamma=3
            )
            interim_mask_loss = interim_mask_loss.mean(dim=(1, 2, 3))
            interim_mask_loss = (interim_mask_loss * is_non_empty).sum() / (is_non_empty.sum() + 1e-6)

        train_loss = loss_seg + loss_object + loss_box + 100 * interim_mask_loss

        # metrics（你的语义转换逻辑）
        _pred = pred.clone().detach()
        _pred[object_score_logits < 0] = -1
        pred_semantic = convert_to_semantic(_pred.detach().view(b, c, h, w))
        gt_semantic = convert_to_semantic(gt.view(b, c, h, w))

        self.train_dice_metric.update(self.dice(pred_semantic, gt_semantic), b)
        self.train_iou_metric.update(self.iou(pred_semantic, gt_semantic), b)

        self.log_dict(
            {
                "train_loss_seg": loss_seg,
                "interim_mask_loss": interim_mask_loss,
                "train_loss_box": loss_box,
                "train_loss_object": loss_object,
                "train_iou": self.train_iou_metric.get_avg(),
                "train_dice": self.train_dice_metric.get_avg(),
            },
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )

        return {
            "loss": train_loss,
            "iou": self.train_iou_metric.get_avg(),
            "dice": self.train_dice_metric.get_avg(),
            "confidence": object_score_logits,
            "images": self.denormalize(batch["images"]),
            "target_masks": gt_semantic,
            "pred_masks": pred_semantic,
            # 提示可视化需要的键（训练阶段也返回，便于保存）
            "pred_boxes": outputs.get("pred_boxes", None),
            "interim_mask_output": outputs.get("interim_mask_output", None),
            "gt_boxes": batch.get("boxes_normalized", None),
        }

    def denormalize(self, img):
        img = img * self.pixel_std.to(img.device) + self.pixel_mean.to(img.device)
        return (img * 255).type(torch.uint8).permute(0, 2, 3, 1).cpu()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # NOTE: 不要把 dataloader_idx 传进 model；只传 inference=True
        b, c, h, w = batch["masks"].shape
        batch = self.reshape_inputs(batch)
        gt = batch["masks"].float()  # (B, 1, H, W)
        # === 验证集同样记录 ===
        with torch.no_grad():
            true_boxes_norm = self._masks_to_boxes_norm(gt)
            biased_boxes_norm = batch["boxes_normalized"]
            ious = ops.box_iou(biased_boxes_norm, true_boxes_norm).diag()

            valid = (true_boxes_norm[:, 2] > true_boxes_norm[:, 0]) & (true_boxes_norm[:, 3] > true_boxes_norm[:, 1])
            iou_mean = (ious[valid].mean().item() if valid.any() else 0.0)
            rate_same = ((ious >= 0.999) & valid).float().mean().item()

            self.log("bias/iou_mean_val", iou_mean, on_step=False, on_epoch=True)
            self.log("bias/rate_same_ge_0.999_val", rate_same, on_step=False, on_epoch=True)

        outputs = self.model(batch, inference=False)  # <— 关键修正
        pred = outputs["pred"]

        # segmentation/object losses（用于日志）
        loss_seg = self.loss_fn(pred, gt)  # (B,)
        is_non_empty = (gt.sum(dim=(1, 2, 3)) > 1).float()
        loss_seg = (loss_seg * is_non_empty).sum() / (is_non_empty.sum() + 1e-6)

        object_score_logits = outputs["object_score_logits"].view(-1)
        loss_object = F.binary_cross_entropy_with_logits(object_score_logits, is_non_empty)

        # 语义图（与你训练一致）
        pred[object_score_logits < 0] = -1
        pred_semantic = convert_to_semantic(pred.detach().view(b, c, h, w))
        gt_semantic = convert_to_semantic(gt.view(b, c, h, w))

        self.val_dice_metric.update(self.dice(pred_semantic, gt_semantic), b)
        self.val_iou_metric.update(self.iou(pred_semantic, gt_semantic), b)

        self.log_dict(
            {
                "val_loss_seg": loss_seg,
                "val_loss_object": loss_object,
                "val_iou": self.val_iou_metric.get_avg(),
                "val_dice": self.val_dice_metric.get_avg(),
            },
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )

        # —— 关键：把提示输出也返回，给回调保存 —— #
        return {
            "images": self.denormalize(batch["images"]),
            "target_masks": gt_semantic,
            "pred_masks": pred_semantic,
            "confidence": object_score_logits,
            "pred_boxes": outputs.get("pred_boxes", None),                 # [0,1] xyxy
            "interim_mask_output": outputs.get("interim_mask_output", None),  # logits
            "gt_boxes": batch.get("boxes_normalized", None),               # [0,1] xyxy
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-4,
            betas=(0.9, 0.999),
            weight_decay=0.1,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.training.max_epochs, eta_min=1e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

# class Learner(pl.LightningModule):
#     def __init__(
#         self,
#         model: torch.nn.Module,
#         loss_fn,
#         lr: List[float],
#         pixel_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
#         pixel_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
#     ):
#         super().__init__()
#         self.model = model
#         self.loss_fn = loss_fn
#         self.lr = lr
#         self.pixel_mean = torch.tensor(pixel_mean).view(1, 3, 1, 1)
#         self.pixel_std = torch.tensor(pixel_std).view(1, 3, 1, 1)
#
#         self.label_smoothing = 0.1
#         self.image_size = (
#             self.model.model.prompt_sampler.prompt_encoder.input_image_size[0]
#         )
#
#         self.train_dice_metric = AverageMeter()
#         self.train_iou_metric = AverageMeter()
#         self.val_dice_metric = AverageMeter()
#         self.val_iou_metric = AverageMeter()
#
#         self.iou = IoU(
#             num_classes=self.model.num_classes + 1,
#             ignore_index=0,  # ignore background
#             average="micro",
#         )
#
#         self.dice = Dice(
#             num_classes=self.model.num_classes + 1,
#             ignore_index=0,  # ignore background
#             average="micro",
#         )
#
#     def on_validation_epoch_end(self) -> None:
#         self.val_dice_metric.reset()
#         self.val_iou_metric.reset()
#         return super().on_validation_epoch_end()
#
#     def on_train_epoch_end(self) -> None:
#         self.train_dice_metric.reset()
#         self.train_iou_metric.reset()
#         return super().on_train_epoch_end()
#
#     @staticmethod
#     def generalized_box_iou_loss(pred_boxes, target_boxes, ignore_boxes=None):
#         """
#         Generalized box iou loss.
#         pred_boxes: (B, 4) x1, y1, x2, y2
#         target_boxes: (B, 4) x1, y1, x2, y2
#         """
#         if ignore_boxes is None:
#             ignore_boxes = torch.zeros_like(pred_boxes).bool()
#         loss = ops.generalized_box_iou_loss(pred_boxes, target_boxes, reduction="none")
#         loss = (loss * (1 - ignore_boxes)).sum() / (1 - ignore_boxes).sum()
#
#         return loss
#
#     @staticmethod
#     def reshape_inputs(batch):
#         batch["boxes"] = batch["boxes"].reshape(-1, 4)
#         batch["boxes_normalized"] = batch["boxes_normalized"].reshape(-1, 4)
#         batch["ignore"] = batch["ignore"].reshape(-1)
#         lr_masks = batch["low_res_masks"]
#         batch["low_res_masks"] = lr_masks.reshape(
#             -1, 1, lr_masks.size(2), lr_masks.size(3)
#         )
#         masks = batch["masks"]
#         batch["masks"] = masks.reshape(-1, 1, masks.size(2), masks.size(3))
#
#         return batch
#
#     def training_step(self, batch, batch_idx):
#         b, c, h, w = batch["masks"].shape
#         batch = self.reshape_inputs(batch)
#         gt = batch["masks"].float()  # (B*C, 1, H, W)
#         outputs = self.model(batch)
#         pred = outputs["pred"]
#         loss_seg = self.loss_fn(pred, gt)  # (B,)
#         # Compute loss for non-empty masks only
#         is_non_empty = (gt.sum(dim=(1, 2, 3)) > 10).float()
#         loss_seg = (loss_seg * is_non_empty).sum() / (is_non_empty.sum() + 1e-6)
#         # Bounding box regression loss
#         loss_box = 0.0
#         if outputs["pred_boxes"] is not None:
#             pred_boxes = outputs["pred_boxes"]  # x1, y1, x2, y2
#             target_boxes = batch["boxes_normalized"]  # x1, y1, x2, y2
#             ignore_boxes = batch["ignore"].float()
#             loss_box = self.generalized_box_iou_loss(
#                 pred_boxes, target_boxes, ignore_boxes
#             )
#
#         # Object prediction head
#         object_score_logits = torch.clip(
#             outputs["object_score_logits"].view(-1), -10, 10
#         )
#         if self.label_smoothing > 0:
#             target = (
#                 is_non_empty * (1 - self.label_smoothing) + self.label_smoothing / 2
#             )
#
#         else:
#             target = is_non_empty
#
#         loss_object = F.binary_cross_entropy_with_logits(object_score_logits, target)
#
#         interim_mask_loss = 0.0
#         if outputs["interim_mask_output"] is not None:
#             interim_mask_loss = ops.sigmoid_focal_loss(
#                 outputs["interim_mask_output"], gt, reduction="none", alpha=0.6, gamma=3
#             )
#
#             interim_mask_loss = interim_mask_loss.mean(dim=(1, 2, 3))
#             interim_mask_loss = (interim_mask_loss * is_non_empty).sum() / (
#                 is_non_empty.sum() + 1e-6
#             )
#
#         train_loss = loss_seg + loss_object + loss_box + 100 * interim_mask_loss
#         # train_loss = (
#         #         loss_seg * 1.0 +  # Dice + 5*Focal
#         #         loss_object * 0.5 +
#         #         loss_box * 1.0 +
#         #         interim_mask_loss * 10.0
#         # )
#
#         # Compute metrics
#         _pred = pred.clone().detach()
#         _pred[object_score_logits < 0] = -1
#         pred_semantic = convert_to_semantic(_pred.detach().view(b, c, h, w))
#         gt_semantic = convert_to_semantic(gt.view(b, c, h, w))
#
#         self.train_dice_metric.update(self.dice(pred_semantic, gt_semantic), b)
#         self.train_iou_metric.update(self.iou(pred_semantic, gt_semantic), b)
#
#         self.log_dict(
#             {
#                 "train_loss_seg": loss_seg,
#                 "interim_mask_loss": interim_mask_loss,
#                 "train_loss_box": loss_box,
#                 "train_loss_object": loss_object,
#                 "train_iou": self.train_iou_metric.get_avg(),
#                 "train_dice": self.train_dice_metric.get_avg(),
#             },
#             prog_bar=True,
#             sync_dist=True,
#             on_step=False,
#             on_epoch=True,
#         )
#
#         return {
#             "loss": train_loss,
#             "iou": self.train_iou_metric.get_avg(),
#             "dice": self.train_dice_metric.get_avg(),
#             "confidence": object_score_logits,
#             "images": self.denormalize(batch["images"]),
#             "target_masks": gt_semantic,
#             "pred_masks": pred_semantic,
#             # DEBUG
#             "pred_boxes": outputs["pred_boxes"],
#             "interim_mask_output": outputs["interim_mask_output"],
#             "gt_boxes": batch["boxes_normalized"],
#         }
#
#     def denormalize(self, img):
#         img = img * self.pixel_std.to(img.device) + self.pixel_mean.to(img.device)
#         return (img * 255).type(torch.uint8).permute(0, 2, 3, 1).cpu()
#
#     def validation_step(self, batch, batch_idx, dataloader_idx=0):
#         print("batch keys:", batch.keys())
#
#         b, c, h, w = batch["masks"].shape
#         batch = self.reshape_inputs(batch)
#         gt = batch["masks"].float()  # (B, C, H, W)
#         outputs = self.model(batch, dataloader_idx, inference=True)
#         pred = outputs["pred"]
#         print(f"[DEBUG] pred shape: {pred.shape}, gt shape: {gt.shape}")
#         loss_seg = self.loss_fn(pred, gt)  # (B,)
#         # train on non-empty masks only
#         is_non_empty = (gt.sum(dim=(1, 2, 3)) > 1).float()
#         loss_seg = (loss_seg * is_non_empty).sum() / (is_non_empty.sum() + 1e-6)
#         object_score_logits = outputs["object_score_logits"].view(-1)
#         loss_object = F.binary_cross_entropy_with_logits(
#             object_score_logits, is_non_empty
#         )
#
#         pred[object_score_logits < 0] = -1
#         pred_semantic = convert_to_semantic(pred.detach().view(b, c, h, w))
#         gt_semantic = convert_to_semantic(gt.view(b, c, h, w))
#
#         self.val_dice_metric.update(self.dice(pred_semantic, gt_semantic), b)
#         self.val_iou_metric.update(self.iou(pred_semantic, gt_semantic), b)
#
#         # log the loss and metrics
#         self.log_dict(
#             {
#                 "val_loss_seg": loss_seg,
#                 "val_loss_object": loss_object,
#                 "val_iou": self.val_iou_metric.get_avg(),
#                 "val_dice": self.val_dice_metric.get_avg(),
#             },
#             prog_bar=True,
#             sync_dist=True,
#             on_step=False,
#             on_epoch=True,
#         )
#
#         return {
#             "images": self.denormalize(batch["images"]),
#             "target_masks": gt_semantic,
#             "pred_masks": pred_semantic,
#             "confidence": object_score_logits,
#         }
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(
#             filter(lambda p: p.requires_grad, self.model.parameters()),
#             lr=1e-4,
#             betas=(0.9, 0.999),
#             weight_decay=0.1,
#         )
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer, T_max=config.training.max_epochs, eta_min=1e-5
#         )
#
#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": {
#                 "scheduler": scheduler,
#                 "interval": "epoch",
#             },
#         }

import random
if __name__ == "__main__":
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        pl.seed_everything(seed, workers=True)
        # 强制cudnn确定性和关闭benchmark（速度会略慢，但最稳妥）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(seed=42)


    args = parser.parse_args()

    if args.config is None:
        args.config = r"D:\zhuhe\sam2rad\sam2rad\configs\landslide.yaml"  # 写你实际的 config 路径
    # if args.config is None:
    #     args.config = r"D:\zhuhe\sam2rad\sam2rad\configs\nodem.yaml"
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
        config = DotDict(config)
    print(config)
    # Register a custom dataset or use a default one, e.g., dataset_obj = DATASETS["default_segmentation"]
    dataset_obj = DATASETS[config.dataset.name]

    trn_ds, val_ds = dataset_obj.from_path(config.dataset)
    # Debug: faster validation
    val_ds = torch.utils.data.Subset(val_ds, range(0, 64))

    trn_dl = get_dataloaders(config.dataset, trn_ds)
    val_dl = get_dataloaders(config.dataset, val_ds)

    logger.info(f"Train dataset size: {len(trn_dl.dataset)}")
    logger.info(f"Validation dataset size: {len(val_dl.dataset)}")

    # Initialize learnable prompts for each dataset
    class_tokens = torch.nn.Parameter(
        torch.randn(
            config.dataset.num_classes,
            config.dataset.num_tokens,
            256,
        )
        / math.sqrt(256)
    )
    loss_fn = CompositeLoss(
        [
            partial(dice_loss, reduction="none"),
            partial(focal_loss, reduction="none", alpha=0.7, gamma=3),
        ],
        weights=[1.0, 10.0],
    )

    model = SegmentationModule(config, {config.dataset.name: class_tokens})
    for name, param in model.model.mask_decoder.named_parameters():
        param.requires_grad = True
    print("[Info] mask_decoder 已解冻参数")
    logger.info(model)
    termcolor.colored("Trainable parameters:", "red")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(termcolor.colored(f"{name} | {param.size()}", "red"))

    learner = Learner(model, loss_fn=loss_fn, lr=1e-4)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    tb_logger = TensorBoardLogger("bias_logs", name=config.dataset.name)
    # early_stop_callback = EarlyStopping(
    #     monitor="val_dice",  # 要监控的指标
    #     min_delta=0.001,  # 最小变化幅度，小于这个不认为有提升
    #     patience=10,  # 连续多少轮没有提升就停止训练
    #     verbose=True,
    #     mode="max",  # val_dice 越大越好
    # )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_dice",
        dirpath=r"D:\zhuhe\sam2rad\model_save\BJ"
        if config.get("save_path") is None
        else config.get("save_path"),
        save_last=True,
        filename="model_{epoch:02d}-{val_dice:.2f}",
        save_top_k=3,
        mode="max",
    )

    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        enable_progress_bar=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        logger=tb_logger,
        callbacks=[
            checkpoint_callback,
            lr_monitor,
            SavePredictionsCallback()
            # early_stop_callback  # ✅ 添加早停回调
        ],
        accelerator="gpu",
    )

    trainer.fit(
        learner,
        train_dataloaders=trn_dl,
        val_dataloaders=val_dl,
        # ckpt_path=config.training.get("resume"),
        ckpt_path=None
    )


# tensorboard --logdir bias_logs
