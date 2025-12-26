# import argparse
# import logging
# import math
# import os
# import sys
# from typing import Dict
#
# import matplotlib.pyplot as plt
# import torch
# import torch.nn.functional as F
# import yaml
# from monai.metrics import DiceMetric, MeanIoU
# from tqdm import tqdm
#
# from sam2rad import (
#     DATASETS,
#     DotDict,
#     build_sam2rad,
#     build_samrad,
#     convert_to_semantic,
# )
#
# logger = logging.getLogger("sam2rad")
#
# def build_model(config):
#     """
#     Choose tor build SAM or SAM2 model based on the config.
#     """
#     if "sam2" in config.image_encoder:
#         return build_sam2rad(config)
#
#     return build_samrad(config)
#
#
# parser = argparse.ArgumentParser(description="Evaluate a segmentation model")
# parser.add_argument(
#     "--config",
#     default=r"D:\zhuhe\sam2rad\sam2rad\configs\landslide.yaml",  # 你自己的 config 路径
#     help="Path to config file"
# )
#
#
# DEBUG = False
#
#
# class SegmentationModel(torch.nn.Module):
#     """
#     Combines segment anything with learnable prompts.
#     """
#
#     def __init__(
#         self,
#         config,
#         prompts: Dict[str, torch.nn.Parameter],
#     ):
#         super(SegmentationModel, self).__init__()
#         assert "sam_checkpoint" in config, "SAM checkpoint is required."
#         self.model = build_model(config)
#         logger.info(self.model)
#         self.dataset_names = list(prompts.keys())
#         self.num_classes = list(prompts.values())[0].shape[0]
#         self.learnable_prompts = torch.nn.ParameterDict(prompts)
#         self.model.prompt_sampler.p[0] = 1.0 # Learnable prompts
#         self.model.prompt_sampler.p[1] = 0
#         self.model.prompt_sampler.p[2] = 0.0  # Ground truth box prompts
#         self.model.prompt_sampler.p[3] = 0
#
#     def forward(self, batch, dataset_index):
#         """Get the learnable prompts for the dataset and make predictions"""
#         prompts = self.learnable_prompts[
#             self.dataset_names[dataset_index]
#         ]  # (num_classes, num_tokens, 256)
#
#         outputs = self.model(batch, prompts, inference=False)
#         return outputs
#
#     @property
#     def device(self):
#         return next(self.parameters()).device
#
#
# class Eval:
#     def __init__(self, model):
#         self.model = model
#         self.image_size = (
#             self.model.model.prompt_sampler.prompt_encoder.input_image_size[0]
#         )
#         self.dice_score = 0.0
#         self.iou_score = 0.0
#         self.count = 0
#
#         self.num_classes = self.model.num_classes
#         self.dice_metric = DiceMetric(
#             include_background=False,
#             reduction="mean_batch",
#             get_not_nans=False,
#             ignore_empty=True,
#         )
#         self.iou_metric = MeanIoU(
#             include_background=False,
#             reduction="mean_batch",
#             get_not_nans=False,
#             ignore_empty=True,
#         )
#
#     def one_hot(self, masks):
#         return (
#             F.one_hot(masks.long(), num_classes=self.model.num_classes + 1)
#             .permute(0, 3, 1, 2)
#             .float()
#         )
#
#     def reset(self):
#         self.dice_metric.reset()
#         self.iou_metric.reset()
#
#     @staticmethod
#     def post_process(pred, input_size, original_size):
#         pred = pred[:, :, : input_size[0], : input_size[1]]
#         return F.interpolate(pred, original_size, mode="bilinear", align_corners=False)
#
#     def eval_batch(self, batch):
#         images = batch["images"].to(self.model.device)
#         gt = batch["masks"].to(self.model.device)
#         boxes = batch["boxes"].to(self.model.device).view(-1, 4)
#         filename = batch["filename"]
#
#         # 下面两行可选：原图和 input_size（如果你有用原图做可视化或还原大小）
#         img_orig = images.cpu().permute(0, 2, 3, 1).numpy()  # for visualization only
#         input_size = images.shape[2:]  # (H, W)
#
#         images, gt, boxes = (
#             images.to(self.model.device),
#             gt.to(self.model.device),
#             boxes.to(self.model.device).view(-1, 4),
#         )
#         _, num_classes, *original_size = gt.shape
#         w_factor, h_factor = (
#             input_size[1] / original_size[1],
#             input_size[0] / original_size[0],
#         )
#         boxes_input = boxes * torch.tensor(
#             [w_factor, h_factor, w_factor, h_factor], device=boxes.device
#         )
#         outputs = self.model({"images": images, "masks": gt, "boxes": boxes_input}, 0)
#         pred = outputs["pred"]
#         pred = self.post_process(pred, input_size, original_size).view(
#             -1, num_classes, *original_size
#         )
#         pred = convert_to_semantic(pred)
#         gt = convert_to_semantic(gt)
#
#         if DEBUG:
#             boxes = boxes.cpu()
#             plt.subplot(1, 2, 1)
#
#             plt.imshow(img_orig[0])
#             plt.imshow(gt[0].cpu(), alpha=0.5)
#             plt.subplot(1, 2, 2)
#             plt.imshow(img_orig[0])
#             plt.imshow(pred[0].cpu(), alpha=0.5)
#             plt.savefig("eval_prompt_learner_debug.png")
#             plt.close()
#             import pdb
#
#             pdb.set_trace()
#
#         # Calculate dice and iou scores
#         gt_onehot = self.one_hot(gt)
#         pred_onehot = self.one_hot(pred)
#         self.dice_metric(pred_onehot, gt_onehot)
#         self.iou_metric(gt_onehot, pred_onehot)
#
#     @torch.no_grad()
#     def eval(self, dataloader):
#         self.reset()
#         with tqdm(dataloader, total=len(dataloader)) as pbar:
#             for batch in dataloader:
#                 self.eval_batch(batch)
#                 pbar.set_description(
#                     f"Dice: {self.dice_metric.aggregate(reduction='none').nanmean(0).nanmean(0).item():.4f}, IoU: {self.iou_metric.aggregate(reduction='none').nanmean(0).nanmean(0).item():.4f}"
#                 )
#                 pbar.update(1)
#
#
# if __name__ == "__main__":
#     import logging
#     import sys
#
#     logging.basicConfig(
#         level=logging.INFO,
#         handlers=[logging.StreamHandler(sys.stdout)]
#     )
#     logger = logging.getLogger("sam2rad")
#
#     args = parser.parse_args()
#
#     with open(args.config) as f:
#         config = DotDict.from_dict(yaml.safe_load(f))
#
#     dataset_name = config.dataset.name
#     train_val_tuple = DATASETS.get(dataset_name, DATASETS["landslide"]).from_path(
#         config.dataset, mode="Train"
#     )
#     train_ds, val_ds = train_val_tuple
#     ds = val_ds  # 使用验证集作为推理集
#
#     tst_dl = torch.utils.data.DataLoader(
#         ds,
#         batch_size=1,  # Images can be of different size
#         num_workers=config.get("num_workers", 1),
#         pin_memory=True,
#         shuffle=False,
#     )
#
#     class_tokens = torch.nn.Parameter(
#         torch.randn(
#             config.dataset.num_classes,
#             config.dataset.num_tokens,
#             256,
#         )
#         / math.sqrt(256)
#     )
#
#     learnable_prompts = {config.dataset.name: class_tokens}
#     logger.info(f"Test dataset size: {len(tst_dl.dataset)}")
#     model = SegmentationModel(config, learnable_prompts)
#     # Load checkpoint
#     checkpoint = torch.load(config.inference.checkpoint_path, map_location="cpu")
#     epoch = checkpoint["epoch"]
#     checkpoint = checkpoint["state_dict"]
#     checkpoint = {k[len("model.") :]: v for k, v in checkpoint.items()}
#     logger.info(model.load_state_dict(checkpoint))
#
#     model = model.to("cuda:0")
#     model.eval()
#
#     evaluator = Eval(model)
#
#     evaluator.eval(tst_dl)
#
#     # Prepare the output directory
#     output_dir = os.path.join(
#         os.path.dirname(config.sam_checkpoint),
#         "eval_results",
#     )
#     os.makedirs(output_dir, exist_ok=True)
#     output_file = os.path.join(output_dir, "eval_results.txt")
#
#     # Collect the per-class Dice and IoU scores
#     dice_scores = evaluator.dice_metric.aggregate("none").nanmean(0)
#     iou_scores = evaluator.iou_metric.aggregate("none").nanmean(0)
#
#     # Collect the average Dice and IoU scores
#     avg_dice = dice_scores.nanmean()
#     avg_iou = iou_scores.nanmean()
#
#     # Prepare the results as a formatted string
#     results_str = "=== Evaluation Results ===\n\n"
#     results_str += "Configuration:\n"
#     results_str += "\nPer-Class Metrics:\n"
#
#     # Format the per-class metrics
#     results_str += "{:<10} {:>10} {:>10}\n".format("Class", "Dice", "IoU")
#     results_str += "-" * 32 + "\n"
#     for i, (dice_score, iou_score) in enumerate(zip(dice_scores, iou_scores)):
#         results_str += "{:<10} {:>10.4f} {:>10.4f}\n".format(
#             f"Class {i}", dice_score.item(), iou_score.item()
#         )
#
#     # Add average metrics
#     results_str += "\nAverage Metrics:\n"
#     results_str += "-" * 32 + "\n"
#     results_str += "{:<10} {:>10.4f} {:>10.4f}\n".format("Average", avg_dice, avg_iou)
#
#     logger.info(results_str)
#
#     results_str = yaml.dump(config) + results_str
#
#     # Topk images with the lowest Dice score
#     dice_scores_per_img = evaluator.dice_metric.aggregate("none").nanmean(dim=1)
#     values, indices = torch.topk(dice_scores_per_img, k=5, largest=False)
#     logger.info("Top 5 images with the least Dice score:")
#     results_str += "\nTop 5 images with the least Dice score:\n"
#     for dice, idx in zip(values, indices):
#         logger.info(f"{tst_dl.dataset.img_files[idx.item()]}, Dice: {dice.item()}")
#         results_str += f"{tst_dl.dataset.img_files[idx.item()]}, Dice: {dice.item()}\n"
#
#     # Write the results to the output file
#     with open(output_file, encoding="utf-8", mode="w") as f:
#         f.write(results_str)
#         f.write("\n")
#
#     logger.info("Results saved to %s" % output_file)



# import argparse
# import logging
# import math
# import os
# import sys
# from typing import Dict
#
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import torch.nn.functional as F
# import yaml
# from monai.metrics import DiceMetric, MeanIoU, ConfusionMatrixMetric
# from tqdm import tqdm
#
# from sam2rad import (
#     DATASETS,
#     DotDict,
#     build_sam2rad,
#     build_samrad,
#     convert_to_semantic,
# )
#
# logger = logging.getLogger("sam2rad")
#
#
# def build_model(config):
#     if "sam2" in config.image_encoder:
#         return build_sam2rad(config)
#     return build_samrad(config)
#
#
# parser = argparse.ArgumentParser(description="Evaluate a segmentation model")
# parser.add_argument(
#     "--config",
#     default=r"D:\zhuhe\sam2rad\sam2rad\configs\landslide.yaml",
#     help="Path to config file",
# )
#
# DEBUG = False
#
#
# class SegmentationModel(torch.nn.Module):
#     def __init__(self, config, prompts: Dict[str, torch.nn.Parameter]):
#         super().__init__()
#         assert "sam_checkpoint" in config, "SAM checkpoint is required."
#         self.model = build_model(config)
#         logger.info(self.model)
#
#         self.dataset_names = list(prompts.keys())
#         self.num_classes = list(prompts.values())[0].shape[0]
#         self.learnable_prompts = torch.nn.ParameterDict(prompts)
#
#         self.model.prompt_sampler.p[0] = 1.0
#         self.model.prompt_sampler.p[1] = 0.0
#         self.model.prompt_sampler.p[2] = 0.0
#         self.model.prompt_sampler.p[3] = 0.0
#
#     def forward(self, batch, dataset_index: int):
#         prompts = self.learnable_prompts[self.dataset_names[dataset_index]]
#         outputs = self.model(batch, prompts, inference=False)
#         return outputs
#
#     @property
#     def device(self):
#         return next(self.parameters()).device
#
#
# class Eval:
#     def __init__(self, model: SegmentationModel):
#         self.model = model
#         self.image_size = (
#             self.model.model.prompt_sampler.prompt_encoder.input_image_size[0]
#         )
#         self.num_classes = self.model.num_classes
#
#         self.dice_metric = DiceMetric(
#             include_background=False,
#             reduction="mean_batch",
#             get_not_nans=False,
#             ignore_empty=True,
#         )
#         self.iou_metric = MeanIoU(
#             include_background=False,
#             reduction="mean_batch",
#             get_not_nans=False,
#             ignore_empty=True,
#         )
#         self.confusion_metric = ConfusionMatrixMetric(
#             include_background=False,
#             reduction="mean_batch",
#             metric_name=["precision", "recall", "accuracy", "f1_score"],
#             compute_sample=False,
#         )
#         self.vis_output_dir = os.path.join(r"D:\zhuhe\sam2rad\sam2rad\weights\eval_results")
#         os.makedirs(self.vis_output_dir, exist_ok=True)
#         self.counter = 0
#
#     def one_hot(self, masks: torch.Tensor) -> torch.Tensor:
#         return (
#             F.one_hot(masks.long(), num_classes=self.num_classes + 1)
#             .permute(0, 3, 1, 2)
#             .float()
#         )
#
#     def reset(self):
#         self.dice_metric.reset()
#         self.iou_metric.reset()
#         self.confusion_metric.reset()
#
#     @staticmethod
#     def post_process(pred: torch.Tensor, input_size, original_size):
#         pred = pred[:, :, : input_size[0], : input_size[1]]
#         return F.interpolate(pred, original_size, mode="bilinear", align_corners=False)
#
#     def eval_batch(self, batch):
#         images = batch["images"].to(self.model.device)
#         gt = batch["masks"].to(self.model.device)
#         boxes = batch["boxes"].to(self.model.device).view(-1, 4)
#         filenames = batch["filename"]
#
#         img_orig = images.cpu().permute(0, 2, 3, 1).numpy()
#
#         _, _, h_orig, w_orig = images.shape
#         input_size = (h_orig, w_orig)
#
#         _, num_classes, *original_size = gt.shape
#         w_factor, h_factor = (
#             input_size[1] / original_size[1],
#             input_size[0] / original_size[0],
#         )
#         boxes_input = boxes * torch.tensor(
#             [w_factor, h_factor, w_factor, h_factor], device=boxes.device
#         )
#
#         outputs = self.model({"images": images, "masks": gt, "boxes": boxes_input}, 0)
#         pred = outputs["pred"]
#         pred = (
#             self.post_process(pred, input_size, original_size)
#             .view(-1, num_classes, *original_size)
#         )
#         pred = convert_to_semantic(pred)
#         gt = convert_to_semantic(gt)
#
#         gt_onehot = self.one_hot(gt)
#         pred_onehot = self.one_hot(pred)
#
#         self.dice_metric(pred_onehot, gt_onehot)
#         self.iou_metric(gt_onehot, pred_onehot)
#         self.confusion_metric(pred_onehot, gt_onehot)
#
#         # save visualization
#         for i in range(len(pred)):
#             img = img_orig[i]
#             gt_mask = gt[i].cpu().numpy()
#             pred_mask = pred[i].cpu().numpy()
#
#             fig, axs = plt.subplots(1, 3, figsize=(12, 4))
#             axs[0].imshow(img.astype(np.uint8))
#             axs[0].set_title("Image")
#             axs[1].imshow(gt_mask, cmap="jet", alpha=0.8)
#             axs[1].set_title("GT")
#             axs[2].imshow(pred_mask, cmap="jet", alpha=0.8)
#             axs[2].set_title("Prediction")
#
#             for ax in axs:
#                 ax.axis("off")
#             plt.tight_layout()
#
#             vis_name = os.path.basename(filenames[i]) if isinstance(filenames[i], str) else f"img_{self.counter}.png"
#             save_path = os.path.join(self.vis_output_dir, vis_name)
#             plt.savefig(save_path, bbox_inches="tight")
#             plt.close()
#             self.counter += 1
#
#     @torch.no_grad()
#     def eval(self, dataloader):
#         self.reset()
#         with tqdm(dataloader, total=len(dataloader)) as pbar:
#             for batch in dataloader:
#                 self.eval_batch(batch)
#                 running_dice = (
#                     self.dice_metric.aggregate(reduction="none").nanmean(0).nanmean(0)
#                 )
#                 running_iou = (
#                     self.iou_metric.aggregate(reduction="none").nanmean(0).nanmean(0)
#                 )
#                 pbar.set_description(
#                     f"Dice: {running_dice:.4f}, IoU: {running_iou:.4f}"
#                 )
#                 pbar.update(1)
#
#
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
#
#     args = parser.parse_args()
#     with open(args.config, "r", encoding="utf-8") as f:
#         config = DotDict.from_dict(yaml.safe_load(f))
#
#     dataset_name = config.dataset.name
#     train_val_tuple = DATASETS.get(dataset_name, DATASETS["landslide"]).from_path(
#         config.dataset, mode="Train"
#     )
#     _, val_ds = train_val_tuple
#
#     tst_dl = torch.utils.data.DataLoader(
#         val_ds,
#         batch_size=1,
#         num_workers=config.get("num_workers", 1),
#         pin_memory=True,
#         shuffle=False,
#     )
#
#     class_tokens = torch.nn.Parameter(
#         torch.randn(
#             config.dataset.num_classes, config.dataset.num_tokens, 256
#         ) / math.sqrt(256)
#     )
#     learnable_prompts = {config.dataset.name: class_tokens}
#
#     logger.info(f"Validation set size: {len(tst_dl.dataset)}")
#
#     model = SegmentationModel(config, learnable_prompts)
#
#     checkpoint = torch.load(config.inference.checkpoint_path, map_location="cpu")
#     epoch = checkpoint.get("epoch", "-")
#     state = {k[len("model."):]: v for k, v in checkpoint["state_dict"].items()}
#     logger.info(model.load_state_dict(state))
#
#     model = model.to("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.eval()
#
#     evaluator = Eval(model)
#     evaluator.eval(tst_dl)
#
#     dice_scores = evaluator.dice_metric.aggregate(reduction="none").nanmean(0)
#     iou_scores = evaluator.iou_metric.aggregate(reduction="none").nanmean(0)
#
#     confusion_metrics = evaluator.confusion_metric.aggregate(reduction="none")
#     precision_scores = confusion_metrics[0].nanmean(0)
#     recall_scores    = confusion_metrics[1].nanmean(0)
#     acc_scores       = confusion_metrics[2].nanmean(0)
#     f1_scores        = confusion_metrics[3].nanmean(0)
#
#     avg_dice = dice_scores.nanmean()
#     avg_iou = iou_scores.nanmean()
#     avg_precision = precision_scores.nanmean()
#     avg_recall = recall_scores.nanmean()
#     avg_acc = acc_scores.nanmean()
#     avg_f1 = f1_scores.nanmean()
#
#     output_dir = os.path.join(os.path.dirname(config.sam_checkpoint), "eval_results")
#     os.makedirs(output_dir, exist_ok=True)
#     output_file = os.path.join(output_dir, "eval_results.txt")
#
#     results_str = "=== Evaluation Results ===\n\n"
#     results_str += "Configuration:\n" + yaml.dump(config) + "\n"
#     results_str += "Per-Class Metrics (skip background)\n"
#     header = (
#         f"{'Class':<10}{'Dice':>10}{'IoU':>10}{'F1':>10}{'Prec':>10}{'Recall':>10}{'Acc':>10}\n"
#     )
#     results_str += header
#     results_str += "-" * len(header) + "\n"
#     for i in range(len(dice_scores)):
#         results_str += (
#             f"Class {i:<4}{dice_scores[i]:>10.4f}{iou_scores[i]:>10.4f}{f1_scores[i]:>10.4f}"
#             f"{precision_scores[i]:>10.4f}{recall_scores[i]:>10.4f}{acc_scores[i]:>10.4f}\n"
#         )
#
#     results_str += "\nAverage Metrics\n"
#     results_str += "-" * len(header) + "\n"
#     results_str += (
#         f"Average  {avg_dice:>10.4f}{avg_iou:>10.4f}{avg_f1:>10.4f}{avg_precision:>10.4f}"
#         f"{avg_recall:>10.4f}{avg_acc:>10.4f}\n"
#     )
#
#     dice_scores_per_img = evaluator.dice_metric.aggregate("none").nanmean(dim=1)
#     values, indices = torch.topk(dice_scores_per_img, k=5, largest=False)
#     results_str += "\nTop 5 images with the lowest Dice score:\n"
#     for dice, idx in zip(values, indices):
#         img_path = tst_dl.dataset.img_files[idx.item()]
#         results_str += f"{img_path}, Dice: {dice.item():.4f}\n"
#
#     with open(output_file, "w", encoding="utf-8") as f:
#         f.write(results_str)
#
#     logger.info("Results saved to %s", output_file)


# -*- coding: utf-8 -*-
import argparse
import logging
import math
import os
import sys
from typing import Dict

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import yaml
import numpy as np
from PIL import Image

from sam2rad import (
    DATASETS,           # 不再用，但保留导入以兼容工程结构
    DotDict,
    build_sam2rad,
    build_samrad,
)

logger = logging.getLogger("sam2rad")

def build_model(config):
    """Choose to build SAM or SAM2 model based on the config."""
    if "sam2" in config.image_encoder:
        return build_sam2rad(config)
    return build_samrad(config)


parser = argparse.ArgumentParser(description="Infer on tiles and save masks")
parser.add_argument(
    "--config",
    default=r"D:\zhuhe\sam2rad\sam2rad\configs\landslide.yaml",  # ← 改为你的 config
    help="Path to config file"
)
parser.add_argument(
    "--tiles_dir",
    default=r"D:\zhuhe\sam2rad\sam2rad\datasets\TL6\test256",  # ← 你的 tiles 目录
    help="Directory containing cropped tile images"
)
parser.add_argument(
    "--output_dir",
    default=r"D:\zhuhe\sam2rad\sam2rad\tl6\256\pred_masks",  # ← 输出目录
    help="Directory to save predicted masks"
)
parser.add_argument(
    "--checkpoint",
    default=r"D:\zhuhe\sam2rad\model_save\sw\model_epoch=118-val_dice=0.82.ckpt",  # ← 你的权重
    help="Path to model checkpoint"
)

DEBUG = False


class SegmentationModel(torch.nn.Module):
    """
    Combines segment anything with learnable prompts.
    """

    def __init__(
        self,
        config,
        prompts: Dict[str, torch.nn.Parameter],
    ):
        super(SegmentationModel, self).__init__()
        assert "sam_checkpoint" in config, "SAM checkpoint is required."
        self.model = build_model(config)
        logger.info(self.model)
        self.dataset_names = list(prompts.keys())
        self.num_classes = list(prompts.values())[0].shape[0]
        self.learnable_prompts = torch.nn.ParameterDict(prompts)
        # 只使用可学习提示
        self.model.prompt_sampler.p[0] = 1.0 # Learnable prompts
        self.model.prompt_sampler.p[1] = 0.0
        self.model.prompt_sampler.p[2] = 0.0  # Ground truth box prompts
        self.model.prompt_sampler.p[3] = 0.0

    def forward(self, batch, dataset_index, inference=True):
        """Inference path: only images are required."""
        prompts = self.learnable_prompts[
            self.dataset_names[dataset_index]
        ]  # (num_classes, num_tokens, 256)
        outputs = self.model(batch, prompts, inference=inference)  # ← 推理分支
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger("sam2rad")

    args = parser.parse_args()

    # 1) 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        config = DotDict.from_dict(yaml.safe_load(f))

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # 2) 准备可学习 prompts（形状需与训练一致）
    class_tokens = torch.nn.Parameter(
        torch.randn(
            config.dataset.num_classes,   # 如 2（滑坡/非滑坡）
            config.dataset.num_tokens,    # 与训练一致
            256,
        ) / math.sqrt(256)
    )
    learnable_prompts = {config.dataset.name: class_tokens}

    # 3) 构建模型并加载权重（兼容多种保存格式）
    model = SegmentationModel(config, learnable_prompts)
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    def _strip_prefix(state):
        return { (k[6:] if k.startswith("model.") else k): v for k, v in state.items() }

    loaded = False
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        msg = model.load_state_dict(_strip_prefix(ckpt["state_dict"]), strict=False)
        loaded = True
        print("==> load_state_dict (state_dict) done")
        print(" - missing:", len(msg.missing_keys))
        if msg.missing_keys[:1]: print("   e.g.", msg.missing_keys[:3])
        print(" - unexpected:", len(msg.unexpected_keys))
        if msg.unexpected_keys[:1]: print("   e.g.", msg.unexpected_keys[:3])
        # 若 ckpt 内含 learnable_prompts，可在此按键名覆盖：
        for k, v in ckpt["state_dict"].items():
            if "learnable_prompts" in k and config.dataset.name in k:
                with torch.no_grad():
                    model.learnable_prompts[config.dataset.name].copy_(v)
                print("Loaded learnable prompts:", k)
                break
    elif isinstance(ckpt, dict):
        msg = model.load_state_dict(_strip_prefix(ckpt), strict=False)
        loaded = True
        print("==> load_state_dict (raw dict) done")
        print(" - missing:", len(msg.missing_keys))
        if msg.missing_keys[:1]: print("   e.g.", msg.missing_keys[:3])
        print(" - unexpected:", len(msg.unexpected_keys))
        if msg.unexpected_keys[:1]: print("   e.g.", msg.unexpected_keys[:3])
        for k, v in ckpt.items():
            if isinstance(k, str) and "learnable_prompts" in k and config.dataset.name in k:
                with torch.no_grad():
                    model.learnable_prompts[config.dataset.name].copy_(v)
                print("Loaded learnable prompts:", k)
                break
    else:
        # 保存的是整个模型对象（少见）
        model = ckpt
        loaded = True
        print("==> loaded full model object")

    model = model.to(DEVICE).eval()

    # 4) 读取期望输入尺寸并设定预处理（与训练一致）
    expected_sz = model.model.prompt_sampler.prompt_encoder.input_image_size[0]  # 常见为 1024
    transform = T.Compose([
        T.Resize((expected_sz, expected_sz), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        # 如果训练未做 ImageNet 标准化，请改成训练时的 mean/std，或去掉 Normalize
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    # 5) 遍历 tiles 做推理并保存 mask
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    with torch.no_grad():
        for fname in os.listdir(args.tiles_dir):
            if not fname.lower().endswith(exts):
                continue
            fpath = os.path.join(args.tiles_dir, fname)

            img = Image.open(fpath).convert("RGB")
            H, W = img.size[1], img.size[0]

            x = transform(img).unsqueeze(0).to(DEVICE)
            outputs = model({"images": x}, dataset_index=0, inference=True)
            logits = outputs["pred"]  # (B, C, H', W')

            # 还原到原 tile 尺寸
            logits_up = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)

            # 根据通道数选择后处理
            if logits_up.shape[1] == 1:
                # 二分类(单通道)：sigmoid + 阈值
                prob = torch.sigmoid(logits_up)
                pred = (prob > 0.5).to(torch.uint8)[:, 0]       # (B,H,W)
            else:
                # 多分类：softmax + argmax
                pred = torch.argmax(logits_up, dim=1)           # (B,H,W)

            mask = pred[0].cpu().numpy().astype(np.uint8)

            # 可视化版（非背景=白），便于快速检查
            vis = (mask > 0).astype(np.uint8) * 255

            out_mask = os.path.join(args.output_dir, os.path.splitext(fname)[0] + "_mask.png")
            Image.fromarray(vis).save(out_mask)
            print("saved:", out_mask)

            if DEBUG:
                # 打印分布做自检
                if logits_up.shape[1] > 1:
                    mmp = torch.softmax(logits_up, dim=1).max(dim=1).values.mean().item()
                    print("mean max prob:", f"{mmp:.3f}")
                else:
                    print("prob stats:",
                          float(prob.min().cpu()), float(prob.max().cpu()))
