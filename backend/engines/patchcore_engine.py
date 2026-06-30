# backend/engines/patchcore_engine.py
import os
# pyrefly: ignore [missing-import]
import cv2
import torch
import numpy as np
import anomalib
from anomalib.models import Patchcore
from PIL import Image as PILImage
from torchvision.transforms.v2 import Resize
from torchvision.transforms.v2 import functional as F

class PatchcoreEngine:
    def __init__(self, model_path, threshold, device=None):
        self.model_path = model_path
        self.threshold = threshold
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"[INFO] Loading PatchCore model from {self.model_path} on {self.device}...")
        torch.serialization.add_safe_globals([anomalib.PrecisionType])
        self.model = Patchcore.load_from_checkpoint(
            self.model_path,
            map_location=self.device,
            weights_only=False
        )
        self.model.to(self.device)
        self.model.eval()
        print("[INFO] PatchCore model ready")

    def run_inference(self, bgr, roi_coords):
        """
        Runs PatchCore inference on the provided BGR image within specified ROI coordinates.
        Returns a dictionary with anomaly score, classification, and visual overlays.
        """
        ey1, ey2, ex1, ex2 = roi_coords
        roi = bgr[ey1:ey2, ex1:ex2]  # Crop ROI

        # BGR → RGB → PIL → PyTorch Tensor
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)

        resize = Resize((512, 512))
        img_tensor = F.to_tensor(resize(pil_img)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)

        score = float(output.pred_score)
        is_ng = score > self.threshold

        h, w = bgr.shape[:2]
        vis_image = None

        try:
            # 1. Anomaly map → JET colormap blended onto image (middle panel)
            amap = output.anomaly_map.squeeze().cpu().numpy()
            amap_resized = cv2.resize(amap, (w, h))
            amap_norm = cv2.normalize(amap_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            heatmap_bgr = cv2.applyColorMap(amap_norm, cv2.COLORMAP_JET)
            blended = cv2.addWeighted(bgr, 0.5, heatmap_bgr, 0.5, 0)  # Image + Anomaly Map

            # 2. Pred mask → red contour overlay on image (right panel)
            mask_panel = bgr.copy()
            if output.pred_mask is not None:
                pred_mask = output.pred_mask.squeeze().cpu().numpy().astype(np.uint8)  # 0 or 1
                pred_mask_resized = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                pred_mask_255 = (pred_mask_resized * 255).astype(np.uint8)
                contours, _ = cv2.findContours(pred_mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(mask_panel, contours, -1, (0, 0, 255), 2)  # red contours

            # 3. Add panel titles
            label_cfg = dict(fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 255, 255), thickness=2)
            
            orig_labeled  = bgr.copy()
            cv2.putText(orig_labeled,  "Image",              (10, 30), **label_cfg)
            blend_labeled = blended.copy()
            cv2.putText(blend_labeled, "Image + Anomaly Map", (10, 30), **label_cfg)
            mask_labeled  = mask_panel.copy()
            cv2.putText(mask_labeled,  "Image + Pred Mask",  (10, 30), **label_cfg)

            # 4. Concat all 3 side-by-side
            vis_image = np.concatenate([orig_labeled, blend_labeled, mask_labeled], axis=1)

        except Exception as e:
            print(f"[DEBUG] Heatmap visualization failed: {e}")

        return {
            "class":       "NG" if is_ng else "OK",
            "confidence":  score,
            "is_ng":       is_ng,
            "score":       score,
            "vis_image":   vis_image,   # full 3-panel BGR image
        }
