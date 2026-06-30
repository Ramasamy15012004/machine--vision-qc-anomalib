# tools/generate_sample_visualizations.py
import os
import sys
import json
import cv2
import numpy as np
import torch

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.engines.patchcore_engine import PatchcoreEngine
from backend.engines import template_engine

def main():
    # 1. Setup folders
    docs_dir = os.path.join(project_root, "docs")
    sample_out_dir = os.path.join(docs_dir, "sample_images")
    os.makedirs(sample_out_dir, exist_ok=True)
    
    # 2. Read config
    config_path = os.path.join(project_root, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
        
    model_path = cfg["patchcore_model_path"]
    if not os.path.isabs(model_path):
        model_path = os.path.join(project_root, model_path)
        
    threshold = cfg["patchcore_threshold"]
    
    # 3. Initialize PatchCore Engine
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing PatchCore on {device}...")
    engine_inst = PatchcoreEngine(model_path, threshold, device)
    
    # 4. Templates config
    templates_cfg = [
        {
            "name": "Part_LH_T",
            "template_indices": list(range(0, 2)),
            "template_crops": [
                (318, 767, 380, 807),
                (341, 773, 431, 812),
            ],
            "expected_roi": (250, 800, 350, 950),
            "threshold": 0.50
        },
        {
            "name": "Part_LH_B",
            "template_indices": list(range(2, 4)),
            "template_crops": [
                (305, 753, 455, 852),
                (292, 750, 544, 938),
            ],
            "expected_roi": (250, 800, 350, 950),
            "threshold": 0.50
        },
        {
            "name": "Part_RH_T",
            "template_indices": list(range(4, 6)),
            "template_crops": [
                (300, 757, 504, 892),
                (285, 761, 534, 904),
            ],
            "expected_roi": (250, 800, 350, 950),
            "threshold": 0.50
        },
        {
            "name": "Part_RH_B",
            "template_indices": list(range(6, 8)),
            "template_crops": [
                (337, 781, 497, 875),
                (298, 753, 471, 846),
            ],
            "expected_roi": (250, 800, 350, 950),
            "threshold": 0.50
        }
    ]
    
    template_images_dir = os.path.join(project_root, "models", "templates")
    template_images = [cv2.imread(os.path.join(template_images_dir, f"{i}.jpg")) for i in range(1, 9)]
    
    prepped_templates = []
    for tpl in templates_cfg:
        crops = []
        for local_i, img_i in enumerate(tpl["template_indices"]):
            ty1, ty2, tx1, tx2 = tpl["template_crops"][local_i]
            crops.append(template_images[img_i][ty1:ty2, tx1:tx2])
        prepped_templates.append(crops)
        
    # 5. Define test files
    test_files = [
        ("OK", "0012.jpg", os.path.join(project_root, "OK", "0012.jpg")),
        ("OK", "0013.jpg", os.path.join(project_root, "OK", "0013.jpg")),
        ("OK", "0014.jpg", os.path.join(project_root, "OK", "0014.jpg")),
        ("NG", "0153.jpg", os.path.join(project_root, "NG", "0153.jpg")),
        ("NG", "0156.jpg", os.path.join(project_root, "NG", "0156.jpg")),
        ("NG", "0158.jpg", os.path.join(project_root, "NG", "0158.jpg")),
    ]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for cat, name, filepath in test_files:
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}, skipping...")
            continue
            
        print(f"Processing {cat} sample: {name}...")
        bgr = cv2.imread(filepath)
        
        # Run template matching
        results = template_engine.inspect_frame(bgr, template_images, templates_cfg, prepped_templates)
        
        matched_result = None
        matched_template = None
        for res, tpl in zip(results, templates_cfg):
            if res["ok"] and res["match_loc"] is not None:
                if matched_result is None or res["score"] > matched_result["score"]:
                    matched_result = res
                    matched_template = tpl
                    
        # Run PatchCore
        roi_coords = matched_template["expected_roi"] if matched_template else templates_cfg[0]["expected_roi"]
        patchcore_res = engine_inst.run_inference(bgr, roi_coords)
        
        # Determine status
        qc_fail = False
        if matched_result is not None and patchcore_res is not None and patchcore_res["is_ng"]:
            qc_fail = True
            
        if matched_result is None:
            final_status = "NO_PART"
            color = (0, 0, 255)
        elif qc_fail or (patchcore_res and patchcore_res["is_ng"]):
            final_status = "NG"
            color = (0, 0, 255)
        else:
            final_status = "OK"
            color = (0, 255, 0)
            
        # Draw overlay
        overlay = bgr.copy()
        
        # Bounding box for template match
        if matched_result is not None:
            ey1, ey2, ex1, ex2 = matched_template["expected_roi"]
            mx, my = matched_result["match_loc"]
            h, w, _ = matched_result["match_shape"]
            x, y = mx + ex1, my + ey1
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(overlay, f"{matched_result['part_id']} | {matched_result['score']:.2f}",
                        (x, y - 10), font, 0.8, (0, 255, 0), 2)
                        
        # PatchCore result overlay
        if patchcore_res is not None:
            ey1, ey2, ex1, ex2 = roi_coords
            result_color = (0, 0, 255) if patchcore_res["is_ng"] else (0, 255, 0)
            cv2.putText(overlay, f"PatchCore: {'NG' if patchcore_res['is_ng'] else 'OK'} {patchcore_res['confidence']*100:.1f}%",
                        (ex1 + 10, ey1 + 30), font, 0.9, result_color, 2)
                        
        # Final Decision overlay
        cv2.putText(overlay, f"FINAL: {final_status}", (30, 40), font, 1.2, color, 3)
        
        # Save overlay file
        overlay_out_path = os.path.join(sample_out_dir, f"inspect_{cat}_{name}")
        cv2.imwrite(overlay_out_path, overlay)
        print(f"Saved annotated result to {overlay_out_path}")
        
        # Save PatchCore 3-panel visualization
        if patchcore_res is not None and patchcore_res["vis_image"] is not None:
            vis_out_path = os.path.join(sample_out_dir, f"patchcore_vis_{cat}_{name}")
            cv2.imwrite(vis_out_path, patchcore_res["vis_image"])
            print(f"Saved PatchCore vis to {vis_out_path}")

    print("Sample generation complete!")

if __name__ == "__main__":
    main()
