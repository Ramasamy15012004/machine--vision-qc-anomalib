# template_engine.py
import cv2
import numpy as np

def inspect_frame(frame_bgr, template_images, templates, prepped_templates=None):
    results = []
    method = cv2.TM_CCOEFF_NORMED

    for tpl_i, tpl in enumerate(templates):
        name = tpl["name"]
        ey1, ey2, ex1, ex2 = tpl["expected_roi"]
        threshold = tpl["threshold"]

        roi = frame_bgr[ey1:ey2, ex1:ex2]  # no cvtColor — work in BGR directly

        best_score = 0.0
        best_template_idx = None
        best_loc = None
        best_shape = None

        # Use pre-cropped templates if available, else fall back to slicing
        if prepped_templates is not None:
            crops = prepped_templates[tpl_i]
            indices = tpl["template_indices"]
        else:
            crops = []
            indices = []
            for local_i, img_i in enumerate(tpl["template_indices"]):
                ty1, ty2, tx1, tx2 = tpl["template_crops"][local_i]
                crops.append(template_images[img_i][ty1:ty2, tx1:tx2])
                indices.append(img_i)

        for local_i, (template, img_i) in enumerate(zip(crops, indices)):
            if template.size == 0:
                continue
            if roi.shape[0] < template.shape[0] or roi.shape[1] < template.shape[1]:
                continue

            res = cv2.matchTemplate(roi, template, method)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val > best_score:
                best_score = max_val
                best_template_idx = img_i
                best_loc = max_loc
                best_shape = template.shape

        results.append({
            "part_id": name,
            "ok": best_score >= threshold,
            "score": float(best_score),
            "threshold": threshold,
            "template_index": best_template_idx,
            "match_loc": best_loc,
            "match_shape": best_shape
        })

    return results