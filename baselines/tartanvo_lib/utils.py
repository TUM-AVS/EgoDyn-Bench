# Vendored from https://github.com/castacks/tartanvo (BSD License, CMU AirLab)
# Minimal subset: only intrinsics + preprocessing utilities needed for inference.

import numpy as np
import cv2


def dataset_intrinsics(dataset='tartanair'):
    if dataset == 'kitti':
        focalx, focaly, centerx, centery = 707.0912, 707.0912, 601.8873, 183.1104
    elif dataset == 'euroc':
        focalx, focaly, centerx, centery = 458.6539916992, 457.2959899902, 367.2149963379, 248.3750000000
    elif dataset == 'tartanair':
        focalx, focaly, centerx, centery = 320.0, 320.0, 320.0, 240.0
    else:
        return None
    return focalx, focaly, centerx, centery


def make_intrinsics_layer(w, h, fx, fy, ox, oy):
    """Build a (H, W, 2) intrinsic layer of normalised pixel coordinates."""
    ww, hh = np.meshgrid(range(w), range(h))
    ww = (ww.astype(np.float32) - ox + 0.5) / fx
    hh = (hh.astype(np.float32) - oy + 0.5) / fy
    intrinsicLayer = np.stack((ww, hh)).transpose(1, 2, 0)
    return intrinsicLayer


def crop_center(img, th, tw):
    """Center-crop an image (H, W, C) or (H, W) to (th, tw), upscaling if needed."""
    h, w = img.shape[:2]
    scale_h, scale_w, scale = 1., 1., 1.
    if th > h:
        scale_h = float(th) / h
    if tw > w:
        scale_w = float(tw) / w
    if scale_h > 1 or scale_w > 1:
        scale = max(scale_h, scale_w)
        w = int(round(w * scale))
        h = int(round(h * scale))
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    x1 = int((w - tw) / 2)
    y1 = int((h - th) / 2)
    if len(img.shape) == 3:
        return img[y1:y1+th, x1:x1+tw, :]
    return img[y1:y1+th, x1:x1+tw]
