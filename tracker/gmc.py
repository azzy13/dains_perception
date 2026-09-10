#!/usr/bin/env python3
"""Global (camera) motion compensation for image-space trackers.

A Kalman filter in image coordinates assumes the image frame is fixed, so an
object's predicted box is where it would be if only *it* had moved. Under a
moving camera that assumption fails: the whole scene shifts between frames, the
prediction lands off the object, IoU association misses, and the track is lost
and re-acquired under a new id.

Measured on rmot3's flown-drone footage: a single target across 300 frames was
covered by up to 15 different track ids, with the dominant id holding only 16%
of its trajectory (`eval/identity_metrics.py`).

This estimates the frame-to-frame 2-D affine of the background with ORB features
plus RANSAC, and the tracker applies it to every track state before predicting,
so the Kalman filter only has to model the object's own motion. Same idea as
BoT-SORT's GMC.

ORB + ``estimateAffinePartial2D`` is used rather than ECC: it is far cheaper at
1080p, and a partial affine (rotation, uniform scale, translation) is the right
model for a drone that translates and yaws — a full homography has more freedom
than the data supports and degenerates on low-texture road.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class GMC:
    """Frame-to-frame affine estimator.

    ``downscale`` trades accuracy for speed; 2 is plenty at 1080p and keeps the
    cost to roughly a millisecond per frame.
    """

    def __init__(self, downscale: int = 2, max_features: int = 1000,
                 min_matches: int = 8, ransac_thresh: float = 3.0):
        self.downscale = max(1, int(downscale))
        self.min_matches = min_matches
        self.ransac_thresh = ransac_thresh
        self._orb = cv2.ORB_create(nfeatures=max_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_kp = None
        self._prev_desc = None

    def reset(self) -> None:
        """Forget the previous frame. Call between sequences."""
        self._prev_gray = None
        self._prev_kp = None
        self._prev_desc = None

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return the 2x3 affine mapping the PREVIOUS frame onto this one.

        Identity on the first frame, or whenever the estimate is unreliable —
        a bad warp is worse than none, so this fails closed.
        """
        eye = np.eye(2, 3, dtype=np.float64)
        if frame_bgr is None or frame_bgr.size == 0:
            return eye

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.downscale > 1:
            gray = cv2.resize(gray, (gray.shape[1] // self.downscale,
                                     gray.shape[0] // self.downscale))

        kp, desc = self._orb.detectAndCompute(gray, None)
        prev_gray, prev_kp, prev_desc = self._prev_gray, self._prev_kp, self._prev_desc
        self._prev_gray, self._prev_kp, self._prev_desc = gray, kp, desc

        if prev_desc is None or desc is None:
            return eye
        if len(prev_desc) < self.min_matches or len(desc) < self.min_matches:
            return eye

        try:
            matches = self._matcher.match(prev_desc, desc)
        except cv2.error:
            return eye
        if len(matches) < self.min_matches:
            return eye

        matches = sorted(matches, key=lambda m: m.distance)[:400]
        src = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, inliers = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_thresh, maxIters=500)
        if H is None or inliers is None or int(inliers.sum()) < self.min_matches:
            return eye

        H = H.astype(np.float64)
        if self.downscale > 1:
            # Translation is in downscaled pixels; the linear part is scale-free.
            H[:, 2] *= self.downscale

        # Sanity: a per-frame warp that implies a huge scale change is a bad fit.
        det = float(np.linalg.det(H[:2, :2]))
        if not np.isfinite(det) or det <= 0.25 or det >= 4.0:
            return eye
        return H


def apply_to_tracks(stracks, H: np.ndarray) -> None:
    """Warp Kalman states in place by a 2x3 affine.

    State is ``[x, y, a, h, vx, vy, va, vh]`` in xyah: centre, aspect, height and
    their velocities. Position and velocity transform by the linear part (the
    velocity is a difference of positions, so it takes no translation); height
    scales by the similarity's uniform scale; aspect is invariant under a
    rotation-plus-uniform-scale, so it is left alone.
    """
    if not stracks:
        return
    R = H[:2, :2]
    t = H[:2, 2]
    det = float(np.linalg.det(R))
    scale = float(np.sqrt(det)) if det > 0 else 1.0

    R8 = np.eye(8, dtype=np.float64)
    R8[:2, :2] = R
    R8[4:6, 4:6] = R

    for st in stracks:
        if st.mean is None:
            continue
        mean = R8.dot(st.mean)
        mean[:2] += t
        mean[3] *= scale          # height
        st.mean = mean
        if st.covariance is not None:
            st.covariance = R8.dot(st.covariance).dot(R8.T)
