# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
Minimal OC-SORT-style tracker compatible with Ultralytics tracking pipeline.

This implementation follows a SORT-like pipeline with Kalman filtering and IoU-based association,
using the same STrack structure as ByteTrack for compatibility with Ultralytics' Results postprocess.

Config keys expected via YAML (see ultralytics/cfg/trackers/ocsort.yaml):
  - track_high_thresh: detection score threshold used for association
  - new_track_thresh: score threshold to initialize new tracks
  - match_thresh: IoU matching threshold
  - track_buffer: max lost time (frames)
  - fuse_score: whether to fuse detection scores into distance metric

Returned tracks format matches ByteTrack: [coords..., track_id, score, cls, idx]
so ultralytics/trackers/track.py can index the last column as id.
"""

from __future__ import annotations

import numpy as np

from .byte_tracker import STrack  # reuse the same track structure and KF shape (XYAH)
from .basetrack import TrackState
from .utils import matching
from .utils.kalman_filter import KalmanFilterXYAH


class OCSORT:
    def __init__(self, args, frame_rate: int = 30):
        self.tracked_stracks: list[STrack] = []
        self.lost_stracks: list[STrack] = []
        self.removed_stracks: list[STrack] = []

        self.frame_id = 0
        self.args = args
        self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

    def get_kalmanfilter(self):
        return KalmanFilterXYAH()

    @staticmethod
    def reset_id():
        STrack.reset_id()

    def reset(self):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

    def init_track(self, dets, scores, cls, img=None):
        return [STrack(xyxy, s, c) for (xyxy, s, c) in zip(dets, scores, cls)] if len(dets) else []

    def get_dists(self, tracks, detections):
        # IoU距离
        dists = matching.iou_distance(tracks, detections)
        # 置信度加权优化：低置信度目标关联距离适当放宽
        if len(detections) > 0:
            scores = np.array([det.score for det in detections])
            # 置信度归一化（避免极端值）
            scores = np.clip(scores, 0.05, 1.0)
            # 置信度越低，距离越小（更容易关联）
            score_weights = 1.0 - scores * 0.5  # 置信度高则权重低，低则权重高
            dists = dists * score_weights[None, :]
        # 原有 fuse_score 逻辑
        if getattr(self.args, "fuse_score", False):
            dists = matching.fuse_score(dists, detections)
        return dists

    @staticmethod
    def multi_predict(tracks):
        STrack.multi_predict(tracks)

    def update(self, results, img=None):
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        scores = results.conf
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        cls = results.cls

        # 双阶段分数筛选
        inds_high = scores >= self.args.track_high_thresh
        inds_low = (scores > getattr(self.args, "track_low_thresh", 0.1)) & (scores < self.args.track_high_thresh)

        dets_high = bboxes[inds_high]
        scores_high = scores[inds_high]
        cls_high = cls[inds_high]

        dets_low = bboxes[inds_low]
        scores_low = scores[inds_low]
        cls_low = cls[inds_low]

        detections_high = self.init_track(dets_high, scores_high, cls_high, img)
        detections_low = self.init_track(dets_low, scores_low, cls_low, img)

        # Split tracked vs unconfirmed
        unconfirmed = []
        tracked_stracks = []
        for t in self.tracked_stracks:
            (unconfirmed if not t.is_activated else tracked_stracks).append(t)

        # Predict
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        self.multi_predict(strack_pool)

        # Optional GMC
        if hasattr(self, "gmc") and img is not None:
            warp = self.gmc.apply(img, dets_high)
            STrack.multi_gmc(strack_pool, warp)
            STrack.multi_gmc(unconfirmed, warp)

        # 第一阶段关联（高分）
        dists = self.get_dists(strack_pool, detections_high)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections_high[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # 第二阶段关联（低分，阈值更宽松）
        r_tracked = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists_low = matching.iou_distance(r_tracked, detections_low)
        matches_low, u_track_low, u_detection_low = matching.linear_assignment(dists_low, thresh=0.5)
        for itracked, idet in matches_low:
            track = r_tracked[itracked]
            det = detections_low[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # 标记未匹配的为lost
        for it in u_track_low:
            track = r_tracked[it]
            track.mark_lost()
            lost_stracks.append(track)

        # 处理unconfirmed
        rem_dets = [detections_high[i] for i in u_detection] + [detections_low[i] for i in u_detection_low]
        dists = self.get_dists(unconfirmed, rem_dets)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
        for itracked, idet in matches:
            unconfirmed[itracked].update(rem_dets[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # 新轨迹初始化
        for inew in u_detection:
            track = rem_dets[inew]
            if track.score < self.args.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        # 清理lost
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # 合并状态
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        self.removed_stracks.extend(removed_stracks)
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-999:]

        return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)

    @staticmethod
    def joint_stracks(tlista, tlistb):
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_stracks(tlista, tlistb):
        track_ids_b = {t.track_id for t in tlistb}
        return [t for t in tlista if t.track_id not in track_ids_b]

    @staticmethod
    def remove_duplicate_stracks(stracksa, stracksb):
        pdist = matching.iou_distance(stracksa, stracksb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb
