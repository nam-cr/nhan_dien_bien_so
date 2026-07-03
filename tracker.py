import numpy as np
import supervision as sv

DEFAULT_MAX_MISSED = 5


class Track:
    def __init__(
        self,
        tid,
        bbox,
        cls_id,
        conf,
        frame_idx,
        pos_smooth: float = 0.7,
        size_smooth: float = 0.7,
    ):
        self.id = int(tid)
        self.cls = int(cls_id)
        self.conf = float(conf)
        self.last_frame = int(frame_idx)
        self.missed = 0
        self.age = 0
        self.total_visible = 1

        x1, y1, x2, y2 = map(float, bbox)
        self.cx = (x1 + x2) / 2.0
        self.cy = (y1 + y2) / 2.0
        self.w = max(1.0, x2 - x1)
        self.h = max(1.0, y2 - y1)

        self.vx = 0.0
        self.vy = 0.0

        self.pos_smooth = float(pos_smooth)
        self.size_smooth = float(size_smooth)

    def bbox_xyxy(self) -> np.ndarray:
        x1 = self.cx - self.w / 2.0
        y1 = self.cy - self.h / 2.0
        x2 = self.cx + self.w / 2.0
        y2 = self.cy + self.h / 2.0
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def predict(self) -> None:
        decay = 0.9 ** max(self.missed, 0)
        self.cx += self.vx * decay
        self.cy += self.vy * decay

    def update(self, bbox, conf, frame_idx) -> None:
        self.age += 1
        self.total_visible += 1

        x1_new, y1_new, x2_new, y2_new = map(float, bbox)
        cx_det = (x1_new + x2_new) / 2.0
        cy_det = (y1_new + y2_new) / 2.0
        w_det = max(1.0, x2_new - x1_new)
        h_det = max(1.0, y2_new - y1_new)

        prev_cx, prev_cy = self.cx, self.cy
        prev_w, prev_h = self.w, self.h

        alpha_p = self.pos_smooth
        alpha_s = self.size_smooth

        self.cx = alpha_p * cx_det + (1.0 - alpha_p) * prev_cx
        self.cy = alpha_p * cy_det + (1.0 - alpha_p) * prev_cy
        self.w = alpha_s * w_det + (1.0 - alpha_s) * prev_w
        self.h = alpha_s * h_det + (1.0 - alpha_s) * prev_h

        self.vx = self.cx - prev_cx
        self.vy = self.cy - prev_cy

        self.conf = float(conf)
        self.last_frame = int(frame_idx)
        self.missed = 0


class SimpleTracker:
    def __init__(
        self,
        iou_threshold: float = 0.4,
        max_missed: int = DEFAULT_MAX_MISSED,
        min_conf_new_track: float = 0.0,
        max_center_dist: float = 1.5,
        w_iou: float = 0.7,
        w_dist: float = 0.3,
        max_match_cost: float = 0.9,
        pos_smooth: float = 0.7,
        size_smooth: float = 0.7,
    ):
        self.iou_threshold = float(iou_threshold)
        self.max_missed = int(max_missed)
        self.min_conf_new_track = float(min_conf_new_track)

        self.max_center_dist = float(max_center_dist)
        self.w_iou = float(w_iou)
        self.w_dist = float(w_dist)
        self.max_match_cost = float(max_match_cost)
        self.HUGE_COST = 1e6

        self.pos_smooth = float(pos_smooth)
        self.size_smooth = float(size_smooth)

        self.tracks: list[Track] = []
        self.next_track_id = 1

    @staticmethod
    def _deduplicate_detections(
        detections: sv.Detections,
        contain_thr: float = 0.9,
        center_thr_ratio: float = 0.25,
    ) -> sv.Detections:
        if detections is None or len(detections) <= 1:
            return detections

        xyxy = detections.xyxy.astype(np.float32)
        n = xyxy.shape[0]

        if getattr(detections, "confidence", None) is not None:
            conf = detections.confidence.astype(np.float32)
        else:
            conf = np.ones((n,), dtype=np.float32)

        if getattr(detections, "class_id", None) is not None:
            cls = detections.class_id.astype(np.int32)
        else:
            cls = np.zeros((n,), dtype=np.int32)

        keep = np.ones(n, dtype=bool)
        order = np.argsort(-conf)

        for idx_i in range(n):
            i = order[idx_i]
            if not keep[i]:
                continue

            box_i = xyxy[i]
            cx_i = (box_i[0] + box_i[2]) / 2.0
            cy_i = (box_i[1] + box_i[3]) / 2.0
            w_i = box_i[2] - box_i[0]
            h_i = box_i[3] - box_i[1]
            area_i = max(w_i * h_i, 1.0)

            for idx_j in range(idx_i + 1, n):
                j = order[idx_j]
                if not keep[j]:
                    continue
                if cls[i] != cls[j]:
                    continue

                box_j = xyxy[j]
                cx_j = (box_j[0] + box_j[2]) / 2.0
                cy_j = (box_j[1] + box_j[3]) / 2.0
                w_j = box_j[2] - box_j[0]
                h_j = box_j[3] - box_j[1]
                area_j = max(w_j * h_j, 1.0)

                x1 = max(box_i[0], box_j[0])
                y1 = max(box_i[1], box_j[1])
                x2 = min(box_i[2], box_j[2])
                y2 = min(box_i[3], box_j[3])

                inter_w = max(0.0, x2 - x1)
                inter_h = max(0.0, y2 - y1)
                inter_area = inter_w * inter_h
                if inter_area <= 0:
                    continue

                area_small = min(area_i, area_j)
                overlap_small = inter_area / area_small

                dx = cx_i - cx_j
                dy = cy_i - cy_j
                center_dist = np.sqrt(dx * dx + dy * dy)
                w_small = min(w_i, w_j)
                h_small = min(h_i, h_j)
                diag_small = np.sqrt(w_small * w_small + h_small * h_small)
                diag_small = max(diag_small, 1.0)
                center_thr = center_thr_ratio * diag_small

                if overlap_small >= contain_thr and center_dist <= center_thr:
                    keep[j] = False

        return detections[keep]

    @staticmethod
    def _iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        N = boxes1.shape[0]
        M = boxes2.shape[0]
        if N == 0 or M == 0:
            return np.zeros((N, M), dtype=np.float32)

        b1 = boxes1[:, None, :]
        b2 = boxes2[None, :, :]

        x1 = np.maximum(b1[..., 0], b2[..., 0])
        y1 = np.maximum(b1[..., 1], b2[..., 1])
        x2 = np.minimum(b1[..., 2], b2[..., 2])
        y2 = np.minimum(b1[..., 3], b2[..., 3])

        inter_w = np.clip(x2 - x1, a_min=0, a_max=None)
        inter_h = np.clip(y2 - y1, a_min=0, a_max=None)
        inter_area = inter_w * inter_h

        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        union = area1[:, None] + area2[None, :] - inter_area
        union = np.clip(union, a_min=1e-6, a_max=None)
        iou = inter_area / union
        return iou.astype(np.float32)

    def _build_cost_matrix(self, track_boxes: np.ndarray, det_boxes: np.ndarray) -> np.ndarray:
        T = track_boxes.shape[0]
        D = det_boxes.shape[0]
        if T == 0 or D == 0:
            return np.zeros((T, D), dtype=np.float32) + self.HUGE_COST

        iou_mat = self._iou_matrix(track_boxes, det_boxes)

        t_cx = (track_boxes[:, 0] + track_boxes[:, 2]) / 2.0
        t_cy = (track_boxes[:, 1] + track_boxes[:, 3]) / 2.0
        d_cx = (det_boxes[:, 0] + det_boxes[:, 2]) / 2.0
        d_cy = (det_boxes[:, 1] + det_boxes[:, 3]) / 2.0

        dx = t_cx[:, None] - d_cx[None, :]
        dy = t_cy[:, None] - d_cy[None, :]
        dist_mat = np.sqrt(dx * dx + dy * dy)

        track_w = track_boxes[:, 2] - track_boxes[:, 0]
        track_h = track_boxes[:, 3] - track_boxes[:, 1]
        track_diag = np.sqrt(track_w * track_w + track_h * track_h)
        track_diag = np.clip(track_diag, 1.0, None)
        dist_norm = dist_mat / track_diag[:, None]

        min_iou_for_match = self.iou_threshold * 0.5
        iou_valid = iou_mat >= min_iou_for_match
        dist_valid = dist_norm <= self.max_center_dist
        valid = iou_valid | dist_valid

        cost_mat = np.full_like(iou_mat, fill_value=self.HUGE_COST, dtype=np.float32)
        if np.any(valid):
            cost_mat[valid] = self.w_iou * (1.0 - iou_mat[valid]) + self.w_dist * dist_norm[valid]
        return cost_mat

    def update(self, detections: sv.Detections | None, frame_idx: int) -> sv.Detections:
        if detections is not None and len(detections) > 1:
            detections = self._deduplicate_detections(detections)

        if detections is None or len(detections) == 0:
            det_xyxy = np.empty((0, 4), dtype=np.float32)
            det_conf = np.empty((0,), dtype=np.float32)
            det_cls = np.empty((0,), dtype=np.int32)
        else:
            det_xyxy = detections.xyxy.astype(np.float32)
            det_conf = detections.confidence.astype(np.float32) if getattr(detections, "confidence", None) is not None else np.ones((len(detections),), dtype=np.float32)
            det_cls = detections.class_id.astype(np.int32) if getattr(detections, "class_id", None) is not None else np.zeros((len(detections),), dtype=np.int32)

        num_tracks = len(self.tracks)
        num_dets = det_xyxy.shape[0]

        for t in self.tracks:
            t.predict()

        matches: list[tuple[int, int]] = []
        unmatched_tracks_idx = list(range(num_tracks))
        unmatched_dets_idx = list(range(num_dets))

        if num_tracks > 0 and num_dets > 0:
            track_boxes = np.stack([t.bbox_xyxy() for t in self.tracks], axis=0).astype(np.float32)
            cost_mat = self._build_cost_matrix(track_boxes, det_xyxy)

            candidate_pairs: list[tuple[int, int, float]] = []
            for t_idx in range(num_tracks):
                for d_idx in range(num_dets):
                    c = float(cost_mat[t_idx, d_idx])
                    if c < self.max_match_cost:
                        candidate_pairs.append((t_idx, d_idx, c))

            candidate_pairs.sort(key=lambda x: x[2])

            matched_tracks = set()
            matched_dets = set()

            for t_idx, d_idx, c in candidate_pairs:
                if t_idx in matched_tracks or d_idx in matched_dets:
                    continue

                if num_dets > 0 and getattr(detections, "class_id", None) is not None:
                    if self.tracks[t_idx].total_visible > 1:
                        if self.tracks[t_idx].cls != int(det_cls[d_idx]):
                            continue

                matches.append((t_idx, d_idx))
                matched_tracks.add(t_idx)
                matched_dets.add(d_idx)

            unmatched_tracks_idx = [i for i in range(num_tracks) if i not in matched_tracks]
            unmatched_dets_idx = [j for j in range(num_dets) if j not in matched_dets]

        for t_idx, d_idx in matches:
            self.tracks[t_idx].update(det_xyxy[d_idx], det_conf[d_idx], frame_idx)
            self.tracks[t_idx].cls = int(det_cls[d_idx])

        for t_idx in unmatched_tracks_idx:
            self.tracks[t_idx].missed += 1

        for d_idx in unmatched_dets_idx:
            if det_conf[d_idx] < self.min_conf_new_track:
                continue
            new_track = Track(
                tid=self.next_track_id,
                bbox=det_xyxy[d_idx],
                cls_id=det_cls[d_idx],
                conf=det_conf[d_idx],
                frame_idx=frame_idx,
                pos_smooth=self.pos_smooth,
                size_smooth=self.size_smooth,
            )
            self.tracks.append(new_track)
            self.next_track_id += 1

        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

        visible_tracks = [t for t in self.tracks if t.missed == 0]
        if len(visible_tracks) == 0:
            out_xyxy = np.empty((0, 4), dtype=np.float32)
            out_conf = np.empty((0,), dtype=np.float32)
            out_cls = np.empty((0,), dtype=np.int32)
            out_ids = np.empty((0,), dtype=np.int32)
        else:
            out_xyxy = np.stack([t.bbox_xyxy() for t in visible_tracks], axis=0)
            out_conf = np.array([t.conf for t in visible_tracks], dtype=np.float32)
            out_cls = np.array([t.cls for t in visible_tracks], dtype=np.int32)
            out_ids = np.array([t.id for t in visible_tracks], dtype=np.int32)

        out_det = sv.Detections(xyxy=out_xyxy, confidence=out_conf, class_id=out_cls)
        try:
            out_det.tracker_id = out_ids
        except Exception:
            pass
        return out_det
