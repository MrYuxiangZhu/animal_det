import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


class MetricTracker:
    """统一指标追踪器，同时写入 JSONL 和 CSV，便于训练曲线分析。"""

    def __init__(self, log_dir: str, name: str) -> None:
        """初始化对象，保存后续训练、推理或数据处理所需的配置和状态。
        
        所属类: ``MetricTracker``。
        
        Args:
            log_dir: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
            name: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        
        Returns:
            该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.log_dir / f"{name}_metrics.jsonl"
        self.csv_path = self.log_dir / f"{name}_metrics.csv"
        self._csv_fields: Optional[List[str]] = None

    def log(self, payload: Dict) -> None:
        """封装该模块中的一个可复用业务步骤，供训练、推理或工具流程调用。
        
        所属类: ``MetricTracker``。
        
        Args:
            payload: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        
        Returns:
            该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
        """
        row = dict(payload)
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        fields = list(row.keys())
        if self._csv_fields is None:
            self._csv_fields = fields
            if not self.csv_path.exists():
                with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=self._csv_fields).writeheader()
        if self._csv_fields == fields:
            with self.csv_path.open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self._csv_fields).writerow(row)


class InferenceTracker:
    """推理跟踪器，记录每帧/每张图的 2D box、类别和置信度。"""

    fieldnames = ["source", "frame_idx", "det_id", "class_id", "class_name", "score", "x1", "y1", "x2", "y2"]

    def __init__(self, log_dir: str, name: str) -> None:
        """初始化对象，保存后续训练、推理或数据处理所需的配置和状态。
        
        所属类: ``InferenceTracker``。
        
        Args:
            log_dir: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
            name: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        
        Returns:
            该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.log_dir / f"{name}_detections.jsonl"
        self.csv_path = self.log_dir / f"{name}_detections.csv"
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()
        self.total_frames = 0
        self.total_detections = 0

    def log_detections(self, source: str, frame_idx: int, detections: Iterable, class_names: List[str]) -> None:
        """封装该模块中的一个可复用业务步骤，供训练、推理或工具流程调用。
        
        所属类: ``InferenceTracker``。
        
        Args:
            source: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
            frame_idx: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
            detections: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
            class_names: 调用方传入的业务参数，具体含义由当前模块配置和上下文决定。
        
        Returns:
            该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
        """
        rows = []
        for det_id, (box, score, cls_id) in enumerate(detections):
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            class_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            rows.append({
                "source": source,
                "frame_idx": frame_idx,
                "det_id": det_id,
                "class_id": int(cls_id),
                "class_name": class_name,
                "score": float(score),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            })
        self.total_frames += 1
        self.total_detections += len(rows)
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"source": source, "frame_idx": frame_idx, "num_detections": len(rows), "boxes_2d": rows}, ensure_ascii=False) + "\n")
        if rows:
            with self.csv_path.open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.fieldnames).writerows(rows)

    def summary(self) -> Dict[str, float]:
        """封装该模块中的一个可复用业务步骤，供训练、推理或工具流程调用。
        
        所属类: ``InferenceTracker``。
        
        Returns:
            该函数的返回值或副作用由调用场景决定；入口函数通常直接完成流程调度。
        """
        return {
            "total_frames": self.total_frames,
            "total_detections": self.total_detections,
            "avg_detections_per_frame": self.total_detections / max(self.total_frames, 1),
        }
