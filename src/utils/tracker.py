import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from src.utils.logger import get_run_log_dir


class MetricTracker:
    """统一指标追踪器，同时写入 JSONL 和 CSV，便于训练曲线分析。"""

    def __init__(self, log_dir: str, name: str) -> None:
        """初始化指标跟踪器，并把输出绑定到当前运行日志目录。
        
        所属类: ``MetricTracker``。
        
        Args: Args 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            log_dir: 日志根目录；运行时会在其中创建日期时间命名的子目录。
            name: 任务或 logger 名称，用于生成日志文件名和运行子目录名。
        
        Returns: Returns 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            无返回值；会创建 ``*_metrics.jsonl`` 和 ``*_metrics.csv`` 的输出路径。
        """
        self.log_dir = get_run_log_dir(name, log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.log_dir / f"{name}_metrics.jsonl"
        self.csv_path = self.log_dir / f"{name}_metrics.csv"
        self._csv_fields: Optional[List[str]] = None

    def log(self, payload: Dict) -> None:
        """封装该模块中的一个可复用业务步骤，供训练、推理或工具流程调用。
        
        所属类: ``MetricTracker``。
        
        Args: Args 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            payload: 待写入指标日志的一行字典，包含 epoch、phase、loss、acc 等字段。
        
        Returns: Returns 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
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
        """初始化推理跟踪器，并把检测结果写入当前运行日志目录。
        
        所属类: ``InferenceTracker``。
        
        Args: Args 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            log_dir: 日志根目录；运行时会在其中创建日期时间命名的子目录。
            name: 任务或 logger 名称，用于生成日志文件名和运行子目录名。
        
        Returns: Returns 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            无返回值；会创建 ``*_detections.jsonl`` 和 ``*_detections.csv`` 的输出路径。
        """
        self.log_dir = get_run_log_dir(name, log_dir)
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
        
        Args: Args 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            source: 输入图片、视频、目录或数据源路径，由推理入口传入。
            frame_idx: 视频帧编号，从 0 开始，用于追踪每帧检测结果。
            detections: 检测结果列表，每项通常为 box、score、class_id。
            class_names: 类别名称列表；列表顺序就是训练标签 ID 和推理类别 ID 的映射关系。
        
        Returns: Returns 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
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
        
        Returns: Returns 参数；请结合函数职责理解其业务含义，调用时应传入与当前任务匹配的值。
            函数返回处理结果；如果是入口或写文件流程，则主要副作用是启动任务、保存结果或写入日志。
        """
        return {
            "total_frames": self.total_frames,
            "total_detections": self.total_detections,
            "avg_detections_per_frame": self.total_detections / max(self.total_frames, 1),
        }
