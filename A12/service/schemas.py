from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class PipelineMetadata:
    model_version: str
    inference_time_ms: float
    frame_count_original: int
    frame_count_cut: int
    fps: float
    cut_start_frame: int
    cut_end_frame: int
    smoothing_strategy: str
    classifier_mode: str


@dataclass
class ClassificationResult:
    label: str
    is_good: bool
    confidence: float
    probabilities: Dict[str, float]


@dataclass
class PipelineOutput:
    annotated_video_path: str
    animation_data_path: str
    keypoints_csv_path: str
    classification: ClassificationResult
    metadata: PipelineMetadata
    warnings: List[str]

    def to_json(self) -> Dict[str, Any]:
        return {
            "annotated_video_path": self.annotated_video_path,
            "animation_data_path": self.animation_data_path,
            "keypoints_csv_path": self.keypoints_csv_path,
            "classification": asdict(self.classification),
            "metadata": asdict(self.metadata),
            "warnings": self.warnings,
        }
