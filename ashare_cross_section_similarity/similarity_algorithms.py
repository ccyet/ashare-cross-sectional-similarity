from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
import pandas as pd

from ashare_cross_section_similarity.features import resample_path, z_normalize

AlgorithmName = Literal[
    "baseline_price_feature",
    "return_shape",
    "hybrid_shape_v2",
    "dtw_optional",
    "mass_optional_history",
]

BASELINE_ALGORITHM = "baseline_price_feature"
ALGORITHM_CHOICES: tuple[str, ...] = (
    "baseline_price_feature",
    "return_shape",
    "hybrid_shape_v2",
    "dtw_optional",
    "mass_optional_history",
)

_ALGORITHM_LABELS = {
    "baseline_price_feature": "Baseline：价格路径 + 手工特征",
    "return_shape": "收益形态：对数收益路径",
    "hybrid_shape_v2": "Hybrid v2：价格 + 收益 + 特征",
    "dtw_optional": "DTW 可选：弹性距离",
    "mass_optional_history": "MASS 可选：历史预筛",
}


@dataclass(frozen=True)
class AlgorithmStatus:
    name: str
    label: str
    available: bool
    reason: str = ""


@dataclass(frozen=True)
class AlgorithmTarget:
    name: str
    target_length: int
    price_path: np.ndarray
    return_path: np.ndarray


def available_algorithm_names() -> list[str]:
    return list(ALGORITHM_CHOICES)


def algorithm_label(name: str) -> str:
    return _ALGORITHM_LABELS.get(name, name)


def get_algorithm_status(name: str) -> AlgorithmStatus:
    if name not in ALGORITHM_CHOICES:
        return AlgorithmStatus(name=name, label=name, available=False, reason="未知算法")
    if name == "dtw_optional" and not _has_optional_dtw():
        return AlgorithmStatus(
            name=name,
            label=algorithm_label(name),
            available=False,
            reason="需要安装 aeon 或 tslearn 后才能启用。",
        )
    if name == "mass_optional_history" and not _has_optional_stumpy():
        return AlgorithmStatus(
            name=name,
            label=algorithm_label(name),
            available=False,
            reason="需要安装 stumpy 后才能启用；当前 benchmark 会显式跳过。",
        )
    return AlgorithmStatus(name=name, label=algorithm_label(name), available=True)


def ensure_algorithm_available(name: str, *, mode: str) -> str:
    if name not in ALGORITHM_CHOICES:
        raise ValueError(f"未知相似算法：{name}")
    if name == "mass_optional_history" and mode != "history":
        raise ValueError("mass_optional_history 仅用于历史时序 benchmark。")
    status = get_algorithm_status(name)
    if not status.available:
        raise ValueError(f"{name} 不可用：{status.reason}")
    return name


def build_algorithm_target(window: pd.DataFrame, algorithm: str) -> AlgorithmTarget:
    close = pd.to_numeric(window["close"], errors="coerce").astype(float).to_numpy()
    target_length = len(close)
    return AlgorithmTarget(
        name=algorithm,
        target_length=target_length,
        price_path=_price_path(close, target_length),
        return_path=_return_path(close, target_length),
    )


def distance_for_window(window: pd.DataFrame, target: AlgorithmTarget) -> dict[str, float]:
    close = pd.to_numeric(window["close"], errors="coerce").astype(float).to_numpy()
    if target.name == "dtw_optional":
        return _dtw_window_distance(close, target)
    price_distance = _price_distance(close, target)
    return_distance = _return_distance(close, target)
    path_distance = _combine_path_distance(target.name, price_distance, return_distance)
    return {
        "路径距离": path_distance,
        "价格路径距离": price_distance,
        "收益路径距离": return_distance,
    }


def distance_for_close_matrix(close_windows: np.ndarray, target: AlgorithmTarget) -> dict[str, np.ndarray]:
    if target.name == "dtw_optional":
        distances = []
        for row in np.asarray(close_windows, dtype=float):
            distance = _optional_dtw_distance(_price_path(row, target.target_length), target.price_path)
            if distance is None:
                raise ValueError("dtw_optional 不可用：需要安装 aeon 或 tslearn。")
            distances.append(float(distance) / math.sqrt(max(1, target.target_length)))
        values = np.asarray(distances, dtype=float)
        return {
            "路径距离": values,
            "价格路径距离": values,
            "收益路径距离": np.full(len(values), np.nan, dtype=float),
        }

    price_distance = _price_distance_matrix(close_windows, target)
    return_distance = _return_distance_matrix(close_windows, target)
    path_distance = _combine_path_distance(target.name, price_distance, return_distance)
    return {
        "路径距离": path_distance,
        "价格路径距离": price_distance,
        "收益路径距离": return_distance,
    }


def _combine_path_distance(
    algorithm: str,
    price_distance: float | np.ndarray,
    return_distance: float | np.ndarray,
) -> float | np.ndarray:
    if algorithm == "return_shape":
        return return_distance
    if algorithm == "hybrid_shape_v2":
        return np.asarray(price_distance) * 0.55 + np.asarray(return_distance) * 0.45
    return price_distance


def _price_distance(close: np.ndarray, target: AlgorithmTarget) -> float:
    path = _price_path(close, target.target_length)
    return float(np.linalg.norm(target.price_path - path) / math.sqrt(max(1, target.target_length)))


def _return_distance(close: np.ndarray, target: AlgorithmTarget) -> float:
    path = _return_path(close, target.target_length)
    length = max(1, len(target.return_path))
    return float(np.linalg.norm(target.return_path - path) / math.sqrt(length))


def _price_distance_matrix(close_windows: np.ndarray, target: AlgorithmTarget) -> np.ndarray:
    path_matrix = _z_normalize_rows(_normalized_close_paths(close_windows))
    return np.linalg.norm(path_matrix - target.price_path, axis=1) / math.sqrt(max(1, target.target_length))


def _return_distance_matrix(close_windows: np.ndarray, target: AlgorithmTarget) -> np.ndarray:
    path_matrix = _normalized_close_paths(close_windows)
    with np.errstate(invalid="ignore", divide="ignore"):
        returns = np.diff(np.log(np.where(path_matrix > 0, path_matrix, np.nan)), axis=1)
    returns = _z_normalize_rows(returns)
    length = max(1, len(target.return_path))
    return np.linalg.norm(returns - target.return_path, axis=1) / math.sqrt(length)


def _price_path(close: np.ndarray, target_length: int) -> np.ndarray:
    path = _normalized_close_path(close)
    if len(path) != target_length:
        path = resample_path(path, target_length)
    return z_normalize(path)


def _return_path(close: np.ndarray, target_length: int) -> np.ndarray:
    path = _normalized_close_path(close)
    if len(path) != target_length:
        path = resample_path(path, target_length)
    with np.errstate(invalid="ignore", divide="ignore"):
        returns = np.diff(np.log(np.where(path > 0, path, np.nan)))
    return z_normalize(np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0))


def _dtw_window_distance(close: np.ndarray, target: AlgorithmTarget) -> dict[str, float]:
    candidate_path = _price_path(close, target.target_length)
    distance = _optional_dtw_distance(candidate_path, target.price_path)
    if distance is None:
        raise ValueError("dtw_optional 不可用：需要安装 aeon 或 tslearn。")
    scaled = float(distance) / math.sqrt(max(1, target.target_length))
    return {
        "路径距离": scaled,
        "价格路径距离": scaled,
        "收益路径距离": float("nan"),
    }


def _normalized_close_paths(close_windows: np.ndarray) -> np.ndarray:
    close_windows = np.asarray(close_windows, dtype=float)
    first = close_windows[:, [0]]
    valid = np.isfinite(first) & (first != 0)
    return np.divide(close_windows, first, out=np.zeros_like(close_windows, dtype=float), where=valid) * 100.0


def _normalized_close_path(close: np.ndarray) -> np.ndarray:
    close = np.asarray(close, dtype=float)
    if len(close) == 0:
        return np.array([], dtype=float)
    first = close[0]
    if not np.isfinite(first) or first == 0:
        return np.zeros(len(close), dtype=float)
    return close / first * 100.0


def _z_normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    means = np.nanmean(values, axis=1, keepdims=True)
    stds = np.nanstd(values, axis=1, keepdims=True)
    centered = values - means
    valid = np.isfinite(stds) & (stds != 0)
    return np.divide(centered, stds, out=centered.copy(), where=valid)


def _has_optional_dtw() -> bool:
    return _optional_dtw_distance(np.array([0.0, 1.0]), np.array([0.0, 1.0])) is not None


def _has_optional_stumpy() -> bool:
    try:
        import stumpy  # noqa: F401
    except ImportError:
        return False
    return True


def _optional_dtw_distance(left: np.ndarray, right: np.ndarray) -> float | None:
    try:
        from aeon.distances import dtw_distance

        return float(dtw_distance(np.asarray(left, dtype=float), np.asarray(right, dtype=float)))
    except ImportError:
        pass
    try:
        from tslearn.metrics import dtw

        return float(dtw(np.asarray(left, dtype=float), np.asarray(right, dtype=float)))
    except ImportError:
        return None
