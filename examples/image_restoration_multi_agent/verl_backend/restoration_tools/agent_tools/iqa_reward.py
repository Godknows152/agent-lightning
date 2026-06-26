
import io
import json
import math
import os
from pathlib import Path

import pyiqa
import torch
from PIL import Image
from transformers import AutoModelForCausalLM

# 临时绕过 torch.load 安全检查 (CVE-2025-32434)
# 建议后续升级 PyTorch >= 2.6 后移除此设置
os.environ['TRANSFORMERS_TORCH_LOAD_IS_SAFE'] = '1'

# Dynamic path resolution: q_align checkpoints are at restoration_tools/checkpoints/q_align/
_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_QALIGN_PATH = str(_THIS_DIR.parent / 'checkpoints' / 'q_align')
_METRIC_NAMES = ('qalign', 'maniqa', 'musiq', 'clipiqa', 'niqe')
_METRIC_ALIASES = {'niqe_transformed': 'niqe'}
_PYIQA_METRICS = {'maniqa', 'musiq', 'clipiqa', 'niqe', 'topiq_nr'}
_MIN_STD = 1e-6
_DEFAULT_NORMALIZATION_STATS = {
    'maniqa': {'mean': 0.2092878860158957, 'std': 0.1245655639250264},
    'musiq': {'mean': 40.525839667740804, 'std': 16.94055156979329},
    'qalign': {'mean': 0.25762742254801686, 'std': 0.1830440687605635},
}

class IQAScore:
    """
    Image Quality Assessment class that implements multiple IQA metrics.

    This class provides methods to evaluate image quality using state-of-the-art
    IQA models like QAlign, MANIQA, MUSIQ, CLIPIQA, and NIQE.
    """

    def __init__(
        self,
        device='cuda',
        score_weight=None,
        qalign_path=None,
        normalize_scores=False,
        normalization_stats_path=None,
        normalization_stats=None,
        metric_config_path=None,
    ):
        """
        Initialize the IQA Score calculator.

        Args:
            device (str): Computing device ('cuda' or 'cpu').
            score_weight (list, optional): Weight for each metric [qalign, maniqa, musiq, clipiqa, niqe].
                                          Defaults to [1,1,1,1,1].
            qalign_path (str, optional): Path to the QAlign model. Defaults to restoration_tools/checkpoints/q_align.
            normalize_scores (bool, optional): Whether to return z-score normalized IQA values.
            normalization_stats_path (str, optional): JSON file containing frozen dataset statistics.
            normalization_stats (dict, optional): In-memory normalization stats. Overrides normalization_stats_path.
            metric_config_path (str, optional): Current project IQA calibration JSON.
        """
        # 处理设备参数：确保获取正确的本地 GPU 设备
        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.score_weight = score_weight if score_weight is not None else [1, 1, 1, 1, 1]
        self.normalize_scores = bool(normalize_scores)
        self.metric_names = tuple(_METRIC_NAMES)
        self.metric_transforms = {
            'qalign': 'identity',
            'maniqa': 'identity',
            'musiq': 'identity',
            'clipiqa': 'identity',
            'niqe': 'legacy_exp_neg_10',
        }
        self._metric_config_stats = None
        if metric_config_path:
            self._load_metric_config(metric_config_path)

        # Path to the QAlign model (use relative path by default)
        if qalign_path is None:
            qalign_path = _DEFAULT_QALIGN_PATH

        print(f"Loading IQA metrics on {self.device}...")

        # 注意：pyiqa 内部可能使用 DataParallel，在多 GPU DeepSpeed 训练中可能有设备冲突
        # 我们通过设置 CUDA_VISIBLE_DEVICES 环境变量或使用特定 device 来缓解
        if self.score_weight:
            if 'qalign' in self.metric_names:
                self.qalign_metric = AutoModelForCausalLM.from_pretrained(
                    qalign_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map={"": str(self.device)},
                )
            device_str = str(self.device)
            for metric_name in self.metric_names:
                if metric_name == 'qalign':
                    continue
                if metric_name not in _PYIQA_METRICS:
                    raise ValueError(f"Unsupported IQA metric '{metric_name}'")
                setattr(self, f'{metric_name}_metric', pyiqa.create_metric(metric_name, device=device_str))

        self.mean_std = self._resolve_normalization_stats(
            normalization_stats=normalization_stats,
            normalization_stats_path=normalization_stats_path,
        )
        if self.normalize_scores:
            missing_metrics = [metric for metric in self.metric_names if metric not in self.mean_std]
            if missing_metrics:
                raise ValueError(
                    "Normalization is enabled but stats are missing metrics: "
                    + ", ".join(missing_metrics)
                )

        print("IQA metrics loaded successfully")

    def _canonical_metric_name(self, metric):
        return _METRIC_ALIASES.get(metric, metric)

    def _resolve_path(self, path):
        resolved = Path(path).expanduser()
        if not resolved.is_absolute():
            resolved = (Path.cwd() / resolved).resolve()
        return resolved

    def _load_metric_config(self, metric_config_path):
        config_path = self._resolve_path(metric_config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"IQA metric config file not found: {config_path}")

        with config_path.open('r', encoding='utf-8') as f:
            payload = json.load(f)

        metrics_payload = payload.get('metrics')
        if not isinstance(metrics_payload, dict) or not metrics_payload:
            raise ValueError(f"Invalid IQA metric config payload in {config_path}")

        self.metric_names = tuple(self._canonical_metric_name(name) for name in metrics_payload.keys())
        self.metric_transforms = {}
        self._metric_config_stats = {}
        for raw_name, metric_cfg in metrics_payload.items():
            metric_name = self._canonical_metric_name(raw_name)
            if not isinstance(metric_cfg, dict):
                raise ValueError(f"Metric config for '{raw_name}' must be a mapping")
            self.metric_transforms[metric_name] = str(metric_cfg.get('raw_transform', 'identity'))
            if 'mean' in metric_cfg and 'std' in metric_cfg:
                self._metric_config_stats[metric_name] = {
                    'mean': float(metric_cfg['mean']),
                    'std': max(abs(float(metric_cfg['std'])), _MIN_STD),
                }

    def _normalize_stats_payload(self, payload):
        metrics_payload = payload.get('metrics', payload)
        normalized = {}
        for metric_name, stats in metrics_payload.items():
            canonical_name = self._canonical_metric_name(metric_name)
            if canonical_name not in self.metric_names:
                continue
            if not isinstance(stats, dict):
                continue
            if 'mean' not in stats or 'std' not in stats:
                continue
            normalized[canonical_name] = {
                'mean': float(stats['mean']),
                'std': max(abs(float(stats['std'])), _MIN_STD),
            }
        return normalized

    def _resolve_normalization_stats(self, normalization_stats=None, normalization_stats_path=None):
        if normalization_stats is not None:
            return self._normalize_stats_payload(normalization_stats)

        if self._metric_config_stats is not None:
            return {
                metric_name: dict(metric_stats)
                for metric_name, metric_stats in self._metric_config_stats.items()
            }

        if normalization_stats_path is None:
            return {
                metric_name: dict(metric_stats)
                for metric_name, metric_stats in _DEFAULT_NORMALIZATION_STATS.items()
            }

        stats_path = self._resolve_path(normalization_stats_path)
        if not stats_path.exists():
            raise FileNotFoundError(f"IQA normalization stats file not found: {stats_path}")

        with stats_path.open('r', encoding='utf-8') as f:
            payload = json.load(f)
        return self._normalize_stats_payload(payload)

    def imread2tensor(self, img_source, rgb=False):
        """
        Convert various image sources to PIL Image.

        Args:
            img_source: Can be bytes, file path string, or PIL Image.
            rgb (bool): Whether to convert to RGB.

        Returns:
            PIL.Image: Loaded image.

        Raises:
            Exception: If the source type is not supported.
        """
        if isinstance(img_source, bytes):
            img = Image.open(io.BytesIO(img_source))
        elif isinstance(img_source, str):
            img = Image.open(img_source)
        elif isinstance(img_source, Image.Image):
            img = img_source
        else:
            raise Exception("Unsupported source type")

        if rgb:
            img = img.convert('RGB')

        return img

    def preprocess_image(self, source_image_path):
        """
        Preprocess the image for quality assessment.

        Args:
            source_image_path (str or bytes or PIL.Image): Image to preprocess.

        Returns:
            PIL.Image: Preprocessed image.
        """
        return self.imread2tensor(source_image_path)

    def normalize_iqa_score(self, score, metric):
        """
        Normalize an IQA score using pre-computed mean and standard deviation.

        Args:
            score (float): Raw score.
            metric (str): Metric name ('qalign', 'maniqa', 'musiq', 'clipiqa', or 'niqe').

        Returns:
            float: Normalized score (z-score).
        """
        metric = self._canonical_metric_name(metric)
        if metric not in self.mean_std:
            raise KeyError(f"Normalization stats for metric '{metric}' are not available")
        mean = self.mean_std[metric]['mean']
        std = self.mean_std[metric]['std']
        return (score - mean) / std

    def _apply_metric_transform(self, metric, raw_score):
        transform = self.metric_transforms.get(metric, 'identity')
        if transform in {'identity', 'none'}:
            return raw_score
        if transform == 'negate':
            return -raw_score
        if transform == 'legacy_exp_neg_10':
            return math.exp(-raw_score / 10.0)
        raise ValueError(f"Unsupported raw_transform '{transform}' for IQA metric '{metric}'")

    def get_raw_iqa_scores(self, source_image_path):
        """Return raw IQA scores in the runtime reward space."""
        with torch.no_grad():
            source_image = self.preprocess_image(source_image_path)

            scores = {}
            for metric_name in self.metric_names:
                if metric_name == 'qalign':
                    raw_score = self.qalign_metric.score(
                        [source_image],
                        task_="quality",
                        input_="image",
                    ).item()
                else:
                    raw_score = getattr(self, f'{metric_name}_metric')(source_image).item()
                scores[metric_name] = self._apply_metric_transform(metric_name, raw_score)
                if metric_name == 'niqe':
                    scores['niqe_raw'] = raw_score
            return scores

    def get_iqa_score(self, source_image_path, normalize=None):
        """
        Calculate quality scores for an image using multiple metrics.

        Args:
            source_image_path (str): Path to the image.
            eval (bool): Whether in evaluation mode.
            is_score_weight (bool): Whether to use weighted scoring.

        Returns:
            list[float]: IQA score list in configured metric order.
                         When normalization is enabled, returns z-score normalized values.
        """
        raw_scores = self.get_raw_iqa_scores(source_image_path)
        should_normalize = self.normalize_scores if normalize is None else bool(normalize)
        if should_normalize:
            return [self.normalize_iqa_score(raw_scores[metric], metric) for metric in self.metric_names]
        return [raw_scores[metric] for metric in self.metric_names]
