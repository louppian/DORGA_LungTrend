# utils_severity.py : DORGA-Full 모델을 사용한 중증도 예측 모듈
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
import os

import matplotlib
matplotlib.use('Agg')

from modules.dorga import DORGAFullModel, vit_base_patch16_512
from config import MARKOV_ALPHA, MARKOV_ORDER
from scipy.special import softmax

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════
# DORGA-Full 모델 로드
# ═══════════════════════════════════════════════════════════
MODEL_PATH = "assets/weights/DORGAFull_posthoc.pth"
K_PATTERNS = 7

# dummy prior (load_state_dict로 실제 값이 덮어씌워짐)
_pi_global_dummy = torch.zeros(4, 4, 5, 5)
_pi_patterns_dummy = torch.zeros(K_PATTERNS, 4, 4, 5, 5)

_vit = vit_base_patch16_512(
    num_classes=5, in_chans=1, drop_path_rate=0.1, global_pool="avg"
)
model = DORGAFullModel(
    _vit, R=4, C=5, K=K_PATTERNS, proj_dim=768,
    pi_global=_pi_global_dummy, pi_patterns=_pi_patterns_dummy,
).to(device)

MARKOV_P = None
if os.path.exists(MODEL_PATH):
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    # posthoc 전이행렬 로드
    if 'posthoc' in ckpt:
        _ph = ckpt['posthoc']
        _key = f'P{MARKOV_ORDER}'
        MARKOV_P = _ph[_key] if _key in _ph else _ph.get('P1')
        print(f"[DORGA] posthoc P{MARKOV_ORDER} loaded: {MARKOV_P.shape}")
    print(f"[DORGA] model loaded: {MODEL_PATH}")
else:
    print(f"[DORGA] WARNING: weight file not found: {MODEL_PATH}")

model.eval()

# ═══════════════════════════════════════════════════════════
# 전처리 & ROI 설정
# ═══════════════════════════════════════════════════════════
to_tensor = transforms.ToTensor()
image_resize_transform = transforms.Resize(
    (512, 512), interpolation=transforms.InterpolationMode.BILINEAR
)
mask_resize_transform = transforms.Resize(
    (512, 512), interpolation=transforms.InterpolationMode.NEAREST
)
dorga_normalize = transforms.Normalize([0.56], [0.17])

# ROI 순서: DORGA box 기준 [RT, LT, RB, LB] (idx0=좌상, idx1=우상, idx2=좌하, idx3=우하)

REL_BOXES = torch.tensor([
    [0.0, 0.0, 0.5, 0.5],   # RT (이미지 좌상단)
    [0.5, 0.0, 1.0, 0.5],   # RB (이미지 좌하단)
    [0.0, 0.5, 0.5, 1.0],   # LT (이미지 우상단)
    [0.5, 0.5, 1.0, 1.0],   # LB (이미지 우하단)
], dtype=torch.float32)


# ═══════════════════════════════════════════════════════════
# 추론 함수
# ═══════════════════════════════════════════════════════════

def all_image_inference(images):
    """
    images: torch.Tensor [N, 1, 512, 512]
    returns: (inference, logits_np)
        inference: list of list [[RT, RB, LT, LB], ...] (DORGA 순서)
        logits_np: np.ndarray [N, 4, 5] — raw logits (post-hoc 보정용)
    """
    resized_images = torch.stack([image_resize_transform(img) for img in images])

    resized_images = dorga_normalize(resized_images)
    resized_images = resized_images.to(device)

    B = resized_images.shape[0]
    rel = REL_BOXES.unsqueeze(0).expand(B, -1, -1).to(device)

    with torch.no_grad():
        out = model(resized_images, rel)

    logits = out["logits_s3"]  # [B, 4, 5]
    preds = logits.argmax(-1)  # [B, 4] (DORGA 순서: RT, RB, LT, LB)

    inference = preds.cpu().numpy().tolist()
    logits_np = logits.cpu().numpy()
    return inference, logits_np


def apply_posthoc_correction(logits_np, user_edit_history, alpha=None):
    """
    logits_np: np.ndarray [N, 4, 5] — raw DORGA logits
    user_edit_history: list of list [[RT, RB, LT, LB], ...] — 사용자 수정 점수 (GT history 역할)
    alpha: float — prior temperature (default: config.MARKOV_ALPHA)
    returns: list of list [[RT, RB, LT, LB], ...] — 보정된 예측

    Order-2: P2[r, prev2, prev1] → 이전 2개 시점 사용
    """
    if MARKOV_P is None:
        return logits_np.argmax(axis=-1).tolist()

    if alpha is None:
        alpha = MARKOV_ALPHA

    order = MARKOV_ORDER
    N, R, C = logits_np.shape
    probs = softmax(logits_np, axis=-1)
    corrected = []

    for i in range(N):
        row = []
        for r in range(R):
            p_model = probs[i, r]
            if i < order or not user_edit_history:
                row.append(int(p_model.argmax()))
            else:
                idx = tuple(int(user_edit_history[i - order + k][r]) for k in range(order))
                p_markov = MARKOV_P[r][idx]
                log_p = np.log(p_model + 1e-10) + alpha * np.log(p_markov + 1e-10)
                row.append(int(log_p.argmax()))
        corrected.append(row)

    return corrected


# ═══════════════════════════════════════════════════════════
# 시각화 함수
# ═══════════════════════════════════════════════════════════

def get_severity_color(severity):
    severity = max(0, min(4, severity))
    mapping = {
        0: (255, 255, 255),  # 흰색
        1: (0, 255, 0),      # 초록색
        2: (255, 255, 0),    # 노란색
        3: (255, 100, 0),    # 주황색
        4: (255, 0, 0),      # 빨간색
    }
    return mapping[severity]


def inference_with_overlay(img_list, inference_list, selected_idx1, selected_idx2):
    """
    선택된 이미지에 2분할 선 및 severity 점수를 오버레이합니다.

    img_list: 이미지 목록 (numpy array)
    selected_idx1, selected_idx2: 선택된 이미지 인덱스
    inference_list: 각 이미지에 대한 [RT, RB, LT, LB] 추론 결과 리스트
    """

    def get_image(idx):
        if idx is not None and idx < len(img_list):
            return img_list[idx]
        return None

    def get_inference(idx):
        if idx is not None and idx < len(inference_list):
            return inference_list[idx]
        return None

    img1 = get_image(selected_idx1)
    img2 = get_image(selected_idx2)
    inf1 = get_inference(selected_idx1)
    inf2 = get_inference(selected_idx2)

    def image_to_tensor(image):
        if image is not None:
            arr = np.array(image)
            arr = np.ascontiguousarray(arr)
            return to_tensor(arr).to(device)
        return None

    tensor1 = image_to_tensor(img1)
    tensor2 = image_to_tensor(img2)

    tensors = []
    inferences = []
    if tensor1 is not None:
        tensors.append(tensor1)
        inferences.append(inf1)
    if tensor2 is not None:
        tensors.append(tensor2)
        inferences.append(inf2)
    if not tensors:
        return None

    data_tensor = torch.stack(tensors, dim=0)   # (B, C, H, W)
    inference_arr = np.array(inferences)          # (B, 4)

    batch = data_tensor.shape[0]
    overlay_images = []
    data_tensor_np = data_tensor.cpu().numpy()

    # 4분면 매핑 (DORGA box 기준: idx0=좌상, idx1=우상, idx2=좌하, idx3=우하)
    quad_map = {
        (0, 0): ("RT", 0),
        (0, 1): ("LT", 1),
        (1, 0): ("RB", 2),
        (1, 1): ("LB", 3),
    }

    for i in range(batch):
        original_img = np.transpose(data_tensor_np[i], (1, 2, 0))
        original_img = (original_img * 255).astype(np.uint8)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)

        overlay = original_img.copy()

        cell_h = 512 // 2
        cell_w = 512 // 2
        cv2.line(overlay, (0, cell_h), (512, cell_h), (0, 0, 0), 1)
        cv2.line(overlay, (cell_w, 0), (cell_w, 512), (0, 0, 0), 1)

        margin = 10
        text_offset = 30
        positions = {
            (0, 0): (margin, margin + text_offset),
            (0, 1): (cell_w * 2 - margin * 7, margin + text_offset),
            (1, 0): (margin, cell_h * 2 - margin),
            (1, 1): (cell_w * 2 - margin * 7, cell_h * 2 - margin),
        }

        for (row, col), (quad_label, idx) in quad_map.items():
            severity_val = inference_arr[i, idx]
            text_color = get_severity_color(severity_val)
            text_x, text_y = positions[(row, col)]
            cv2.putText(
                overlay,
                f"{quad_label}: {severity_val}",
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                text_color,
                2,
                cv2.LINE_AA,
            )

        overlay_images.append(overlay)

    return overlay_images