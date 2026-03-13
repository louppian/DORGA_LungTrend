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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════
# DORGA-Full 모델 로드
# ═══════════════════════════════════════════════════════════
MODEL_PATH = "assets/weights/DORGAFull.pth"
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

if os.path.exists(MODEL_PATH):
    sd = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if 'state_dict' in sd:
        sd = sd['state_dict']
    model.load_state_dict(sd, strict=False)
    print(f"[DORGA] 모델 로드 완료: {MODEL_PATH}")
else:
    print(f"[DORGA] 경고: 가중치 파일을 찾을 수 없습니다 → {MODEL_PATH}")

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

# DORGA ROI 순서: [RT, RB, LT, LB]
# GUI  ROI 순서: [RT, LT, RB, LB]
# 변환 인덱스: DORGA[0]→GUI[0], DORGA[2]→GUI[1], DORGA[1]→GUI[2], DORGA[3]→GUI[3]
_DORGA_TO_GUI_ORDER = [0, 2, 1, 3]

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
    returns: list of list  [[RT, LT, RB, LB], ...] (GUI 순서)
    """
    resized_images = torch.stack([image_resize_transform(img) for img in images])

    # DORGA는 1채널 입력 + 정규화
    resized_images = dorga_normalize(resized_images)
    resized_images = resized_images.to(device)

    B = resized_images.shape[0]
    rel = REL_BOXES.unsqueeze(0).expand(B, -1, -1).to(device)

    with torch.no_grad():
        out = model(resized_images, rel)

    # logits_s3: [B, 4, 5] → argmax → [B, 4] (DORGA 순서: RT, RB, LT, LB)
    preds = out["logits_s3"].argmax(-1)

    # GUI 순서로 재정렬: [RT, LT, RB, LB]
    preds = preds[:, _DORGA_TO_GUI_ORDER]

    inference = preds.cpu().numpy().tolist()
    return inference


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


def inference_with_overlay(img_list, inference_list, selected_idx1, selected_idx2, flip_flag):
    """
    선택된 이미지에 2분할 선 및 severity 점수를 오버레이합니다.

    img_list: 이미지 목록 (numpy array)
    selected_idx1, selected_idx2: 선택된 이미지 인덱스
    inference_list: 각 이미지에 대한 [RT, LT, RB, LB] 추론 결과 리스트
    flip_flag: 좌우 반전 여부
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

    # flip_flag에 따른 4분면 매핑
    quad_map_0 = {
        (0, 0): ("RT", 0),
        (0, 1): ("LT", 1),
        (1, 0): ("RB", 2),
        (1, 1): ("LB", 3),
    }
    quad_map_1 = {
        (0, 0): ("LT", 1),
        (0, 1): ("RT", 0),
        (1, 0): ("LB", 3),
        (1, 1): ("RB", 2),
    }
    quad_map = quad_map_0 if flip_flag == 0 else quad_map_1

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