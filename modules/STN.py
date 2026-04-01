import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# ═══════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════

class LungSegmentationDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        pre_mask = cv2.imread(self.images[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=pre_mask)
            pre_mask = augmented['image']

        pre_mask = pre_mask / 255.0
        mask = mask / 255.0

        return pre_mask, mask


# ═══════════════════════════════════════════════════════════
# Loss
# ═══════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.contiguous()
        targets = targets.contiguous()
        intersection = (inputs * targets).sum(dim=2).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (
            inputs.sum(dim=2).sum(dim=1) + targets.sum(dim=2).sum(dim=1) + self.smooth
        )
        return 1 - dice.mean()


# ═══════════════════════════════════════════════════════════
# Identity 행렬 상수
# ═══════════════════════════════════════════════════════════

IDENTITY_THETA = torch.tensor(
    [[1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0]], dtype=torch.float32
)


# ═══════════════════════════════════════════════════════════
# STN (Idempotent 버전)
# ═══════════════════════════════════════════════════════════

class STN(nn.Module):
    """
    Spatial Transformer Network with idempotency support.

    변경 사항 (기존 대비):
      1. forward()가 theta만 반환 (기존과 동일)
      2. predict_theta(): 입력으로부터 θ만 예측 (변환 적용 안 함)
      3. transform(): θ를 받아서 변환만 적용
      4. forward_with_idempotency(): 학습 시 사용 — θ1, aligned, θ2를 모두 반환
      5. identity_gate_threshold: 추론 시 θ가 identity에 가까우면 스킵
    """

    def __init__(self, in_shape=(1, 512, 512), mask_resize: int = 512,
                 dense_neurons=50, freeze_align_model=False,
                 identity_gate_threshold=0.01):
        super(STN, self).__init__()

        assert not in_shape[1] % mask_resize, "The STN size must be a multiple of mask size"
        trainable = not freeze_align_model

        self.identity_gate_threshold = identity_gate_threshold

        # MaxPooling for input adaptation
        self.pool1 = nn.MaxPool2d(
            kernel_size=(in_shape[1] // mask_resize, in_shape[2] // mask_resize)
        )

        # Convolutional and pooling layers
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(in_channels=in_shape[0], out_channels=20, kernel_size=5, stride=1)
        if not trainable:
            self.conv1.requires_grad_(False)

        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=5, stride=1)
        if not trainable:
            self.conv2.requires_grad_(False)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(
            in_features=self.calculate_flatten_size(in_shape),
            out_features=dense_neurons,
        )
        if not trainable:
            self.fc1.requires_grad_(False)

        self.relu = nn.ReLU()

        # Final alignment layer (6 affine parameters)
        self.fc2 = nn.Linear(in_features=dense_neurons, out_features=6)
        if not trainable:
            self.fc2.requires_grad_(False)

        # Initialize to identity transformation
        self.fc2.weight.data.zero_()
        self.fc2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def calculate_flatten_size(self, in_shape):
        dummy_input = torch.randn(1, in_shape[0], in_shape[1], in_shape[2])
        x = self.pool1(dummy_input)
        x = self.pool2(self.conv1(x))
        x = self.pool3(self.conv2(x))
        return x.numel()

    # ─────────────────────────────────────────
    # 핵심 분리: θ 예측 / 변환 적용
    # ─────────────────────────────────────────

    def predict_theta(self, x):
        """입력 이미지로부터 아핀 파라미터 θ만 예측 (변환 적용 안 함)"""
        xs = self.pool1(x)
        xs = self.pool2(self.conv1(xs))
        xs = self.pool3(self.conv2(xs))
        xs = self.flatten(xs)
        xs = self.relu(self.fc1(xs))
        theta = self.fc2(xs)
        theta = theta.view(-1, 2, 3)
        return theta

    @staticmethod
    def transform(x, theta):
        """θ를 사용해 이미지에 아핀 변환 적용"""
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)

    # ─────────────────────────────────────────
    # forward: 기존 인터페이스 유지 (θ 반환)
    # ─────────────────────────────────────────

    def forward(self, x):
        """기존과 동일: θ 반환. 추론 시 identity gate 적용."""
        theta = self.predict_theta(x)

        # 추론 모드에서 identity gate
        if not self.training and self.identity_gate_threshold > 0:
            theta = self._apply_identity_gate(theta)

        return theta

    # ─────────────────────────────────────────
    # 학습용: Idempotency forward
    # ─────────────────────────────────────────

    def forward_with_idempotency(self, x):
        """
        학습 시 idempotency loss 계산을 위한 forward.

        Returns:
            theta1:  첫 번째 변환 파라미터 (2×3)
            aligned: θ1으로 변환된 이미지
            theta2:  aligned를 다시 넣었을 때의 θ (identity에 가까워야 함)
        """
        # 1차 변환
        theta1 = self.predict_theta(x)
        aligned = self.transform(x, theta1)

        # 2차 변환 (idempotency 체크)
        theta2 = self.predict_theta(aligned.detach())  # detach로 1차 그래디언트 분리 가능

        return theta1, aligned, theta2

    # ─────────────────────────────────────────
    # Identity gate (추론 시)
    # ─────────────────────────────────────────

    def _apply_identity_gate(self, theta):
        """
        θ가 identity에 충분히 가까우면 identity로 스냅.
        이미 정렬된 이미지의 이중 변환을 방지.
        """
        identity = IDENTITY_THETA.to(theta.device).unsqueeze(0)  # (1, 2, 3)
        diff = (theta - identity).abs().mean(dim=(1, 2))          # (B,)
        mask = (diff < self.identity_gate_threshold).float()       # (B,)
        mask = mask.view(-1, 1, 1)                                 # (B, 1, 1)

        # 가까우면 identity, 아니면 원래 theta
        return mask * identity + (1 - mask) * theta


# ═══════════════════════════════════════════════════════════
# Idempotency Loss
# ═══════════════════════════════════════════════════════════

class IdempotencyLoss(nn.Module):
    """
    θ2가 identity matrix에 가까워지도록 하는 loss.
    L = ||θ2 - I||²
    """
    def __init__(self):
        super().__init__()

    def forward(self, theta2):
        """
        theta2: (B, 2, 3) — 두 번째 통과에서 나온 아핀 파라미터
        """
        identity = IDENTITY_THETA.to(theta2.device).unsqueeze(0).expand_as(theta2)
        return F.mse_loss(theta2, identity)


# ═══════════════════════════════════════════════════════════
# 학습 예시
# ═══════════════════════════════════════════════════════════

"""
사용 예시 (기존 학습 코드 기반):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stn_model = STN(identity_gate_threshold=0.01).to(device)

    alignment_criterion = F.mse_loss       # preseg → postseg 정렬 loss
    idempotency_criterion = IdempotencyLoss()
    optimizer = torch.optim.Adam(stn_model.parameters(), lr=0.0001)

    lambda_idem = 1.0  # idempotency loss 가중치 (0.5~2.0 범위에서 튜닝)

    for epoch in range(10):
        stn_model.train()
        running_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device, dtype=torch.float)
            masks  = masks.to(device, dtype=torch.float)

            optimizer.zero_grad()

            # Idempotency-aware forward
            theta1, aligned, theta2 = stn_model.forward_with_idempotency(images)

            # 1) 정렬 loss: 변환된 이미지 ≈ 목표 마스크
            aligned_squeezed = aligned.squeeze(1)
            loss_align = alignment_criterion(aligned_squeezed, masks)

            # 2) Idempotency loss: 두 번째 θ ≈ identity
            loss_idem = idempotency_criterion(theta2)

            # 총 loss
            loss = loss_align + lambda_idem * loss_idem
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/10], Loss: {running_loss / len(train_loader):.4f}")

    torch.save(stn_model.state_dict(), 'stn_model_idempotent.pth')
"""