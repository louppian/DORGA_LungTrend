# utils/utils_back.py — Flip 제거 & Post-hoc Markov 적용 버전

import gradio as gr
from utils.utils_severity import inference_with_overlay, get_severity_color, apply_posthoc_correction
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import cv2
import torchvision.transforms as transforms
import torch
from datetime import datetime
from matplotlib import gridspec
from config import RESULTS_DIR

to_tensor = transforms.ToTensor()
image_resize_transform = transforms.Resize(
    (512, 512), interpolation=transforms.InterpolationMode.BILINEAR
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ROI 순서: DORGA 네이티브 [RT, RB, LT, LB]
ROI_MAPPING = {"RT": 0, "RB": 1, "LT": 2, "LB": 3}


# ════════════════════════════════════════════════
# 섹션 전환
# ════════════════════════════════════════════════

def switch_section(files):
    if files:
        return gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(visible=True), gr.update(visible=False)


# ════════════════════════════════════════════════
# 갤러리 선택 관련
# ════════════════════════════════════════════════

def strip_gallery_prefix(label_text: str) -> str:
    """갤러리 라벨에서 '✔ 1. ' 같은 선택 표시와 번호를 모두 제거하고 원본 파일명만 반환"""
    cleaned = re.sub(r'^[✔✓]\s*', '', label_text)
    cleaned = re.sub(r'^\d+\.\s*', '', cleaned)
    cleaned = re.sub(r'^[✔✓]\s*', '', cleaned)
    cleaned = re.sub(r'^\d+\.\s*', '', cleaned)
    return cleaned.strip()


GALLERY_BORDER_COLOR = (0, 180, 255)
GALLERY_BORDER_THICKNESS = 4


def _add_selection_border(img):
    if img is None:
        return img
    arr = np.array(img, copy=True)
    if arr.dtype.kind == 'f':
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.shape[2] == 1:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    h, w = arr.shape[:2]
    t = GALLERY_BORDER_THICKNESS
    cv2.rectangle(arr, (0, 0), (w - 1, h - 1), GALLERY_BORDER_COLOR, t)
    return arr


def on_select_gallery(gallery, img_list, datetime_list, selected_indices, evt: gr.SelectData):
    overlay_chk = False
    if not gallery:
        return None, None, None, None, selected_indices, gallery, False

    idx = evt.index
    if idx < 0 or idx >= len(gallery):
        return None, None, None, None, selected_indices, gallery, False

    on_select_gallery.cached_names = [strip_gallery_prefix(text) for _, text in gallery]

    if idx in selected_indices:
        selected_indices.remove(idx)
    else:
        if len(selected_indices) < 4:
            selected_indices.append(idx)
        else:
            selected_indices.pop(0)
            selected_indices.append(idx)

    updated_gallery = []
    for i, (img, _) in enumerate(gallery):
        original_filename = on_select_gallery.cached_names[i]
        is_selected = i in selected_indices
        if is_selected:
            new_text = f"✔ {i+1}. {original_filename}"
            display_img = _add_selection_border(img_list[i] if img_list else img)
        else:
            new_text = f"{i+1}. {original_filename}"
            display_img = img_list[i] if img_list else img
        updated_gallery.append((display_img, new_text))
    gallery_update = gr.update(value=updated_gallery)

    main_images = []
    for i in range(4):
        if i < len(selected_indices):
            main_images.append(img_list[selected_indices[i]])
        else:
            main_images.append(None)

    if len(selected_indices) >= 2:
        overlay_chk = True

    return (
        main_images[0], main_images[1], main_images[2], main_images[3],
        selected_indices, gallery_update, overlay_chk
    )


# ════════════════════════════════════════════════
# 오버레이 / 클릭 헬퍼
# ════════════════════════════════════════════════

def inference_overlay(image1, image2, image3, image4, orgn_img_list,
                      overlay_flag, selected_indices, inference_list):
    new_images = [None, None, None, None]
    if len(selected_indices) == 0:
        return image1, image2, image3, image4, overlay_flag
    for pos, idx in enumerate(selected_indices):
        processed = inference_with_overlay(orgn_img_list, inference_list, idx, idx)
        new_images[pos] = processed[0] if processed and len(processed) > 0 else orgn_img_list[idx]
    for pos in range(len(selected_indices), 4):
        new_images[pos] = None
    return new_images[0], new_images[1], new_images[2], new_images[3], overlay_flag


def get_quadrant_label(x, y, width, height):
    center_x = width / 2
    center_y = height / 2
    row = 0 if y < center_y else 1
    col = 0 if x < center_x else 1
    mapping = {(0, 0): "RT", (0, 1): "LT",
               (1, 0): "RB", (1, 1): "LB"}
    return mapping[(row, col)]


def draw_click_border(image, quadrant_label):
    if image is None:
        return None
    image_copy = image.copy()
    if image_copy.dtype.kind == 'f':
        if np.max(image_copy) <= 1.0:
            image_copy = (image_copy * 255).astype(np.uint8)
        else:
            image_copy = image_copy.astype(np.uint8)
    h, w = image_copy.shape[:2]
    half_w, half_h = w // 2, h // 2
    quadrant_boxes = {
        "RT": (0, 0, half_w, half_h),
        "LT": (half_w, 0, w - 1, half_h),
        "RB": (0, half_h, half_w, h - 1),
        "LB": (half_w, half_h, w - 1, h - 1),
    }
    if quadrant_label in quadrant_boxes:
        x1, y1, x2, y2 = quadrant_boxes[quadrant_label]
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), (255, 255, 255), 1)
    return image_copy


def safe_copy_image(img, fallback=None):
    if img is None:
        return fallback
    if isinstance(img, np.ndarray):
        return img.copy()
    if isinstance(img, Image.Image):
        return img.copy()
    return fallback


def get_adjusted_clean_image(first_img, idx, placeholder_img,
                             orgn_img_list, inference_list, overlay_flag):
    if overlay_flag:
        processed = inference_with_overlay(orgn_img_list, inference_list, idx, idx)
        if processed and len(processed) > 0:
            return processed[0]
        else:
            return safe_copy_image(first_img, fallback=placeholder_img)
    else:
        return safe_copy_image(first_img, fallback=placeholder_img)


# ════════════════════════════════════════════════
# 이미지 클릭 핸들러 (통합)
# ════════════════════════════════════════════════

def _on_image_click(slot_index,
                    image1, image2, image3, image4,
                    first_image1, first_image2, first_image3, first_image4,
                    overlay_flag, selected_indices, click_info,
                    orgn_img_list, inference_list, evt: gr.SelectData):
    x, y = evt.index
    quadrant_label = get_quadrant_label(x, y, 512, 512)

    images = [image1, image2, image3, image4]
    firsts = [first_image1, first_image2, first_image3, first_image4]

    fresh = []
    for pos in range(4):
        idx = selected_indices[pos] if pos < len(selected_indices) else 0
        fresh.append(get_adjusted_clean_image(
            firsts[pos], idx, images[pos],
            orgn_img_list, inference_list, overlay_flag
        ))

    used_idx = selected_indices[slot_index] if slot_index < len(selected_indices) else 0
    bordered = draw_click_border(fresh[slot_index], quadrant_label)
    new_click_info = f"Image {used_idx + 1}, {quadrant_label}"

    result_images = list(fresh)
    result_images[slot_index] = bordered

    return (
        new_click_info,
        result_images[0], result_images[1], result_images[2], result_images[3],
        fresh[0], fresh[1], fresh[2], fresh[3],
    )


def on_image1_click(image1, image2, image3, image4,
                    first_image1, first_image2, first_image3, first_image4,
                    overlay_flag, selected_indices, click_info,
                    orgn_img_list, inference_list, evt: gr.SelectData):
    return _on_image_click(0, image1, image2, image3, image4,
                           first_image1, first_image2, first_image3, first_image4,
                           overlay_flag, selected_indices, click_info,
                           orgn_img_list, inference_list, evt)


def on_image2_click(image1, image2, image3, image4,
                    first_image1, first_image2, first_image3, first_image4,
                    overlay_flag, selected_indices, click_info,
                    orgn_img_list, inference_list, evt: gr.SelectData):
    return _on_image_click(1, image1, image2, image3, image4,
                           first_image1, first_image2, first_image3, first_image4,
                           overlay_flag, selected_indices, click_info,
                           orgn_img_list, inference_list, evt)


def on_image3_click(image1, image2, image3, image4,
                    first_image1, first_image2, first_image3, first_image4,
                    overlay_flag, selected_indices, click_info,
                    orgn_img_list, inference_list, evt: gr.SelectData):
    return _on_image_click(2, image1, image2, image3, image4,
                           first_image1, first_image2, first_image3, first_image4,
                           overlay_flag, selected_indices, click_info,
                           orgn_img_list, inference_list, evt)


def on_image4_click(image1, image2, image3, image4,
                    first_image1, first_image2, first_image3, first_image4,
                    overlay_flag, selected_indices, click_info,
                    orgn_img_list, inference_list, evt: gr.SelectData):
    return _on_image_click(3, image1, image2, image3, image4,
                           first_image1, first_image2, first_image3, first_image4,
                           overlay_flag, selected_indices, click_info,
                           orgn_img_list, inference_list, evt)


# ════════════════════════════════════════════════
# Inference / Score 관련
# ════════════════════════════════════════════════

def copy_inference_list(inference_list_val):
    print('2. Inference Completes')
    return [list(inference) for inference in inference_list_val]


def update_dropdown_default(click_info, inference_list, inference_list_edit):
    if click_info == "Unknown" or not inference_list_edit:
        return "0"
    try:
        parts = click_info.split(",")
        image_str = parts[0].strip()
        region_str = parts[1].strip()
        image_idx = int(image_str.split(" ")[1]) - 1
        region_idx = ROI_MAPPING.get(region_str, 0)
        original_score = inference_list[image_idx][region_idx]
        current_score = inference_list_edit[image_idx][region_idx]
        score_str = str(current_score)
        if current_score != original_score:
            score_str += "*"
        return score_str
    except Exception as e:
        print("update_dropdown_default error:", e)
        return "0"


def update_score(new_score, click_info, inference_list_edit):
    if click_info == "Unknown" or not inference_list_edit:
        return inference_list_edit
    try:
        parts = click_info.split(",")
        image_str = parts[0].strip()
        region_str = parts[1].strip()
        image_idx = int(image_str.split(" ")[1]) - 1
        region_idx = ROI_MAPPING.get(region_str, 0)
        cleaned_score = new_score.replace("*", "").strip()
        inference_list_edit[image_idx][region_idx] = int(cleaned_score)
    except Exception as e:
        print("Error : update_score: ", e)
    return inference_list_edit


# ════════════════════════════════════════════════
# Post-hoc Markov 토글
# ════════════════════════════════════════════════

def toggle_posthoc(logits_state, posthoc_flag):
    new_flag = not posthoc_flag
    if new_flag and logits_state is not None:
        posthoc_list = apply_posthoc_correction(logits_state)
        return posthoc_list, new_flag
    return None, new_flag


# ════════════════════════════════════════════════
# 그래프
# ════════════════════════════════════════════════

def update_graph(inference_list, inference_list_edit, click_info, selected_indices,
                 inference_list_posthoc=None):
    if not inference_list:
        return None

    plt.close('all')

    roi_names = ['RT', 'RB', 'LT', 'LB']
    n_imgs = len(inference_list)
    xs = list(range(1, n_imgs + 1))

    upd_i = upd_roi = None
    if click_info != "Unknown":
        try:
            img_str, roi_str = [p.strip() for p in click_info.split(",")]
            upd_i = int(img_str.split(" ")[1]) - 1
            upd_roi = ROI_MAPPING.get(roi_str)
        except:
            pass

    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1.2, 1, 1])
    axT = fig.add_subplot(gs[0, :])
    axRois = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
              fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]
    fig.patch.set_facecolor("#0F0F11")

    handles, labels = [], []

    # ── Total ──
    tot_ai = [sum(r) for r in inference_list]
    tot_edit = ([sum(r) for r in inference_list_edit]
                if inference_list_edit and len(inference_list_edit) == n_imgs else None)
    tot_posthoc = ([sum(r) for r in inference_list_posthoc]
                   if inference_list_posthoc and len(inference_list_posthoc) == n_imgs else None)

    axT.set_facecolor("#0F0F11")
    if tot_posthoc:
        axT.plot(xs, tot_posthoc, 's-', ms=4, lw=1.2, color='#00FF88', label="Total Post-Hoc", zorder=4)
    else:
        axT.plot(xs, tot_ai, 's-', ms=4, lw=1.2, color='#00FFFF', label="Total AI", zorder=3)
    if tot_edit:
        axT.plot(xs, tot_edit, 's--', ms=4, lw=1.2, color='#E26114', label="Total Edit", zorder=2)

    if tot_edit and upd_i is not None:
        axT.scatter(upd_i + 1, tot_edit[upd_i], s=60, color="#FF00FF", zorder=5)
    tot_base = tot_posthoc or tot_ai
    for sel in selected_indices or []:
        if isinstance(sel, int) and 0 <= sel < n_imgs:
            y = tot_edit[sel] if tot_edit else tot_base[sel]
            axT.annotate("", xy=(sel + 1, y), xytext=(sel + 1, y + 1),
                         arrowprops=dict(arrowstyle="->", color="red", lw=2))

    axT.set_xlabel("Shots", color='white')
    axT.set_ylabel("Severity", color='white')
    axT.set_xticks(xs)
    axT.set_yticks(range(0, 17, 2))
    axT.tick_params(axis='both', colors='white')
    for sp in axT.spines.values():
        sp.set_color("white")

    # ── 각 ROI ──
    for i, (ax, name) in enumerate(zip(axRois, roi_names)):
        ax.set_facecolor("#0F0F11")
        ai_vals = [row[i] for row in inference_list]
        ed_vals = ([row[i] for row in inference_list_edit]
                   if inference_list_edit and len(inference_list_edit) == n_imgs else None)
        ph_vals = ([row[i] for row in inference_list_posthoc]
                   if inference_list_posthoc and len(inference_list_posthoc) == n_imgs else None)

        if ph_vals:
            h1, = ax.plot(xs, ph_vals, 'o-', ms=3, lw=1, color='#00FF88', label="Post-Hoc", zorder=4)
            handles.append(h1)
            labels.append("Post-Hoc")
        else:
            h1, = ax.plot(xs, ai_vals, 'o-', ms=3, lw=1, color='#00FFFF', label="AI Inference", zorder=3)
            handles.append(h1)
            labels.append("AI Inference")

        if ed_vals:
            h2, = ax.plot(xs, ed_vals, 'o--', ms=3, lw=1, color='#E26114', label="User Edit", zorder=2)
            handles.append(h2)
            labels.append("User Edit")
            if upd_roi == i and upd_i is not None:
                ax.scatter(upd_i + 1, ed_vals[upd_i], color='#FF00FF', s=40, zorder=5)

        base_vals = ph_vals or ai_vals
        for sel in selected_indices or []:
            if isinstance(sel, int) and 0 <= sel < n_imgs:
                y = ed_vals[sel] if ed_vals else base_vals[sel]
                ax.annotate("", xy=(sel + 1, y), xytext=(sel + 1, y + 0.5),
                            arrowprops=dict(arrowstyle="->", color="red", lw=2))

        ax.set_title(name, color='white')
        ax.set_xlabel("Shots", color='white')
        ax.set_ylabel("Severity", color='white')
        ax.set_xticks(xs)
        ax.set_yticks(range(5))
        ax.tick_params(axis='both', colors='white')
        if i % 2 == 1:
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
        for sp in ax.spines.values():
            sp.set_color("white")

    # ── 범례 ──
    order = ["AI Inference", "Post-Hoc", "User Edit"]
    uniq = {}
    for lbl in order:
        for h, l in zip(handles, labels):
            if l == lbl and l not in uniq:
                uniq[l] = h
    legend = fig.legend(list(uniq.values()), list(uniq.keys()),
                        loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=3,
                        frameon=False)
    for t in legend.get_texts():
        t.set_color("white")

    fig.tight_layout(pad=0.6)
    fig.subplots_adjust(top=0.88)
    return fig


# ════════════════════════════════════════════════
# 이미지 유틸
# ════════════════════════════════════════════════

def ensure_pil_image(obj):
    if obj is None:
        return None
    if isinstance(obj, Image.Image):
        return obj
    if isinstance(obj, np.ndarray):
        return Image.fromarray(obj)
    if isinstance(obj, str):
        if os.path.exists(obj):
            return Image.open(obj)
        else:
            raise ValueError(f"해당 경로를 찾을 수 없습니다: {obj}")
    if isinstance(obj, dict) and "name" in obj:
        path = obj["name"]
        if os.path.exists(path):
            return Image.open(path)
        else:
            raise ValueError(f"해당 path를 찾을 수 없습니다: {path}")
    raise ValueError(f"지원하지 않는 이미지 타입: {type(obj)}")


def ensure_four(img_list, fallback):
    if img_list is None or len(img_list) < 4:
        if fallback is None or len(fallback) < 4:
            return [None, None, None, None]
        else:
            return fallback
    return img_list


# ════════════════════════════════════════════════
# 메인 이미지 새로고침 (Overlay 토글 시)
# ════════════════════════════════════════════════

def refresh_main_images(orgn_img_list, inference_list, selected_indices,
                        overlay_flag, popup_click_info):
    new_main_images = [None, None, None, None]
    if not selected_indices or len(selected_indices) == 0:
        return new_main_images

    parse_idx = None
    parse_quad = None
    if popup_click_info and "," in popup_click_info:
        try:
            parts = popup_click_info.split(",")
            image_part = parts[0].strip()
            quad_part = parts[1].strip()
            x_num = image_part.split(" ")[1]
            parse_idx = int(x_num) - 1
            parse_quad = quad_part
        except:
            pass

    for pos, idx in enumerate(selected_indices):
        if pos >= 4:
            break

        if overlay_flag:
            processed = inference_with_overlay(
                orgn_img_list, inference_list, idx, idx
            )
            final_img = processed[0] if processed else orgn_img_list[idx]
        else:
            final_img = orgn_img_list[idx]

        if parse_idx == idx and parse_quad:
            final_img = draw_click_border(final_img, parse_quad)
        new_main_images[pos] = final_img

    return new_main_images


# ════════════════════════════════════════════════
# 토글 버튼
# ════════════════════════════════════════════════

def toggle_overlay(orgn_img_list, inference_list, selected_indices,
                   popup_click_info, overlay_flag):
    new_overlay_flag = not overlay_flag
    main_images = refresh_main_images(
        orgn_img_list, inference_list, selected_indices,
        new_overlay_flag, popup_click_info
    )
    btn_label = "[ O ]" if new_overlay_flag else "O"
    return (
        main_images[0], main_images[1], main_images[2], main_images[3],
        new_overlay_flag, btn_label
    )


# ════════════════════════════════════════════════
# 리셋
# ════════════════════════════════════════════════

def reset_page():
    print("페이지를 초기화 합니다")
    if hasattr(on_select_gallery, "cached_names"):
        delattr(on_select_gallery, "cached_names")

    return (
        # ① 파일·섹션
        gr.update(value=None),          # image_files
        gr.update(visible=True),        # upload_section
        gr.update(visible=False),       # main_section
        gr.update(value=[]),            # gallery

        # ② 4-Viewer
        gr.update(value=None), gr.update(value=None),
        gr.update(value=None), gr.update(value=None),

        # ③ 인덱스 & 이미지 리스트
        gr.update(value=[]),            # selected_indices
        gr.update(value=None),          # orgn_img_list

        # ④ first_image1-4
        gr.update(value=None), gr.update(value=None),
        gr.update(value=None), gr.update(value=None),

        # ⑤ 환자 정보
        gr.update(value="File Type: Unknown\nPatient ID: Unknown\nImage Count: Unknown"),

        # ⑥ inference
        gr.update(value=None),          # inference_list
        gr.update(value=None),          # inference_list_edit

        # ⑦ Popup
        gr.update(value="Unknown"),     # popup_click_info
        gr.update(value="0"),           # popup_score_edit

        # ⑧ 하단 마크다운·Textbox
        gr.update(value="File Format: Unknown"),  # ext_str
        gr.update(value="File Count: Unknown"),   # ext_cnt
        gr.update(value="Patient ID : "),         # ext_pid
        gr.update(value=""),                       # user_id_input

        gr.update(value=False),   # overlay_flag
        gr.update(value=False),   # file_is_npz_state
        gr.update(value=None),    # logits_state
        gr.update(value=None),    # inference_list_posthoc
        gr.update(value=False),   # posthoc_flag
    )


# ════════════════════════════════════════════════
# NPZ + 엑셀 내보내기
# ════════════════════════════════════════════════

def _safe(txt):
    return re.sub(r"[\\/:*?\"<>|]", "_", str(txt).strip())


def _make_npz_dir(base, user_id, date_dir):
    path = os.path.join(base, user_id, date_dir)
    os.makedirs(path, exist_ok=True)
    return path


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.kind == "f":
        arr = np.clip(arr, 0, 1) * 255
    return arr.astype(np.uint8)


def _clean_gallery_label(label: str) -> str:
    cleaned = re.sub(r"^[✔✓]\s*", "", label)
    cleaned = re.sub(r"^\d+\.\s*", "", cleaned)
    return cleaned.strip()


def export_npy(inference_list, inference_list_edit,
               ext_pid, user_id_input, gallery, orgn_img_list):

    patient_id = _safe(ext_pid.split("Patient ID:")[-1]) if "Patient ID:" in ext_pid else _safe(ext_pid)
    user_id = _safe(user_id_input or "Unknown") or "Unknown"

    now = datetime.now()
    yymmdd = now.strftime("%y%m%d")
    hhmmss = now.strftime("%H%M%S")

    BASE_DIR = RESULTS_DIR
    npz_dir = _make_npz_dir(BASE_DIR, user_id, yymmdd)
    npz_path = os.path.join(npz_dir, f"{patient_id}_{hhmmss}.npz")

    img_dir = os.path.join(npz_dir, f"{patient_id}_{hhmmss}_images")
    os.makedirs(img_dir, exist_ok=True)

    for idx, (img_arr, gallery_item) in enumerate(zip(orgn_img_list or [], gallery or [])):
        raw_label = gallery_item[1] if isinstance(gallery_item, (list, tuple)) and len(gallery_item) >= 2 else str(gallery_item)
        true_name = _clean_gallery_label(raw_label)
        root, _ = os.path.splitext(true_name)
        save_name = f"{idx+1:02d}_{root}.png"
        save_path = os.path.join(img_dir, save_name)
        arr = _to_uint8(np.asarray(img_arr))
        mode = "L" if arr.ndim == 2 else "RGB"
        Image.fromarray(arr, mode=mode).save(save_path)

    print(f"[IMG]  saved → {img_dir} ({len(orgn_img_list or [])} files)")

    export_data = {
        "patient_id":      patient_id,
        "user_id":         user_id,
        "timestamp":       now.strftime("%Y-%m-%d %H:%M:%S"),
        "gallery":         list(gallery or []),
        "original_images": list(orgn_img_list or []),
        "inference":       inference_list,
        "inference_edit":  inference_list_edit,
    }
    np.savez(npz_path, **export_data)
    print(f"[NPZ] saved → {npz_path}")

    excel_path = os.path.join(BASE_DIR, "roi_export.xlsx")
    columns = ["PatientID", "UserID", "Timestamp"]
    for i in range(1, 21):
        columns += [f"RT{i}", f"RB{i}", f"LT{i}", f"LB{i}"]

    def _row(scores, label):
        row = [patient_id, label, now.strftime("%Y-%m-%d %H:%M:%S")]
        scores = scores or []
        for i in range(20):
            if i < len(scores):
                row.extend(scores[i])
            else:
                row.extend([""] * 4)
        return row

    rows_new = [
        _row(inference_list, "Original"),
        _row(inference_list_edit, user_id),
    ]
    df_new = pd.DataFrame(rows_new, columns=columns)

    if os.path.exists(excel_path):
        df_old = pd.read_excel(excel_path)
        mask = ((df_old["PatientID"] == patient_id)
                & (df_old["UserID"] == user_id)
                & (df_old["Timestamp"] == now.strftime("%Y-%m-%d %H:%M:%S")))
        df_old = df_old.loc[~mask]
        df_final = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_final = df_new

    df_final.to_excel(excel_path, index=False)
    print(f"[Excel] updated → {excel_path}")
    gr.Info("Export Complete ✅")
