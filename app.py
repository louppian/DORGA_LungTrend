# app.py — Flip 제거 & Post-hoc Markov Correction 적용 버전

import gradio as gr
import os
import re
from utils.utils_image import files_info
from utils.utils_back import *
from utils.utils_severity import apply_posthoc_correction
from config import DEFAULT_USER_ID
import numpy as np
from datetime import datetime
from PIL import Image

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

css = """
#main-content { flex: 5 !important; }
#main-image { width: 400px !important; height: 400px !important; margin: 0 !important; }
#reset-button { width: 200px !important; margin: 0 !important; }
#gallery img { width: 200px !important; height: 200px !important; object-fit: contain !important; cursor: pointer !important; }
#sidebar { flex: 1 !important; position: fixed !important; right: 0 !important; top: 0 !important; bottom: 0 !important; width: 400px !important; background-color: #f6f7f9 !important; box-shadow: -2px 0 5px rgba(0,0,0,0.1) !important; overflow-y: auto !important; z-index: 100 !important; }
#toggle-wrapper { display: flex !important; flex-direction: row-reverse !important; align-items: center !important; justify-content: flex-end !important; gap: 8px !important; }
#colorbar, #graph { border: none !important; box-shadow: none !important; }
#popup-container { position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0px 0px 10px rgba(0,0,0,0.5); width: 300px; text-align: center; z-index: 3000 !important; overflow: visible !important; }
#popup-container ul.options.svelte-y6qw75 { position: fixed; top: 50%; left: 50%; z-index: 9999 !important; overflow: visible !important; background-color: #FFFFFF !important; color: #000000 !important; opacity: 1 !important; }
#popup-container ul.options.svelte-y6qw75 li { background-color: #000000 !important; color: #FFFFFF !important; opacity: 1 !important; }
#popup-container ul.options.svelte-y6qw75 li:hover { background-color: #EEEEEE !important; }
#overlay { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.5); z-index: 2999; }
.dark #sidebar { background-color: #2a2b32 !important; box-shadow: -2px 0 5px rgba(0,0,0,0.3) !important; }
"""


def parse_after_colon(txt: str) -> str:
    if ":" in txt:
        return txt.split(":", 1)[-1].strip()
    return txt.strip()


def convert_to_image(item):
    if isinstance(item, np.ndarray):
        return item
    elif isinstance(item, list):
        try:
            arr = np.array(item)
            if arr.ndim in (2, 3):
                return arr
            return None
        except Exception as e:
            print("Error converting list to image:", e)
            return None
    elif isinstance(item, str):
        if os.path.exists(item):
            try:
                return np.array(Image.open(item).convert("RGB"))
            except Exception as e:
                print("Error loading image from path:", item, e)
                return None
        return None
    elif isinstance(item, Image.Image):
        return np.array(item)
    return item


def process_upload(files):
    # 출력: gallery, orgn_img_list, inference_list, logits_state, datetime_state,
    #        patient_id_info, file_type_info, image_count_info,
    #        ext_pid, ext_str, ext_cnt,
    #        user_id_input, file_is_npz_state, inference_edit, selected_indices,
    #        inference_list_posthoc
    if not files:
        return (None, None, None, None, "",
                "File Type: Unknown", "Image Count: Unknown",
                "Patient ID: Unknown", "Patient ID : ",
                "File Format: Unknown", "File Count: Unknown",
                "Unknown", False, [], [], None)

    first_file = files[0]
    fname = first_file.name.lower()

    if fname.endswith(".npy") or fname.endswith(".npz"):
        try:
            data = np.load(first_file.name, allow_pickle=True)
        except Exception as e:
            return (None, None, None, None, "",
                    f"File Type: npy (error: {e})",
                    "Image Count: Unknown", "Patient ID: Unknown",
                    "Patient ID : ", "File Format: Unknown", "File Count: Unknown",
                    "Unknown", False, [], [], None)

        if isinstance(data, np.lib.npyio.NpzFile):
            orgn_images = data["original_images"] if "original_images" in data.files else None
            gallery_images = orgn_images
            patient_id = data["patient_id"] if "patient_id" in data.files else "Unknown"
            inference_list = data["inference"] if "inference" in data.files else []
            inference_edit = data["inference_edit"] if "inference_edit" in data.files else inference_list
            user_id_loaded = data["user_id"] if "user_id" in data.files else "Unknown"
            timestamp_loaded = data["timestamp"] if "timestamp" in data.files else np.datetime_as_string(np.datetime64('now'))
        elif isinstance(data, dict):
            orgn_images = data.get("original_images", None)
            gallery_images = orgn_images if orgn_images is not None else data.get("gallery", None)
            patient_id = data.get("patient_id", "Unknown")
            inference_list = data.get("inference", [])
            inference_edit = data.get("inference_edit", inference_list)
            user_id_loaded = data.get("user_id", "Unknown")
            timestamp_loaded = data.get("timestamp", np.datetime_as_string(np.datetime64('now')))
        elif isinstance(data, np.ndarray) and data.shape == () and isinstance(data.item(), dict):
            data = data.item()
            orgn_images = data.get("original_images", None)
            gallery_images = orgn_images if orgn_images is not None else data.get("gallery", None)
            patient_id = data.get("patient_id", "Unknown")
            inference_list = data.get("inference", [])
            inference_edit = data.get("inference_edit", inference_list)
            user_id_loaded = data.get("user_id", "Unknown")
            timestamp_loaded = data.get("timestamp", np.datetime_as_string(np.datetime64('now')))
        else:
            gallery_images = None
            orgn_images = None
            patient_id = "Unknown"
            inference_list = []
            inference_edit = []
            user_id_loaded = DEFAULT_USER_ID
            timestamp_loaded = np.datetime_as_string(np.datetime64('now'))

        if isinstance(inference_list, np.ndarray):
            inference_list = inference_list.tolist()
        if isinstance(inference_edit, np.ndarray):
            inference_edit = inference_edit.tolist()
        if isinstance(gallery_images, np.ndarray):
            gallery_images = gallery_images.tolist()
        if isinstance(orgn_images, np.ndarray):
            orgn_images = orgn_images.tolist()

        def _strip_check_prefix(label, fallback):
            if not label or label.strip() == "":
                label = fallback
            name = os.path.basename(label)
            name = re.sub(r"^[✔✓]\s*", "", name)
            name = re.sub(r"^\d+\.\s*", "", name)
            return name.strip()

        if gallery_images is not None:
            new_gallery, new_orgn = [], []
            for item in gallery_images:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    img_raw, label_raw = item
                else:
                    img_raw, label_raw = item, ""
                img_arr = convert_to_image(img_raw)
                clean_lbl = _strip_check_prefix(label_raw, first_file.name)
                new_gallery.append((img_arr, clean_lbl))
                new_orgn.append(img_arr)
            gallery_images = new_gallery
            orgn_images = new_orgn

        image_count = len(gallery_images) if gallery_images is not None else 0
        file_is_npz = fname.endswith(".npz")

        return (
            gallery_images,
            orgn_images,
            inference_list,
            None,                              # logits_state (NPZ에는 없음)
            timestamp_loaded,
            f"{patient_id}",
            "npz" if file_is_npz else "npy",
            f"{image_count}",
            f"Patient ID : {patient_id}",
            f"File Format: {'npz' if file_is_npz else 'npy'}",
            f"File Count: {image_count}",
            user_id_loaded,
            file_is_npz,
            inference_edit,
            [],
            None,                              # inference_list_posthoc (NPZ에는 없음)
        )
    else:
        result = files_info(files)
        # result: gallery_items, origin_items, inference_list, logits_np, datetimes, ext_info, cnt_info, pid_info
        logits_np = result[3]
        # 초기 업로드 시에는 AI inference 자체를 history로 사용 (아직 user edit 없음)
        posthoc_list = apply_posthoc_correction(logits_np, result[2]) if logits_np is not None else None
        return (
            result[0],                        # gallery
            result[1],                        # orgn_img_list
            result[2],                        # inference_list
            logits_np,                        # logits_state
            result[4],                        # datetime_state
            parse_after_colon(result[7]),     # patient_id_info
            parse_after_colon(result[5]),     # file_type_info
            parse_after_colon(result[6]),     # image_count_info
            result[7],                        # ext_pid
            result[5],                        # ext_str
            result[6],                        # ext_cnt
            DEFAULT_USER_ID,                   # user_id_loaded
            False,                             # file_is_npz
            result[2],                        # inference_edit
            [],
            posthoc_list,                     # inference_list_posthoc (자동 계산)
        )


def conditional_copy_inference_list(inference_list_val, inference_edit_val, file_is_npz_val):
    if file_is_npz_val:
        return inference_edit_val
    if inference_list_val is None:
        return None
    try:
        return [list(inference) for inference in inference_list_val]
    except Exception as e:
        print("conditional_copy_inference_list error:", e)
        return inference_list_val


def do_toggle_posthoc(logits_state, posthoc_flag, inference_list, inference_list_posthoc,
                      inference_list_edit, popup_click_info, selected_indices,
                      orgn_img_list, overlay_flag):
    new_flag = not posthoc_flag

    # P ON → user edit를 history로 사용하여 보정 재계산
    if new_flag and logits_state is not None and inference_list_edit:
        posthoc_list = apply_posthoc_correction(np.array(logits_state), inference_list_edit)
    else:
        posthoc_list = inference_list_posthoc

    display_inference = posthoc_list if (new_flag and posthoc_list) else inference_list

    # 이미지 갱신
    imgs = [None, None, None, None]
    if selected_indices and orgn_img_list and overlay_flag:
        for pos, idx in enumerate(selected_indices):
            if pos >= 4:
                break
            processed = inference_with_overlay(orgn_img_list, display_inference, idx, idx)
            imgs[pos] = processed[0] if processed and len(processed) > 0 else orgn_img_list[idx]

    elif selected_indices and orgn_img_list:
        for pos, idx in enumerate(selected_indices):
            if pos >= 4:
                break
            imgs[pos] = orgn_img_list[idx]

    fig = update_graph(inference_list, inference_list_edit, popup_click_info,
                       selected_indices, posthoc_list)
    btn_label = "[ P ]" if new_flag else "P"
    return imgs[0], imgs[1], imgs[2], imgs[3], new_flag, fig, gr.update(value=btn_label)


# ════════════════════════════════════════════════
# Gradio UI
# ════════════════════════════════════════════════

with gr.Blocks(css=css) as demo:

    # 상태 변수
    overlay_flag = gr.State(value=True)
    selected_indices = gr.State(value=[])
    orgn_img_list = gr.State(value=None)
    inference_list = gr.State(value=None)
    inference_list_edit = gr.State(value=None)
    datetime_state = gr.State(value=None)
    first_image1 = gr.State(value=None)
    first_image2 = gr.State(value=None)
    first_image3 = gr.State(value=None)
    first_image4 = gr.State(value=None)
    file_is_npz_state = gr.State(value=False)
    logits_state = gr.State(value=None)
    inference_list_posthoc = gr.State(value=None)
    posthoc_flag = gr.State(value=False)

    # ── 업로드 섹션 ──
    with gr.Column(visible=True) as upload_section:
        with gr.Row():
            with gr.Column():
                gr.Markdown("## File Upload")
                image_files = gr.File(
                    label="Upload Dicom Files", file_count="multiple",
                    file_types=[".dcm", ".dicom", ".jpg", ".jpeg", ".png", ".npy", ".npz"]
                )
            with gr.Column():
                gr.Markdown("## Loaded File Information")
                user_id_input = gr.Textbox(label="User ID", value=DEFAULT_USER_ID, interactive=False)
                patient_id_info = gr.Textbox(label="Patient ID", placeholder="Patient ID: Unknown")
                file_type_info = gr.Textbox(label="File Type", placeholder="File Type: Unknown", interactive=False)
                image_count_info = gr.Textbox(label="Image Count", placeholder="Image Count: Unknown", interactive=False)
        with gr.Row():
            process_button = gr.Button("Process Files")

    # ── 메인 섹션 ──
    with gr.Row(visible=False) as main_section:
        with gr.Column(elem_id="main-content"):
            with gr.Row(max_height=100):
                ext_pid = gr.Markdown(value="Patient ID : ")
                ext_str = gr.Markdown(value="File Format: Unknown")
                ext_cnt = gr.Markdown(value="File Count: Unknown")
                overlay_button = gr.Button("O", elem_id="overlay-button")
                posthoc_button = gr.Button("P", elem_id="posthoc-button")
                reset_button = gr.Button("Reset", elem_id="reset-button")
                export_button = gr.Button("Export")
            with gr.Row(max_height=400):
                gallery = gr.Gallery(columns=10, allow_preview=False, show_label=False, height=300)
            with gr.Row():
                image1 = gr.Image(show_label=False, show_download_button=False, show_fullscreen_button=False, sources=[], elem_id='main-image')
                image2 = gr.Image(show_label=False, show_download_button=False, show_fullscreen_button=False, sources=[], elem_id='main-image')
                image3 = gr.Image(show_label=False, show_download_button=False, show_fullscreen_button=False, sources=[], elem_id='main-image')
                image4 = gr.Image(show_label=False, show_download_button=False, show_fullscreen_button=False, sources=[], elem_id='main-image')
            with gr.Row(visible=True):
                graph = gr.Plot(label="Graph", show_label=False, elem_id="graph")

    # ── 팝업 ──
    with gr.Column(visible=False, elem_id="popup-container") as popup:
        popup_click_info = gr.Textbox(label="Selected Region", value="Unknown")
        popup_score_edit = gr.Dropdown(
            ["0", "1", "2", "3", "4"], allow_custom_value=True, label="Score Edit"
        )
        with gr.Row():
            popup_save = gr.Button("Save")
            popup_cancel = gr.Button("Cancel")

    # ════════════════════════════════════════════
    # 이벤트 바인딩
    # ════════════════════════════════════════════

    # 파일 업로드
    image_files.change(
        fn=process_upload,
        inputs=[image_files],
        outputs=[
            gallery, orgn_img_list, inference_list, logits_state, datetime_state,
            patient_id_info, file_type_info, image_count_info,
            ext_pid, ext_str, ext_cnt,
            user_id_input, file_is_npz_state, inference_list_edit, selected_indices,
            inference_list_posthoc
        ],
    )

    # Process Files 버튼
    process_button.click(
        fn=switch_section,
        inputs=[image_files],
        outputs=[upload_section, main_section],
    )

    # inference_list 변경 시 inference_edit 동기화
    inference_list.change(
        conditional_copy_inference_list,
        inputs=[inference_list, inference_list_edit, file_is_npz_state],
        outputs=inference_list_edit,
    )
    inference_list.change(
        update_graph,
        inputs=[inference_list, inference_list_edit, popup_click_info, selected_indices,
                inference_list_posthoc],
        outputs=[graph],
    )

    # 갤러리 선택 → 메인 이미지 업데이트 → 오버레이 → first_image 캐싱 → 그래프
    gallery.select(
        fn=on_select_gallery,
        inputs=[gallery, orgn_img_list, datetime_state, selected_indices],
        outputs=[image1, image2, image3, image4, selected_indices, gallery, overlay_flag],
    ).then(
        fn=inference_overlay,
        inputs=[
            image1, image2, image3, image4,
            orgn_img_list,
            overlay_flag, selected_indices, inference_list
        ],
        outputs=[image1, image2, image3, image4, overlay_flag],
    ).then(
        fn=lambda i1, i2, i3, i4, sel_idx, orgn_imgs: (
            orgn_imgs[sel_idx[0]] if len(sel_idx) > 0 else None,
            orgn_imgs[sel_idx[1]] if len(sel_idx) > 1 else None,
            orgn_imgs[sel_idx[2]] if len(sel_idx) > 2 else None,
            orgn_imgs[sel_idx[3]] if len(sel_idx) > 3 else None,
        ),
        inputs=[image1, image2, image3, image4, selected_indices, orgn_img_list],
        outputs=[first_image1, first_image2, first_image3, first_image4],
    ).then(
        fn=update_graph,
        inputs=[inference_list, inference_list_edit, popup_click_info, selected_indices,
                inference_list_posthoc],
        outputs=[graph],
    )

    # 이미지 클릭 → 팝업 표시
    for img, handler in zip(
        [image1, image2, image3, image4],
        [on_image1_click, on_image2_click, on_image3_click, on_image4_click],
    ):
        img.select(
            fn=handler,
            inputs=[
                image1, image2, image3, image4,
                first_image1, first_image2, first_image3, first_image4,
                overlay_flag, selected_indices, popup_click_info,
                orgn_img_list, inference_list,
            ],
            outputs=[
                popup_click_info,
                image1, image2, image3, image4,
                first_image1, first_image2, first_image3, first_image4,
            ],
        ).then(
            fn=lambda info: gr.update(visible=True),
            inputs=[popup_click_info],
            outputs=[popup],
        )

    # 팝업 저장/취소
    popup_save.click(
        fn=update_score,
        inputs=[popup_score_edit, popup_click_info, inference_list_edit],
        outputs=[inference_list_edit],
    ).then(
        fn=update_dropdown_default,
        inputs=[popup_click_info, inference_list, inference_list_edit],
        outputs=[popup_score_edit],
    ).then(
        fn=lambda: gr.update(visible=False),
        inputs=[],
        outputs=[popup],
    )

    popup_cancel.click(
        fn=lambda: gr.update(visible=False),
        inputs=[],
        outputs=[popup],
    )

    # Overlay 토글
    overlay_button.click(
        fn=toggle_overlay,
        inputs=[
            orgn_img_list, inference_list, selected_indices,
            popup_click_info, overlay_flag,
        ],
        outputs=[image1, image2, image3, image4, overlay_flag, overlay_button],
    )

    # Post-hoc Markov 토글
    posthoc_button.click(
        fn=do_toggle_posthoc,
        inputs=[
            logits_state, posthoc_flag, inference_list, inference_list_posthoc,
            inference_list_edit, popup_click_info, selected_indices,
            orgn_img_list, overlay_flag,
        ],
        outputs=[image1, image2, image3, image4, posthoc_flag, graph, posthoc_button],
    )

    # 팝업 값 변경
    popup_click_info.change(
        fn=update_dropdown_default,
        inputs=[popup_click_info, inference_list, inference_list_edit],
        outputs=[popup_score_edit],
    )

    popup_score_edit.change(
        fn=update_score,
        inputs=[popup_score_edit, popup_click_info, inference_list_edit],
        outputs=inference_list_edit,
    ).then(
        fn=update_dropdown_default,
        inputs=[popup_click_info, inference_list, inference_list_edit],
        outputs=[popup_score_edit],
    )

    def recalc_posthoc_and_graph(inference_list, inference_list_edit, popup_click_info,
                                 selected_indices, logits_state, posthoc_flag):
        """user edit 변경 시 posthoc 재계산 + 그래프 갱신"""
        posthoc_list = None
        if posthoc_flag and logits_state is not None and inference_list_edit:
            posthoc_list = apply_posthoc_correction(np.array(logits_state), inference_list_edit)
        fig = update_graph(inference_list, inference_list_edit, popup_click_info,
                           selected_indices, posthoc_list)
        return posthoc_list, fig

    inference_list_edit.change(
        fn=recalc_posthoc_and_graph,
        inputs=[inference_list, inference_list_edit, popup_click_info, selected_indices,
                logits_state, posthoc_flag],
        outputs=[inference_list_posthoc, graph],
    )

    # Export
    export_button.click(
        fn=export_npy,
        inputs=[inference_list, inference_list_edit, patient_id_info, user_id_input, gallery, orgn_img_list],
        outputs=[],
    )

    # Reset
    reset_button.click(
        fn=reset_page,
        inputs=[],
        outputs=[
            image_files, upload_section, main_section,     # 1~3
            gallery,                                        # 4
            image1, image2, image3, image4,                 # 5~8
            selected_indices,                               # 9
            orgn_img_list,                                  # 10
            first_image1, first_image2, first_image3, first_image4,  # 11~14
            patient_id_info,                                # 15
            inference_list, inference_list_edit,             # 16~17
            popup_click_info, popup_score_edit,              # 18~19
            ext_str, ext_cnt, ext_pid,                      # 20~22
            user_id_input,                                  # 23
            overlay_flag, file_is_npz_state,                # 24~25
            logits_state, inference_list_posthoc, posthoc_flag,  # 26~28
        ],
    )


demo.launch(share=False, debug=True)
