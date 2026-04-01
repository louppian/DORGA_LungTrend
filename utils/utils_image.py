# utils_image.py: 이미지 전처리 및 파일 정보 추출을 위한 모듈
import torch
import pydicom
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import os
from utils.utils_severity import all_image_inference
from utils.utils_segmentation import segmentation
import cv2
import gradio as gr
import random
from datetime import datetime
from modules.STN import STN
import torch.nn.functional as F
import gdown
from config import DEFAULT_DATETIME

to_tensor = transforms.ToTensor()
image_resize_transform = transforms.Resize((512,512), interpolation=transforms.InterpolationMode.BILINEAR)
mask_resize_transform = transforms.Resize((512,512), interpolation=transforms.InterpolationMode.NEAREST)

def spatial_normalize(original_batch, preseg_batch):

    import torch
    import torch.nn.functional as F

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stn_model = STN(in_shape=(1, 512, 512))
    MODEL_URL = "https://drive.google.com/uc?id=1WSACPIC_nGsnHksl4jiYzMyqK-79tRi0"
    MODEL_PATH = "assets/weights/stn_weights.pth"
    if not os.path.exists(MODEL_PATH): gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    stn_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    stn_model.to(device)
    stn_model.eval()

    if preseg_batch.shape[-2:] != (512, 512):
        preseg_batch = F.interpolate(preseg_batch.float(), size=(512, 512), mode='nearest')

    # 디바이스 이동
    preseg_batch = preseg_batch.to(device)
    original_batch = original_batch.to(device)

    with torch.no_grad():
        # STN 모델에 PreSeg 이미지 입력 → 아핀 파라미터 theta 반환
        theta = stn_model(preseg_batch)

    # theta를 원본 이미지에 적용하기 위한 grid 생성
    grid = F.affine_grid(theta, original_batch.size(), align_corners=False)
    aligned_original = F.grid_sample(original_batch, grid, align_corners=False)


    return aligned_original



### pad 이후 필요한 경우 1024 로 이미지 크기 조정
def preprocess(input_img):
    img = np.array(input_img, dtype=np.uint8)  # 명시적으로 uint8로 변환
    h, w = img.shape[:2]  # .shape 사용하여 height, width 가져오기
    
    if h != 1024 or w != 1024:
        img = cv2.resize(img, (1024, 1024))
    return img


### pad 이후 필요한 경우 512로 이미지 크기 조정
def preprocess_(input_img):
    img = np.array(input_img, dtype=np.uint8)  # 명시적으로 uint8로 변환
    h, w = img.shape[:2]
    if h != 512 or w != 512:
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
    return img

### 확장자 추출
def convert_ext(file_fullname):
    ext = os.path.splitext(file_fullname)[1].lower()
    if ext == ".jpg" or ext =='.jpeg' :    ext_str = "JPEG"
    elif ext == ".png":                    ext_str = "PNG"
    elif ext == ".dcm" or ext == ".dicom": ext_str = "DICOM"
    else: ext_str = "Unknown"
    return f'File Format: {ext_str}'

def convert_cnt(cnt_image) : return f'File Count: {cnt_image}'
def convert_pid(patient_id): return f'Patient ID: {patient_id}'


### DICOM 파일 정보 추출
def dicom_info(ds):
    
    patient_id = ds.PatientID
    study_date = ds.get("StudyDate", "") or ds.get("AcquisitionDate", "")
    study_time = ds.get("StudyTime", "") or ds.get("AcquisitionTime", "")

    if len(study_date) == 8 and study_date.isdigit():
        date_str = f"{study_date[:4]}-{study_date[4:6]}-{study_date[6:]}"
    else:
        date_str = "Unknown"

    if "." in study_time:
        study_time = study_time.split(".")[0]
    if len(study_time) == 6 and study_time.isdigit():
        time_str = f"{study_time[:2]}:{study_time[2:4]}:{study_time[4:]}"
    else:
        time_str = "Unknown"

    lines = [f"{date_str}", f"{time_str}"]
    lines_str = " ".join(lines)

    return patient_id, lines_str

def flip_images(image_list):
    flipped_list = []
    print('1. Flip Images Cashing Completes')
    for img in image_list:
        if img is not None:
            # 이미지가 NumPy 배열이면 np.fliplr로 좌우반전
            if isinstance(img, np.ndarray):
                flipped_img = np.fliplr(img)
            else:
                flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_list.append(flipped_img)
        else:
            flipped_list.append(img)
    return flipped_list


# 파일 정보 추출
def files_info(files): 
    info_list = []  
    hex_digits = '0123456789abcdef'
    np.random.seed(42)
    
    if files:
        files = list(files)
        for file in files:
            file_fullname = file.name.lower()
            file_name = os.path.basename(file_fullname)
            if file_fullname.endswith((".jpg", ".jpeg", ".png")):
                pil_image = Image.open(file).convert("L")
                pid = ''.join(random.choice(hex_digits) for _ in range(16))
                dt_str = DEFAULT_DATETIME
            elif file_fullname.endswith((".dcm", ".dicom")):
                dicom_data = pydicom.dcmread(file, force=True)
                pid, dt_str = dicom_info(dicom_data)
                image_array = dicom_data.pixel_array.astype(np.float32)
                image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array)) * 255   
                image_array = image_array.astype(np.uint8)
                pil_image = Image.fromarray(image_array).convert("L")
            else:
                # 지원하지 않는 확장자의 경우 pass
                continue

            pil_1024_image = preprocess(pil_image)  # (1024, 1024) 사이즈 (segmentation용)
            pil_512_image = preprocess_(pil_image)   # (512, 512) 사이즈 (정규화용)
            tensor_1024_image = to_tensor(pil_1024_image)
            tensor_512_image = to_tensor(pil_512_image)

            # info_list에 필요한 정보들 저장 (날짜, 1024이미지, segmentation용 tensor, 정규화용 tensor, pid)
            info_list.append((dt_str, (pil_1024_image, file_name), tensor_1024_image, tensor_512_image, pid))

        # 날짜/시간 문자열로 정렬하기 위한 함수
        def parse_dt(dt_str):
            try:
                return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            except:
                return datetime.min

        info_list.sort(key=lambda x: parse_dt(x[0]))

        # 정렬 후 각 요소 분리
        sorted_dt, gallery_items, image_for_segmentation, image_for_normalization, patient_ids = zip(*info_list)

        # 리스트로 변환
        gallery_items           = list(gallery_items)           # [(pil_이미지_1024, 파일명), ...]
        datetimes               = list(sorted_dt)
        image_for_segmentation  = list(image_for_segmentation)  # [tensor(1×1024×1024), ...]
        image_for_normalization = list(image_for_normalization) # [tensor(1×512×512), ...]
        patient_ids             = list(patient_ids)

        # 환자 ID 중복 체크 등...
        if len(set(patient_ids)) <= 1:
            if all(item == "" for item in patient_ids):
                gr.Warning("환자 ID가 없습니다. 임의로 생성합니다.")
                patient_ids.clear()
                patient_ids.append(''.join(random.choice(hex_digits) for _ in range(16)))
            else:
                gr.Warning("환자 ID가 중복되어 있습니다. 파일을 확인해주세요.")
        #if len(set(datetimes)) <= 1:
        #    gr.Warning("Acquisition Date Time이 중복되어 있습니다.")

        cnt_image = len(gallery_items)

        # 텐서 스택
        img_seg_list = torch.stack(image_for_segmentation).float()  # [N, 1, 1024, 1024]
        img_nrm_list = torch.stack(image_for_normalization).float()  # [N, 1, 512, 512]

        print('segmentation and normalization start')
        # Segmentation 후 마스크 얻기
        mask_tensor = segmentation(img_seg_list).unsqueeze(1)     # [N, 1, 1024, 1024]
        img_list_tensor = spatial_normalize(img_nrm_list, mask_tensor)  # [N, 1, 512, 512] (float)
        img_list_np = img_list_tensor.cpu().numpy()  # [N, 1, 512, 512]
       

        # origin_items를 (이미지, 파일명) 튜플 대신, 이미지 리스트로 저장
        origin_items = []
        for i in range(len(img_list_np)):
            norm_img_2d = img_list_np[i, 0]  # shape: (512, 512)
            origin_items.append(norm_img_2d)

        # 좌우반전 이미지 리스트 생성 (flip_images는 위에서 수정된 함수 사용)
        fliped_items = flip_images(origin_items)
        gallery_mask = mask_tensor.cpu().numpy()           # 마스크 NumPy
        inference_list = all_image_inference(img_list_tensor)  # 중증도 추론결과 
                
        ext_info = convert_ext(file_fullname)
        cnt_info = convert_cnt(cnt_image)
        pid_info = convert_pid(patient_ids[0])

    # gallery_items는 1024×1024 버전, origin_items와 fliped_items는 512×512 정규화/좌우반전 이미지 리스트
    return gallery_items, origin_items, fliped_items, inference_list, datetimes, ext_info, cnt_info, pid_info
