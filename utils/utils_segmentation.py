# Segmentation_mask.py : image_for_segmentation을 받아서 segmentation을 수행하고 마스크를 반환하는 모듈
import os
import numpy as np
import cv2
import scipy.sparse as sp
import torch
import matplotlib.pyplot as plt
from glob import glob
from GUNet.GUNet_model import GUNet 
from GUNet.utils import scipy_to_torch_sparse, genMatrixesLungsHeart


# 전역 변수 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hybrid = None

#하이브리드 모델 로드
def loadModel(device):
    # lungs and heart에 해당하는 그래프 행렬 생성
    A, AD, D, U = genMatrixesLungsHeart()
    N1 = A.shape[0]
    N2 = AD.shape[0]

    A = sp.csc_matrix(A).tocoo()
    AD = sp.csc_matrix(AD).tocoo()
    D = sp.csc_matrix(D).tocoo()
    U = sp.csc_matrix(U).tocoo()

    D_ = [D.copy()]
    U_ = [U.copy()]

    config = {}
    config['n_nodes'] = [N1, N1, N1, N2, N2, N2]
    A_ = [A.copy(), A.copy(), A.copy(), AD.copy(), AD.copy(), AD.copy()]
    
    A_t, D_t, U_t = ([scipy_to_torch_sparse(x).to(device) for x in X] 
                      for X in (A_, D_, U_))

    config['latents'] = 64
    config['inputsize'] = 1024

    f = 32
    config['filters'] = [2, f, f, f, f//2, f//2, f//2]
    config['skip_features'] = f

    model = GUNet(config.copy(), D_t, U_t, A_t).to(device)
    model.load_state_dict(torch.load("assets/weights/finetuned_9601.pt", map_location=device))
    model.eval()
    
    return model

# 랜드마크 바탕으로 바이너리 마스크 생성 후 정규화
def get_mask(landmarks):
    h, w = 1024,1024  # img -> (32,1,1024,1024)
    np.set_printoptions(threshold=np.inf)  # 모든 값 출력
    batch_size = landmarks.shape[0]

    masks = torch.zeros((batch_size, h, w), dtype=torch.uint8, device=landmarks.device)
    
    for i in range(batch_size):
        RL = landmarks[i, 0:44].reshape(-1, 1, 2).to(torch.int32)
        LL = landmarks[i, 44:94].reshape(-1, 1, 2).to(torch.int32)
        mask_np = masks[i].cpu().numpy()
        
        # 오른쪽 폐(RL) 추가
        cv2.drawContours(mask_np, [RL.cpu().numpy()], -1, 255, -1)
        # 왼쪽 폐(LL) 추가
        cv2.drawContours(mask_np, [LL.cpu().numpy()], -1, 255, -1)

        masks[i] = torch.from_numpy(mask_np) / 255
    return masks

# 마스크 생성
def segmentation(data): # data tensor (32,1,1024,1024)
    global hybrid, device
    
    if hybrid is None:
        hybrid = loadModel(device)

    with torch.no_grad():
        # 모델 출력은 [landmarks, ...] 형태 중 첫 번째 값 사용
        data = data.to(device)
        output = hybrid(data)[0]    #.cpu() # 32,120,2 tensor [0,1]
    
    # output이 텐서라서 0~1값을 가짐. -> 1024 곱하긴 했는데 이렇게 해도 되나? 
    output = output*1024 ; output = output.int()
    image_mask = get_mask(output) 
    return image_mask.cpu()      

if __name__ == "__main__":
    pass