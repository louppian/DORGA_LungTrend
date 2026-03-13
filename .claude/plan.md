# PAFE → DORGA 전환 마무리 & 버그 수정 계획

## Phase 1 — 버그 수정 (완료)

| # | 우선순위 | 내용 | 상태 |
|---|---------|------|------|
| 1 | FATAL | `utils_severity.py:12` import `dorga_model` → `dorga` | ✅ |
| 2 | HIGH | `utils_back.py:626` V3 → V4 경로 | ✅ |
| 3 | HIGH | `utils_back.py:634` gallery 튜플 언패킹 | ✅ |
| 4 | MEDIUM | `utils_segmentation.py:47` 백슬래시 경로 | ✅ |
| 5 | MEDIUM | `utils_image.py:134` `np.seed` → `np.random.seed(42)` | ✅ |
| 6 | MEDIUM | 미사용 import 정리 (app.py, STN.py, utils_image.py) | ✅ |
| 7 | MEDIUM | `requirements.txt` 정리 (faiss-gpu 제거, 중복 제거, timm 추가) | ✅ |
| 8 | LOW | `conditional_copy_inference_list` None guard 추가 | ✅ |
| 9 | LOW | `update_graph` matplotlib figure 메모리 누수 수정 (`plt.close('all')`) | ✅ |

---

## Phase 2 — UX 개선 (예정)

### P0: 클릭 피드백 개선
- **`utils_back.py` `draw_click_border`** — 1px 흰 테두리 → 3~4px 컬러(#FF00FF 등) 테두리로 변경
  - X-ray 밝은 영역에서도 확실히 보이도록 contrast 높은 색상 사용

### P1: 갤러리 선택 가시성
- 캡션 ✔ 만으로는 구분이 어려움
- 선택된 이미지에 슬롯 번호 표시 (예: `"[1] ✔ 3. filename"`)
- 4장 초과 선택 시 FIFO 제거 경고 또는 안내

### P1: Overlay/Flip 토글 상태 표시
- "O" / "F" 버튼에 ON/OFF 상태 시각적 구분 (색상 변경 또는 라벨 변경)

### P1: 4-Viewer 이미지 식별
- 각 뷰어 슬롯에 번호 또는 파일명 label 표시 (`show_label=True`)

### P1: Reset 확인
- Reset 클릭 시 확인 절차 추가 (편집 데이터 유실 방지)

### P2: 점수 편집 `*` 표시 설명
- UI에 범례 또는 툴팁 추가 — `*`가 "사용자가 수정한 값"임을 안내

---

## 미수정 (의도적으로 남김)
- `modules/backbones.py` — 미사용이지만, 향후 필요할 수 있으므로 보류
- `DORGAFull.pth` 가중치 파일 — 이미 `assets/weights/`에 존재
