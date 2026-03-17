import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

# 1. 페이지 설정 및 제목
st.set_page_config(page_title="고화질 이미지 스캐너", layout="centered")
st.title("📸 스마트 이미지 스캐너")
st.write("이미지를 업로드하고, 서류의 **네 모서리**를 클릭하세요. (큰 이미지도 자동으로 조절됩니다)")

# 2. 세션 상태 초기화 (좌표 저장용)
if 'pts' not in st.session_state:
    st.session_state.pts = []

# 사이드바에 초기화 버튼 배치
if st.sidebar.button("💫 좌표 초기화"):
    st.session_state.pts = []
    st.rerun()

# 3. 파일 업로드
uploaded_file = st.file_uploader("이미지 파일을 선택하세요", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # 이미지 로드
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    orig_h, orig_w = img_array.shape[:2]

    # --- [핵심] 이미지 스케일링 로직 ---
    # 화면에 보여줄 가로 폭을 800px로 고정합니다.
    display_width = 800
    ratio = orig_w / display_width  # 원본과 화면 표시용 이미지의 비율
    display_height = int(orig_h / ratio)

    # 화면 표시용으로 리사이즈 (마킹을 위해 복사본 생성)
    display_img = cv2.resize(img_array, (display_width, display_height))
    
    # 4. 점 그리기 (사용자가 클릭한 위치 표시)
    for i, p in enumerate(st.session_state.pts):
        # 화면 좌표(p)에 바로 그립니다.
        cv2.circle(display_img, (int(p[0]), int(p[1])), 10, (0, 255, 0), -1)
        cv2.putText(display_img, str(i+1), (int(p[0])+15, int(p[1])+15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # PIL 이미지로 변환하여 클릭 컴포넌트에 전달
    display_pil = Image.fromarray(display_img)

    # 5. 이미지 표시 및 클릭 감지
    # 고정된 너비(800px)를 지정하여 이미지가 잘리는 현상을 방지합니다.
    value = streamlit_image_coordinates(display_pil, width=display_width, key="coords")

    if value:
        point = (value["x"], value["y"])
        
        # 중복 클릭 방지 및 4개까지만 기록
        if not st.session_state.pts or (point != st.session_state.pts[-1]):
            if len(st.session_state.pts) < 4:
                st.session_state.pts.append(point)
                st.rerun()

    # 6. 원근 변환 실행 (4개의 점이 모두 찍혔을 때)
    if len(st.session_state.pts) == 4:
        st.divider()
        st.subheader("✨ 변환 결과 (원본 화질)")
        
        # [중요] 화면 좌표를 원본 이미지 좌표로 변환
        # 공식: 원본 좌표 = 화면 좌표 * (원본 너비 / 화면 너비)
        real_pts = []
        for p in st.session_state.pts:
            real_pts.append([p[0] * ratio, p[1] * ratio])
        
        pts = np.array(real_pts, dtype=np.float32)
        
        # 상하좌우 좌표 정렬
        sm = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        topLeft = pts[np.argmin(sm)]
        bottomRight = pts[np.argmax(sm)]
        topRight = pts[np.argmin(diff)]
        bottomLeft = pts[np.argmax(diff)]

        pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])

        # 변환 후 이미지 크기 계산
        w1 = abs(bottomRight[0] - bottomLeft[0])
        w2 = abs(topRight[0] - topLeft[0])
        h1 = abs(topRight[1] - bottomRight[1])
        h2 = abs(topLeft[1] - bottomLeft[1])
        width = int(max([w1, w2]))
        height = int(max([h1, h2]))

        pts2 = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

        # 원본 이미지(img_array)에 변환 적용하여 고화질 유지
        mtrx = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(img_array, mtrx, (width, height))
        
        # 최종 결과 표시
        st.image(result, caption=f"스캔 완료 ({width}x{height})")
        
        # 다운로드 버튼 추가
        result_pil = Image.fromarray(result)
        st.download_button("이미지 저장하기", 
                           data=cv2.imencode('.png', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))[1].tobytes(), 
                           file_name="scanned_image.png", 
                           mime="image/png")
