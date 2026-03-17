import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="웹 이미지 스캐너")
st.title("📸 이미지 원근 변환 스캐너")
st.write("이미지를 업로드하고, 서류의 네 모서리를 순서대로 클릭하세요.")

# 1. 파일 업로드
uploaded_file = st.file_uploader("이미지 파일을 선택하세요...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # 이미지 로드
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # 세션 상태로 클릭 좌표 저장 (웹은 새로고침이 잦으므로 상태 저장이 필수입니다)
    if 'pts' not in st.session_state:
        st.session_state.pts = []

    # 2. 이미지 표시 및 좌표 수집
    # 'streamlit_image_coordinates' 라이브러리가 필요합니다.
    value = streamlit_image_coordinates(image, key="coords")

    if value:
        point = (value["x"], value["y"])
        if point not in st.session_state.pts and len(st.session_state.pts) < 4:
            st.session_state.pts.append(point)
            st.rerun()

    # 클릭한 지점 표시
    for i, p in enumerate(st.session_state.pts):
        st.write(f"점 {i+1}: {p}")

    # 3. 4개의 점이 모두 선택되었을 때 변환 로직 실행
    if len(st.session_state.pts) == 4:
        pts = np.array(st.session_state.pts, dtype=np.float32)
        
        # 상하좌우 계산 로직 (기존 코드 활용)
        sm = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        topLeft = pts[np.argmin(sm)]
        bottomRight = pts[np.argmax(sm)]
        topRight = pts[np.argmin(diff)]
        bottomLeft = pts[np.argmax(diff)]

        pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])

        w1 = abs(bottomRight[0] - bottomLeft[0])
        w2 = abs(topRight[0] - topLeft[0])
        h1 = abs(topRight[1] - bottomRight[1])
        h2 = abs(topLeft[1] - bottomLeft[1])
        width = int(max([w1, w2]))
        height = int(max([h1, h2]))

        pts2 = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

        # 변환 실행
        mtrx = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(img_bgr, mtrx, (width, height))
        
        # 결과 출력
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        st.subheader("✅ 변환 결과")
        st.image(result_rgb, caption="스캔된 이미지")

        if st.button("좌표 초기화"):
            st.session_state.pts = []
            st.rerun()