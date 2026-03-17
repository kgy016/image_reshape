import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

# 페이지 설정
st.set_page_config(page_title="이미지 스캐너", layout="centered")

st.title("📸 웹 이미지 스캐너 (원근 변환)")
st.write("이미지를 업로드하고, 서류의 **네 모서리**를 순서대로 클릭하세요.")

# 1. 좌표를 저장할 세션 상태 초기화
if 'pts' not in st.session_state:
    st.session_state.pts = []

# 초기화 버튼
if st.sidebar.button("좌표 전체 초기화"):
    st.session_state.pts = []
    st.rerun()

# 2. 파일 업로드
uploaded_file = st.file_uploader("이미지 파일을 선택하세요 (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # 이미지 로드 (PIL -> numpy)
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # 3. 이미지 위에 점 그리기 (마킹 기능)
    # 이미지가 변경되어도 점을 유지하기 위해 매번 복사본에 그립니다.
    draw_img = img_array.copy()
    for i, p in enumerate(st.session_state.pts):
        # 초록색 점 찍기
        cv2.circle(draw_img, (int(p[0]), int(p[1])), 10, (0, 255, 0), -1)
        # 점 번호 표시
        cv2.putText(draw_img, str(i+1), (int(p[0])+15, int(p[1])+15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # 마킹된 이미지를 다시 PIL로 변환하여 클릭 컴포넌트에 전달
    display_pil = Image.fromarray(draw_img)

    # 4. 이미지 표시 및 좌표 수집
    # 클릭하면 'value'에 {x: , y: } 좌표가 담깁니다.
    value = streamlit_image_coordinates(display_pil, key="coords")

    if value:
        point = (value["x"], value["y"])
        
        # [중요] 자동 클릭 방지 및 중복 클릭 방지 로직
        # 1. 이전에 찍은 점과 좌표가 다를 때만 저장
        # 2. 4개까지만 저장
        if not st.session_state.pts or (point != st.session_state.pts[-1]):
            if len(st.session_state.pts) < 4:
                st.session_state.pts.append(point)
                st.rerun()

    # 현재 찍힌 좌표 현황 보여주기
    if st.session_state.pts:
        cols = st.columns(len(st.session_state.pts))
        for i, p in enumerate(st.session_state.pts):
            cols[i].metric(f"점 {i+1}", f"{int(p[0])}, {int(p[1])}")

    # 5. 4개의 점이 모두 찍혔을 때 변환 로직 실행
    if len(st.session_state.pts) == 4:
        st.divider()
        st.subheader("✅ 변환 결과")
        
        pts = np.array(st.session_state.pts, dtype=np.float32)
        
        # 상하좌우 좌표 자동 정렬
        sm = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        topLeft = pts[np.argmin(sm)]
        bottomRight = pts[np.argmax(sm)]
        topRight = pts[np.argmin(diff)]
        bottomLeft = pts[np.argmax(diff)]

        pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])

        # 결과 이미지 크기 계산
        w1 = abs(bottomRight[0] - bottomLeft[0])
        w2 = abs(topRight[0] - topLeft[0])
        h1 = abs(topRight[1] - bottomRight[1])
        h2 = abs(topLeft[1] - bottomLeft[1])
        width = int(max([w1, w2]))
        height = int(max([h1, h2]))

        pts2 = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

        # OpenCV 원근 변환 적용 (BGR 변환 없이 RGB 상태에서 진행)
        mtrx = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(img_array, mtrx, (width, height))
        
        # 결과 출력
        st.image(result, caption="스캔 완료! 우클릭하여 저장하세요.")
        
        if st.button("다시 찍기"):
            st.session_state.pts = []
            st.rerun()