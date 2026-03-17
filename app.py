import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

# 1. 페이지 설정 (좌우를 넓게 쓰는 wide 모드)
st.set_page_config(page_title="이미지 스캐너", layout="wide")

st.title("📸 스마트 이미지 스캐너")
st.info("이미지의 네 모서리를 [좌상단 -> 우상단 -> 우하단 -> 좌하단] 순서로 클릭하세요.")

# 2. 좌표 저장을 위한 세션 상태 관리
if 'pts' not in st.session_state:
    st.session_state.pts = []

# 사이드바 초기화 버튼 (깔끔하게 분리)
if st.sidebar.button("💫 모든 좌표 초기화"):
    st.session_state.pts = []
    st.rerun()

# 3. 파일 업로드
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # 이미지 열기
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    orig_h, orig_w = img_array.shape[:2]

    # --- [수정] 이미지 크기 계산 로직 (잘림 방지) ---
    # 화면 너비에 따라 적절한 비율로 보여주기 위해 가로를 800px 기준으로 잡습니다.
    # 만약 브라우저가 이보다 작으면 자동으로 줄어듭니다.
    display_width = 800 
    if orig_w < display_width:
        display_width = orig_w
        
    ratio = orig_w / display_width
    display_height = int(orig_h / ratio)

    # 표시용 이미지 리사이즈
    display_img = cv2.resize(img_array, (display_width, display_height))
    
    # 4. 이미지 위에 클릭한 점 마킹하기
    for i, p in enumerate(st.session_state.pts):
        # 초록색 원 그리기
        cv2.circle(display_img, (int(p[0]), int(p[1])), 10, (0, 255, 0), -1)
        # 번호 표시 (글자 크기 키움)
        cv2.putText(display_img, str(i+1), (int(p[0])+15, int(p[1])+15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # 마킹된 이미지를 컴포넌트로 전달
    display_pil = Image.fromarray(display_img)

    # 5. 좌표 수집 (잘림 방지를 위해 width 설정을 컴포넌트 내부에서 자동 처리)
    # 이 부분에서 컴포넌트가 잘리지 않도록 레이아웃을 중앙으로 배치합니다.
    value = streamlit_image_coordinates(display_pil, key="coords")

    if value:
        point = (value["x"], value["y"])
        
        # 중복 클릭 방지 로직 (이전 점과 다를 때만 저장)
        if not st.session_state.pts or (point != st.session_state.pts[-1]):
            if len(st.session_state.pts) < 4:
                st.session_state.pts.append(point)
                st.rerun()

    # 6. 결과 처리 (점 4개가 모두 찍혔을 때)
    if len(st.session_state.pts) == 4:
        st.divider()
        st.success("✅ 4개의 지점이 선택되었습니다. 보정 결과를 확인하세요.")
        
        # 화면 좌표를 원본 고화질 좌표로 복원
        real_pts = []
        for p in st.session_state.pts:
            real_pts.append([p[0] * ratio, p[1] * ratio])
        
        pts = np.array(real_pts, dtype=np.float32)
        
        # 4개 좌표의 상하좌우 정렬 (사용자 기존 로직)
        sm = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        topLeft = pts[np.argmin(sm)]
        bottomRight = pts[np.argmax(sm)]
        topRight = pts[np.argmin(diff)]
        bottomLeft = pts[np.argmax(diff)]

        pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])

        # 보정 후 이미지 크기 계산
        w1 = abs(bottomRight[0] - bottomLeft[0])
        w2 = abs(topRight[0] - topLeft[0])
        h1 = abs(topRight[1] - bottomRight[1])
        h2 = abs(topLeft[1] - bottomLeft[1])
        width = int(max([w1, w2]))
        height = int(max([h1, h2]))

        pts2 = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

        # 원본 이미지에 변환 적용
        mtrx = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(img_array, mtrx, (width, height))
        
        # 결과 화면 출력
        st.image(result, caption=f"보정 완료 ({width}x{height})", use_container_width=True)
        
        # 다운로드 버튼
        btn_col1, btn_col2 = st.columns([1, 4])
        with btn_col1:
            st.download_button(
                label="💾 이미지 저장하기",
                data=cv2.imencode('.png', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))[1].tobytes(),
                file_name="scanned_result.png",
                mime="image/png"
            )
