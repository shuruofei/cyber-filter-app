import subprocess
import sys

# ==========================================
# 🔪 终极黑客补丁：内存级拦截与暗杀
# ==========================================
# 扫描环境中是否存在毒瘤（带界面的 OpenCV）
installed_packages = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode('utf-8')
if 'opencv-contrib-python' in installed_packages:
    # 强行卸载带屏幕的冲突版本
    subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', 'opencv-contrib-python'])
    # 顺手把 headless 重新干净地装一遍，抢占 cv2 命名空间
    subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', 'opencv-python-headless'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'opencv-python-headless'])

# 下面才是咱们原本正常的代码
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# ==========================================
# 🖐️ 路线三：魔法进阶 —— 隔空手势追踪 (骨骼激活篇)
# ==========================================

st.set_page_config(page_title="隔空手势追踪", page_icon="🖐️")
st.title("🖐️ 隔空手势追踪：骨骼激活")
st.write("坐直，将手伸到镜头前，点击拍摄，我们将实时绘制出你的 21 个手部关键点！")

# --- 🛠️ 召唤 Google 视觉大厨 (初始化 MediaPipe Hands) ---
mp_hands = mp.solutions.hands # 启用手部追踪模型
mp_drawing = mp.solutions.drawing_utils # 启用绘图工具 (帮我们画点和线)

# 初始化核心追踪器
# static_image_mode=True：这里我们处理静态照片，精度更高
# max_num_hands=2：最多同时找两只手
# min_detection_confidence=0.5：检测置信度需超过 50%
hands_tracker = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=2,
    min_detection_confidence=0.5
)

# 1. 摄像头输入
picture = st.camera_input("📸 张开手，掌心面对镜头...")

if picture is not None:
    with st.spinner("正在进行像素级骨骼解剖..."):
        # 读取照片并转为 CV2 BGR 格式
        bytes_data = picture.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # 魔法核心必须：MediaPipe 要求输入必须是 RGB 格式，所以必须反转颜色通道
        cv2_img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        
        # --- 🔮 终极扫描：调用 MediaPipe 算法处理图像 ---
        results = hands_tracker.process(cv2_img_rgb)
        
        # 准备一个画特效的图层
        annotated_image = cv2_img.copy()

        # --- 处理扫描结果 ---
        # 如果能在画面里找到手 (multi_hand_landmarks 不为空)
        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)
            st.success(f"🎯 扫描完成！在像素汪洋中成功解剖出 {num_hands} 只手。")
            
            # 遍历每一只找出来的手
            for hand_landmarks in results.multi_hand_landmarks:
                # 打印其中一个点的坐标看看 (比如食指指尖 8 号点)
                # print(hand_landmarks.landmark[8]) # 取消注释可在后台看到坐标数字
                
                # 🔮 绘图魔法：在照片上精准画出 21 个点和它们的骨骼连接线
                mp_drawing.draw_landmarks(
                    annotated_image, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS, # 指定要连接哪些点
                    # 设定点的样式 (青色圆点)
                    mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=3, circle_radius=4),
                    # 设定线的样式 (白色线条)
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
        else:
            st.warning("👀 扫描失败，请确保你的手完全伸到镜头前，并掌心面对镜头！")

        # 转换 BGR 为 RGB 用于 Streamlit 显示
        final_img_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # 展示结果
    st.image(final_img_rgb, channels="RGB", use_container_width=True)

# 释放资源
hands_tracker.close()