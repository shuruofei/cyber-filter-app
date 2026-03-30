import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ==========================================
# 👁️ 路线三：赛博滤镜工坊 (自定义换装版)
# ==========================================

st.set_page_config(page_title="百变滤镜工坊", page_icon="😎")
st.title("😎 百变面部贴纸滤镜")
st.write("不仅能虚化背景，还能自由切换或上传你喜欢的眼镜样式！")

# --- 🛠️ 核心工具：智能处理外部图片 ---
def process_glasses_image(image_bytes, face_w):
    """
    接收用户上传或预设的图片，并智能缩放适配脸型
    """
    # 1. 把传进来的图片数据转换成 OpenCV 能处理的矩阵
    glasses_img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    
    # 2. 安全检查：确认图片是不是自带透明通道 (必须是 4 个通道：B, G, R, Alpha)
    if glasses_img.shape[2] != 4:
        # 如果用户传了一张没有透明背景的普通 JPG，这里会报错拦截
        return None
        
    # 3. 动态缩放：算出眼镜现在的宽高比例，把宽度缩放到脸宽的 90%
    orig_h, orig_w = glasses_img.shape[:2]
    target_w = int(face_w * 0.9)
    target_h = int((target_w / orig_w) * orig_h) # 保持原有比例不变形
    
    # 使用 cv2.resize 瞬间完成图片的拉伸或压缩
    resized_glasses = cv2.resize(glasses_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return resized_glasses

# --- 🛠️ 备用工具：如果你没传图，我还是给你画一个经典的 ---
def draw_default_sunglasses(face_w, face_h):
    glass_w = int(face_w * 0.9)
    glass_h = int(face_h * 0.3)
    sunglasses_rgba = np.zeros((glass_h, glass_w, 4), dtype=np.uint8)
    color = (0, 0, 0, 255) 
    eye_w = glass_w // 2
    cv2.ellipse(sunglasses_rgba, (eye_w // 2, glass_h // 2), (eye_w // 2 - 5, glass_h // 2), 0, 0, 360, color, -1)
    cv2.ellipse(sunglasses_rgba, (eye_w + eye_w // 2, glass_h // 2), (eye_w // 2 - 5, glass_h // 2), 0, 0, 360, color, -1)
    bridge_w = int(glass_w * 0.1)
    cv2.rectangle(sunglasses_rgba, (eye_w - bridge_w, glass_h // 2 - 5), (eye_w + bridge_w, glass_h // 2 + 5), color, -1)
    return sunglasses_rgba

# --- 侧边栏：搭建“衣柜”交互界面 ---
with st.sidebar:
    st.header("🎛️ 滤镜控制台")
    
    # 让用户选择样式
    glasses_style = st.radio(
        "选择你的专属眼镜样式：",
        ["经典黑客帝国 (默认)", "自定义上传 (需透明底 PNG)"]
    )
    
    # 如果用户选了自定义，就弹出一个上传框
    uploaded_glasses = None
    if glasses_style == "自定义上传 (需透明底 PNG)":
        uploaded_glasses = st.file_uploader("请上传一张透明背景的眼镜图片", type=["png"])
        if uploaded_glasses:
            # 在侧边栏预览一下用户传的眼镜
            st.image(uploaded_glasses, caption="你的专属道具", width=150)

    st.markdown("---")
    blur_intensity = st.slider("赛博虚化强度", min_value=1, max_value=51, value=25, step=2)

# 1. 摄像头输入
picture = st.camera_input("📸 准备就绪，点击拍摄！")

if picture is not None:
    with st.spinner("正在为你进行精准换装..."):
        bytes_data = picture.getvalue()
        orig_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        height, width = orig_img.shape[:2]
        effect_img = orig_img.copy()
        
        gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))

        if len(faces) > 0:
            mask = np.zeros((height, width), dtype=np.uint8)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
                
                # 🔮 关键逻辑：根据用户的选择，决定拿什么眼镜往脸上贴
                sunglasses = None
                if glasses_style == "自定义上传 (需透明底 PNG)" and uploaded_glasses is not None:
                    # 尝试处理用户上传的图片
                    sunglasses = process_glasses_image(uploaded_glasses.getvalue(), w)
                    if sunglasses is None:
                        st.error("❌ 哎呀，你上传的图片好像没有透明通道（不是抠好图的 PNG），我只能先给你换回默认款啦！")
                
                # 如果没传、处理失败，或者选了默认款，就画一个黑客眼镜
                if sunglasses is None:
                    sunglasses = draw_default_sunglasses(w, h)
                
                sg_h, sg_w = sunglasses.shape[:2]
                
                # 计算贴纸坐标，这里的 0.25 是调整眼镜在脸部上下的位置
                offset_y = int(y + h * 0.25)
                offset_x = int(x + (w - sg_w) / 2)
                
                # 防止眼镜超出画面边界导致程序崩溃
                if offset_y + sg_h <= height and offset_x + sg_w <= width:
                    # Alpha 通道融合算法
                    sg_rgb = sunglasses[:, :, :3]
                    sg_alpha = sunglasses[:, :, 3] / 255.0 

                    roi = effect_img[offset_y:offset_y+sg_h, offset_x:offset_x+sg_w]
                    for c in range(3):
                        roi[:, :, c] = (roi[:, :, c] * (1.0 - sg_alpha) + sg_rgb[:, :, c] * sg_alpha).astype(np.uint8)

            blurred_img = cv2.GaussianBlur(orig_img, (blur_intensity, blur_intensity), 0)
            final_img = np.where(mask[:, :, np.newaxis] == 255, effect_img, blurred_img)
            st.success("🎯 换装成功！你的专属造型已上线。")
            
        else:
            final_img = cv2.GaussianBlur(orig_img, (blur_intensity, blur_intensity), 0)
            st.warning("👀 扫描失败，请稍微调整一下面对镜头的角度！")

        final_img_rgb = cv2.cvtColor(final_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    st.image(final_img_rgb, channels="RGB", use_container_width=True)