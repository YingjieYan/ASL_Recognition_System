import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

# 加载模型（只加载一次）
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("asl_cnn_model.h5")
    return model

model = load_model()

# 类别标签
class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
    'Y', 'Z', 'del', 'nothing', 'space'
]

# 图像预处理
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = ImageOps.fit(image, (224, 224))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# Streamlit 页面配置
st.set_page_config(page_title="手语识别系统", page_icon="🤟")
st.title("✋ 美国手语（ASL）识别")
st.markdown("上传一张手语图片，系统将识别它代表的字母。")

uploaded_file = st.file_uploader("📤 上传图片", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="上传的图片", use_column_width=True)

    with st.spinner("识别中..."):
        img_array = preprocess_image(image)
        prediction = model.predict(img_array)[0]
        pred_index = np.argmax(prediction)
        pred_class = class_names[pred_index]
        confidence = prediction[pred_index] * 100

    st.success(f"✅ 识别结果：**{pred_class}** （置信度：{confidence:.2f}%）")
