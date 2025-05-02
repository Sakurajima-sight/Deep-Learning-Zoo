import torch
import streamlit as st
from PIL import Image
import numpy as np
import sys
import os

Image.MAX_IMAGE_PIXELS = None

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import clip

# 模型名称与下载链接字典
_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


st.title("CLIP 图文匹配演示（多模型支持）")

device = "cuda" if torch.cuda.is_available() else "cpu"

# 模型选择
model_name = st.selectbox("选择 CLIP 模型", list(_MODELS.keys()), index=5)

# 加载模型
@st.cache_resource(show_spinner=True)
def load_model(name):
    return clip.load(name, device=device)

model, preprocess = load_model(model_name)

st.write("上传一张图片，并输入多行文本，点击按钮开始推理。")

uploaded_image = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])

# ✅ 上传图片后立即显示
if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="上传的图片", use_container_width=True)

text_input = st.text_area("输入候选文本（每行一个）", "a photo of a cat\na photo of a dog\na diagram")

# ✅ 推理按钮
if st.button("开始推理") and uploaded_image and text_input.strip():
    text_list = [line.strip() for line in text_input.strip().split("\n") if line.strip()]
    text_tokens = clip.tokenize(text_list).to(device)
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_tokens)
        logits_per_image, _ = model(image_input, text_tokens)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    st.subheader("匹配概率（越高越相关）")
    for label, prob in zip(text_list, probs):
        st.write(f"**{label}**: {prob:.4f}")
