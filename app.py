import os

import streamlit as st

from src.inference import load_model


@st.cache_resource
def load_ai_core(model_path):
    return load_model(model_path)


def main():
    st.set_page_config(page_title="Shopee Sentiment AI", page_icon="🛒", layout="centered")
    st.title("Phân Tích Cảm Xúc Đánh Giá Shopee")
    model_path = os.environ.get("SENTIMENT_MODEL_PATH")
    if not model_path:
        st.info("Đặt SENTIMENT_MODEL_PATH trỏ tới sentiment_pipeline.joblib của lần huấn luyện mới.")
        st.stop()
    try:
        model = load_ai_core(model_path)
    except (OSError, ValueError) as error:
        st.error(str(error))
        st.stop()

    user_input = st.text_area("Nhập bình luận của bạn tại đây:", height=150,
                              placeholder="Ví dụ: Sản phẩm này giao hàng nhanh, chất lượng tốt!")
    if st.button("Phân tích cảm xúc"):
        with st.spinner("AI đang xử lý..."):
            result = model.infer([user_input], include_probabilities=True)[0]
        if result["empty_after_preprocessing"]:
            st.warning("Không còn từ vựng sau tiền xử lý. Dự đoán dưới đây dùng đầu vào không có đặc trưng văn bản.")
        st.subheader("Kết quả dự đoán:")
        prediction = result["label"]
        if prediction == "positive":
            st.success("🟢 TÍCH CỰC")
        elif prediction == "negative":
            st.error("🔴 TIÊU CỰC")
        else:
            st.write(prediction)
        st.caption("Xác suất do mô hình ước tính, chưa hiệu chỉnh; không phải xác suất bảo đảm dự đoán đúng.")
        for label, probability in result["probabilities"].items():
            st.metric(label=label, value=f"{probability:.2%}")
            st.progress(float(probability))


if __name__ == "__main__":
    main()
