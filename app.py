import streamlit as st
import joblib
import sys
import os

# Trỏ đường dẫn hệ thống để gọi được class tiền xử lý
sys.path.append(os.path.abspath('.'))
from src.preprocessor import VietnameseTextProcessor

# Cấu hình giao diện trang web
st.set_page_config(page_title="Shopee Sentiment AI", page_icon="🛒", layout="centered")

# --- 1. TẢI MÔ HÌNH VÀ PREPROCESSOR VÀO BỘ NHỚ (CACHE) ---
# Sử dụng st.cache_resource để ứng dụng không phải load lại model mỗi khi người dùng bấm nút
@st.cache_resource
def load_ai_core():
    # Khởi tạo bộ tiền xử lý
    preprocessor = VietnameseTextProcessor()
    
    # Thay đổi tên thư mục dưới đây thành tên thư mục model bạn vừa train ra
    # Ví dụ: 'models/20260709_180026/sentiment_pipeline.joblib'
    model_path = "models/20260710_224302/sentiment_pipeline.joblib" 
    model = joblib.load(model_path)
    
    return preprocessor, model

preprocessor, model = load_ai_core()

# --- 2. XÂY DỰNG GIAO DIỆN NGƯỜI DÙNG ---
st.title("Phân Tích Cảm Xúc Đánh Giá Shopee")
st.markdown("Nhập một đoạn đánh giá sản phẩm bằng tiếng Việt vào ô bên dưới, AI sẽ dự đoán xem đây là đánh giá **Tích cực** hay **Tiêu cực**.")

# Ô nhập liệu
user_input = st.text_area("Nhập bình luận của bạn tại đây:", height=150, placeholder="Ví dụ: Sản phẩm này giao hàng nhanh, chất lượng tuyệt vời!")

# Nút bấm phân tích
if st.button("Phân tích cảm xúc"):
    if not user_input.strip():
        st.warning("Vui lòng nhập nội dung đánh giá trước khi phân tích!")
    else:
        with st.spinner("AI đang xử lý..."):
            # Bước 1: Làm sạch dữ liệu thông qua Preprocessor
            # Hàm transform nhận vào một list và trả ra một list
            cleaned_text_list = preprocessor.transform([user_input])
            
            # Kiểm tra xem sau khi làm sạch câu có bị rỗng không (ví dụ: chỉ gõ toàn icon)
            if not cleaned_text_list or not cleaned_text_list[0].strip():
                st.error("Câu văn sau khi làm sạch không chứa từ vựng hợp lệ. Vui lòng thử lại!")
            else:
                cleaned_text = cleaned_text_list[0]
                
                # Bước 2: Đưa vào mô hình dự đoán (Lấy xác suất)
                # Hàm predict_proba trả về mảng xác suất cho từng nhãn
                probabilities = model.predict_proba([cleaned_text])[0]
                
                # Lấy danh sách các nhãn mà mô hình đã học (thường là ['negative', 'positive'])
                classes = model.classes_
                
                # Ghép nhãn và xác suất tương ứng thành một dictionary
                prob_dict = dict(zip(classes, probabilities))
                
                # Chuyển đổi sang phần trăm
                pos_prob = prob_dict.get('positive', 0.0) * 100
                neg_prob = prob_dict.get('negative', 0.0) * 100
                
                # Xác định nhãn chiến thắng (nhãn có phần trăm cao hơn)
                prediction = 'positive' if pos_prob > neg_prob else 'negative'
                
                # Bước 3: Hiển thị kết quả trực quan
                st.divider()
                st.subheader("Kết quả dự đoán:")
                
                # Hiển thị thông báo chính
                if prediction == 'positive':
                    st.success(f"🟢 Đây là đánh giá TÍCH CỰC (Độ tự tin: {pos_prob:.2f}%)")
                else:
                    st.error(f"🔴 Đây là đánh giá TIÊU CỰC (Độ tự tin: {neg_prob:.2f}%)")
                
                # Hiển thị thanh tiến trình (Progress bar) cho từng nhãn
                st.markdown("### Chi tiết phân tích:")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(label="Tích cực (Positive)", value=f"{pos_prob:.2f}%")
                    # Streamlit progress bar nhận giá trị từ 0 đến 100
                    st.progress(int(pos_prob))
                    
                with col2:
                    st.metric(label="Tiêu cực (Negative)", value=f"{neg_prob:.2f}%")
                    st.progress(int(neg_prob))
                
                # Debug
                # st.info(f"**Văn bản sau khi làm sạch (Tiền xử lý):** {cleaned_text}")