import cv2
import streamlit as st
import numpy as np
import time
# 确保这两个文件在你的仓库中存在
from hand_landmarks import HandLandmarkDetector
# from gesture_classifier import GestureClassifier # 如果还没上传这个文件，请先注释掉

# -------------------------------------------------------------------------
# 为了防止报错，如果你还没有 GestureClassifier，我加了一个模拟类。
# 如果你已经上传了 gesture_classifier.py，请删除下面这个 Mock 类，
# 并取消上面 from gesture_classifier... 的注释
class MockGestureClassifier:
    def __init__(self):
        self.LABELS = ['A', 'B', 'C', 'OK', 'Five']
    def predict(self, features):
        # 随机返回一个结果用于测试
        return np.random.choice(self.LABELS), np.random.random()

# 请根据实际情况决定使用哪一个
try:
    from gesture_classifier import GestureClassifier
except ImportError:
    GestureClassifier = MockGestureClassifier
# -------------------------------------------------------------------------

# 设置页面配置
st.set_page_config(
    page_title="汉语手指字母识别",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化模型 (使用缓存避免重复加载)
@st.cache_resource
def init_models():
    detector = HandLandmarkDetector()
    classifier = GestureClassifier()
    return detector, classifier

try:
    detector, classifier = init_models()
except Exception as e:
    st.error(f"模型加载失败: {e}")
    st.stop()

# 应用状态管理 (使用 session_state 持久化数据)
if 'history' not in st.session_state:
    st.session_state.history = []
if 'stats' not in st.session_state:
    st.session_state.stats = {}

# 页面标题
st.title("👋 汉语手指字母识别 (云端版)")

# 侧边栏设置
with st.sidebar:
    st.header("设置")
    
    # 显示/隐藏手部关键点
    show_landmarks = st.toggle("显示手部关键点", value=True)
    
    # 置信度阈值滑块
    confidence_threshold = st.slider(
        "置信度阈值",
        min_value=0.1,
        max_value=1.0,
        value=0.6,
        step=0.05
    )
    
    # 清除历史按钮
    if st.button("清除历史数据"):
        st.session_state.history = []
        st.session_state.stats = {}
        st.rerun()

# 主内容区
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📷 摄像头输入")
    # 核心修改：使用 st.camera_input 替代 cv2.VideoCapture
    img_file_buffer = st.camera_input("点击下方按钮拍照进行识别")

with col2:
    st.subheader("📊 识别结果")
    result_placeholder = st.empty()
    metrics_placeholder = st.empty()
    stats_placeholder = st.empty()

# 处理逻辑
if img_file_buffer is not None:
    start_time = time.time()
    
    # 1. 将上传的图片转换为 OpenCV 格式
    bytes_data = img_file_buffer.getvalue()
    # imdecode 读取的是 BGR 格式
    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # 2. 镜像翻转 (让自拍看起来更自然)
    frame = cv2.flip(frame, 1)
    
    # 3. 检测关键点
    detection_start = time.time()
    landmarks, annotated_frame = detector.detect(frame)
    detection_time = (time.time() - detection_start) * 1000
    
    prediction = None
    confidence = 0.0
    classification_time = 0
    
    # 4. 手势分类
    if landmarks is not None:
        features = detector.extract_features(landmarks)
        if features is not None:
            cls_start = time.time()
            prediction, confidence = classifier.predict(features)
            classification_time = (time.time() - cls_start) * 1000
            
            # 只有置信度足够高才记录
            if confidence > confidence_threshold:
                # 更新历史和统计
                st.session_state.history.append((prediction, confidence))
                # 保持历史记录只有最近 20 条
                if len(st.session_state.history) > 20:
                    st.session_state.history.pop(0)
                
                # 更新统计
                st.session_state.stats[prediction] = st.session_state.stats.get(prediction, 0) + 1

                # 在图片上绘制英文结果 (OpenCV不支持中文)
                cv2.putText(annotated_frame, f"Gesture: {prediction}", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Conf: {confidence:.2f}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 5. 显示处理后的图像
    # 如果用户选择不显示关键点，就用原图
    final_image = annotated_frame if show_landmarks else frame
    # OpenCV 是 BGR，Streamlit 需要 RGB
    final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
    
    # 在左侧列显示处理后的图
    with col1:
        st.image(final_image, caption="识别处理视图", use_container_width=True)

    # 6. 显示右侧数据面板
    with result_placeholder:
        if prediction and confidence > confidence_threshold:
            st.success(f"识别结果: **{prediction}**")
            st.progress(float(confidence))
        elif landmarks is None:
            st.warning("未检测到手部")
        else:
            st.info(f"置信度过低 (<{confidence_threshold})")

    with metrics_placeholder:
        m1, m2 = st.columns(2)
        m1.metric("检测耗时", f"{detection_time:.1f} ms")
        m2.metric("分类耗时", f"{classification_time:.1f} ms")

    with stats_placeholder:
        st.markdown("### 📈 历史统计")
        if st.session_state.stats:
            st.bar_chart(st.session_state.stats)
        
        if st.session_state.history:
            st.markdown("### 🕒 最近记录")
            # 显示最近5条
            for i, (pred, conf) in enumerate(reversed(st.session_state.history[-5:])):
                st.text(f"{i+1}. 手势: {pred} (置信度: {conf:.1%})")

else:
    # 初始状态提示
    with col1:
        st.info("👋 请允许浏览器使用摄像头，并点击 'Take Photo' 按钮开始识别。")
