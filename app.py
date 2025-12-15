import cv2
import streamlit as st
import numpy as np
import av
import threading
from collections import deque

# 1. 修正导入：单独导入 WebRtcMode 和 VideoProcessorBase
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode

# 导入你的模型文件
from hand_landmarks import HandLandmarkDetector

# -------------------------------------------------------------
# 兼容性处理
try:
    from gesture_classifier import GestureClassifier
except ImportError:
    class GestureClassifier:
        def __init__(self):
            self.LABELS = ['No Model', 'Test']
        def predict(self, features):
            return "Test", 0.0
# -------------------------------------------------------------

st.set_page_config(
    page_title="汉语手指字母识别 (WebRTC)",
    page_icon="👋",
    layout="wide"
)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# 2. 类名修正：VideoTransformerBase -> VideoProcessorBase
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = HandLandmarkDetector()
        self.classifier = GestureClassifier()
        
        self.confidence_threshold = 0.6
        self.smooth_window = 5
        self.show_landmarks = True
        
        self.prediction_queue = deque(maxlen=self.smooth_window)
        self.lock = threading.Lock()

    def update_params(self, threshold, window, show_lm):
        with self.lock:
            self.confidence_threshold = threshold
            self.smooth_window = window
            self.prediction_queue = deque(maxlen=window)
            self.show_landmarks = show_lm

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # WebRTC 接收到的帧处理逻辑
        img = frame.to_ndarray(format="bgr24")
        
        # 镜像翻转
        img = cv2.flip(img, 1)
        
        with self.lock:
            thresh = self.confidence_threshold
            show_lm = self.show_landmarks
        
        # 检测
        landmarks, annotated_frame = self.detector.detect(img)
        
        final_img = annotated_frame if (landmarks is not None and show_lm) else img
        
        if landmarks is not None:
            features = self.detector.extract_features(landmarks)
            if features is not None:
                prediction, confidence = self.classifier.predict(features)
                
                if confidence > thresh:
                    self.prediction_queue.append(prediction)
                    
                    if len(self.prediction_queue) >= 1:
                        final_prediction = max(set(self.prediction_queue), key=self.prediction_queue.count)
                        
                        # 绘制结果 (只能英文)
                        color = (0, 255, 0) if confidence > 0.8 else (0, 165, 255)
                        cv2.putText(final_img, f"Gesture: {final_prediction}", (20, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                        cv2.putText(final_img, f"Conf: {confidence:.2f}", (20, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return av.VideoFrame.from_ndarray(final_img, format="bgr24")

# -------------------------------------------------------------
# UI 布局
# -------------------------------------------------------------

st.title("👋 汉语手指字母识别 (实时流)")

with st.sidebar:
    st.header("设置")
    show_landmarks = st.checkbox("显示骨架", value=True)
    conf_threshold = st.slider("置信度阈值", 0.1, 1.0, 0.6, 0.05)
    smooth_win = st.slider("平滑窗口", 1, 10, 5, 1)
    st.info("提示：请允许浏览器使用摄像头权限。")

col1, col2 = st.columns([0.7, 0.3])

with col1:
    st.subheader("实时摄像头")
    
    # 启动 WebRTC
    ctx = webrtc_streamer(
        key="hand-gesture",
        # 3. 修正调用：直接使用 WebRtcMode.SENDRECV
        mode=WebRtcMode.SENDRECV, 
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        ctx.video_processor.update_params(conf_threshold, smooth_win, show_landmarks)

with col2:
    st.subheader("状态")
    if ctx.state.playing:
        st.success("摄像头运行中 ✅")
    else:
        st.warning("摄像头已停止 🛑")
