import cv2
import streamlit as st
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import threading
from collections import deque

# 导入你的模型文件
from hand_landmarks import HandLandmarkDetector

# -------------------------------------------------------------
# 兼容性处理：防止没有 gesture_classifier 文件导致报错
try:
    from gesture_classifier import GestureClassifier
except ImportError:
    # 模拟分类器（占位符）
    class GestureClassifier:
        def __init__(self):
            self.LABELS = ['No Model', 'Test']
        def predict(self, features):
            return "Test", 0.0
# -------------------------------------------------------------

# 1. 页面配置
st.set_page_config(
    page_title="汉语手指字母识别 (WebRTC)",
    page_icon="👋",
    layout="wide"
)

# 2. WebRTC 配置 (关键！这是云端穿透内网必须的)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# 3. 定义视频处理器 (WebRTC 的核心)
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        # 初始化模型
        self.detector = HandLandmarkDetector()
        self.classifier = GestureClassifier()
        
        # 状态参数 (从全局 session_state 或默认值获取)
        self.confidence_threshold = 0.6
        self.smooth_window = 5
        self.show_landmarks = True
        
        # 平滑队列
        self.prediction_queue = deque(maxlen=self.smooth_window)
        self.lock = threading.Lock() # 线程锁

    def update_params(self, threshold, window, show_lm):
        # 用于从 UI 线程更新参数
        with self.lock:
            self.confidence_threshold = threshold
            self.smooth_window = window
            self.prediction_queue = deque(maxlen=window)
            self.show_landmarks = show_lm

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        每一帧都会在这个函数里被处理
        """
        # 1. 将 av.VideoFrame 转为 OpenCV 格式
        img = frame.to_ndarray(format="bgr24")
        
        # 2. 镜像翻转
        img = cv2.flip(img, 1)
        
        # 3. 获取当前参数
        with self.lock:
            thresh = self.confidence_threshold
            show_lm = self.show_landmarks
        
        # 4. 手部检测
        landmarks, annotated_frame = self.detector.detect(img)
        
        # 如果不显示骨架，就用原图
        final_img = annotated_frame if (landmarks is not None and show_lm) else img
        
        # 5. 手势识别
        if landmarks is not None:
            features = self.detector.extract_features(landmarks)
            if features is not None:
                prediction, confidence = self.classifier.predict(features)
                
                if confidence > thresh:
                    # 平滑处理
                    self.prediction_queue.append(prediction)
                    
                    # 统计窗口内最高频的手势
                    if len(self.prediction_queue) >= 1:
                        # 简单的投票机制
                        final_prediction = max(set(self.prediction_queue), key=self.prediction_queue.count)
                        
                        # 绘制结果 (OpenCV 不支持中文，使用英文显示)
                        # 绿色表示高置信度，橙色表示低置信度
                        color = (0, 255, 0) if confidence > 0.8 else (0, 165, 255)
                        
                        cv2.putText(final_img, f"Gesture: {final_prediction}", (20, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                        cv2.putText(final_img, f"Conf: {confidence:.2f}", (20, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 6. 将处理后的图像转回 av.VideoFrame 返回给浏览器
        return av.VideoFrame.from_ndarray(final_img, format="bgr24")

# -------------------------------------------------------------
# UI 界面布局
# -------------------------------------------------------------

st.title("👋 汉语手指字母识别 (实时流)")

# 侧边栏设置
with st.sidebar:
    st.header("设置")
    
    # 获取用户输入
    show_landmarks = st.checkbox("显示骨架", value=True)
    conf_threshold = st.slider("置信度阈值", 0.1, 1.0, 0.6, 0.05)
    smooth_win = st.slider("平滑窗口", 1, 10, 5, 1)
    
    st.info("提示：\n1. 首次运行请允许摄像头权限\n2. 手机端请使用 Chrome/Safari\n3. 结果将直接显示在视频上方")

# 主界面布局
col1, col2 = st.columns([0.7, 0.3])

with col1:
    st.subheader("实时摄像头")
    
    # 启动 WebRTC
    ctx = webrtc_streamer(
        key="hand-gesture",
        mode=webrtc_streamer.WebRtcMode.SENDRECV, # 发送并接收视频
        rtc_configuration=RTC_CONFIGURATION,      # STUN 服务器配置
        video_processor_factory=VideoProcessor,   # 指定处理器
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # 实时更新处理器参数
    if ctx.video_processor:
        ctx.video_processor.update_params(conf_threshold, smooth_win, show_landmarks)

with col2:
    st.subheader("状态")
    if ctx.state.playing:
        st.success("摄像头运行中 ✅")
        st.write("正在进行云端实时推理...")
    else:
        st.warning("摄像头已停止 🛑")
        st.write("请点击左侧 'START' 按钮")
        
    st.markdown("---")
    st.markdown("""
    **如何使用：**
    1. 点击 **SELECT DEVICE** 选择摄像头。
    2. 点击 **START** 开启视频流。
    3. 将手放入画面框内。
    4. 识别结果会以 **绿色文字** 显示在视频左上角。
    """)
