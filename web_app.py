"""
web_app.py - 优化版
包含高级平滑处理和严格置信度过滤，对齐 main.py 的识别体验
"""
import os
import cv2
import time
import numpy as np
import traceback
import collections
from flask import Flask, render_template, Response, jsonify, request
import datetime

# 引入你的核心模块 (确保这两个文件在同级目录)
from hand_landmarks import HandLandmarkDetector
from gesture_classifier import GestureClassifier

app = Flask(__name__)

# ===========================
# 1. 初始化核心模型
# ===========================
detector = HandLandmarkDetector()
classifier = GestureClassifier()

# ===========================
# 2. 全局状态管理 (核心)
# ===========================
class AppState:
    def __init__(self):
        self.mode = 'recognition'       # 'recognition' 或 'learning'

        # --- 识别核心参数 ---
        self.confidence_threshold = 0.6 # 默认阈值，低于此值不显示
        self.smooth_window = 2          # 平滑窗口大小 (5帧)

        # --- 实时状态 ---
        self.last_prediction = None     # 最终输出给前端的预测结果
        self.last_confidence = 0.0      # 最终置信度
        self.hand_detected = False      # 是否检测到手

        # --- 历史与统计 ---
        # 用于平滑的队列 (存储最近5帧的原始预测)
        self.raw_predictions_queue = collections.deque(maxlen=self.smooth_window)
        self.prediction_history = []    # 发送给前端的历史列表
        self.gesture_statistics = {}    # 统计数据
        self.performance_metrics = {
            'frame_rate': 0,
            'detection_time': 0,
            'classification_time': 0
        }

        # --- 控制开关 ---
        self.is_running = False         # 是否启动摄像头
        self.show_landmarks = True      # 是否绘制骨架
        self.is_recording = False       # 是否记录
        self.current_gesture = None     # 学习模式下的目标手势
        self.demo_gesture = None        # 演示模式手势

state = AppState()

# ===========================
# 3. 核心逻辑：视频流处理
# ===========================
def generate_frames():
    import time

    while not state.is_running:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.5)

    # 尝试开启 DSHOW 加速
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    # 🔴 设置摄像头分辨率 (降低分辨率 = 极大提速)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    frame_count = 0
    start_time = time.time()

    # 🔴 跳帧计数器
    process_interval = 2  # 每 2 帧检测一次 (1表示每帧都检，2表示隔一帧，3表示隔两帧)
    current_frame_index = 0
    last_annotated_frame = None

    try:
        while state.is_running:
            frame_count += 1
            current_frame_index += 1

            # 计算FPS
            if frame_count % 15 == 0:
                elapsed = time.time() - start_time
                if elapsed > 0:
                    state.performance_metrics['frame_rate'] = round(15 / elapsed, 1)
                start_time = time.time()
                frame_count = 0

            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)

            # 🔴 只有满足间隔时，才进行重型计算 (AI识别)
            if current_frame_index % process_interval == 0:

                # 1. 检测
                t1 = time.time()
                landmarks, annotated_frame = detector.detect(frame)
                state.performance_metrics['detection_time'] = round((time.time() - t1) * 1000, 2)
                state.hand_detected = (landmarks is not None)

                # 缓存这一帧的画面，下一帧直接用，省算力
                last_annotated_frame = annotated_frame

                # 2. 识别
                current_raw_pred = None
                current_raw_conf = 0.0

                if landmarks is not None and len(landmarks) > 0:
                    features = detector.extract_features(landmarks)
                    if features is not None and len(features) > 0:
                        t2 = time.time()
                        raw_pred, raw_conf = classifier.predict(features)
                        state.performance_metrics['classification_time'] = round((time.time() - t2) * 1000, 2)
                        current_raw_pred = raw_pred
                        current_raw_conf = raw_conf

                # 3. 数据平滑
                if current_raw_pred:
                    state.raw_predictions_queue.append((current_raw_pred, current_raw_conf))
                else:
                    # 注意：跳帧时不要清空队列，只有真正没检测到才清空
                    if landmarks is None:
                        state.raw_predictions_queue.clear()
                        state.last_prediction = None

                if len(state.raw_predictions_queue) >= min(3, state.smooth_window):
                    gestures = [p[0] for p in state.raw_predictions_queue]
                    most_common_gesture = max(set(gestures), key=gestures.count)
                    avg_confidence = np.mean([c for g, c in state.raw_predictions_queue if g == most_common_gesture])

                    if avg_confidence >= state.confidence_threshold:
                        state.last_prediction = most_common_gesture
                        state.last_confidence = float(avg_confidence)
                        state.gesture_statistics[most_common_gesture] = state.gesture_statistics.get(
                            most_common_gesture, 0) + 1

                        if not state.prediction_history or state.prediction_history[-1][0] != most_common_gesture:
                            state.prediction_history.append((most_common_gesture, float(avg_confidence)))
                            if len(state.prediction_history) > 20:
                                state.prediction_history.pop(0)
                    else:
                        state.last_prediction = None
                        state.last_confidence = float(avg_confidence)

            else:
                # 🔴 跳过的帧：直接使用上一帧的画面 (或者只显示原图)
                if last_annotated_frame is not None:
                    annotated_frame = last_annotated_frame
                else:
                    annotated_frame = frame

            # 显示
            final_frame = annotated_frame if state.show_landmarks else frame
            ret, buffer = cv2.imencode('.jpg', final_frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            # 🔴 极速模式下，sleep可以减少一点点，比如 0.02
            time.sleep(0.02)

    except Exception as e:
        print(f"流处理错误: {e}")
        traceback.print_exc()
    finally:
        cap.release()

# ===========================
# 4. Flask 路由
# ===========================

@app.route('/')
def index():
    # 传递必要变量给模板
    current_year = datetime.datetime.now().year
    gestures_list = classifier.LABELS if hasattr(classifier, 'LABELS') else []
    return render_template('index.html', gestures=gestures_list, current_year=current_year)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_recognition', methods=['POST'])
def toggle_recognition():
    data = request.json
    if 'is_running' in data:
        state.is_running = bool(data['is_running'])
        # 如果停止，清空一下队列
        if not state.is_running:
            state.raw_predictions_queue.clear()
            state.last_prediction = None
        return jsonify({'status': 'success', 'running': state.is_running})
    return jsonify({'status': 'error'}), 400

@app.route('/api/state', methods=['GET'])
def get_state():
    # 前端轮询这个接口获取最新数据
    return jsonify({
        'mode': state.mode,
        'is_running': state.is_running,
        'hand_detected': state.hand_detected,

        # 这里的 last_prediction 已经是经过 平滑+阈值过滤 后的结果了
        'last_prediction': state.last_prediction,
        'last_confidence': state.last_confidence,

        'prediction_history': state.prediction_history,
        'gesture_statistics': state.gesture_statistics,
        'performance_metrics': state.performance_metrics,
        'show_landmarks': state.show_landmarks,
        'confidence_threshold': state.confidence_threshold,
        'total_predictions': sum(state.gesture_statistics.values())
    })

@app.route('/api/settings', methods=['POST'])
def update_settings():
    data = request.json
    # 更新设置
    if 'confidence_threshold' in data:
        state.confidence_threshold = float(data['confidence_threshold'])
        print(f"阈值更新为: {state.confidence_threshold}")

    if 'smooth_window' in data:
        val = int(data['smooth_window'])
        state.smooth_window = max(1, min(10, val))
        # 调整队列长度
        state.raw_predictions_queue = collections.deque(state.raw_predictions_queue, maxlen=state.smooth_window)

    if 'show_landmarks' in data:
        state.show_landmarks = bool(data['show_landmarks'])

    if 'mode' in data:
        state.mode = data['mode']

    if 'is_recording' in data:
        state.is_recording = bool(data['is_recording'])

    return jsonify({'status': 'success'})

@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    state.prediction_history = []
    return jsonify({'status': 'success'})

@app.route('/api/reset_statistics', methods=['POST'])
def reset_statistics():
    state.gesture_statistics = {}
    return jsonify({'status': 'success'})


# 找到 web_app.py 最底部的这几行
if __name__ == '__main__':
    # 确保文件夹存在
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)

    print("启动 Web 应用...")
    print("请确保 hand_landmarks.py 和 gesture_classifier.py 在同一目录")

    # 🔴 修改这里：
    # 1. debug=False (关闭调试模式，防止重载器冲突)
    # 2. threaded=True (开启多线程，让视频流和API请求互不干扰)
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)