"""
main.py - Web 界面启动入口
运行此文件将启动服务器并自动打开浏览器界面
"""
import os
import cv2
import time
import numpy as np
import traceback
import collections
import datetime
import webbrowser  # 🟢 新增：用于自动打开浏览器
from flask import Flask, render_template, Response, jsonify, request

# 引入你的核心模块
from hand_landmarks import HandLandmarkDetector
from gesture_classifier import GestureClassifier

# 创建 Flask 应用
app = Flask(__name__)

# ===========================
# 1. 初始化核心模型
# ===========================
detector = HandLandmarkDetector()
# 使用无需模型的规则分类器（最稳定），如果你有模型文件，可以换回机器学习版
classifier = GestureClassifier()


# ===========================
# 2. 全局状态管理
# ===========================
class AppState:
    def __init__(self):
        self.mode = 'recognition'
        self.confidence_threshold = 0.6
        self.smooth_window = 3  # 默认平滑窗口

        self.last_prediction = None
        self.last_confidence = 0.0
        self.hand_detected = False

        # 队列用于平滑处理
        self.raw_predictions_queue = collections.deque(maxlen=self.smooth_window)
        self.prediction_history = []
        self.gesture_statistics = {}
        self.performance_metrics = {
            'frame_rate': 0,
            'detection_time': 0,
            'classification_time': 0
        }

        self.is_running = False
        self.show_landmarks = True
        self.is_recording = False
        self.current_gesture = None


state = AppState()


# ===========================
# 3. 视频流处理 (包含所有优化)
# ===========================
def generate_frames():
    # 初始等待
    while not state.is_running:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.5)

    # 尝试开启 DSHOW 加速 (Windows)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    # ⚡ 速度优化：降低分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    frame_count = 0
    start_time = time.time()

    try:
        while state.is_running:
            # FPS 计算
            frame_count += 1
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

            # 1. 检测
            t1 = time.time()
            landmarks, annotated_frame = detector.detect(frame)
            state.performance_metrics['detection_time'] = round((time.time() - t1) * 1000, 2)
            state.hand_detected = (landmarks is not None)

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

            # 3. 平滑与投票
            if current_raw_pred:
                state.raw_predictions_queue.append((current_raw_pred, current_raw_conf))
            else:
                if landmarks is None:
                    state.raw_predictions_queue.clear()
                    state.last_prediction = None

            if len(state.raw_predictions_queue) >= min(2, state.smooth_window):
                gestures = [p[0] for p in state.raw_predictions_queue]
                most_common_gesture = max(set(gestures), key=gestures.count)
                avg_confidence = np.mean([c for g, c in state.raw_predictions_queue if g == most_common_gesture])

                # 严格阈值过滤
                if avg_confidence >= state.confidence_threshold:
                    state.last_prediction = most_common_gesture
                    state.last_confidence = float(avg_confidence)
                    state.gesture_statistics[most_common_gesture] = state.gesture_statistics.get(most_common_gesture,
                                                                                                 0) + 1

                    if not state.prediction_history or state.prediction_history[-1][0] != most_common_gesture:
                        state.prediction_history.append((most_common_gesture, float(avg_confidence)))
                        if len(state.prediction_history) > 20: state.prediction_history.pop(0)
                else:
                    state.last_prediction = None
                    state.last_confidence = float(avg_confidence)

            # 显示处理
            final_frame = annotated_frame if state.show_landmarks else frame
            ret, buffer = cv2.imencode('.jpg', final_frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            # 防卡顿休眠
            time.sleep(0.03)

    except Exception as e:
        print(f"流错误: {e}")
        traceback.print_exc()
    finally:
        cap.release()


# ===========================
# 4. Flask 路由
# ===========================
@app.route('/')
def index():
    return render_template('index.html',
                           gestures=classifier.LABELS,
                           current_year=datetime.datetime.now().year)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/toggle_recognition', methods=['POST'])
def toggle_recognition():
    data = request.json
    if 'is_running' in data:
        state.is_running = bool(data['is_running'])
        if not state.is_running:
            state.raw_predictions_queue.clear()
            state.last_prediction = None
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error'}), 400


@app.route('/api/state', methods=['GET'])
def get_state():
    return jsonify({
        'mode': state.mode,
        'hand_detected': state.hand_detected,
        'last_prediction': state.last_prediction,
        'last_confidence': state.last_confidence,
        'prediction_history': state.prediction_history,
        'gesture_statistics': state.gesture_statistics,
        'performance_metrics': state.performance_metrics,
        'total_predictions': sum(state.gesture_statistics.values())
    })


@app.route('/api/settings', methods=['POST'])
def update_settings():
    data = request.json
    if 'confidence_threshold' in data: state.confidence_threshold = float(data['confidence_threshold'])
    if 'smooth_window' in data:
        state.smooth_window = max(1, min(10, int(data['smooth_window'])))
        state.raw_predictions_queue = collections.deque(state.raw_predictions_queue, maxlen=state.smooth_window)
    if 'show_landmarks' in data: state.show_landmarks = bool(data['show_landmarks'])
    if 'mode' in data: state.mode = data['mode']
    if 'is_recording' in data: state.is_recording = bool(data['is_recording'])
    return jsonify({'status': 'success'})


@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    state.prediction_history = []
    return jsonify({'status': 'success'})


@app.route('/api/reset_statistics', methods=['POST'])
def reset_statistics():
    state.gesture_statistics = {}
    return jsonify({'status': 'success'})


# ===========================
# 5. 主程序启动
# ===========================
if __name__ == '__main__':
    # 确保文件夹存在
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)

    # 🟢 核心功能：自动打开浏览器
    port = 5001
    url = f"http://localhost:{port}"
    print(f"🚀 系统启动中... 正在打开浏览器: {url}")

    # 延迟1.5秒打开浏览器，等待Flask启动
    from threading import Timer

    Timer(1.5, lambda: webbrowser.open(url)).start()

    # 启动服务器 (多线程模式防止卡死)
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)