"""
手势分类器模块 - 优化版
使用机器学习模型识别手语字母，增强了置信度处理
"""
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

class GestureClassifier:
    """手势分类器"""

    # 30个手语字母标签
    LABELS = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        'ZH', 'CH', 'SH', 'NG'
    ]

    def __init__(self, model_path=None):
        # 寻找模型路径逻辑保持不变
        if model_path is None:
            current_dir = Path(__file__).parent
            possible_paths = [
                current_dir / 'gesture_model.pkl',
                Path('gesture_model.pkl'),
                Path.cwd() / 'gesture_model.pkl',
            ]

            model_path = None
            for path in possible_paths:
                if path.exists():
                    model_path = str(path)
                    break

            if model_path is None:
                model_path = 'gesture_model.pkl'

        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        """加载训练好的模型"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"模型已加载: {self.model_path}")
            except Exception as e:
                print(f"模型加载失败: {e}")
                self.model = None
        else:
            print(f"警告: 模型文件不存在: {self.model_path}")
            print("系统将无法进行有效预测，直到运行 train_model.py")
            self.model = None # 标记为未加载

    def predict(self, features):
        """
        预测手势 - 包含置信度平滑处理
        """
        # 1. 检查模型是否加载
        if self.model is None:
            return "No Model", 0.0

        # 2. 检查特征有效性
        if features is None or len(features) == 0:
            return None, 0.0

        # 转换为 numpy 数组
        features_np = np.array(features)

        # 3. 形状检查 (确保是二维数组 [1, n_features])
        if features_np.ndim == 1:
            features_np = features_np.reshape(1, -1)

        try:
            # 4. 获取预测概率
            probabilities = self.model.predict_proba(features_np)[0]

            # 获取最大概率的索引和值
            max_index = np.argmax(probabilities)
            confidence = probabilities[max_index]
            label = self.model.classes_[max_index] # 使用模型中保存的类标签

            # --- 🌟 关键优化：置信度处理 🌟 ---

            # 如果是随机森林，有时候它会对于没见过的手势给出 0.4 左右的均匀概率
            # 我们希望这种情况下置信度能显得更低，以便被阈值过滤掉

            # 策略：如果第二高的概率也非常接近最高概率，说明模型很犹豫
            sorted_probs = np.sort(probabilities)
            if len(sorted_probs) >= 2:
                top1 = sorted_probs[-1]
                top2 = sorted_probs[-2]

                # 如果前两名差距很小 (比如 0.45 和 0.40)，说明模型不确定
                if (top1 - top2) < 0.1:
                    confidence = confidence * 0.7 # 惩罚置信度

            return label, float(confidence)

        except Exception as e:
            print(f"预测出错: {e}")
            return "Error", 0.0

    # 训练相关代码可以保留，也可以省略，因为 Web App 主要只用 predict
    def get_label_index(self, label):
        try:
            return self.LABELS.index(label.upper())
        except ValueError:
            return -1