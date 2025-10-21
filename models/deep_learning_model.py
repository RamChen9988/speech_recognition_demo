"""
深度学习声学模型实现
对应课程第四节课内容
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import os
import pickle
import joblib


class DeepLearningModel:
    """
    深度学习声学模型类
    对应PPT: 第4-8页 (深度学习在语音识别中的应用)
    """
    
    def __init__(self, input_dim=13, num_classes=5, model_type='dnn'):
        """
        初始化深度学习模型
        
        参数:
        - input_dim: 输入特征维度 (MFCC维度)
        - num_classes: 分类类别数
        - model_type: 模型类型 ('dnn', 'cnn', 'rnn')
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.history = None
        
    def build_dnn_model(self):
        """
        构建DNN模型
        对应PPT: 第4页 (深度神经网络DNN在语音识别中的应用)
        """
        print("构建DNN模型...")
        model = keras.Sequential([
            # 输入层
            layers.Dense(256, activation='relu', input_shape=(self.input_dim,)),
            layers.Dropout(0.3),  # 防止过拟合
            
            # 隐藏层
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            
            # 输出层
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # 编译模型
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def build_cnn_model(self, sequence_length=100):
        """
        构建CNN模型
        对应PPT: 第5页 (卷积神经网络CNN - 语音的"局部特征检测器")
        """
        print("构建CNN模型...")
        model = keras.Sequential([
            # 输入层 - 重塑为2D (时间帧, 特征维度, 1个通道)
            layers.Reshape((sequence_length, self.input_dim, 1), 
                          input_shape=(sequence_length, self.input_dim)),
            
            # 卷积层1 - 检测局部特征
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # 卷积层2
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # 展平后接全连接层
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            
            # 输出层
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def build_rnn_model(self, sequence_length=100):
        """
        构建RNN模型
        对应PPT: 第6页 (循环神经网络RNN - 语音的"记忆大师")
        """
        print("构建RNN模型...")
        model = keras.Sequential([
            # LSTM层 - 处理序列数据
            layers.LSTM(128, return_sequences=True, 
                       input_shape=(sequence_length, self.input_dim)),
            layers.Dropout(0.2),
            
            # 第二个LSTM层
            layers.LSTM(64),
            layers.Dropout(0.2),
            
            # 全连接层
            layers.Dense(32, activation='relu'),
            
            # 输出层
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def prepare_features(self, features_list, fixed_length=100):
        """
        准备特征数据，统一序列长度
        """
        processed_features = []
        
        for features in features_list:
            # 如果序列太长，截断
            if len(features) > fixed_length:
                features = features[:fixed_length]
            # 如果序列太短，填充
            elif len(features) < fixed_length:
                padding = np.zeros((fixed_length - len(features), features.shape[1]))
                features = np.vstack([features, padding])
            
            processed_features.append(features)
        
        return np.array(processed_features)

    def train(self, training_data, epochs=50, validation_split=0.2):
        """
        训练深度学习模型
        对应PPT: 第7页 (深度学习声学模型搭建实战)
        """
        print("开始训练深度学习模型...")
        
        # 准备数据
        X_train = []
        y_train = []
        
        for label, features_list in training_data.items():
            for features in features_list:
                X_train.append(features)
                y_train.append(label)
        
        # 特征标准化
        X_train_flat = np.vstack(X_train)
        self.scaler.fit(X_train_flat)
        
        # 标签编码
        y_encoded = self.label_encoder.fit_transform(y_train)
        
        # 根据模型类型准备数据
        if self.model_type == 'dnn':
            # DNN使用帧级别的特征 (取均值)
            X_processed = np.array([np.mean(features, axis=0) for features in X_train])
            X_processed = self.scaler.transform(X_processed)
            
            # 构建DNN模型
            self.model = self.build_dnn_model()
            
        elif self.model_type in ['cnn', 'rnn']:
            # CNN和RNN使用序列数据
            X_processed = self.prepare_features(X_train)
            
            # 标准化每个特征维度
            original_shape = X_processed.shape
            X_flat = X_processed.reshape(-1, self.input_dim)
            X_flat = self.scaler.transform(X_flat)
            X_processed = X_flat.reshape(original_shape)
            
            if self.model_type == 'cnn':
                self.model = self.build_cnn_model(sequence_length=X_processed.shape[1])
            else:
                self.model = self.build_rnn_model(sequence_length=X_processed.shape[1])
        
        # 训练模型
        print("开始训练...")
        self.history = self.model.fit(
            X_processed, y_encoded,
            epochs=epochs,
            validation_split=validation_split,
            batch_size=32,
            verbose=1
        )
        
        self.is_trained = True
        
        # 绘制训练历史
        self.plot_training_history()
        
        return self.history

    def plot_training_history(self):
        """
        绘制训练历史
        """
        if self.history is None:
            print("没有训练历史可显示")
            return
            
        plt.figure(figsize=(12, 4))
        
        # 准确率
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='训练准确率')
        plt.plot(self.history.history['val_accuracy'], label='验证准确率')
        plt.title(f'{self.model_type.upper()}模型准确率')
        plt.xlabel('Epoch')
        plt.ylabel('准确率')
        plt.legend()
        
        # 损失
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='训练损失')
        plt.plot(self.history.history['val_loss'], label='验证损失')
        plt.title(f'{self.model_type.upper()}模型损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def predict(self, audio, sr=22050):
        """
        使用深度学习模型进行预测
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        # 提取特征 (导入TraditionalAcousticModel来使用特征提取方法)
        from models.traditional_model import TraditionalAcousticModel
        traditional_model = TraditionalAcousticModel(n_mfcc=self.input_dim)
        features = traditional_model.extract_features(audio, sr)
        
        # 根据模型类型准备特征
        if self.model_type == 'dnn':
            features_processed = np.mean(features, axis=0).reshape(1, -1)
            features_processed = self.scaler.transform(features_processed)
        else:
            features_processed = self.prepare_features([features])
            original_shape = features_processed.shape
            features_flat = features_processed.reshape(-1, self.input_dim)
            features_flat = self.scaler.transform(features_flat)
            features_processed = features_flat.reshape(original_shape)
        
        # 预测
        predictions = self.model.predict(features_processed)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_label = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        
        # 计算所有类别的概率
        scores = {}
        for i, label in enumerate(self.label_encoder.classes_):
            scores[label] = predictions[0][i]
        
        return predicted_label, scores

    def save(self, model_path):
        """
        保存模型到指定路径
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，无法保存")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # 保存Keras模型
        self.model.save(f"{model_path}_model.h5")
        
        # 保存预处理器和编码器
        joblib.dump(self.scaler, f"{model_path}_scaler.pkl")
        joblib.dump(self.label_encoder, f"{model_path}_label_encoder.pkl")
        
        # 保存模型配置
        config = {
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        with open(f"{model_path}_config.pkl", 'wb') as f:
            pickle.dump(config, f)
        
        print(f"模型已保存到: {model_path}_*")

    def load(self, model_path):
        """
        从指定路径加载模型
        """
        try:
            # 加载模型配置
            with open(f"{model_path}_config.pkl", 'rb') as f:
                config = pickle.load(f)
            
            # 验证配置
            if (config['input_dim'] != self.input_dim or 
                config['num_classes'] != self.num_classes or
                config['model_type'] != self.model_type):
                print("警告: 模型配置不匹配，可能影响预测效果")
            
            # 加载Keras模型
            self.model = keras.models.load_model(f"{model_path}_model.h5")
            
            # 加载预处理器和编码器
            self.scaler = joblib.load(f"{model_path}_scaler.pkl")
            self.label_encoder = joblib.load(f"{model_path}_label_encoder.pkl")
            
            # 更新状态
            self.is_trained = True
            
            print(f"模型已从 {model_path}_* 加载")
            return True
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False

    def model_exists(self, model_path):
        """
        检查模型文件是否存在
        """
        required_files = [
            f"{model_path}_model.h5",
            f"{model_path}_scaler.pkl", 
            f"{model_path}_label_encoder.pkl",
            f"{model_path}_config.pkl"
        ]
        
        return all(os.path.exists(f) for f in required_files)
