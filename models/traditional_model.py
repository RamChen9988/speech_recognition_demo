"""
传统声学模型实现 - GMM-HMM
对应课程第三节课内容
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import librosa
import warnings
import os
import pickle
import joblib
warnings.filterwarnings('ignore')

# 尝试导入hmmlearn，如果失败则使用简化版本
try:
    from hmmlearn import hmm
    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False
    print("警告: 未安装hmmlearn，使用简化HMM实现")


class TraditionalAcousticModel:
    """
    传统声学模型类 - 实现GMM-HMM
    对应PPT: 第3-6页 (声学模型基础 - HMM与GMM)
    """
    
    def __init__(self, n_states=3, n_mfcc=13, n_gmm_components=3):
        """
        初始化传统声学模型
        
        参数:
        - n_states: HMM状态数 (对应PPT第5页: 每个音素3-5个状态)
        - n_mfcc: MFCC特征维度
        - n_gmm_components: GMM混合分量数 (对应PPT第6页)
        """
        self.n_states = n_states
        self.n_mfcc = n_mfcc
        self.n_gmm_components = n_gmm_components
        self.models = {}  # 存储每个类别的HMM模型
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def extract_features(self, audio, sr=22050):
        """
        提取MFCC特征
        对应PPT: 第二节课回顾 (语音信号预处理)
        """
        # 预加重 - 增强高频成分
        audio_pre = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
        
        # 提取MFCC特征
        mfccs = librosa.feature.mfcc(
            y=audio_pre, 
            sr=sr, 
            n_mfcc=self.n_mfcc,
            n_fft=2048, 
            hop_length=512
        )
        
        return mfccs.T  # 返回 (时间帧, 特征维度)

    def train_simple_hmm(self, features_list):
        """
        简化版HMM训练 (用于教学演示)
        对应PPT: 第5页 (HMM在语音识别中的应用)
        """
        n_samples = len(features_list)
        feature_dim = features_list[0].shape[1]
        
        # 简化实现: 随机初始化模型参数
        # 实际应该使用EM算法训练
        means = np.random.randn(self.n_states, feature_dim) * 0.1
        covars = np.ones((self.n_states, feature_dim)) * 0.1
        
        print(f"简化HMM训练完成: {n_samples}个样本, {feature_dim}维特征")
        return {"means": means, "covars": covars}

    def train_gmm_hmm(self, training_data):
        """
        训练GMM-HMM模型
        对应PPT: 第6-7页 (GMM-HMM声学模型)
        """
        print("开始训练传统GMM-HMM声学模型...")
        
        # 收集所有特征用于标准化
        all_features = []
        for label, features_list in training_data.items():
            for features in features_list:
                all_features.append(features)
        
        # 特征标准化
        all_features_combined = np.vstack(all_features)
        self.scaler.fit(all_features_combined)
        print("特征标准化完成")
        
        # 为每个类别训练模型
        for label, features_list in training_data.items():
            print(f"训练类别 '{label}' 的模型...")
            
            # 标准化特征
            normalized_features_list = []
            for features in features_list:
                normalized_features = self.scaler.transform(features)
                normalized_features_list.append(normalized_features)
            
            if HAS_HMMLEARN:
                # 使用hmmlearn训练HMM
                lengths = [len(features) for features in normalized_features_list]
                features_combined = np.vstack(normalized_features_list)
                
                # 创建并训练GMM-HMM模型
                model = hmm.GaussianHMM(
                    n_components=self.n_states,
                    covariance_type="diag",
                    n_iter=100
                )
                model.fit(features_combined, lengths)
                self.models[label] = model
            else:
                # 使用简化版本
                self.models[label] = self.train_simple_hmm(normalized_features_list)
        
        self.is_trained = True
        print("传统GMM-HMM模型训练完成!")

    def predict(self, audio, sr=22050):
        """
        使用传统模型进行预测
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        # 提取特征
        features = self.extract_features(audio, sr)
        features = self.scaler.transform(features)
        
        # 计算每个模型的得分
        scores = {}
        for label, model in self.models.items():
            if HAS_HMMLEARN:
                try:
                    score = model.score(features)
                    scores[label] = score
                except:
                    scores[label] = -np.inf
            else:
                # 简化版本: 使用最近邻距离
                dist = np.mean(np.sqrt(np.sum((features - model["means"][0])**2, axis=1)))
                scores[label] = -dist  # 距离越小，得分越高
        
        # 返回最佳匹配类别
        best_label = max(scores, key=scores.get)
        return best_label, scores

    def save(self, model_path):
        """
        保存传统模型到指定路径
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，无法保存")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # 保存scaler
        joblib.dump(self.scaler, f"{model_path}_scaler.pkl")
        
        # 保存模型配置
        config = {
            'n_states': self.n_states,
            'n_mfcc': self.n_mfcc,
            'n_gmm_components': self.n_gmm_components,
            'is_trained': self.is_trained,
            'has_hmmlearn': HAS_HMMLEARN
        }
        with open(f"{model_path}_config.pkl", 'wb') as f:
            pickle.dump(config, f)
        
        # 保存HMM模型
        if HAS_HMMLEARN:
            # 保存hmmlearn模型
            for label, model in self.models.items():
                with open(f"{model_path}_{label}_hmm.pkl", 'wb') as f:
                    pickle.dump(model, f)
        else:
            # 保存简化模型
            with open(f"{model_path}_simple_models.pkl", 'wb') as f:
                pickle.dump(self.models, f)
        
        print(f"传统模型已保存到: {model_path}_*")

    def load(self, model_path):
        """
        从指定路径加载传统模型
        """
        try:
            # 加载模型配置
            with open(f"{model_path}_config.pkl", 'rb') as f:
                config = pickle.load(f)
            
            # 验证配置
            if (config['n_states'] != self.n_states or 
                config['n_mfcc'] != self.n_mfcc or
                config['n_gmm_components'] != self.n_gmm_components):
                print("警告: 模型配置不匹配，可能影响预测效果")
            
            # 检查hmmlearn状态是否一致
            if config['has_hmmlearn'] != HAS_HMMLEARN:
                print("警告: hmmlearn状态不一致，可能影响模型加载")
            
            # 加载scaler
            self.scaler = joblib.load(f"{model_path}_scaler.pkl")
            
            # 加载HMM模型
            if HAS_HMMLEARN and config['has_hmmlearn']:
                # 加载hmmlearn模型
                self.models = {}
                # 从配置文件中获取命令词列表
                from config import settings
                commands = settings.DATA_CONFIG['commands']
                for label in commands:
                    try:
                        with open(f"{model_path}_{label}_hmm.pkl", 'rb') as f:
                            self.models[label] = pickle.load(f)
                    except FileNotFoundError:
                        print(f"警告: 类别 {label} 的模型文件不存在")
            else:
                # 加载简化模型
                with open(f"{model_path}_simple_models.pkl", 'rb') as f:
                    self.models = pickle.load(f)
            
            # 更新状态
            self.is_trained = True
            
            print(f"传统模型已从 {model_path}_* 加载")
            return True
            
        except Exception as e:
            print(f"传统模型加载失败: {e}")
            return False

    def model_exists(self, model_path):
        """
        检查传统模型文件是否存在
        """
        required_files = [
            f"{model_path}_scaler.pkl",
            f"{model_path}_config.pkl"
        ]
        
        # 检查HMM模型文件
        if HAS_HMMLEARN:
            # 检查每个类别的HMM模型文件
            # 从配置文件中获取命令词列表
            from config import settings
            commands = settings.DATA_CONFIG['commands']
            for label in commands:
                required_files.append(f"{model_path}_{label}_hmm.pkl")
        else:
            required_files.append(f"{model_path}_simple_models.pkl")
        
        return all(os.path.exists(f) for f in required_files)
