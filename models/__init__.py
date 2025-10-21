"""
语音识别模型模块
包含传统声学模型和深度学习模型
"""

from .traditional_model import TraditionalAcousticModel
from .deep_learning_model import DeepLearningModel

__all__ = ['TraditionalAcousticModel', 'DeepLearningModel']
