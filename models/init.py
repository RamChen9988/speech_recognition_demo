"""
模型包初始化文件
"""

from .traditional_model import TraditionalAcousticModel
from .deep_learning_model import DeepLearningModel

__all__ = ['TraditionalAcousticModel', 'DeepLearningModel']