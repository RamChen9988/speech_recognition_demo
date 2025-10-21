"""
项目配置文件
包含所有可调整的参数和设置
"""

# 音频参数配置
AUDIO_CONFIG = {
    'sample_rate': 22050,
    'duration': 1.0,  # 录音时长(秒)
    'n_mfcc': 13,     # MFCC特征维度
}

# 传统模型配置
TRADITIONAL_MODEL_CONFIG = {
    'n_states': 3,           # HMM状态数
    'n_gmm_components': 3,   # GMM混合分量数
}

# 深度学习模型配置
DEEP_LEARNING_CONFIG = {
    'dnn': {
        'hidden_layers': [256, 128],
        'dropout_rate': 0.3,
    },
    'cnn': {
        'filters': [32, 64],
        'kernel_size': (3, 3),
        'pool_size': (2, 2),
    },
    'rnn': {
        'lstm_units': [128, 64],
        'dropout_rate': 0.2,
    },
    'training': {
        'epochs': 50,
        'batch_size': 32,
        'validation_split': 0.2,
    }
}

# 数据配置
DATA_CONFIG = {
    'commands': ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'],  # 命令词列表
    'samples_per_command': 200,  # 每个命令的样本数
    'audio_duration': 1.0,      # 生成音频的时长
}

# 路径配置
PATH_CONFIG = {
    'data_dir': 'data/commands',
    'model_dir': 'saved_models',
}
