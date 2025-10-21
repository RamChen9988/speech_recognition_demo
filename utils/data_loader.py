"""
数据加载和生成工具
"""

import os
import numpy as np
import soundfile as sf
import librosa


class DataLoader:
    """
    数据加载器类
    负责生成和加载训练数据
    """
    
    def __init__(self):
        self.dataset_loaded = False
    
    def generate_sample_data(self, commands=None, samples_per_command=10, 
                           audio_duration=1.0, sample_rate=22050):
        """
        生成示例训练数据
        创建简单的语音命令数据集用于演示
        """
        if commands is None:
            commands = ['start', 'stop', 'left', 'right', 'go']
            
        print("生成示例训练数据...")
        
        # 创建数据目录
        os.makedirs('data/commands', exist_ok=True)
        
        # 定义命令词和对应的基础频率
        base_frequencies = {
            'start': 200,    # 开始 - 较低频率
            'stop': 400,     # 停止 - 中等频率  
            'left': 600,     # 左 - 较高频率
            'right': 800,    # 右 - 高频率
            'go': 300        # 前进 - 中低频率
        }
        
        # 为每个命令生成多个样本
        for command in commands:
            command_dir = f'data/commands/{command}'
            os.makedirs(command_dir, exist_ok=True)
            
            base_freq = base_frequencies.get(command, 300)
            
            for i in range(samples_per_command):
                # 生成带频率变化的音频
                t = np.linspace(0, audio_duration, int(sample_rate * audio_duration))
                
                # 基础频率加上小的随机变化
                freq_variation = base_freq + np.random.randint(-50, 50)
                
                # 生成音频信号
                audio = 0.8 * np.sin(2 * np.pi * freq_variation * t)
                
                # 添加谐波
                audio += 0.3 * np.sin(2 * np.pi * freq_variation * 2 * t)
                audio += 0.2 * np.sin(2 * np.pi * freq_variation * 3 * t)
                
                # 添加噪声
                noise = 0.05 * np.random.randn(len(audio))
                audio += noise
                
                # 保存音频
                filename = f'{command_dir}/{command}_{i:02d}.wav'
                sf.write(filename, audio, sample_rate)
        
        print("示例数据生成完成!")
        self.dataset_loaded = True
        return True
    
    def load_training_data(self, n_mfcc=13, use_real_data=True):
        """
        加载训练数据并提取特征
        
        Args:
            n_mfcc: MFCC特征维度
            use_real_data: 是否使用真实数据集，如果为False则使用合成数据
        """
        from models.traditional_model import TraditionalAcousticModel
        
        training_data = {}
        traditional_model = TraditionalAcousticModel(n_mfcc=n_mfcc)
        
        if use_real_data:
            # 优先使用Google Speech Commands数据集
            speech_commands_dir = 'data/speech_commands'
            if os.path.exists(speech_commands_dir):
                print("使用Google Speech Commands数据集")
                data_dir = speech_commands_dir
                # 只使用项目需要的命令
                target_commands = ['go', 'left', 'right', 'stop']
            else:
                print("Google数据集未找到，使用本地命令数据")
                data_dir = 'data/commands'
                target_commands = None
        else:
            # 使用合成数据
            if not self.dataset_loaded:
                self.generate_sample_data()
            data_dir = 'data/commands'
            target_commands = None
        
        # 加载命令数据
        if target_commands:
            # 只加载指定的命令
            command_dirs = [d for d in target_commands 
                           if os.path.isdir(os.path.join(data_dir, d))]
        else:
            # 加载所有命令
            command_dirs = [d for d in os.listdir(data_dir) 
                           if os.path.isdir(os.path.join(data_dir, d))]
        
        print(f"加载的命令: {command_dirs}")
        
        for command in command_dirs:
            command_path = os.path.join(data_dir, command)
            audio_files = [f for f in os.listdir(command_path) if f.endswith('.wav')]
            
            # 限制每个命令的样本数量，避免内存问题
            max_samples = min(100, len(audio_files))
            selected_files = audio_files[:max_samples]
            
            print(f"  {command}: 使用 {len(selected_files)}/{len(audio_files)} 个样本")
            
            features_list = []
            for audio_file in selected_files:
                filepath = os.path.join(command_path, audio_file)
                try:
                    audio, sr = librosa.load(filepath, sr=22050)
                    
                    # 提取特征
                    features = traditional_model.extract_features(audio, sr)
                    features_list.append(features)
                except Exception as e:
                    print(f"处理文件 {filepath} 时出错: {e}")
                    continue
            
            if features_list:
                training_data[command] = features_list
        
        if not training_data:
            print("警告: 未加载到任何训练数据！")
            print("请运行 download_dataset.py 下载真实数据集")
        
        return training_data
