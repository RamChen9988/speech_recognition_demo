"""
语音预处理工具
对应课程第二节课内容
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa


class VoicePreprocessing:
    """
    语音预处理类
    整合语音信号处理功能
    """
    
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.audio_data = None
        
    def record_audio(self, duration=3, sample_rate=22050):
        """录制音频"""
        print(f"开始录音，请说话... ({duration}秒)")
        self.sample_rate = sample_rate
        self.audio_data = sd.rec(int(duration * sample_rate), 
                                samplerate=sample_rate, 
                                channels=1)
        sd.wait()
        self.audio_data = self.audio_data.flatten()
        print("录音完成!")
        return self.audio_data
    
    def load_audio(self, filepath):
        """加载音频文件"""
        self.audio_data, self.sample_rate = librosa.load(filepath, sr=self.sample_rate)
        return self.audio_data
    
    def demonstrate_preprocessing(self, audio):
        """
        演示语音预处理流程
        对应PPT: 第二节课内容回顾
        """
        # 预加重
        audio_pre = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
        
        # 分帧
        frame_length = int(0.025 * self.sample_rate)  # 25ms
        frame_step = int(0.01 * self.sample_rate)     # 10ms
        
        frames = []
        for i in range(0, len(audio_pre) - frame_length, frame_step):
            frames.append(audio_pre[i:i+frame_length])
        frames = np.array(frames)
        
        # 加窗
        hamming_window = np.hamming(frame_length)
        windowed_frames = frames * hamming_window
        
        print(f"预处理完成: {len(frames)}帧, 每帧{frame_length}个样本")
        return audio_pre, frames, windowed_frames