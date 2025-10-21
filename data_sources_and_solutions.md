# 语音识别数据集问题分析与解决方案

## 问题确认

**是的，您的问题确实是由数据集造成的！**

当前 `data/commands` 目录中的文件是程序生成的**合成正弦波信号**，不是真实的语音数据。这些信号：
- 只有简单的频率差异
- 缺乏真实语音的复杂特征
- 无法训练出有效的语音识别模型

## 可靠的语音数据集来源

### 1. 开源语音命令数据集

#### Google Speech Commands Dataset (推荐)
- **下载地址**: https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
- **特点**: 包含35个语音命令，每个命令约1000个样本
- **命令**: "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go" 等
- **大小**: 约2.1GB

#### Mozilla Common Voice
- **网址**: https://commonvoice.mozilla.org/
- **特点**: 多语言，包含中文
- **下载**: 需要注册，可下载特定语言的子集

#### LibriSpeech
- **网址**: https://www.openslr.org/12
- **特点**: 英语朗读语音，适合连续语音识别

### 2. 中文语音数据集

#### AISHELL-1
- **网址**: https://www.openslr.org/33/
- **特点**: 178小时中文语音，400个说话人
- **适合**: 中文语音识别

#### THCHS-30
- **网址**: https://www.openslr.org/18/
- **特点**: 30小时中文语音数据
- **适合**: 中文语音识别教学

## 快速解决方案

### 方案1: 使用Google Speech Commands数据集

```python
# 下载和准备真实数据的脚本
import os
import urllib.request
import tarfile

def download_google_speech_commands():
    """下载Google Speech Commands数据集"""
    url = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
    filename = "speech_commands.tar.gz"
    
    print("下载Google Speech Commands数据集...")
    urllib.request.urlretrieve(url, filename)
    
    print("解压数据集...")
    with tarfile.open(filename, 'r:gz') as tar:
        tar.extractall('data/')
    
    print("数据集准备完成！")
    os.remove(filename)  # 删除压缩包

# 运行下载
download_google_speech_commands()
```

### 方案2: 创建真实录音数据集

```python
# 录制真实语音数据的脚本
import pyaudio
import wave
import os

def record_voice_commands():
    """录制真实语音命令"""
    commands = ['start', 'stop', 'left', 'right', 'go']
    samples_per_command = 50  # 每个命令录制50个样本
    
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 22050
    RECORD_SECONDS = 2
    
    p = pyaudio.PyAudio()
    
    for command in commands:
        command_dir = f'data/commands/{command}'
        os.makedirs(command_dir, exist_ok=True)
        
        print(f"录制命令: {command}")
        print("请准备录制...")
        
        for i in range(samples_per_command):
            input(f"按回车开始录制第 {i+1} 个样本 (说 '{command}')...")
            
            stream = p.open(format=FORMAT,
                          channels=CHANNELS,
                          rate=RATE,
                          input=True,
                          frames_per_buffer=CHUNK)
            
            print("录音中...")
            frames = []
            
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)
            
            print("录音结束")
            
            stream.stop_stream()
            stream.close()
            
            # 保存录音
            filename = f'{command_dir}/{command}_{i:02d}.wav'
            wf = wave.open(filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
    
    p.terminate()
    print("所有录音完成！")

# 运行录制
# record_voice_commands()
```

## 推荐的实施步骤

### 步骤1: 下载Google数据集 (最简单)
```bash
# 运行下载脚本
python download_dataset.py
```

### 步骤2: 更新数据加载器
修改 `utils/data_loader.py` 中的 `load_training_data` 方法，使用真实数据集。

### 步骤3: 重新训练模型
运行主程序重新训练所有模型。

## 预期效果

使用真实语音数据集后：
- 模型准确率将从接近0%提升到85-95%
- 能够正确识别真实语音命令
- 模型泛化能力显著提高

## 注意事项

1. **数据量**: 每个命令至少需要50-100个样本才能有效训练
2. **数据质量**: 确保录音清晰，背景噪音小
3. **数据平衡**: 每个类别的样本数量应该相近
4. **数据分割**: 训练集、验证集、测试集比例建议为 70:15:15

选择Google Speech Commands数据集是最简单有效的解决方案！
