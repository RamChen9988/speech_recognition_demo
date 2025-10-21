#!/usr/bin/env python3
"""
下载Google Speech Commands数据集的脚本
"""

import os
import urllib.request
import tarfile
import shutil

def download_google_speech_commands():
    """下载Google Speech Commands数据集"""
    url = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
    filename = "speech_commands.tar.gz"
    
    print("=" * 60)
    print("下载Google Speech Commands数据集")
    print("=" * 60)
    
    # 检查是否已存在数据集
    if os.path.exists('data/speech_commands'):
        print("检测到已存在speech_commands数据集")
        response = input("是否重新下载？(y/n): ").strip().lower()
        if response != 'y':
            print("跳过下载")
            return
    
    print("开始下载数据集...")
    print("文件大小约2.1GB，下载可能需要几分钟...")
    print("如果下载失败，请尝试以下替代方案:")
    print("1. 使用浏览器手动下载: https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz")
    print("2. 将下载的文件放在项目根目录，命名为 'speech_commands.tar.gz'")
    print("3. 重新运行此脚本")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"\n下载尝试 {attempt + 1}/{max_retries}...")
            
            # 使用带进度显示的下载
            def report_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(100, int(downloaded * 100 / total_size))
                if total_size > 0:
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(f"\r下载进度: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)
            
            # 下载文件
            urllib.request.urlretrieve(url, filename, report_progress)
            print("\n下载完成！")
            break
            
        except Exception as e:
            print(f"\n下载尝试 {attempt + 1} 失败: {e}")
            if attempt < max_retries - 1:
                print("等待5秒后重试...")
                import time
                time.sleep(5)
                if os.path.exists(filename):
                    os.remove(filename)
            else:
                print(f"\n所有下载尝试都失败了。")
                print("请尝试手动下载:")
                print("1. 访问: https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz")
                print("2. 下载文件到项目根目录")
                print("3. 确保文件名为 'speech_commands.tar.gz'")
                print("4. 重新运行此脚本")
                return
    
    # 如果下载成功，继续解压
    if os.path.exists(filename):
        try:
            print("解压数据集...")
            # 解压到data目录
            with tarfile.open(filename, 'r:gz') as tar:
                tar.extractall('data/')
            
            print("数据集解压完成！")
            
            # 清理临时文件
            os.remove(filename)
            
            # 检查解压后的目录结构
            speech_commands_dir = 'data/speech_commands'
            if os.path.exists(speech_commands_dir):
                commands = [d for d in os.listdir(speech_commands_dir) 
                           if os.path.isdir(os.path.join(speech_commands_dir, d))]
                print(f"\n数据集包含的命令: {len(commands)} 个")
                print("部分命令示例:", commands[:10])
                
                # 统计样本数量
                total_samples = 0
                for command in commands[:5]:  # 只检查前5个命令
                    command_dir = os.path.join(speech_commands_dir, command)
                    samples = len([f for f in os.listdir(command_dir) if f.endswith('.wav')])
                    total_samples += samples
                    print(f"  {command}: {samples} 个样本")
                
                print(f"\n数据集准备完成！")
                print("现在您可以运行 main.py 来使用真实语音数据训练模型")
                
            else:
                print("警告: 解压后未找到speech_commands目录")
                
        except Exception as e:
            print(f"解压过程中出现错误: {e}")
            if os.path.exists(filename):
                os.remove(filename)

def setup_project_structure():
    """设置项目数据结构"""
    print("\n设置项目数据结构...")
    
    # 创建必要的目录
    os.makedirs('data/commands', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)
    
    print("项目结构设置完成")

if __name__ == "__main__":
    setup_project_structure()
    download_google_speech_commands()
