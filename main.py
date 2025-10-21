"""
语音识别教学项目主程序
整合所有模块，提供用户交互界面
"""

import os
import numpy as np
from config import settings
from models import TraditionalAcousticModel, DeepLearningModel
from utils import VoicePreprocessing, DataLoader, Visualization


class SpeechRecognitionProject:
    """
    语音识别主项目类
    整合所有功能模块
    """
    
    def __init__(self):
        self.preprocessor = VoicePreprocessing()
        self.data_loader = DataLoader()
        self.traditional_model = TraditionalAcousticModel(
            n_states=settings.TRADITIONAL_MODEL_CONFIG['n_states'],
            n_mfcc=settings.AUDIO_CONFIG['n_mfcc'],
            n_gmm_components=settings.TRADITIONAL_MODEL_CONFIG['n_gmm_components']
        )
        self.dl_models = {}
        
    def compare_models_performance(self):
        """
        比较不同模型的性能
        对应PPT: 第8页 (实际应用案例与效果对比)
        """
        print("=== 模型性能对比 ===")
        
        # 模型路径配置
        model_dir = settings.PATH_CONFIG['model_dir']
        traditional_model_path = f"{model_dir}/traditional_model"
        dl_model_paths = {
            'dnn': f"{model_dir}/dnn_model",
            'cnn': f"{model_dir}/cnn_model", 
            'rnn': f"{model_dir}/rnn_model"
        }
        
        # 加载数据
        training_data = self.data_loader.load_training_data()
        
        # 训练传统模型
        print("训练传统GMM-HMM模型...")
        self.traditional_model.train_gmm_hmm(training_data)
        
        # 自动保存传统模型
        if self.traditional_model.is_trained:
            self.traditional_model.save(traditional_model_path)
            print(f"传统模型已自动保存到: {traditional_model_path}")
        
        # 训练深度学习模型
        dl_types = ['dnn', 'cnn', 'rnn']
        accuracies = []
        
        for model_type in dl_types:
            print(f"\n训练{model_type.upper()}模型...")
            dl_model = DeepLearningModel(
                input_dim=settings.AUDIO_CONFIG['n_mfcc'], 
                num_classes=len(training_data),
                model_type=model_type
            )
            
            # 训练模型
            history = dl_model.train(training_data, 
                                   epochs=settings.DEEP_LEARNING_CONFIG['training']['epochs'])
            
            # 记录最佳验证准确率
            best_val_acc = max(history.history['val_accuracy'])
            accuracies.append(best_val_acc)
            
            # 自动保存深度学习模型
            if dl_model.is_trained:
                dl_model.save(dl_model_paths[model_type])
                print(f"{model_type.upper()}模型已自动保存到: {dl_model_paths[model_type]}")
            
            # 保存模型
            self.dl_models[model_type] = dl_model
        
        # 性能对比可视化
        models = ['GMM-HMM', 'DNN', 'CNN', 'RNN']
        
        # 为传统模型估算一个准确率 (简化)
        traditional_acc = 0.75  # 假设传统模型准确率
        
        accuracies_with_traditional = [traditional_acc] + accuracies
        
        Visualization.plot_model_comparison(accuracies_with_traditional, models)
        
        print("\n所有模型训练完成并已自动保存!")
    
    def real_time_recognition_demo(self):
        """
        实时语音识别演示
        先尝试加载已保存的模型，如果不存在再重新训练
        """
        print("=== 实时语音识别演示 ===")
        print("可用的命令: start, stop, left, right, go")
        
        # 模型路径配置
        model_dir = settings.PATH_CONFIG['model_dir']
        traditional_model_path = f"{model_dir}/traditional_model"
        dl_model_paths = {
            'dnn': f"{model_dir}/dnn_model",
            'cnn': f"{model_dir}/cnn_model", 
            'rnn': f"{model_dir}/rnn_model"
        }
        
        # 跟踪哪些模型需要保存
        models_to_save = []
        
        # 尝试加载传统模型
        print("\n尝试加载传统GMM-HMM模型...")
        if self.traditional_model.model_exists(traditional_model_path):
            if self.traditional_model.load(traditional_model_path):
                print("传统模型加载成功!")
            else:
                print("传统模型加载失败，需要重新训练")
                self._train_traditional_model()
                models_to_save.append('traditional')
        else:
            print("传统模型文件不存在，需要重新训练")
            self._train_traditional_model()
            models_to_save.append('traditional')
        
        # 尝试加载深度学习模型
        for model_type, model_path in dl_model_paths.items():
            print(f"\n尝试加载{model_type.upper()}模型...")
            if model_type not in self.dl_models:
                self.dl_models[model_type] = DeepLearningModel(
                    input_dim=settings.AUDIO_CONFIG['n_mfcc'],
                    num_classes=len(settings.DATA_CONFIG['commands']),
                    model_type=model_type
                )
            
            if self.dl_models[model_type].model_exists(model_path):
                if self.dl_models[model_type].load(model_path):
                    print(f"{model_type.upper()}模型加载成功!")
                else:
                    print(f"{model_type.upper()}模型加载失败，需要重新训练")
                    self._train_dl_model(model_type)
                    models_to_save.append(model_type)
            else:
                print(f"{model_type.upper()}模型文件不存在，需要重新训练")
                self._train_dl_model(model_type)
                models_to_save.append(model_type)
        
        # 只保存重新训练的模型
        if models_to_save:
            print("\n保存重新训练的模型...")
            if 'traditional' in models_to_save and self.traditional_model.is_trained:
                self.traditional_model.save(traditional_model_path)
                print("传统模型已保存")
            
            for model_type in models_to_save:
                if model_type != 'traditional' and model_type in self.dl_models:
                    model = self.dl_models[model_type]
                    if model.is_trained:
                        model.save(dl_model_paths[model_type])
                        print(f"{model_type.upper()}模型已保存")
            
            print("模型保存完成!")
        else:
            print("\n所有模型都已加载，无需重新保存")
        
        # 实时识别循环
        while True:
            print("\n" + "-"*40)
            choice = input("选择操作: 1.录音识别 2.切换模型 3.返回\n请输入选择: ")
            
            if choice == '1':
                # 录音并识别
                audio = self.preprocessor.record_audio(
                    duration=settings.AUDIO_CONFIG['duration'],
                    sample_rate=settings.AUDIO_CONFIG['sample_rate']
                )
                
                # 使用所有模型进行识别
                print("\n识别结果:")
                
                # 传统模型
                try:
                    trad_label, trad_scores = self.traditional_model.predict(audio)
                    print(f"传统GMM-HMM: {trad_label} (置信度: {max(trad_scores.values()):.3f})")
                except:
                    print("传统GMM-HMM: 模型未训练")
                
                # 深度学习模型
                for model_type, model in self.dl_models.items():
                    if model.is_trained:
                        dl_label, dl_scores = model.predict(audio)
                        best_score = max(dl_scores.values())
                        print(f"深度学习{model_type.upper()}: {dl_label} (置信度: {best_score:.3f})")
                
                # 播放录制的音频
                print("播放录制的音频:")
                import IPython.display as ipd
                ipd.display(ipd.Audio(audio, rate=settings.AUDIO_CONFIG['sample_rate']))
                
            elif choice == '2':
                # 模型切换演示
                print("当前所有模型都已加载")
                
            elif choice == '3':
                break
            else:
                print("无效选择")

    def _train_traditional_model(self):
        """训练传统模型"""
        print("开始训练传统GMM-HMM模型...")
        training_data = self.data_loader.load_training_data()
        self.traditional_model.train_gmm_hmm(training_data)
        print("传统模型训练完成!")

    def _train_dl_model(self, model_type):
        """训练指定类型的深度学习模型"""
        print(f"开始训练{model_type.upper()}模型...")
        training_data = self.data_loader.load_training_data()
        
        if model_type not in self.dl_models:
            self.dl_models[model_type] = DeepLearningModel(
                input_dim=settings.AUDIO_CONFIG['n_mfcc'],
                num_classes=len(training_data),
                model_type=model_type
            )
        
        self.dl_models[model_type].train(
            training_data, 
            epochs=settings.DEEP_LEARNING_CONFIG['training']['epochs']
        )
        print(f"{model_type.upper()}模型训练完成!")


def main():
    """
    主程序 - 提供交互式菜单
    """
    project = SpeechRecognitionProject()
    
    print("="*60)
    print("       语音识别教学演示系统")
    print("       工程化重构版本")
    print("="*60)
    
    while True:
        print("\n" + "="*50)
        print("主菜单")
        print("="*50)
        print("1. GMM概念演示 (第三课)")
        print("2. 训练并比较所有模型")
        print("3. 实时语音识别演示") 
        print("4. 单独训练深度学习模型")
        print("5. 退出")
        
        choice = input("\n请选择功能 (1-5): ").strip()
        
        if choice == '1':
            Visualization.demonstrate_gmm_concept()
            
        elif choice == '2':
            project.compare_models_performance()
            
        elif choice == '3':
            project.real_time_recognition_demo()
            
        elif choice == '4':
            print("\n选择要训练的深度学习模型:")
            print("1. DNN (深度神经网络)")
            print("2. CNN (卷积神经网络)") 
            print("3. RNN (循环神经网络)")
            
            dl_choice = input("请选择 (1-3): ").strip()
            model_types = {'1': 'dnn', '2': 'cnn', '3': 'rnn'}
            
            if dl_choice in model_types:
                model_type = model_types[dl_choice]
                training_data = project.data_loader.load_training_data()
                
                dl_model = DeepLearningModel(
                    input_dim=settings.AUDIO_CONFIG['n_mfcc'],
                    num_classes=len(training_data),
                    model_type=model_type
                )
                
                dl_model.train(training_data, 
                             epochs=settings.DEEP_LEARNING_CONFIG['training']['epochs'])
                project.dl_models[model_type] = dl_model
                
                # 自动保存单个训练的模型
                model_dir = settings.PATH_CONFIG['model_dir']
                model_path = f"{model_dir}/{model_type}_model"
                if dl_model.is_trained:
                    dl_model.save(model_path)
                    print(f"{model_type.upper()}模型训练完成并已自动保存到: {model_path}")
                else:
                    print(f"{model_type.upper()}模型训练完成!")
            else:
                print("无效选择")
                
        elif choice == '5':
            print("感谢使用语音识别教学系统!")
            break
            
        else:
            print("无效选择，请重新输入!")


if __name__ == "__main__":
    # 创建必要的目录
    os.makedirs('data/commands', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)
    
    # 检查TensorFlow版本
    import tensorflow as tf
    print(f"TensorFlow版本: {tf.__version__}")
    
    # 运行主程序
    main()
