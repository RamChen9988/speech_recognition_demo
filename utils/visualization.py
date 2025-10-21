"""
可视化工具类
提供各种图表和演示功能
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


class Visualization:
    """
    可视化工具类
    提供算法原理和结果的可视化
    """
    
    @staticmethod
    def demonstrate_gmm_concept():
        """
        演示GMM概念
        对应PPT: 第6页 (高斯混合模型GMM - 概率的"计算器")
        """
        print("=== GMM概念演示 ===")
        
        # 生成模拟的MFCC特征数据
        np.random.seed(42)
        
        # 生成三个不同的高斯分布，模拟三个音素
        n_samples = 100
        means = [
            [1, 1],    # 音素1的特征中心
            [4, 4],    # 音素2的特征中心  
            [1, 4]     # 音素3的特征中心
        ]
        
        covs = [
            [[0.3, 0.1], [0.1, 0.3]],
            [[0.4, -0.1], [-0.1, 0.4]],
            [[0.2, 0], [0, 0.5]]
        ]
        
        # 生成数据
        data1 = np.random.multivariate_normal(means[0], covs[0], n_samples)
        data2 = np.random.multivariate_normal(means[1], covs[1], n_samples)
        data3 = np.random.multivariate_normal(means[2], covs[2], n_samples)
        
        # 可视化
        plt.figure(figsize=(15, 5))
        
        # 原始数据分布
        plt.subplot(1, 3, 1)
        plt.scatter(data1[:, 0], data1[:, 1], alpha=0.6, label='音素1')
        plt.scatter(data2[:, 0], data2[:, 1], alpha=0.6, label='音素2')
        plt.scatter(data3[:, 0], data3[:, 1], alpha=0.6, label='音素3')
        plt.title('不同音素的MFCC特征分布')
        plt.xlabel('MFCC系数1')
        plt.ylabel('MFCC系数2')
        plt.legend()
        
        # 单高斯建模
        plt.subplot(1, 3, 2)
        # 对音素1用单高斯建模
        gmm_single = GaussianMixture(n_components=1)
        gmm_single.fit(data1)
        
        # 绘制等高线
        x = np.linspace(-1, 6, 100)
        y = np.linspace(-1, 6, 100)
        X, Y = np.meshgrid(x, y)
        XX = np.array([X.ravel(), Y.ravel()]).T
        Z = gmm_single.score_samples(XX)
        Z = Z.reshape(X.shape)
        
        plt.contourf(X, Y, Z, levels=20, alpha=0.6)
        plt.scatter(data1[:, 0], data1[:, 1], alpha=0.3)
        plt.title('单高斯建模 (局限性)')
        plt.xlabel('MFCC系数1')
        plt.ylabel('MFCC系数2')
        
        # GMM建模
        plt.subplot(1, 3, 3)
        gmm_multi = GaussianMixture(n_components=2)  # 使用2个高斯分量
        gmm_multi.fit(data1)
        
        Z_multi = gmm_multi.score_samples(XX)
        Z_multi = Z_multi.reshape(X.shape)
        
        plt.contourf(X, Y, Z_multi, levels=20, alpha=0.6)
        plt.scatter(data1[:, 0], data1[:, 1], alpha=0.3)
        plt.title('GMM建模 (多个高斯分布)')
        plt.xlabel('MFCC系数1')
        plt.ylabel('MFCC系数2')
        
        plt.tight_layout()
        plt.show()
        
        print("GMM演示完成: 展示了如何用多个高斯分布更好地建模复杂数据分布")
    
    @staticmethod
    def plot_model_comparison(accuracies, model_names):
        """
        绘制模型性能对比图
        对应PPT: 第8页 (实际应用案例与效果对比)
        """
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, accuracies, 
                      color=['lightblue', 'lightgreen', 'lightcoral', 'gold'])
        
        # 在柱状图上添加数值
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.title('不同语音识别模型性能对比')
        plt.ylabel('验证准确率')
        plt.ylim(0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.show()
        
        print("\n性能对比总结:")
        print("传统GMM-HMM: 理论基础强，需要人工设计特征")
        print("深度学习DNN: 自动学习特征，准确率显著提升") 
        print("深度学习CNN: 擅长提取局部特征，抗噪声能力强")
        print("深度学习RNN: 擅长处理序列数据，理解上下文")