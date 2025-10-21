# 语音识别教学项目

这是一个用于教学演示的语音识别项目，整合了传统GMM-HMM方法和现代深度学习方法。

## 项目特点

- **模块化设计**：代码结构清晰，便于理解和扩展
- **教学友好**：详细的注释和文档说明
- **可视化演示**：丰富的图表展示算法原理
- **实时交互**：支持实时录音和识别演示

## 项目结构

## 使用流程

1. **首次运行**：模型不存在，自动训练并保存
2. **后续运行**：检测到已保存模型，直接加载使用
3. **实时演示**：加载所有可用模型进行语音识别

## 文件结构

```
speech_recognition_demo/
├── .gitignore                    # Git忽略文件配置
├── README.md                     # 项目说明文档
├── requirements.txt              # Python依赖包列表
├── main.py                       # 主程序入口
├── download_dataset.py           # 数据集下载脚本
├── 自动保存和加载功能说明.md      # 自动保存功能说明文档
├── data_sources_and_solutions.md # 数据源和解决方案文档
│
├── config/                       # 配置文件目录
│   └── settings.py               # 项目配置设置
│
├── data/                         # 数据文件目录
│   └── (语音数据集文件)
│
├── models/                       # 模型实现目录
│   ├── __init__.py
│   ├── init.py
│   ├── traditional_model.py      # 传统模型实现
│   └── deep_learning_model.py    # 深度学习模型实现
│
├── saved_models/                 # 训练好的模型保存目录
│   ├── traditional_model_*       # 传统模型文件
│   ├── dnn_model_*               # DNN模型文件  
│   ├── cnn_model_*               # CNN模型文件
│   └── rnn_model_*               # RNN模型文件
│
└── utils/                        # 工具函数目录
    ├── __init__.py
    ├── init.py
    ├── data_loader.py            # 数据加载工具
    ├── preprocess.py             # 数据预处理工具
    └── visualization.py          # 可视化工具
```

## 配置说明

配置文件 `config/settings.py` 中的关键设置：
- `DATA_CONFIG['commands']`: 定义所有命令词列表
- `PATH_CONFIG['model_dir']`: 模型保存目录

## 快速开始

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 下载数据集：
   ```bash
   python download_dataset.py
   ```

3. 运行主程序：
   ```bash
   python main.py
   ```

## 模块说明

- **main.py**: 项目主入口，协调各个模块运行
- **models/**: 包含传统和深度学习模型实现
- **utils/**: 提供数据处理、可视化和工具函数
- **config/**: 项目配置管理
- **data/**: 存储语音数据集
- **saved_models/**: 保存训练好的模型文件
