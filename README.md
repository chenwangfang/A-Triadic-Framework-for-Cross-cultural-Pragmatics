# 跨文化语用学的三元框架：数字符号资源作为分布式认知的构成成分

<p align="center">
  <img src="https://img.shields.io/badge/研究状态-进行中-brightgreen" alt="研究状态">
  <img src="https://img.shields.io/badge/数据覆盖-2021--2025-blue" alt="数据覆盖">
  <img src="https://img.shields.io/badge/分析方法-混合方法-orange" alt="分析方法">
  <img src="https://img.shields.io/badge/语言-中文/English-red" alt="语言">
</p>

## 项目概览

本项目通过分析中国外交部例行记者会（2021-2025）的语料，证明**数字符号资源（DSR）是分布式认知的构成性成分**。研究采用混合方法，结合统计分析、信息论和网络分析等多种方法，系统验证三个核心假设。

### 核心假设

- **H1 认知依赖性**：数字符号资源作为认知系统的构成性成分
- **H2 系统调节**：语境因素系统性调节数字符号资源的认知作用
- **H3 动态演化**：数字符号资源-认知关系呈现结构化演化模式

### 理论贡献

1. **跨文化语用学**：提出三元框架（数字工具-认知过程-文化语境）
2. **分布式认知理论**：证明数字符号资源的构成性地位
3. **方法论创新**：开发66+特征的综合分析框架

## 快速开始

### 环境要求

```bash
# Python 3.8+
pip install -r requirements.txt
```

### 数据处理流程

```bash
# 1. 数据验证与预处理
cd Script_cn
python step1_data_validation.py      # 验证XML语料
python step2_data_extraction.py      # 提取66+特征
python step3_cognitive_metrics.py    # 计算DSR、TL、CS指标

# 2. 假设验证分析
python H1验证分析.py                 # H1认知依赖性验证
python H2验证分析.py                 # H2系统调节验证  
python H3验证分析.py                 # H3动态演化验证

# 3. 生成综合报告
python generate_comprehensive_report.py
```

## 项目结构

```
A-Triadic-Framework-for-Cross-cultural-Pragmatics/
├── Corpus/                    # XML语料库（2021-2025）
├── Script_cn/                 # 中文版分析脚本
│   ├── step1-9_*.py          # 顺序分析脚本
│   ├── H[1-3]验证分析.py      # 假设验证脚本
│   └── mixed_methods_*.py     # 混合方法分析
├── Script_en/                 # 英文版分析脚本
├── output_cn/                 # 中文版输出
│   ├── data/                 # CSV和JSON数据
│   ├── figures/              # 高清可视化（1200 DPI）
│   └── md/                   # Markdown报告
├── data/                      # 核心数据文件
├── context/                   # 背景资料
│   ├── 2021年3月-2025年5月/   # 时间段说明
│   ├── 外交话语标注方案/       # 标注体系
│   └── 标注语料实例/          # 标注示例
└── README.md                  # 本文档
```

## 核心指标

### 认知度量

- **DSR (Digital Symbolic Resources)**：数字符号资源使用度
  - 测量数字工具在认知过程中的参与程度
  - 范围：0-1，值越高表示数字工具使用越多

- **TL (Task Load)**：任务负荷
  - 量化认知任务的复杂度和需求
  - 考虑问题复杂性、时间压力等因素

- **CS (Cognitive Success)**：认知成功度
  - 评估认知表现的有效性
  - 基于回答完整性、准确性等多维度

### 分析维度（66+特征）

1. **语言特征**：词汇密度、句法复杂度、模态使用等
2. **语用策略**：10种关系建构策略的使用模式
3. **认知标记**：不确定性表达、认知负荷指标等
4. **互动特征**：话轮转换、问答对应等
5. **时间动态**：季度趋势、演化阶段等

## 主要发现

### H1 认知依赖性（已验证）

- 信息论分析显示显著的功能互补性（FC = 0.303, *p* < .001）
- 虚拟移除测试表明强必要性（λ = 0.905）
- 统计模型展现稳健关系（*R*² = 0.187, *p* < .001）

### H2 系统调节（已验证）

- 中等语境下调节效应最强（*β* = 0.521, *p* < .001）
- 贝叶斯分层模型显示跨语境的*R*²变异
- 因果森林分析识别出高敏感性子群

### H3 动态演化（已验证）

- S型曲线拟合确认三阶段演化模式
- 变点检测识别出2023年Q1的结构性转变
- 网络密度呈现季度性增长趋势

## 技术栈

### 数据处理
- pandas, numpy - 数据操作
- xml.etree - XML解析
- json - 数据存储

### 统计分析
- scipy, statsmodels - 传统统计
- scikit-learn - 机器学习
- pymc3 - 贝叶斯建模

### 可视化
- matplotlib, seaborn - 图表生成
- networkx - 网络可视化

## 使用指南

### 运行完整分析

```bash
# 使用提供的批处理脚本
./run_full_analysis.sh

# 或分步执行
python step1_data_validation.py
python step2_data_extraction.py
# ... 依次运行
```

### 自定义分析

```python
# 导入核心分析器
from step4_information_theory import InformationTheoryAnalyzer
from step5_statistical_models import StatisticalModeler

# 创建分析实例
analyzer = InformationTheoryAnalyzer()
results = analyzer.run_analysis(data_path='your_data.csv')
```

### 生成可视化

所有图表统一使用1200 DPI高清输出：

```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
fig = plt.figure(figsize=(20, 16), dpi=1200)
# 绘图代码
plt.savefig('output.jpg', dpi=1200, format='jpg')
```

## 引用

如果您使用本项目的代码或数据，请引用：

```bibtex
@misc{chen2025triadic,
  title={A Triadic Framework for Cross-cultural Pragmatics: 
         Digital Symbolic Resources as Constitutive Components 
         of Distributed Cognition},
  author={Chen, Fang and Liangwang, De},
  year={2025},
  howpublished={\url{https://github.com/chenwangfang/A-Triadic-Framework-for-Cross-cultural-Pragmatics}}
}
```

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 联系方式

- 项目负责人：Fang Chen and De Liangwang
- 项目主页：[GitHub Repository](https://github.com/chenwangfang/A-Triadic-Framework-for-Cross-cultural-Pragmatics)

## 致谢

感谢所有为本项目提供支持和建议的同事和审稿人。特别感谢外交部例行记者会提供的公开语料资源。

---

<p align="center">
  <i>推进跨文化语用学研究 · 深化分布式认知理论</i>
</p>