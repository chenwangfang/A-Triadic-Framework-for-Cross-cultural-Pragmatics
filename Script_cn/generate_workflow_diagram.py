#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成H1-H3验证分析工作流程图

该脚本生成两个关键的流程图：
1. 整体数据处理流程图
2. 假设验证依赖关系图
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def create_data_flow_diagram():
    """创建整体数据处理流程图"""
    fig, ax = plt.subplots(figsize=(16, 10), dpi=1200)
    
    # 定义节点位置
    nodes = {
        'XML语料库': (2, 9),
        'step1验证': (2, 7.5),
        'step2提取': (2, 6),
        'step3指标': (2, 4.5),
        'metrics.csv': (2, 3),
        'H1验证': (0.5, 1.5),
        'H2验证': (2, 1.5),
        'H3验证': (3.5, 1.5),
        '综合报告': (2, 0)
    }
    
    # 绘制节点
    for node, (x, y) in nodes.items():
        if node == 'metrics.csv':
            # 数据文件用椭圆
            fancy_box = patches.Ellipse((x, y), 1.5, 0.8, 
                                      facecolor='lightblue', 
                                      edgecolor='darkblue', 
                                      linewidth=2)
        elif 'step' in node or 'H' in node and '验证' in node:
            # 处理步骤用矩形
            fancy_box = FancyBboxPatch((x-0.7, y-0.3), 1.4, 0.6,
                                     boxstyle="round,pad=0.1",
                                     facecolor='lightgreen',
                                     edgecolor='darkgreen',
                                     linewidth=2)
        else:
            # 其他用圆角矩形
            fancy_box = FancyBboxPatch((x-0.7, y-0.3), 1.4, 0.6,
                                     boxstyle="round,pad=0.1",
                                     facecolor='lightyellow',
                                     edgecolor='orange',
                                     linewidth=2)
        
        ax.add_patch(fancy_box)
        ax.text(x, y, node, ha='center', va='center', fontsize=10, weight='bold')
    
    # 绘制连接线
    connections = [
        ('XML语料库', 'step1验证'),
        ('step1验证', 'step2提取'),
        ('step2提取', 'step3指标'),
        ('step3指标', 'metrics.csv'),
        ('metrics.csv', 'H1验证'),
        ('metrics.csv', 'H2验证'),
        ('metrics.csv', 'H3验证'),
        ('H1验证', '综合报告'),
        ('H2验证', '综合报告'),
        ('H3验证', '综合报告')
    ]
    
    for start, end in connections:
        x1, y1 = nodes[start]
        x2, y2 = nodes[end]
        
        if start == 'metrics.csv' and 'H' in end:
            # 分叉连接
            ax.annotate('', xy=(x2, y2+0.3), xytext=(x1, y1-0.4),
                       arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
        else:
            # 直接连接
            ax.annotate('', xy=(x2, y2+0.3), xytext=(x1, y1-0.3),
                       arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    # 添加阶段标注
    ax.text(4.5, 6.75, '数据准备阶段', fontsize=12, weight='bold', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='wheat'))
    ax.text(4.5, 1.5, '假设验证阶段', fontsize=12, weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='wheat'))
    
    ax.set_xlim(-1, 5)
    ax.set_ylim(-0.5, 10)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../output_cn/figures/data_flow_diagram.jpg', dpi=1200, bbox_inches='tight')
    plt.close()

def create_hypothesis_dependency_diagram():
    """创建假设验证依赖关系图"""
    fig = plt.figure(figsize=(18, 12), dpi=1200)
    
    # 创建三个子图，每个对应一个假设
    for idx, (hypothesis, components) in enumerate([
        ('H1认知依赖性', [
            'step4信息论分析\n(功能互补性)',
            'step5统计模型\n(双向因果)',
            'step6构成性检验\n(虚拟移除)',
            'step9网络分析\n(中介中心性)',
            'mixed涌现效应\n(协同作用)'
        ]),
        ('H2系统调节', [
            'step7调节分析\n(语境梯度)',
            'step5b贝叶斯\n(R²变化)',
            'step5h因果森林\n(异质性)',
            'step5d信号提取\n(噪声模式)',
            'mixed语境峰值\n(非线性)'
        ]),
        ('H3动态演化', [
            'step8演化分析\n(S曲线)',
            'step5g变点检测\n(结构突变)',
            'mixed时间效应\n(递增趋势)',
            'step4动态涌现\n(时变性)',
            'step9网络演化\n(密度变化)'
        ])
    ]):
        ax = plt.subplot(1, 3, idx+1)
        
        # 中心节点
        center = (0, 0)
        center_circle = plt.Circle(center, 0.8, facecolor='red', alpha=0.3, edgecolor='darkred', linewidth=3)
        ax.add_patch(center_circle)
        ax.text(center[0], center[1], hypothesis, ha='center', va='center', 
                fontsize=11, weight='bold')
        
        # 周围节点
        n = len(components)
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        radius = 2.5
        
        for i, (angle, component) in enumerate(zip(angles, components)):
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            
            # 不同类型的组件用不同颜色
            if 'step' in component:
                color = 'lightblue'
                edge_color = 'darkblue'
            else:
                color = 'lightgreen'
                edge_color = 'darkgreen'
            
            rect = FancyBboxPatch((x-0.9, y-0.35), 1.8, 0.7,
                                boxstyle="round,pad=0.05",
                                facecolor=color,
                                edgecolor=edge_color,
                                linewidth=2)
            ax.add_patch(rect)
            ax.text(x, y, component, ha='center', va='center', fontsize=9)
            
            # 绘制连接线
            ax.plot([center[0] + 0.8*np.cos(angle), x - 0.9*np.cos(angle)],
                   [center[1] + 0.8*np.sin(angle), y - 0.9*np.sin(angle)],
                   'k-', lw=2, alpha=0.5)
        
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../output_cn/figures/hypothesis_dependency_diagram.jpg', dpi=1200, bbox_inches='tight')
    plt.close()

def create_integration_process_diagram():
    """创建脚本整合过程图"""
    fig, ax = plt.subplots(figsize=(14, 10), dpi=1200)
    
    # 定义整合流程
    stages = [
        ('1. 导入模块', 0.5, 8, 'lightblue'),
        ('2. 数据加载', 0.5, 6.5, 'lightgreen'),
        ('3. 分析执行', 0.5, 5, 'lightyellow'),
        ('4. 证据整合', 0.5, 3.5, 'lightcoral'),
        ('5. 结果输出', 0.5, 2, 'lightgray')
    ]
    
    # 绘制主流程
    for stage, x, y, color in stages:
        rect = FancyBboxPatch((x, y-0.4), 3, 0.8,
                            boxstyle="round,pad=0.1",
                            facecolor=color,
                            edgecolor='black',
                            linewidth=2)
        ax.add_patch(rect)
        ax.text(x+1.5, y, stage, ha='center', va='center', fontsize=12, weight='bold')
        
        if y > 2:  # 不是最后一个
            ax.annotate('', xy=(x+1.5, y-0.7), xytext=(x+1.5, y-0.4),
                       arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    
    # 添加详细说明
    details = [
        (4.5, 8, '• InformationTheoryAnalyzer\n• StatisticalModeler\n• ConstitutivenessAnalyzer', 'left'),
        (4.5, 6.5, '• data_with_metrics.csv\n• 各分析结果.json', 'left'),
        (4.5, 5, '• 运行各项分析\n• 收集指标值\n• 计算显著性', 'left'),
        (4.5, 3.5, '• 计算支持度\n• 生成证据矩阵\n• 综合评分', 'left'),
        (4.5, 2, '• CSV数据\n• JSON结果\n• JPG图表\n• MD报告', 'left')
    ]
    
    for x, y, text, align in details:
        ax.text(x, y, text, ha=align, va='center', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax.set_xlim(0, 8)
    ax.set_ylim(1, 9)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../output_cn/figures/integration_process_diagram.jpg', dpi=1200, bbox_inches='tight')
    plt.close()

def main():
    """主函数"""
    output_dir = Path('../output_cn/figures')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("生成数据流程图...")
    create_data_flow_diagram()
    
    print("生成假设依赖关系图...")
    create_hypothesis_dependency_diagram()
    
    print("生成整合过程图...")
    create_integration_process_diagram()
    
    print("所有流程图已生成完毕！")

if __name__ == "__main__":
    main()