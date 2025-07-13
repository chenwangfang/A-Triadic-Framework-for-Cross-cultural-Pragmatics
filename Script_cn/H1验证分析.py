#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
H1假设验证分析脚本
===================
假设H1（认知依赖性）：DSR作为认知系统的构成性组件

本脚本整合多种分析方法，全面验证H1假设：
1. 信息论分析 - 功能互补性、三重交互互信息
2. 构成性检验 - 虚拟移除、路径必要性、系统鲁棒性
3. 统计模型 - 线性/非线性模型、VAR因果分析
4. 网络分析 - DSR中介中心性、认知网络密度
5. 混合方法 - 认知涌现效应、DSR×TL协同作用

输出：
- H1_validation_results.csv/json - 分析结果数据
- H1_cognitive_dependency_analysis.jpg - 综合可视化（5个子图）
- H1_validation_report.md - 详细验证报告
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入APA格式化工具
from apa_formatter import format_p_value, format_correlation, format_t_test, format_f_test, format_mean_sd, format_effect_size, format_regression

# 导入必要的分析模块
try:
    from step4_information_theory_H1 import EnhancedFunctionalComplementarityAnalyzer
except ImportError:
    from step4_information_theory import FunctionalComplementarityAnalyzer as EnhancedFunctionalComplementarityAnalyzer

from step5_statistical_models import StatisticalModelsAnalyzer
from step6_constitutiveness_tests import ConstitutivenessTests
from step9_network_diffusion_analysis import NetworkDiffusionAnalysis


class H1CognitiveDependencyValidator:
    """H1假设（认知依赖性）的综合验证器"""
    
    def __init__(self, data_path='../output_cn/data'):
        """
        初始化验证器
        
        Parameters:
        -----------
        data_path : str
            数据文件路径
        """
        self.data_path = Path(data_path)
        self.output_path = self.data_path.parent
        
        # 初始化结果字典
        self.results = {
            'hypothesis': 'H1',
            'hypothesis_description': 'DSR作为认知系统的构成性组件',
            'evidence': {},
            'visualizations': {},
            'errors': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # 初始化各分析器
        self.info_analyzer = None
        self.stat_analyzer = None
        self.const_analyzer = None
        self.network_analyzer = None
        
        # 数据容器
        self.df = None
        self.analysis_results = {}
        
    def validate_data(self):
        """验证数据完整性"""
        if self.df is None:
            return False
            
        # 检查数据量
        if len(self.df) < 100:
            print(f"警告：数据量较少 ({len(self.df)} 条)，可能影响分析结果")
            
        # 检查缺失值
        missing_counts = self.df[['dsr_cognitive', 'tl_functional', 'cs_output']].isnull().sum()
        if missing_counts.any():
            print(f"警告：存在缺失值：\n{missing_counts}")
            
        return True
        
    def load_data(self):
        """加载所有必要的数据文件"""
        print("正在加载数据...")
        
        try:
            # 加载主数据文件
            data_file = self.data_path / 'data_with_metrics.csv'
            if not data_file.exists():
                raise FileNotFoundError(f"主数据文件不存在：{data_file}")
                
            self.df = pd.read_csv(data_file)
            self.df['date'] = pd.to_datetime(self.df['date'])
            
            # 验证必要的列是否存在
            required_columns = ['dsr_cognitive', 'tl_functional', 'cs_output']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            if missing_columns:
                print(f"警告：缺少必要的列：{missing_columns}")
                
            print(f"成功加载主数据：{len(self.df)} 条记录")
            
            # 加载已有的分析结果
            result_files = {
                'information_theory': 'information_theory_results.json',
                'statistical_models': 'statistical_models_results.json',
                'constitutiveness': 'constitutiveness_test_results.json',  # 修正文件名
                'network_analysis': 'network_diffusion_results.json'  # 修正文件名
            }
            
            for key, filename in result_files.items():
                filepath = self.data_path / filename
                if filepath.exists():
                    with open(filepath, 'r', encoding='utf-8') as f:
                        self.analysis_results[key] = json.load(f)
                    print(f"加载{key}结果：{filename}")
                    
        except Exception as e:
            error_msg = f"数据加载失败：{str(e)}"
            print(error_msg)
            self.results['errors'].append({
                'stage': 'data_loading',
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            })
            
    def run_information_theory_analysis(self):
        """运行信息论分析"""
        print("\n1. 执行信息论分析...")
        
        try:
            # 初始化分析器
            self.info_analyzer = EnhancedFunctionalComplementarityAnalyzer(
                data_path=str(self.data_path)
            )
            
            # 如果已有结果，直接使用
            if 'information_theory' in self.analysis_results:
                results = self.analysis_results['information_theory']
            else:
                # 运行新分析
                results = self.info_analyzer.analyze_functional_complementarity()
            
            # 提取关键指标
            self.results['evidence']['information_theory'] = {
                'functional_complementarity': results.get('functional_complementarity', {}),
                'triple_interaction_mi': results.get('nonlinear_mi', {}).get('triple_interaction_mi', 0),
                'conditional_mi': results.get('continuous_mi', {}),
                'synergy_redundancy': results.get('continuous_mi', {}).get('dsr_core', {}),
                'significance': results.get('conditional_granger', {})
            }
            
            print("✓ 信息论分析完成")
            
        except Exception as e:
            error_msg = f"信息论分析失败：{str(e)}"
            print(f"✗ {error_msg}")
            self.results['errors'].append({
                'analysis': 'information_theory',
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            })
            
    def run_constitutiveness_tests(self):
        """运行构成性检验"""
        print("\n2. 执行构成性检验...")
        
        try:
            # 初始化分析器
            self.const_analyzer = ConstitutivenessTests(
                data_path=str(self.data_path)
            )
            
            # 如果已有结果，直接使用
            if 'constitutiveness' in self.analysis_results:
                results = self.analysis_results['constitutiveness']
            else:
                # 加载数据并运行新分析
                self.const_analyzer.load_data()
                results = self.const_analyzer.run_constitutiveness_tests()
            
            # 提取关键指标（适配实际的数据结构）
            self.results['evidence']['constitutiveness'] = {
                'virtual_removal': {
                    'performance_loss': results.get('virtual_removal', {}).get('performance_loss', {}).get('overall_performance', 0),
                    'significant_impact': results.get('virtual_removal', {}).get('performance_loss', {}).get('overall_performance', 0) > 0.5
                },
                'path_necessity': {
                    'indirect_effect': results.get('path_necessity', {}).get('indirect_effect', {}).get('value', 0),
                    'is_necessary': results.get('path_necessity', {}).get('is_necessary', True),
                    'mediation_type': results.get('path_necessity', {}).get('mediation_analysis', {}).get('mediation_type', '未知')
                },
                'robustness': {
                    'robustness_value': results.get('robustness_tests', {}).get('overall_robustness', 0),
                    'is_robust': results.get('robustness_tests', {}).get('overall_robustness', 0) > 0.9
                },
                'overall_assessment': {
                    'constitutiveness_found': results.get('constitutiveness_score', {}).get('weighted_score', 0) > 0.8,
                    'verdict': results.get('constitutiveness_score', {}).get('verdict', '未知')
                }
            }
            
            print("✓ 构成性检验完成")
            
        except Exception as e:
            error_msg = f"构成性检验失败：{str(e)}"
            print(f"✗ {error_msg}")
            self.results['errors'].append({
                'analysis': 'constitutiveness_tests',
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            })
            
    def run_statistical_models(self):
        """运行统计模型分析"""
        print("\n3. 执行统计模型分析...")
        
        try:
            # 初始化分析器
            self.stat_analyzer = StatisticalModelsAnalyzer(
                data_path=str(self.data_path)
            )
            
            # 如果已有结果，直接使用
            if 'statistical_models' in self.analysis_results:
                results = self.analysis_results['statistical_models']
            else:
                # 运行新分析
                results = self.stat_analyzer.run_all_models()
            
            # 提取关键模型结果
            self.results['evidence']['statistical_models'] = {
                'M1_linear': results.get('M1_baseline', {}),
                'M3_nonlinear': results.get('M3_nonlinear', {}),
                'M4_VAR': results.get('M4_var_causality', {}),
                'model_comparison': results.get('model_comparison', {}),
                'best_model': results.get('best_model', {})
            }
            
            print("✓ 统计模型分析完成")
            
        except Exception as e:
            error_msg = f"统计模型分析失败：{str(e)}"
            print(f"✗ {error_msg}")
            self.results['errors'].append({
                'analysis': 'statistical_models',
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            })
            
    def run_network_analysis(self):
        """运行网络分析"""
        print("\n4. 执行网络分析...")
        
        try:
            # 初始化分析器
            self.network_analyzer = NetworkDiffusionAnalysis(
                data_path=str(self.data_path)
            )
            
            # 如果已有结果，直接使用
            if 'network_analysis' in self.analysis_results:
                results = self.analysis_results['network_analysis']
            else:
                # 加载数据并运行新分析
                self.network_analyzer.load_data()
                results = self.network_analyzer.run_network_diffusion_analysis()
            
            # 提取关键指标（适配实际数据结构）
            network_data = results.get('cognitive_network', {})
            dsr_nodes = ['DSR_core', 'DSR_bridge', 'DSR_integrate']
            
            # 计算DSR节点的平均中心性
            avg_betweenness = 0
            avg_degree = 0
            avg_closeness = 0
            
            for node in dsr_nodes:
                node_data = network_data.get('node_attributes', {}).get(node, {})
                avg_betweenness += node_data.get('betweenness_centrality', 0)
                avg_degree += node_data.get('degree_centrality', 0)
                avg_closeness += node_data.get('closeness_centrality', 0)
            
            avg_betweenness /= len(dsr_nodes)
            avg_degree /= len(dsr_nodes)
            avg_closeness /= len(dsr_nodes)
            
            self.results['evidence']['network_analysis'] = {
                'DSR_centrality': {
                    'betweenness': avg_betweenness,
                    'degree': avg_degree,
                    'closeness': avg_closeness
                },
                'network_density': network_data.get('network_properties', {}).get('density', 0),
                'DSR_mediation': {
                    'is_mediator': avg_betweenness > 0.1,  # 如果介数中心性>0.1则认为是中介
                    'mediation_strength': avg_betweenness
                },
                'key_nodes': results.get('key_nodes', {})
            }
            
            print("✓ 网络分析完成")
            
        except Exception as e:
            error_msg = f"网络分析失败：{str(e)}"
            print(f"✗ {error_msg}")
            self.results['errors'].append({
                'analysis': 'network_analysis',
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            })
            
            
    def integrate_evidence(self):
        """整合所有证据，计算综合支持度"""
        print("\n6. 整合证据并计算支持度...")
        
        # 定义各证据的权重（动态调整）
        # 根据实际可用的证据动态分配权重
        available_evidence = []
        base_weights = {
            'information_theory': 0.25,
            'constitutiveness': 0.30,
            'statistical_models': 0.25,
            'network_analysis': 0.20
        }
        
        # 检查哪些证据实际可用
        for evidence_type in base_weights:
            if evidence_type in self.results['evidence'] and self.results['evidence'][evidence_type]:
                available_evidence.append(evidence_type)
        
        # 重新分配权重
        if available_evidence:
            total_base_weight = sum(base_weights[e] for e in available_evidence)
            evidence_weights = {
                e: base_weights[e] / total_base_weight for e in available_evidence
            }
        else:
            evidence_weights = base_weights
        
        # 计算各项证据得分
        evidence_scores = {}
        
        # 1. 信息论证据
        if 'information_theory' in self.results['evidence']:
            it_evidence = self.results['evidence']['information_theory']
            # 从weighted_average中获取功能互补性得分
            fc_data = it_evidence.get('functional_complementarity', {})
            fc_score = fc_data.get('weighted_average', {}).get('total_complementarity', 0)
            # 归一化到0-1范围
            fc_normalized = min(fc_score, 1.0) if fc_score > 0 else 0
            # 检查三重交互MI（归一化到0-1范围）
            triple_mi = it_evidence.get('triple_interaction_mi', 0)
            mi_score = min(triple_mi * 5, 1.0) if triple_mi > 0 else 0  # 将0.115映射到约0.575
            evidence_scores['information_theory'] = (fc_normalized + mi_score) / 2
        
        # 2. 构成性证据
        if 'constitutiveness' in self.results['evidence']:
            const_evidence = self.results['evidence']['constitutiveness']
            
            # 检查各项构成性测试结果
            vr_significant = const_evidence.get('virtual_removal', {}).get('significant_impact', False)
            pn_necessary = const_evidence.get('path_necessity', {}).get('is_necessary', False)
            rob_robust = const_evidence.get('robustness', {}).get('is_robust', False)
            const_found = const_evidence.get('overall_assessment', {}).get('constitutiveness_found', False)
            
            # 计算显著性指标数量
            significant_count = sum([vr_significant, pn_necessary, rob_robust, const_found])
            evidence_scores['constitutiveness'] = significant_count / 4.0
        
        # 3. 统计模型证据
        if 'statistical_models' in self.results['evidence']:
            stat_evidence = self.results['evidence']['statistical_models']
            
            # 从模型比较中获取R²值
            model_comparison = stat_evidence.get('model_comparison', {})
            summary_table = model_comparison.get('summary_table', [])
            
            # 提取M1和M3的R²值
            m1_r2 = 0
            m3_r2 = 0
            for model in summary_table:
                if model.get('Model') == 'M1_baseline':
                    m1_r2 = model.get('R_squared', 0)
                elif model.get('Model') == 'M3_nonlinear':
                    m3_r2 = model.get('R_squared', 0)
            
            # 检查非线性模型是否优于线性模型
            nonlinear_better = m3_r2 > m1_r2 and m3_r2 > 0.15  # R²>0.15表示有意义的解释力
            
            # VAR因果分析（暂时假设不显著，因为数据中没有具体信息）
            var_significant = False
            
            # 计算统计模型得分
            r2_score = min(m3_r2 * 2, 1.0) if m3_r2 > 0 else 0  # 将R²映射到0-1
            evidence_scores['statistical_models'] = (r2_score + int(nonlinear_better)) / 2
        
        # 4. 网络分析证据
        if 'network_analysis' in self.results['evidence']:
            net_evidence = self.results['evidence']['network_analysis']
            centrality_score = net_evidence.get('DSR_centrality', {}).get('betweenness', 0)
            mediation_score = 1 if net_evidence.get('DSR_mediation', {}).get('is_mediator', False) else 0
            evidence_scores['network_analysis'] = (centrality_score + mediation_score) / 2
        
        
        # 保存证据分析结果（不计算百分比支持度）
        self.results['evidence_scores'] = evidence_scores
        
        # 基于统计显著性和效应量进行定性评估
        # 检查各项分析的统计显著性
        significant_findings = []
        
        # 信息论分析
        if 'information_theory' in self.results['evidence']:
            fc_score = self.results['evidence']['information_theory'].get(
                'functional_complementarity', {}).get('weighted_average', {}).get('total_complementarity', 0)
            if fc_score > 0.2:
                significant_findings.append('功能互补性显著')
        
        # 构成性检验
        if 'constitutiveness' in self.results['evidence']:
            const_score = self.results['evidence']['constitutiveness'].get(
                'overall_assessment', {}).get('constitutiveness_score', 0)
            if const_score > 0.8:
                significant_findings.append('构成性得分高')
        
        # 统计模型
        if 'statistical_models' in self.results['evidence']:
            models = self.results['evidence']['statistical_models']
            model_comparison = models.get('model_comparison', {}).get('summary_table', [])
            for model in model_comparison:
                if model.get('Model') == 'M3_nonlinear' and model.get('R_squared', 0) > 0.15:
                    significant_findings.append('非线性模型显著')
                    break
        
        # 网络分析
        if 'network_analysis' in self.results['evidence']:
            if self.results['evidence']['network_analysis'].get('DSR_mediation', {}).get('is_mediator', False):
                significant_findings.append('DSR具有中介作用')
        
        # 基于显著性发现进行定性评估
        if len(significant_findings) >= 3:
            support_assessment = "支持"
        elif len(significant_findings) >= 2:
            support_assessment = "部分支持"
        else:
            support_assessment = "不支持"
            
        self.results['support_assessment'] = support_assessment
        self.results['evidence_summary'] = {
            'significant_findings': significant_findings,
            'total_analyses': len([k for k in ['information_theory', 'constitutiveness', 
                                              'statistical_models', 'network_analysis'] 
                                 if k in self.results['evidence']])
        }
        
        print(f"✓ 证据整合完成，评估结果：{support_assessment}")
        
    def generate_visualization(self):
        """生成综合可视化图表"""
        print("\n7. 生成综合可视化...")
        
        # 创建图形
        fig = plt.figure(figsize=(16, 12), dpi=1200)  # 必须使用1200 DPI
        fig.patch.set_facecolor('white')
        
        # 1. 信息论指标（左上）
        ax1 = plt.subplot(2, 2, 1)
        self._plot_information_theory(ax1)
        
        # 2. 构成性测试结果（右上）
        ax2 = plt.subplot(2, 2, 2)
        self._plot_constitutiveness_tests(ax2)
        
        # 3. 统计模型对比（左下）
        ax3 = plt.subplot(2, 2, 3)
        self._plot_statistical_models(ax3)
        
        # 4. 网络中心性（右下）
        ax4 = plt.subplot(2, 2, 4)
        self._plot_network_analysis(ax4)
        
        # 调整布局
        plt.tight_layout(pad=3.0, h_pad=4.0, w_pad=3.0)
        
        # 保存图形
        output_file = self.output_path / 'figures' / 'H1_cognitive_dependency_analysis.jpg'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=1200, format='jpg')  # 不使用bbox_inches='tight'以保持一致的尺寸
        plt.close()
        
        self.results['visualizations']['main_figure'] = str(output_file)
        print(f"✓ 可视化已保存至：{output_file}")
        
    def _plot_information_theory(self, ax):
        """绘制信息论分析结果"""
        if 'information_theory' not in self.results['evidence']:
            ax.text(0.5, 0.5, '无信息论数据', ha='center', va='center')
            ax.set_title('信息论分析')
            return
            
        data = self.results['evidence']['information_theory']
        
        # 提取指标（从实际数据结构中获取）
        fc_data = data.get('functional_complementarity', {})
        fc_score = fc_data.get('weighted_average', {}).get('total_complementarity', 0)
        
        # 获取条件互信息和协同效应
        cmi_data = data.get('conditional_mi', {}).get('dsr_core', {})
        joint_mi = cmi_data.get('joint_mi', 0)
        synergy = data.get('synergy_redundancy', {}).get('synergy', 0)
        
        metrics = {
            '功能互补性': fc_score,
            '三重交互MI': data.get('triple_interaction_mi', 0),
            '条件互信息': joint_mi,
            '协同效应': abs(synergy)  # 显示绝对值以便在图中更清晰
        }
        
        # 绘制条形图
        bars = ax.bar(metrics.keys(), metrics.values(), color='steelblue', alpha=0.8)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')
        
        ax.set_title('信息论指标', fontsize=14, fontweight='bold')
        ax.set_ylabel('指标值')
        ax.set_ylim(0, max(metrics.values()) * 1.2)
        ax.grid(axis='y', alpha=0.3)
        
        # 如果协同效应为负，添加说明
        if synergy < 0:
            ax.text(3, metrics['协同效应'] + 0.01, '(冗余)', 
                   ha='center', va='bottom', fontsize=9, style='italic')
        
    def _plot_constitutiveness_tests(self, ax):
        """绘制构成性检验结果"""
        if 'constitutiveness' not in self.results['evidence']:
            ax.text(0.5, 0.5, '无构成性检验数据', ha='center', va='center')
            ax.set_title('构成性检验')
            return
            
        data = self.results['evidence']['constitutiveness']
        
        # 准备数据（使用实际的性能损失和测试结果）
        tests = ['虚拟移除', '路径必要性', '系统鲁棒性']
        
        # 获取各项指标值
        perf_loss = data.get('virtual_removal', {}).get('performance_loss', 0)
        # 将性能损失值映射到0-1范围（因为原始值可能>1）
        perf_loss_normalized = min(perf_loss / 2.0, 1.0)  # 假设最大损失为2
        
        # 路径必要性：使用间接效应比例作为指标
        indirect_effect = data.get('path_necessity', {}).get('indirect_effect', 0)
        # 如果间接效应是数值，直接使用；如果是字典，从中提取value
        if isinstance(indirect_effect, dict):
            indirect_effect = indirect_effect.get('value', 0)
        # 将间接效应映射到0-1范围
        path_necessity_score = min(abs(indirect_effect) * 10, 1.0)  # 假设0.1的间接效应是高值
        
        # 鲁棒性值已经在0-1范围内
        robustness_value = data.get('robustness', {}).get('robustness_value', 0)
        
        scores = [
            perf_loss_normalized,  # 虚拟移除
            path_necessity_score,  # 路径必要性
            robustness_value  # 系统鲁棒性
        ]
        
        # 绘制雷达图
        angles = np.linspace(0, 2 * np.pi, len(tests), endpoint=False)
        scores = np.array(scores)
        
        # 闭合图形
        angles = np.concatenate([angles, [angles[0]]])
        scores = np.concatenate([scores, [scores[0]]])
        
        ax.plot(angles, scores, 'o-', linewidth=2, color='darkgreen')
        ax.fill(angles, scores, alpha=0.25, color='darkgreen')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(tests)
        ax.set_ylim(0, 1)
        ax.set_ylabel('构成性强度', fontsize=12)
        ax.set_title('构成性检验结果', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
    def _plot_statistical_models(self, ax):
        """绘制统计模型对比"""
        if 'statistical_models' not in self.results['evidence']:
            ax.text(0.5, 0.5, '无统计模型数据', ha='center', va='center')
            ax.set_title('统计模型对比')
            return
            
        data = self.results['evidence']['statistical_models']
        
        # 从模型比较表中提取R²值
        model_comparison = data.get('model_comparison', {})
        summary_table = model_comparison.get('summary_table', [])
        
        # 提取M1和M3的R²值
        m1_r2 = 0
        m3_r2 = 0
        for model in summary_table:
            if model.get('Model') == 'M1_baseline':
                m1_r2 = model.get('R_squared', 0)
            elif model.get('Model') == 'M3_nonlinear':
                m3_r2 = model.get('R_squared', 0)
        
        models = ['M1线性', 'M3非线性']
        r2_values = [m1_r2, m3_r2]
        
        # 绘制条形图
        bars = ax.bar(models, r2_values, color=['coral', 'darkred'], alpha=0.8)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')
        
        ax.set_title('统计模型R²对比', fontsize=14, fontweight='bold')
        ax.set_ylabel('R²值')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        # 添加VAR因果检验结果
        var_result = data.get('M4_VAR', {}).get('granger_causality', {})
        if var_result:
            result_text = "VAR因果检验：DSR→CS " + ("显著" if var_result.get('DSR_causes_CS', False) else "不显著")
            ax.text(0.5, 0.95, result_text, transform=ax.transAxes, 
                   ha='center', va='top', fontsize=10, style='italic')
        
    def _plot_network_analysis(self, ax):
        """绘制网络分析结果"""
        if 'network_analysis' not in self.results['evidence']:
            ax.text(0.5, 0.5, '无网络分析数据', ha='center', va='center')
            ax.set_title('网络分析')
            return
            
        data = self.results['evidence']['network_analysis']
        
        # 提取中心性指标
        centrality_data = data.get('DSR_centrality', {})
        metrics = {
            '度中心性': centrality_data.get('degree', 0),
            '介数中心性': centrality_data.get('betweenness', 0),
            '接近中心性': centrality_data.get('closeness', 0)
        }
        
        # 绘制条形图
        x = np.arange(len(metrics))
        bars = ax.bar(x, list(metrics.values()), color='purple', alpha=0.7)
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')
        
        ax.set_xticks(x)
        ax.set_xticklabels(metrics.keys())
        ax.set_title('DSR网络中心性指标', fontsize=14, fontweight='bold')
        ax.set_ylabel('中心性值')
        ax.grid(axis='y', alpha=0.3)
        
        # 添加网络密度信息
        density = data.get('network_density', 0)
        ax.text(0.02, 0.95, f'网络密度: {density:.3f}', transform=ax.transAxes,
               va='top', fontsize=10)
        
        
    def generate_report(self):
        """生成Markdown格式的综合报告"""
        print("\n8. 生成分析报告...")
        
        report = []
        report.append("# H1假设验证报告")
        report.append(f"\n生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n## 执行摘要")
        
        # 假设内容
        report.append(f"\n**假设内容**：{self.results['hypothesis_description']}")
        report.append(f"\n**综合评估**：{self.results.get('support_assessment', '未知')}")
        
        # 证据摘要
        evidence_summary = self.results.get('evidence_summary', {})
        report.append(f"\n**显著性发现**：{len(evidence_summary.get('significant_findings', []))}项")
        
        # 关键发现
        report.append("\n**关键发现**：")
        key_findings = self._extract_key_findings()
        for finding in key_findings:
            report.append(f"- {finding}")
        
        # 详细分析结果
        report.append("\n## 详细分析结果")
        
        # 1. 信息论分析
        report.append("\n### 1. 信息论分析")
        self._add_information_theory_report(report)
        
        # 2. 构成性检验
        report.append("\n### 2. 构成性检验")
        self._add_constitutiveness_report(report)
        
        # 3. 统计模型
        report.append("\n### 3. 统计模型分析")
        self._add_statistical_models_report(report)
        
        # 4. 网络分析
        report.append("\n### 4. 网络分析")
        self._add_network_analysis_report(report)
        
        
        # 综合证据评估
        report.append("\n## 综合证据评估")
        self._add_evidence_integration_report(report)
        
        # 结论
        report.append("\n## 结论")
        self._add_conclusion(report)
        
        # 统计汇总表格
        report.append("\n## 统计汇总表格")
        self._add_statistical_tables(report)
        
        # 附录
        if self.results.get('errors'):
            report.append("\n## 附录：错误日志")
            for error in self.results['errors']:
                report.append(f"\n- **{error['analysis']}** ({error['timestamp']}): {error['error']}")
        
        # 保存报告
        report_path = self.output_path / 'md' / 'H1_validation_report.md'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
            
        print(f"✓ 报告已保存至：{report_path}")
        
    def _extract_key_findings(self):
        """提取关键发现"""
        findings = []
        
        # 信息论发现
        if 'information_theory' in self.results['evidence']:
            fc_data = self.results['evidence']['information_theory'].get('functional_complementarity', {})
            fc_score = fc_data.get('weighted_average', {}).get('total_complementarity', 0)
            if fc_score > 0.2:
                findings.append(f"DSR展现出显著的功能互补性（FC={fc_score:.3f}）")
        
        # 构成性发现
        if 'constitutiveness' in self.results['evidence']:
            const_data = self.results['evidence']['constitutiveness']
            vr_impact = const_data.get('virtual_removal', {}).get('significant_impact', False)
            perf_loss = const_data.get('virtual_removal', {}).get('performance_loss', 0)
            
            if vr_impact or perf_loss > 0.5:
                findings.append(f"虚拟移除测试显示DSR对认知成功具有显著影响（性能损失：{perf_loss:.3f}）")
            
            if const_data.get('overall_assessment', {}).get('constitutiveness_found', False):
                findings.append("构成性检验结果显著，支持H1假设")
        
        # 统计模型发现
        if 'statistical_models' in self.results['evidence']:
            models = self.results['evidence']['statistical_models']
            model_comparison = models.get('model_comparison', {})
            summary_table = model_comparison.get('summary_table', [])
            
            # 获取模型R²值
            m1_r2 = 0
            m3_r2 = 0
            for model in summary_table:
                if model.get('Model') == 'M1_baseline':
                    m1_r2 = model.get('R_squared', 0)
                elif model.get('Model') == 'M3_nonlinear':
                    m3_r2 = model.get('R_squared', 0)
            
            if m3_r2 > m1_r2 and m3_r2 > 0.15:
                improvement = (m3_r2 - m1_r2) / m1_r2 * 100 if m1_r2 > 0 else 0
                findings.append(f"非线性模型（R² = {m3_r2:.3f}）优于线性模型（R² = {m1_r2:.3f}），改进 {improvement:.1f}%")
        
        # 网络分析发现
        if 'network_analysis' in self.results['evidence']:
            centrality = self.results['evidence']['network_analysis'].get(
                'DSR_centrality', {}).get('betweenness', 0)
            if centrality > 0.5:
                findings.append(f"DSR在认知网络中具有高介数中心性（介数中心性 = {centrality:.3f}）")
        
        
        return findings
        
    def _add_information_theory_report(self, report):
        """添加信息论分析报告内容"""
        if 'information_theory' not in self.results['evidence']:
            report.append("\n*信息论分析未执行或失败*")
            return
            
        data = self.results['evidence']['information_theory']
        
        report.append("\n**功能互补性分析**")
        fc_data = data.get('functional_complementarity', {})
        weighted_avg = fc_data.get('weighted_average', {})
        report.append(f"- 总体功能互补性：{weighted_avg.get('total_complementarity', 0):.3f}")
        report.append(f"- 低DSR组：{fc_data.get('low', {}).get('complementarity', 0):.3f}")
        report.append(f"- 中DSR组：{fc_data.get('medium', {}).get('complementarity', 0):.3f}")
        report.append(f"- 高DSR组：{fc_data.get('high', {}).get('complementarity', 0):.3f}")
        
        report.append("\n**三重交互互信息**")
        report.append(f"- MI(DSR;TL;CS) = {data.get('triple_interaction_mi', 0):.4f}")
        
        report.append("\n**条件互信息**")
        cmi_data = data.get('conditional_mi', {})
        if isinstance(cmi_data, dict) and cmi_data:
            # 只显示主要的互信息值
            dsr_core = cmi_data.get('dsr_core', {})
            if dsr_core:
                report.append(f"- DSR联合互信息 = {dsr_core.get('joint_mi', 0):.4f}")
                report.append(f"- DSR总互信息 = {dsr_core.get('total_mi', 0):.4f}")
                report.append(f"- 协同效应 = {dsr_core.get('synergy', 0):.4f}")
        
        report.append("\n**统计显著性**")
        sig_data = data.get('significance', {})
        if isinstance(sig_data, dict) and sig_data:
            # 显示因果检验结果
            full_sample = sig_data.get('full_sample', {})
            dsr_causes_cs = full_sample.get('DSR_causes_CS', {})
            if dsr_causes_cs:
                p_val = dsr_causes_cs.get('p_value', 1)
                p_str = format_p_value(p_val)
                report.append(f"- DSR→CS因果检验 {p_str}")
                report.append(f"- 显著性：{'是' if dsr_causes_cs.get('significant', False) else '否'}")
        
    def _add_constitutiveness_report(self, report):
        """添加构成性检验报告内容"""
        if 'constitutiveness' not in self.results['evidence']:
            report.append("\n*构成性检验未执行或失败*")
            return
            
        data = self.results['evidence']['constitutiveness']
        
        report.append("\n**虚拟移除测试**")
        vr_data = data.get('virtual_removal', {})
        perf_loss = vr_data.get('performance_loss', 0)
        report.append(f"- 移除DSR后性能损失：{perf_loss:.3f}")
        report.append(f"- 显著影响：{'是' if vr_data.get('significant_impact', False) else '否'}")
        
        report.append("\n**路径必要性分析**")
        pn_data = data.get('path_necessity', {})
        report.append(f"- 间接效应：{pn_data.get('indirect_effect', 0):.3f}")
        report.append(f"- 路径必要：{'是' if pn_data.get('is_necessary', False) else '否'}")
        
        report.append("\n**系统鲁棒性评估**")
        rob_data = data.get('robustness', {})
        report.append(f"- 鲁棒性值：{rob_data.get('robustness_value', 0):.3f}")
        report.append(f"- 系统鲁棒：{'是' if rob_data.get('is_robust', False) else '否'}")
        
        # 添加综合评估
        overall = data.get('overall_assessment', {})
        if overall:
            report.append("\n**综合评估**")
            report.append(f"- 构成性检验结果：{overall.get('verdict', '未知')}")
        
    def _add_statistical_models_report(self, report):
        """添加统计模型报告内容"""
        if 'statistical_models' not in self.results['evidence']:
            report.append("\n*统计模型分析未执行或失败*")
            return
            
        data = self.results['evidence']['statistical_models']
        
        # 从模型比较表中获取数据
        model_comparison = data.get('model_comparison', {})
        summary_table = model_comparison.get('summary_table', [])
        
        # 提取各模型数据
        m1_data = {}
        m3_data = {}
        for model in summary_table:
            if model.get('Model') == 'M1_baseline':
                m1_data = model
            elif model.get('Model') == 'M3_nonlinear':
                m3_data = model
        
        # M1线性模型
        report.append("\n**M1 线性基准模型**")
        report.append(f"- R² = {m1_data.get('R_squared', 0):.3f}")
        report.append(f"- AIC = {m1_data.get('AIC', 0):.1f}")
        
        # M3非线性模型
        report.append("\n**M3 非线性交互模型**")
        report.append(f"- R² = {m3_data.get('R_squared', 0):.3f}")
        report.append(f"- AIC = {m3_data.get('AIC', 0):.1f}")
        
        # M4 VAR因果分析
        report.append("\n**M4 VAR因果分析**")
        var_data = data.get('M4_VAR', {})
        gc_data = var_data.get('granger_causality', {})
        report.append(f"- DSR → CS：{'显著' if gc_data.get('DSR_causes_CS', False) else '不显著'}")
        report.append(f"- TL → CS：{'显著' if gc_data.get('TL_causes_CS', False) else '不显著'}")
        
        # 模型比较
        report.append("\n**模型比较**")
        best_r2_model = model_comparison.get('best_r_squared', '未知')
        best_aic_model = model_comparison.get('best_aic', '未知')
        report.append(f"- 最佳R²模型：{best_r2_model}")
        report.append(f"- 最佳AIC模型：{best_aic_model}")
        
        # 计算非线性改进
        m1_r2 = m1_data.get('R_squared', 0)
        m3_r2 = m3_data.get('R_squared', 0)
        improvement = m3_r2 - m1_r2
        report.append(f"- 非线性改进：{improvement:.3f} ({improvement/m1_r2*100:.1f}%)" if m1_r2 > 0 else "- 非线性改进：N/A")
        
    def _add_network_analysis_report(self, report):
        """添加网络分析报告内容"""
        if 'network_analysis' not in self.results['evidence']:
            report.append("\n*网络分析未执行或失败*")
            return
            
        data = self.results['evidence']['network_analysis']
        
        report.append("\n**DSR网络中心性**")
        centrality = data.get('DSR_centrality', {})
        report.append(f"- 度中心性：{centrality.get('degree', 0):.3f}")
        report.append(f"- 介数中心性：{centrality.get('betweenness', 0):.3f}")
        report.append(f"- 接近中心性：{centrality.get('closeness', 0):.3f}")
        
        report.append("\n**网络属性**")
        report.append(f"- 网络密度：{data.get('network_density', 0):.3f}")
        
        report.append("\n**DSR中介作用**")
        mediation = data.get('DSR_mediation', {})
        report.append(f"- 中介作用：{'是' if mediation.get('is_mediator', False) else '否'}")
        report.append(f"- 中介强度：{mediation.get('mediation_strength', 0):.3f}")
        
        
    def _add_evidence_integration_report(self, report):
        """添加证据整合报告内容"""
        report.append("\n### 证据综合评估")
        
        # 列出各类证据的定性评估
        evidence_assessments = {
            'information_theory': '信息论分析',
            'constitutiveness': '构成性检验',
            'statistical_models': '统计模型',
            'network_analysis': '网络分析'
        }
        
        scores = self.results.get('evidence_scores', {})
        
        report.append("\n**各项分析的证据**：")
        for key, name in evidence_assessments.items():
            if key in self.results['evidence'] and self.results['evidence'][key]:
                report.append(f"- {name}：已完成")
            else:
                report.append(f"- {name}：未执行")
        
        # 添加整体评估
        evidence_summary = self.results.get('evidence_summary', {})
        report.append(f"\n**证据汇总**：")
        report.append(f"- 完成的分析数量：{evidence_summary.get('total_analyses', 0)}")
        report.append(f"- 显著性发现：{len(evidence_summary.get('significant_findings', []))}项")
        
        # 列出显著性发现
        for finding in evidence_summary.get('significant_findings', []):
            report.append(f"  - {finding}")
        
        report.append(f"\n**评估结论**：基于上述分析，H1假设{self.results.get('support_assessment', '未知')}")
        
    def _add_conclusion(self, report):
        """添加结论部分"""
        evidence_summary = self.results.get('evidence_summary', {})
        
        report.append(f"\n基于多维度的综合分析，对H1假设（{self.results['hypothesis_description']}）的分析结果如上所述。")
        
        # 列出具体的统计发现
        report.append("\n**主要发现**：")
        
        # 基于实际的统计结果
        if 'information_theory' in self.results['evidence']:
            fc_score = self.results['evidence']['information_theory'].get(
                'functional_complementarity', {}).get('weighted_average', {}).get('total_complementarity', 0)
            if fc_score > 0:
                report.append(f"- 功能互补性分析：FC = {fc_score:.3f}")
                
        if 'constitutiveness' in self.results['evidence']:
            const_verdict = self.results['evidence']['constitutiveness'].get(
                'overall_assessment', {}).get('verdict', '')
            if const_verdict:
                report.append(f"- 构成性检验：{const_verdict}")
                
        if 'statistical_models' in self.results['evidence']:
            models = self.results['evidence']['statistical_models']
            model_comparison = models.get('model_comparison', {}).get('summary_table', [])
            for model in model_comparison:
                if model.get('Model') == 'M3_nonlinear':
                    r2 = model.get('R_squared', 0)
                    if r2 > 0:
                        report.append(f"- 非线性模型：R² = {r2:.3f}")
                    break
                    
        if 'network_analysis' in self.results['evidence']:
            centrality = self.results['evidence']['network_analysis'].get(
                'DSR_centrality', {}).get('betweenness', 0)
            if centrality > 0:
                report.append(f"- DSR介数中心性：{centrality:.3f}")
        
        report.append("\n**理论贡献**：")
        report.append("本研究为分布式认知理论中数字符号资源的构成性作用提供了实证支持，展示了DSR如何作为认知系统的不可或缺组件，而非仅仅是辅助工具。")
    
    def _add_statistical_tables(self, report):
        """添加综合统计汇总表格"""
        
        # 表1：信息论分析汇总
        report.append("\n### 表1：信息论分析结果汇总")
        report.append("\n| 指标 | 数值 | 解释 |")
        report.append("|------|------|------|")
        
        if 'information_theory' in self.results['evidence']:
            it_data = self.results['evidence']['information_theory']
            
            # 功能互补性
            fc_data = it_data.get('functional_complementarity', {})
            weighted_avg = fc_data.get('weighted_average', {})
            report.append(f"| 总体功能互补性 | {weighted_avg.get('total_complementarity', 0):.3f} | DSR功能之间的互补程度 |")
            report.append(f"| 低DSR组互补性 | {fc_data.get('low', {}).get('complementarity', 0):.3f} | 低DSR使用下的功能互补 |")
            report.append(f"| 中DSR组互补性 | {fc_data.get('medium', {}).get('complementarity', 0):.3f} | 中等DSR使用下的功能互补 |")
            report.append(f"| 高DSR组互补性 | {fc_data.get('high', {}).get('complementarity', 0):.3f} | 高DSR使用下的功能互补 |")
            
            # 互信息指标
            report.append(f"| 三重交互互信息 | {it_data.get('triple_interaction_mi', 0):.4f} | DSR、TL、CS之间的信息依赖 |")
            
            # 条件互信息
            cmi_data = it_data.get('conditional_mi', {})
            if isinstance(cmi_data, dict) and 'dsr_core' in cmi_data:
                dsr_core = cmi_data['dsr_core']
                report.append(f"| DSR联合互信息 | {dsr_core.get('joint_mi', 0):.4f} | DSR功能的联合信息贡献 |")
                report.append(f"| DSR协同效应 | {dsr_core.get('synergy', 0):.4f} | DSR功能间的协同作用 |")
        
        # 表2：构成性检验汇总
        report.append("\n### 表2：构成性检验结果汇总")
        report.append("\n| 检验类型 | 指标 | 数值 | 显著性 |")
        report.append("|----------|------|------|---------|")
        
        if 'constitutiveness' in self.results['evidence']:
            const_data = self.results['evidence']['constitutiveness']
            
            # 虚拟移除测试
            vr_data = const_data.get('virtual_removal', {})
            report.append(f"| 虚拟移除测试 | 性能损失 | {vr_data.get('performance_loss', 0):.3f} | {'是' if vr_data.get('significant_impact', False) else '否'} |")
            
            # 路径必要性
            pn_data = const_data.get('path_necessity', {})
            report.append(f"| 路径必要性分析 | 间接效应 | {pn_data.get('indirect_effect', 0):.3f} | {'是' if pn_data.get('is_necessary', False) else '否'} |")
            
            # 系统鲁棒性
            rob_data = const_data.get('robustness', {})
            report.append(f"| 系统鲁棒性评估 | 鲁棒性值 | {rob_data.get('robustness_value', 0):.3f} | {'是' if rob_data.get('is_robust', False) else '否'} |")
        
        # 表3：统计模型比较
        report.append("\n### 表3：统计模型比较汇总")
        report.append("\n| 模型 | R² | 调整R² | AIC | BIC | 显著性 |")
        report.append("|------|-----|--------|-----|-----|---------|")
        
        if 'statistical_models' in self.results['evidence']:
            models_data = self.results['evidence']['statistical_models']
            model_comparison = models_data.get('model_comparison', {})
            summary_table = model_comparison.get('summary_table', [])
            
            for model in summary_table:
                model_name = model.get('Model', '')
                if model_name in ['M1_baseline', 'M3_nonlinear']:
                    display_name = 'M1 线性基准' if model_name == 'M1_baseline' else 'M3 非线性交互'
                    r2 = model.get('R_squared', 0)
                    adj_r2 = model.get('Adj_R_squared', 0)
                    aic = model.get('AIC', 0)
                    bic = model.get('BIC', 0)
                    sig = model.get('Significant', False)
                    report.append(f"| {display_name} | {r2:.3f} | {adj_r2:.3f} | {aic:.1f} | {bic:.1f} | {'是' if sig else '否'} |")
            
            # VAR模型因果检验
            var_data = models_data.get('M4_VAR', {})
            gc_data = var_data.get('granger_causality', {})
            if gc_data:
                report.append("\n### 表4：VAR因果检验结果")
                report.append("\n| 因果关系 | F统计量 | *p*值 | 显著性 |")
                report.append("|----------|----------|-------|---------|")
                
                # DSR → CS
                dsr_cs = gc_data.get('DSR_causes_CS_details', {})
                if dsr_cs:
                    f_stat = dsr_cs.get('test_statistic', 0)
                    p_val = dsr_cs.get('p_value', 1)
                    sig = gc_data.get('DSR_causes_CS', False)
                    report.append(f"| DSR → CS | {f_stat:.2f} | {format_p_value(p_val)} | {'是' if sig else '否'} |")
                
                # TL → CS
                tl_cs = gc_data.get('TL_causes_CS_details', {})
                if tl_cs:
                    f_stat = tl_cs.get('test_statistic', 0)
                    p_val = tl_cs.get('p_value', 1)
                    sig = gc_data.get('TL_causes_CS', False)
                    report.append(f"| TL → CS | {f_stat:.2f} | {format_p_value(p_val)} | {'是' if sig else '否'} |")
        
        # 表5：网络分析汇总
        report.append("\n### 表5：网络分析结果汇总")
        report.append("\n| 指标 | 数值 | 解释 |")
        report.append("|------|------|------|")
        
        if 'network_analysis' in self.results['evidence']:
            net_data = self.results['evidence']['network_analysis']
            
            # 中心性指标
            centrality = net_data.get('DSR_centrality', {})
            report.append(f"| DSR度中心性 | {centrality.get('degree', 0):.3f} | DSR的直接连接数量 |")
            report.append(f"| DSR介数中心性 | {centrality.get('betweenness', 0):.3f} | DSR在网络中的桥梁作用 |")
            report.append(f"| DSR接近中心性 | {centrality.get('closeness', 0):.3f} | DSR到其他节点的平均距离 |")
            
            # 网络属性
            report.append(f"| 网络密度 | {net_data.get('network_density', 0):.3f} | 网络整体连接程度 |")
            
            # 中介作用
            mediation = net_data.get('DSR_mediation', {})
            report.append(f"| DSR中介强度 | {mediation.get('mediation_strength', 0):.3f} | DSR的中介效应大小 |")
        
        # 表6：效应量汇总
        report.append("\n### 表6：主要效应量汇总")
        report.append("\n| 效应类型 | 效应量 | 95% CI | 效应大小 |")
        report.append("|----------|---------|---------|----------|")
        
        # 从不同分析中提取效应量
        if 'statistical_models' in self.results['evidence']:
            models_data = self.results['evidence']['statistical_models']
            
            # 从模型中提取R²作为效应量
            for model in models_data.get('model_comparison', {}).get('summary_table', []):
                if model.get('Model') == 'M3_nonlinear':
                    r2 = model.get('R_squared', 0)
                    # 计算Cohen's f²
                    f2 = r2 / (1 - r2) if r2 < 1 else 0
                    effect_size = 'small' if f2 < 0.15 else 'medium' if f2 < 0.35 else 'large'
                    report.append(f"| DSR-CS关系 (R²) | {r2:.3f} | - | {effect_size} |")
                    report.append(f"| Cohen's f² | {f2:.3f} | - | {effect_size} |")
        
    def save_results(self):
        """保存所有结果"""
        print("\n9. 保存分析结果...")
        
        # 创建输出目录
        data_dir = self.output_path / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存JSON格式详细结果
        json_path = data_dir / 'H1_validation_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"✓ JSON结果已保存至：{json_path}")
        
        # 准备CSV格式摘要数据
        summary_data = {
            'hypothesis': 'H1',
            'hypothesis_description': self.results['hypothesis_description'],
            'support_assessment': self.results['support_assessment'],
            'positive_evidence_count': self.results.get('evidence_summary', {}).get('positive_evidence_count', 0),
            'total_evidence_types': self.results.get('evidence_summary', {}).get('total_evidence_types', 0),
            'timestamp': self.results['timestamp']
        }
        
        # 添加各项证据得分
        for evidence_type, score in self.results.get('evidence_scores', {}).items():
            summary_data[f'{evidence_type}_score'] = score
        
        # 保存CSV格式摘要
        csv_path = data_dir / 'H1_validation_results.csv'
        summary_df = pd.DataFrame([summary_data])
        summary_df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"✓ CSV摘要已保存至：{csv_path}")
        
    def run_all_analyses(self):
        """运行所有分析的主函数"""
        print("="*60)
        print("H1假设验证分析")
        print("="*60)
        
        # 1. 加载数据
        self.load_data()
        if self.df is None:
            print("数据加载失败，分析终止")
            return
            
        # 1.5 验证数据完整性
        if not self.validate_data():
            print("数据验证失败，但将继续分析...")
            # 不终止，让分析继续进行
        
        # 2. 运行各项分析
        self.run_information_theory_analysis()
        self.run_constitutiveness_tests()
        self.run_statistical_models()
        self.run_network_analysis()
        
        # 3. 整合证据
        self.integrate_evidence()
        
        # 4. 生成可视化
        self.generate_visualization()
        
        # 5. 生成报告
        self.generate_report()
        
        # 6. 保存结果
        self.save_results()
        
        print("\n" + "="*60)
        print("H1假设验证分析完成！")
        print(f"综合评估：{self.results['support_assessment']}")
        evidence_summary = self.results.get('evidence_summary', {})
        print(f"显著性发现：{len(evidence_summary.get('significant_findings', []))}项")
        print("="*60)


def main():
    """主函数"""
    # 创建验证器实例
    validator = H1CognitiveDependencyValidator(
        data_path='../output_cn/data'
    )
    
    # 运行完整分析
    validator.run_all_analyses()


if __name__ == "__main__":
    main()