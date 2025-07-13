# step7_moderation_analysis.py
# 第七步：调节分析 - 分析语境和平台因素对DSR构成性的调节作用

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 统计分析
from scipy import stats
from scipy.stats import pearsonr, spearmanr, f_oneway
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from sklearn.preprocessing import StandardScaler

# 机器学习
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class ModerationAnalysis:
    """调节分析类"""
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.df = None
        self.results = {
            'context_moderation': {},
            'platform_moderation': {},
            'interaction_effects': {},
            'threshold_detection': {},
            'moderation_patterns': {}
        }
        
    def load_data(self):
        """加载数据"""
        # 优先加载最新的分析数据
        for filename in ['data_with_pattern_metrics.csv', 
                        'data_with_metrics.csv']:
            file_path = self.data_path / filename
            if file_path.exists():
                self.df = pd.read_csv(file_path, encoding='utf-8-sig')
                break
                
        if self.df is None:
            raise FileNotFoundError("未找到数据文件")
            
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # 处理敏感度编码
        # sensitivity_code: 1=低, 2=中, 3=高
        self.df['context_stratum'] = self.df['sensitivity_code'].map({
            1: 'low',
            2: 'medium', 
            3: 'high'
        })
        
        # 创建平台类型（基于日期或其他特征推断）
        # 这里简化处理，假设不同年份代表不同的平台演进阶段
        self.df['platform_type'] = pd.cut(self.df['date'].dt.year, 
                                         bins=[2020, 2022, 2024, 2026],
                                         labels=['early', 'mature', 'advanced'])
        
        print("="*60)
        print("第七步：调节分析")
        print("="*60)
        print(f"数据加载完成: {len(self.df)} 条记录")
        print(f"语境层级分布: {self.df['context_stratum'].value_counts().to_dict()}")
        print(f"平台类型分布: {self.df['platform_type'].value_counts().to_dict()}")
        
        return self.df
        
    def run_moderation_analysis(self):
        """运行所有调节分析"""
        
        print("\n1. 语境敏感度调节分析")
        context_moderation = self.analyze_context_moderation()
        
        print("\n2. 平台类型调节分析")
        platform_moderation = self.analyze_platform_moderation()
        
        print("\n3. 交互效应分解")
        interaction_effects = self.decompose_interaction_effects()
        
        print("\n4. 局部阈值检测")
        threshold_detection = self.detect_local_thresholds()
        
        print("\n5. 调节模式综合分析")
        moderation_patterns = self.analyze_moderation_patterns()
        
        # 生成可视化
        self.create_visualizations()
        
        # 保存结果
        self.save_results()
        
        return self.results
        
    def analyze_context_moderation(self):
        """分析语境敏感度的调节作用"""
        print("  执行语境敏感度调节分析...")
        
        results = {
            'main_effects': {},
            'interaction_effects': {},
            'simple_slopes': {},
            'moderation_strength': {}
        }
        
        # 1. 主效应分析
        for context in ['low', 'medium', 'high']:
            subset = self.df[self.df['context_stratum'] == context]
            if len(subset) > 10:
                # DSR-CS相关性
                corr = pearsonr(subset['dsr_cognitive'], subset['cs_output'])[0]
                # 平均水平
                mean_dsr = subset['dsr_cognitive'].mean()
                mean_cs = subset['cs_output'].mean()
                
                results['main_effects'][context] = {
                    'n_samples': len(subset),
                    'dsr_cs_correlation': corr,
                    'mean_dsr': mean_dsr,
                    'mean_cs': mean_cs
                }
                
        # 2. 交互效应模型
        # 准备数据
        model_data = self.df[['dsr_cognitive', 'cs_output', 'context_stratum']].dropna()
        
        # 创建虚拟变量
        model_data = pd.get_dummies(model_data, columns=['context_stratum'], prefix='context')
        
        # 创建交互项
        for context in ['medium', 'high']:
            if f'context_{context}' in model_data.columns:
                model_data[f'dsr_x_context_{context}'] = (
                    model_data['dsr_cognitive'] * model_data[f'context_{context}']
                )
        
        # 拟合交互模型
        formula = 'cs_output ~ dsr_cognitive + context_medium + context_high + dsr_x_context_medium + dsr_x_context_high'
        if all(col in model_data.columns for col in ['context_medium', 'context_high', 
                                                     'dsr_x_context_medium', 'dsr_x_context_high']):
            interaction_model = smf.ols(formula, data=model_data).fit()
            
            results['interaction_effects'] = {
                'model_summary': {
                    'r_squared': interaction_model.rsquared,
                    'adj_r_squared': interaction_model.rsquared_adj,
                    'f_statistic': interaction_model.fvalue,
                    'p_value': interaction_model.f_pvalue
                },
                'coefficients': {
                    'dsr_baseline': interaction_model.params.get('dsr_cognitive', 0),
                    'context_medium_effect': interaction_model.params.get('context_medium', 0),
                    'context_high_effect': interaction_model.params.get('context_high', 0),
                    'interaction_medium': interaction_model.params.get('dsr_x_context_medium', 0),
                    'interaction_high': interaction_model.params.get('dsr_x_context_high', 0)
                },
                'p_values': {
                    'interaction_medium': interaction_model.pvalues.get('dsr_x_context_medium', 1),
                    'interaction_high': interaction_model.pvalues.get('dsr_x_context_high', 1)
                }
            }
            
        # 3. 简单斜率分析
        for context in ['low', 'medium', 'high']:
            subset = self.df[self.df['context_stratum'] == context]
            if len(subset) > 30:
                X = sm.add_constant(subset['dsr_cognitive'])
                y = subset['cs_output']
                model = sm.OLS(y, X).fit()
                
                results['simple_slopes'][context] = {
                    'slope': model.params[1],
                    'se': model.bse[1],
                    'p_value': model.pvalues[1],
                    'ci_lower': model.conf_int()[1][0],
                    'ci_upper': model.conf_int()[1][1]
                }
                
        # 4. 调节强度评估
        if results['simple_slopes']:
            slopes = [v['slope'] for v in results['simple_slopes'].values()]
            results['moderation_strength'] = {
                'slope_range': max(slopes) - min(slopes),
                'slope_variance': np.var(slopes),
                'significant_differences': sum(1 for v in results['interaction_effects'].get('p_values', {}).values() if v < 0.05)
            }
            
        print(f"    - 低敏感度相关性: {results['main_effects'].get('low', {}).get('dsr_cs_correlation', 0):.3f}")
        print(f"    - 中敏感度相关性: {results['main_effects'].get('medium', {}).get('dsr_cs_correlation', 0):.3f}")
        print(f"    - 高敏感度相关性: {results['main_effects'].get('high', {}).get('dsr_cs_correlation', 0):.3f}")
        print(f"    - 调节效应强度: {results['moderation_strength'].get('slope_range', 0):.3f}")
        
        self.results['context_moderation'] = results
        return results
        
    def analyze_platform_moderation(self):
        """分析平台类型的调节作用"""
        print("  执行平台类型调节分析...")
        
        results = {
            'platform_effects': {},
            'evolution_patterns': {},
            'platform_x_context': {}
        }
        
        # 1. 不同平台的主效应
        for platform in ['early', 'mature', 'advanced']:
            subset = self.df[self.df['platform_type'] == platform]
            if len(subset) > 10:
                # 基本统计
                corr_dsr_cs = pearsonr(subset['dsr_cognitive'], subset['cs_output'])[0]
                mean_integration = subset['dsr_integration_depth'].mean() if 'dsr_integration_depth' in subset.columns else 0
                
                results['platform_effects'][platform] = {
                    'n_samples': len(subset),
                    'dsr_cs_correlation': corr_dsr_cs,
                    'mean_integration_depth': mean_integration,
                    'dsr_effectiveness': subset['dsr_cognitive'].std() / subset['cs_output'].std() if subset['cs_output'].std() > 0 else 0
                }
                
        # 2. 平台演化模式
        yearly_data = []
        for year in sorted(self.df['date'].dt.year.unique()):
            year_subset = self.df[self.df['date'].dt.year == year]
            if len(year_subset) > 10:
                yearly_data.append({
                    'year': int(year),
                    'dsr_cs_correlation': pearsonr(year_subset['dsr_cognitive'], year_subset['cs_output'])[0],
                    'mean_dsr': year_subset['dsr_cognitive'].mean(),
                    'mean_integration': year_subset['dsr_integration_depth'].mean() if 'dsr_integration_depth' in year_subset.columns else 0
                })
                
        results['evolution_patterns'] = yearly_data
        
        # 3. 平台×语境交互
        for platform in ['early', 'mature', 'advanced']:
            platform_results = {}
            for context in ['low', 'medium', 'high']:
                subset = self.df[(self.df['platform_type'] == platform) & 
                               (self.df['context_stratum'] == context)]
                if len(subset) > 10:
                    platform_results[context] = {
                        'n_samples': len(subset),
                        'dsr_cs_correlation': pearsonr(subset['dsr_cognitive'], subset['cs_output'])[0],
                        'mean_performance': subset['cs_output'].mean()
                    }
            
            if platform_results:
                results['platform_x_context'][platform] = platform_results
                
        print(f"    - 早期平台相关性: {results['platform_effects'].get('early', {}).get('dsr_cs_correlation', 0):.3f}")
        print(f"    - 成熟平台相关性: {results['platform_effects'].get('mature', {}).get('dsr_cs_correlation', 0):.3f}")
        print(f"    - 高级平台相关性: {results['platform_effects'].get('advanced', {}).get('dsr_cs_correlation', 0):.3f}")
        
        self.results['platform_moderation'] = results
        return results
        
    def decompose_interaction_effects(self):
        """分解交互效应"""
        print("  执行交互效应分解...")
        
        results = {
            'three_way_interaction': {},
            'decomposed_effects': {},
            'synergy_analysis': {}
        }
        
        # 1. 三向交互模型 (DSR × Context × Platform)
        model_data = self.df[['dsr_cognitive', 'cs_output', 'context_stratum', 'platform_type']].dropna()
        
        # 创建虚拟变量和交互项
        model_data = pd.get_dummies(model_data, columns=['context_stratum', 'platform_type'])
        
        # 简化：只考虑主要的三向交互
        if 'context_high' in model_data.columns and 'platform_advanced' in model_data.columns:
            model_data['three_way'] = (model_data['dsr_cognitive'] * 
                                      model_data['context_high'] * 
                                      model_data['platform_advanced'])
            
            # 拟合模型
            formula = 'cs_output ~ dsr_cognitive * context_high * platform_advanced'
            try:
                three_way_model = smf.ols(formula, data=model_data).fit()
                
                results['three_way_interaction'] = {
                    'model_fit': {
                        'r_squared': three_way_model.rsquared,
                        'aic': three_way_model.aic,
                        'f_statistic': three_way_model.fvalue
                    },
                    'three_way_coefficient': three_way_model.params.get('dsr_cognitive:context_high:platform_advanced', 0),
                    'three_way_pvalue': three_way_model.pvalues.get('dsr_cognitive:context_high:platform_advanced', 1),
                    'significant': three_way_model.pvalues.get('dsr_cognitive:context_high:platform_advanced', 1) < 0.05
                }
            except Exception as e:
                print(f"    警告：三向交互模型拟合失败: {str(e)}")
                
        # 2. 效应分解
        # 计算不同条件下的边际效应
        conditions = []
        for context in ['low', 'medium', 'high']:
            for platform in ['early', 'mature', 'advanced']:
                subset = self.df[(self.df['context_stratum'] == context) & 
                               (self.df['platform_type'] == platform)]
                if len(subset) > 20:
                    # 计算边际效应
                    X = sm.add_constant(subset['dsr_cognitive'])
                    y = subset['cs_output']
                    try:
                        model = sm.OLS(y, X).fit()
                        conditions.append({
                            'context': context,
                            'platform': platform,
                            'marginal_effect': model.params[1],
                            'se': model.bse[1],
                            'n': len(subset)
                        })
                    except:
                        pass
                        
        results['decomposed_effects'] = conditions
        
        # 3. 协同效应分析
        if conditions:
            effects_matrix = {}
            for cond in conditions:
                key = f"{cond['context']}_{cond['platform']}"
                effects_matrix[key] = cond['marginal_effect']
                
            # 计算协同指标
            effects_values = list(effects_matrix.values())
            results['synergy_analysis'] = {
                'max_effect': max(effects_values),
                'min_effect': min(effects_values),
                'effect_range': max(effects_values) - min(effects_values),
                'optimal_condition': max(effects_matrix.items(), key=lambda x: x[1])[0],
                'suboptimal_condition': min(effects_matrix.items(), key=lambda x: x[1])[0]
            }
            
        print(f"    - 三向交互显著性: {results['three_way_interaction'].get('significant', False)}")
        print(f"    - 最优条件: {results['synergy_analysis'].get('optimal_condition', 'N/A')}")
        print(f"    - 效应范围: {results['synergy_analysis'].get('effect_range', 0):.3f}")
        
        self.results['interaction_effects'] = results
        return results
        
    def detect_local_thresholds(self):
        """检测局部阈值效应"""
        print("  执行局部阈值检测...")
        
        results = {
            'threshold_points': {},
            'nonlinear_regions': {},
            'saturation_effects': {}
        }
        
        # 1. 检测不同语境下的阈值点
        for context in ['low', 'medium', 'high']:
            subset = self.df[self.df['context_stratum'] == context]
            if len(subset) > 50:
                # 使用分段回归检测阈值
                dsr_values = subset['dsr_cognitive'].values
                cs_values = subset['cs_output'].values
                
                # 排序
                sorted_idx = np.argsort(dsr_values)
                dsr_sorted = dsr_values[sorted_idx]
                cs_sorted = cs_values[sorted_idx]
                
                # 简单的阈值检测：找到斜率变化最大的点
                window = 20
                slopes = []
                thresholds = []
                
                for i in range(window, len(dsr_sorted) - window):
                    # 前段斜率
                    slope1 = np.polyfit(dsr_sorted[i-window:i], cs_sorted[i-window:i], 1)[0]
                    # 后段斜率
                    slope2 = np.polyfit(dsr_sorted[i:i+window], cs_sorted[i:i+window], 1)[0]
                    
                    slopes.append(abs(slope2 - slope1))
                    thresholds.append(dsr_sorted[i])
                    
                if slopes:
                    max_change_idx = np.argmax(slopes)
                    threshold_value = thresholds[max_change_idx]
                    
                    results['threshold_points'][context] = {
                        'threshold_value': threshold_value,
                        'slope_change': slopes[max_change_idx],
                        'before_threshold_n': sum(dsr_values < threshold_value),
                        'after_threshold_n': sum(dsr_values >= threshold_value)
                    }
                    
        # 2. 非线性区域识别
        # 使用二次项检验非线性
        for context in ['low', 'medium', 'high']:
            subset = self.df[self.df['context_stratum'] == context]
            if len(subset) > 30:
                # 线性模型
                X_linear = sm.add_constant(subset['dsr_cognitive'])
                model_linear = sm.OLS(subset['cs_output'], X_linear).fit()
                
                # 二次模型
                subset_copy = subset.copy()
                subset_copy['dsr_squared'] = subset_copy['dsr_cognitive'] ** 2
                X_quadratic = sm.add_constant(subset_copy[['dsr_cognitive', 'dsr_squared']])
                model_quadratic = sm.OLS(subset_copy['cs_output'], X_quadratic).fit()
                
                # 比较模型
                results['nonlinear_regions'][context] = {
                    'linear_r2': model_linear.rsquared,
                    'quadratic_r2': model_quadratic.rsquared,
                    'r2_improvement': model_quadratic.rsquared - model_linear.rsquared,
                    'quadratic_coef': model_quadratic.params.get('dsr_squared', 0),
                    'quadratic_pvalue': model_quadratic.pvalues.get('dsr_squared', 1),
                    'is_nonlinear': model_quadratic.pvalues.get('dsr_squared', 1) < 0.05
                }
                
        # 3. 饱和效应检测
        # 检查高DSR值区域的边际效应递减
        high_dsr_threshold = self.df['dsr_cognitive'].quantile(0.75)
        
        for context in ['low', 'medium', 'high']:
            subset_low = self.df[(self.df['context_stratum'] == context) & 
                                (self.df['dsr_cognitive'] <= high_dsr_threshold)]
            subset_high = self.df[(self.df['context_stratum'] == context) & 
                                 (self.df['dsr_cognitive'] > high_dsr_threshold)]
            
            if len(subset_low) > 20 and len(subset_high) > 20:
                # 计算两个区域的边际效应
                try:
                    model_low = sm.OLS(subset_low['cs_output'], 
                                      sm.add_constant(subset_low['dsr_cognitive'])).fit()
                    model_high = sm.OLS(subset_high['cs_output'], 
                                       sm.add_constant(subset_high['dsr_cognitive'])).fit()
                    
                    results['saturation_effects'][context] = {
                        'low_dsr_slope': model_low.params[1],
                        'high_dsr_slope': model_high.params[1],
                        'saturation_ratio': model_high.params[1] / model_low.params[1] if model_low.params[1] != 0 else 0,
                        'is_saturated': model_high.params[1] < model_low.params[1] * 0.5
                    }
                except:
                    pass
                    
        print(f"    - 检测到的阈值点: {len(results['threshold_points'])}")
        print(f"    - 非线性区域: {sum(1 for v in results['nonlinear_regions'].values() if v.get('is_nonlinear', False))}")
        print(f"    - 饱和效应区域: {sum(1 for v in results['saturation_effects'].values() if v.get('is_saturated', False))}")
        
        self.results['threshold_detection'] = results
        return results
        
    def analyze_moderation_patterns(self):
        """综合分析调节模式"""
        print("  执行调节模式综合分析...")
        
        results = {
            'pattern_summary': {},
            'moderation_typology': {},
            'policy_implications': {}
        }
        
        # 1. 模式总结
        # 基于前面的分析结果总结主要模式
        context_effects = self.results.get('context_moderation', {}).get('simple_slopes', {})
        platform_effects = self.results.get('platform_moderation', {}).get('platform_effects', {})
        
        # 语境调节模式
        if context_effects:
            slopes = {k: v['slope'] for k, v in context_effects.items()}
            results['pattern_summary']['context_pattern'] = {
                'type': 'enhancing' if slopes.get('high', 0) > slopes.get('low', 0) else 'buffering',
                'strength': max(slopes.values()) - min(slopes.values()) if slopes else 0,
                'optimal_context': max(slopes.items(), key=lambda x: x[1])[0] if slopes else None
            }
            
        # 平台调节模式
        if platform_effects:
            correlations = {k: v['dsr_cs_correlation'] for k, v in platform_effects.items()}
            results['pattern_summary']['platform_pattern'] = {
                'type': 'progressive' if correlations.get('advanced', 0) > correlations.get('early', 0) else 'diminishing',
                'evolution_trend': correlations.get('advanced', 0) - correlations.get('early', 0),
                'current_stage': 'advanced' if 'advanced' in correlations and correlations['advanced'] == max(correlations.values()) else 'developing'
            }
            
        # 2. 调节类型学
        # 基于Johnson-Neyman技术的分类
        moderation_types = []
        
        # 检查交互效应的显著性
        interaction_pvalues = self.results.get('context_moderation', {}).get('interaction_effects', {}).get('p_values', {})
        
        if any(p < 0.05 for p in interaction_pvalues.values()):
            moderation_types.append('significant_interaction')
            
        # 检查非线性
        nonlinear_regions = self.results.get('threshold_detection', {}).get('nonlinear_regions', {})
        if any(v.get('is_nonlinear', False) for v in nonlinear_regions.values()):
            moderation_types.append('nonlinear_moderation')
            
        # 检查阈值效应
        if self.results.get('threshold_detection', {}).get('threshold_points'):
            moderation_types.append('threshold_moderation')
            
        results['moderation_typology'] = {
            'identified_types': moderation_types,
            'complexity_level': len(moderation_types),
            'primary_type': moderation_types[0] if moderation_types else 'linear_moderation'
        }
        
        # 3. 政策启示
        implications = []
        
        # 基于语境模式
        if results['pattern_summary'].get('context_pattern', {}).get('type') == 'enhancing':
            implications.append({
                'area': 'context_sensitivity',
                'recommendation': '在高敏感度语境中加强DSR应用',
                'priority': 'high'
            })
            
        # 基于平台模式
        if results['pattern_summary'].get('platform_pattern', {}).get('type') == 'progressive':
            implications.append({
                'area': 'platform_development',
                'recommendation': '持续推进平台功能升级以增强DSR效果',
                'priority': 'medium'
            })
            
        # 基于阈值效应
        if 'threshold_moderation' in moderation_types:
            implications.append({
                'area': 'resource_allocation',
                'recommendation': '识别并优先支持接近阈值的应用场景',
                'priority': 'high'
            })
            
        results['policy_implications'] = implications
        
        print(f"    - 语境调节模式: {results['pattern_summary'].get('context_pattern', {}).get('type', 'N/A')}")
        print(f"    - 平台调节模式: {results['pattern_summary'].get('platform_pattern', {}).get('type', 'N/A')}")
        print(f"    - 调节复杂度: {results['moderation_typology'].get('complexity_level', 0)}")
        
        self.results['moderation_patterns'] = results
        return results
        
    def create_visualizations(self):
        """创建可视化"""
        print("\n生成可视化...")
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 标题
        fig.suptitle('调节效应分析结果', fontsize=20, fontweight='bold')
        
        # 1. 语境调节效应（左上）
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_context_moderation(ax1)
        
        # 2. 平台调节效应（中上）
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_platform_moderation(ax2)
        
        # 3. 交互效应3D图（右上）
        ax3 = fig.add_subplot(gs[0, 2], projection='3d')
        self._plot_3d_interaction(ax3)
        
        # 4. 简单斜率图（左中）
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_simple_slopes(ax4)
        
        # 5. 阈值效应图（中中）
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_threshold_effects(ax5)
        
        # 6. 平台演化图（右中）
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_platform_evolution(ax6)
        
        # 7. 效应分解热图（底部左）
        ax7 = fig.add_subplot(gs[2, :2])
        self._plot_effect_decomposition(ax7)
        
        # 8. 调节强度比较（底部右）
        ax8 = fig.add_subplot(gs[2, 2])
        self._plot_moderation_strength(ax8)
        
        # 9. 政策启示图（最底部）
        ax9 = fig.add_subplot(gs[3, :])
        self._plot_policy_implications(ax9)
        
        # 保存图形
        output_path = self.data_path.parent / 'figures' / 'moderation_analysis.jpg'
        plt.savefig(output_path, dpi=1200, bbox_inches='tight', format='jpg')
        plt.close()
        
        print(f"  可视化已保存至: {output_path}")
        
    def _plot_context_moderation(self, ax):
        """绘制语境调节效应"""
        main_effects = self.results.get('context_moderation', {}).get('main_effects', {})
        
        if main_effects:
            contexts = ['low', 'medium', 'high']
            correlations = [main_effects.get(c, {}).get('dsr_cs_correlation', 0) for c in contexts]
            colors = ['lightblue', 'orange', 'red']
            
            bars = ax.bar(contexts, correlations, color=colors, alpha=0.7)
            ax.set_xlabel('语境敏感度')
            ax.set_ylabel('DSR-CS相关系数')
            ax.set_title('语境敏感度的调节效应')
            ax.set_ylim(0, max(correlations) * 1.2 if correlations else 1)
            
            # 添加数值标签
            for bar, corr in zip(bars, correlations):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{corr:.3f}', ha='center')
        else:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center')
            ax.set_title('语境调节效应')
            
    def _plot_platform_moderation(self, ax):
        """绘制平台调节效应"""
        platform_effects = self.results.get('platform_moderation', {}).get('platform_effects', {})
        
        if platform_effects:
            platforms = ['early', 'mature', 'advanced']
            correlations = [platform_effects.get(p, {}).get('dsr_cs_correlation', 0) for p in platforms]
            
            ax.plot(platforms, correlations, 'bo-', linewidth=2, markersize=10)
            ax.fill_between(range(len(platforms)), correlations, alpha=0.3)
            ax.set_xlabel('平台类型')
            ax.set_ylabel('DSR-CS相关系数')
            ax.set_title('平台类型的调节效应')
            ax.grid(True, alpha=0.3)
            
            # 添加趋势线标注
            if len(correlations) > 1:
                trend = correlations[-1] - correlations[0]
                ax.text(0.95, 0.95, f'趋势: {"↑" if trend > 0 else "↓"} {abs(trend):.3f}',
                       transform=ax.transAxes, ha='right', va='top',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        else:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center')
            ax.set_title('平台调节效应')
            
    def _plot_3d_interaction(self, ax):
        """绘制3D交互效应图"""
        # 创建网格数据
        dsr_range = np.linspace(self.df['dsr_cognitive'].min(), self.df['dsr_cognitive'].max(), 20)
        context_levels = [0, 1, 2]  # low, medium, high
        
        DSR, CONTEXT = np.meshgrid(dsr_range, context_levels)
        
        # 基于模型预测创建曲面
        interaction_effects = self.results.get('context_moderation', {}).get('interaction_effects', {}).get('coefficients', {})
        
        if interaction_effects:
            # 简化的预测模型
            baseline = interaction_effects.get('dsr_baseline', 0.1)
            medium_int = interaction_effects.get('interaction_medium', 0.05)
            high_int = interaction_effects.get('interaction_high', 0.1)
            
            CS = baseline * DSR
            CS[1, :] += medium_int * DSR[1, :]  # medium context
            CS[2, :] += high_int * DSR[2, :]     # high context
            
            # 绘制曲面
            surf = ax.plot_surface(DSR, CONTEXT, CS, cmap=cm.coolwarm, alpha=0.8)
            
            ax.set_xlabel('DSR认知功能')
            ax.set_ylabel('语境层级')
            ax.set_zlabel('CS输出')
            ax.set_title('DSR×语境交互效应')
            
            # 设置y轴标签
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(['低', '中', '高'])
            
            # 添加颜色条
            fig = ax.get_figure()
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        else:
            ax.text2D(0.5, 0.5, '无数据', transform=ax.transAxes)
            ax.set_title('3D交互效应')
            
    def _plot_simple_slopes(self, ax):
        """绘制简单斜率图"""
        simple_slopes = self.results.get('context_moderation', {}).get('simple_slopes', {})
        
        if simple_slopes:
            # 为每个语境绘制回归线
            colors = {'low': 'blue', 'medium': 'orange', 'high': 'red'}
            
            for context, slope_info in simple_slopes.items():
                subset = self.df[self.df['context_stratum'] == context]
                if len(subset) > 10:
                    # 绘制散点
                    ax.scatter(subset['dsr_cognitive'], subset['cs_output'], 
                             alpha=0.3, s=10, color=colors.get(context, 'gray'))
                    
                    # 绘制回归线
                    x_range = np.linspace(subset['dsr_cognitive'].min(), 
                                        subset['dsr_cognitive'].max(), 100)
                    y_pred = slope_info['slope'] * x_range + subset['cs_output'].mean() - slope_info['slope'] * subset['dsr_cognitive'].mean()
                    
                    ax.plot(x_range, y_pred, color=colors.get(context, 'gray'), 
                           linewidth=2, label=f'{context} (β={slope_info["slope"]:.3f})')
                    
            ax.set_xlabel('DSR认知功能')
            ax.set_ylabel('CS输出')
            ax.set_title('不同语境下的简单斜率')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center')
            ax.set_title('简单斜率分析')
            
    def _plot_threshold_effects(self, ax):
        """绘制阈值效应"""
        threshold_points = self.results.get('threshold_detection', {}).get('threshold_points', {})
        
        if threshold_points:
            contexts = list(threshold_points.keys())
            thresholds = [threshold_points[c]['threshold_value'] for c in contexts]
            slope_changes = [threshold_points[c]['slope_change'] for c in contexts]
            
            # 创建双轴图
            ax2 = ax.twinx()
            
            # 条形图：阈值位置
            bars = ax.bar(contexts, thresholds, alpha=0.6, color='blue', label='阈值位置')
            
            # 线图：斜率变化
            line = ax2.plot(contexts, slope_changes, 'ro-', linewidth=2, 
                           markersize=8, label='斜率变化')
            
            ax.set_xlabel('语境层级')
            ax.set_ylabel('DSR阈值', color='blue')
            ax2.set_ylabel('斜率变化量', color='red')
            ax.set_title('局部阈值检测结果')
            
            # 合并图例
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            ax.tick_params(axis='y', labelcolor='blue')
            ax2.tick_params(axis='y', labelcolor='red')
        else:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center')
            ax.set_title('阈值效应')
            
    def _plot_platform_evolution(self, ax):
        """绘制平台演化图"""
        evolution_patterns = self.results.get('platform_moderation', {}).get('evolution_patterns', [])
        
        if evolution_patterns:
            years = [d['year'] for d in evolution_patterns]
            correlations = [d['dsr_cs_correlation'] for d in evolution_patterns]
            integrations = [d['mean_integration'] for d in evolution_patterns]
            
            # 双轴图
            ax2 = ax.twinx()
            
            # 相关性演化
            line1 = ax.plot(years, correlations, 'b-', linewidth=2, 
                           marker='o', markersize=8, label='DSR-CS相关性')
            
            # 整合深度演化
            line2 = ax2.plot(years, integrations, 'g-', linewidth=2, 
                            marker='s', markersize=8, label='整合深度')
            
            ax.set_xlabel('年份')
            ax.set_ylabel('相关系数', color='blue')
            ax2.set_ylabel('整合深度', color='green')
            ax.set_title('平台功能演化轨迹')
            
            # 添加趋势线
            z1 = np.polyfit(years, correlations, 1)
            p1 = np.poly1d(z1)
            ax.plot(years, p1(years), "b--", alpha=0.8)
            
            # 合并图例
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left')
            
            ax.tick_params(axis='y', labelcolor='blue')
            ax2.tick_params(axis='y', labelcolor='green')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center')
            ax.set_title('平台演化')
            
    def _plot_effect_decomposition(self, ax):
        """绘制效应分解热图"""
        decomposed_effects = self.results.get('interaction_effects', {}).get('decomposed_effects', [])
        
        if decomposed_effects:
            # 创建效应矩阵
            contexts = ['low', 'medium', 'high']
            platforms = ['early', 'mature', 'advanced']
            
            effect_matrix = np.zeros((len(contexts), len(platforms)))
            
            for effect in decomposed_effects:
                i = contexts.index(effect['context'])
                j = platforms.index(effect['platform'])
                effect_matrix[i, j] = effect['marginal_effect']
                
            # 绘制热图
            im = ax.imshow(effect_matrix, cmap='RdYlBu_r', aspect='auto')
            
            # 设置标签
            ax.set_xticks(np.arange(len(platforms)))
            ax.set_yticks(np.arange(len(contexts)))
            ax.set_xticklabels(platforms)
            ax.set_yticklabels(contexts)
            
            # 添加数值标注
            for i in range(len(contexts)):
                for j in range(len(platforms)):
                    if effect_matrix[i, j] != 0:
                        text = ax.text(j, i, f'{effect_matrix[i, j]:.3f}',
                                     ha="center", va="center", color="black")
                        
            ax.set_xlabel('平台类型')
            ax.set_ylabel('语境层级')
            ax.set_title('交互效应分解：边际效应热图')
            
            # 添加颜色条
            plt.colorbar(im, ax=ax, label='边际效应')
        else:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center')
            ax.set_title('效应分解')
            
    def _plot_moderation_strength(self, ax):
        """绘制调节强度比较"""
        # 收集各种调节效应的强度
        strengths = {}
        
        # 语境调节强度
        context_mod = self.results.get('context_moderation', {}).get('moderation_strength', {})
        if context_mod:
            strengths['语境调节'] = context_mod.get('slope_range', 0)
            
        # 平台调节强度
        platform_effects = self.results.get('platform_moderation', {}).get('platform_effects', {})
        if platform_effects:
            corrs = [v.get('dsr_cs_correlation', 0) for v in platform_effects.values()]
            if corrs:
                strengths['平台调节'] = max(corrs) - min(corrs)
                
        # 交互效应强度
        synergy = self.results.get('interaction_effects', {}).get('synergy_analysis', {})
        if synergy:
            strengths['交互效应'] = synergy.get('effect_range', 0)
            
        if strengths:
            # 绘制雷达图
            categories = list(strengths.keys())
            values = list(strengths.values())
            
            # 添加第一个值到末尾，使图形闭合
            values += values[:1]
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, color='purple')
            ax.fill(angles, values, alpha=0.25, color='purple')
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, max(values) * 1.2 if values else 1)
            ax.set_title('调节效应强度比较')
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center')
            ax.set_title('调节强度')
            
    def _plot_policy_implications(self, ax):
        """绘制政策启示"""
        implications = self.results.get('moderation_patterns', {}).get('policy_implications', [])
        
        if implications:
            # 创建政策建议的可视化
            areas = [imp['area'] for imp in implications]
            priorities = [1 if imp['priority'] == 'high' else 0.5 for imp in implications]
            colors = ['red' if p == 1 else 'orange' for p in priorities]
            
            y_pos = np.arange(len(areas))
            ax.barh(y_pos, priorities, color=colors, alpha=0.7)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels([imp['recommendation'][:30] + '...' if len(imp['recommendation']) > 30 
                               else imp['recommendation'] for imp in implications])
            ax.set_xlabel('优先级')
            ax.set_xlim(0, 1.2)
            ax.set_title('政策启示与建议')
            
            # 添加优先级标签
            for i, (area, priority) in enumerate(zip(areas, priorities)):
                label = '高' if priority == 1 else '中'
                ax.text(priority + 0.05, i, label, va='center')
        else:
            ax.text(0.5, 0.5, '暂无政策建议', ha='center', va='center')
            ax.set_title('政策启示')
            
    def save_results(self):
        """保存结果"""
        output_file = self.data_path / 'moderation_analysis_results.json'
        
        # 转换numpy类型为Python类型
        def convert_numpy(obj):
            if isinstance(obj, (np.integer, np.int_, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float_, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return obj.strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
                
        results_to_save = convert_numpy(self.results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, ensure_ascii=False, indent=2)
            
        print(f"\n结果已保存至: {output_file}")
        
        # 生成Markdown报告
        self.generate_report()
        
    def generate_report(self):
        """生成Markdown报告"""
        report = []
        report.append("# 调节效应分析报告\n")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 1. 总体发现
        report.append("## 总体发现\n")
        
        # 语境调节
        context_pattern = self.results.get('moderation_patterns', {}).get('pattern_summary', {}).get('context_pattern', {})
        if context_pattern:
            report.append(f"- **语境调节模式**: {context_pattern.get('type', 'N/A')}")
            report.append(f"- **调节强度**: {context_pattern.get('strength', 0):.3f}")
            report.append(f"- **最优语境**: {context_pattern.get('optimal_context', 'N/A')}\n")
            
        # 平台调节
        platform_pattern = self.results.get('moderation_patterns', {}).get('pattern_summary', {}).get('platform_pattern', {})
        if platform_pattern:
            report.append(f"- **平台调节模式**: {platform_pattern.get('type', 'N/A')}")
            report.append(f"- **演化趋势**: {platform_pattern.get('evolution_trend', 0):.3f}")
            report.append(f"- **当前阶段**: {platform_pattern.get('current_stage', 'N/A')}\n")
            
        # 2. 语境调节详情
        report.append("## 语境敏感度调节分析\n")
        
        main_effects = self.results.get('context_moderation', {}).get('main_effects', {})
        if main_effects:
            report.append("### 不同语境下的效应\n")
            report.append("| 语境层级 | 样本数 | DSR-CS相关性 | 平均DSR | 平均CS |")
            report.append("|---------|--------|-------------|---------|--------|")
            
            for context in ['low', 'medium', 'high']:
                if context in main_effects:
                    data = main_effects[context]
                    report.append(f"| {context} | {data['n_samples']} | "
                                f"{data['dsr_cs_correlation']:.3f} | "
                                f"{data['mean_dsr']:.3f} | "
                                f"{data['mean_cs']:.3f} |")
                                
        # 简单斜率
        simple_slopes = self.results.get('context_moderation', {}).get('simple_slopes', {})
        if simple_slopes:
            report.append("\n### 简单斜率分析\n")
            for context, slope_data in simple_slopes.items():
                report.append(f"- **{context}语境**: β = {slope_data['slope']:.3f} "
                            f"(p = {slope_data['p_value']:.3f})")
                            
        # 3. 平台调节详情
        report.append("\n## 平台类型调节分析\n")
        
        platform_effects = self.results.get('platform_moderation', {}).get('platform_effects', {})
        if platform_effects:
            report.append("### 不同平台的效应\n")
            report.append("| 平台类型 | 样本数 | DSR-CS相关性 | 整合深度 |")
            report.append("|---------|--------|-------------|---------|")
            
            for platform in ['early', 'mature', 'advanced']:
                if platform in platform_effects:
                    data = platform_effects[platform]
                    report.append(f"| {platform} | {data['n_samples']} | "
                                f"{data['dsr_cs_correlation']:.3f} | "
                                f"{data['mean_integration_depth']:.3f} |")
                                
        # 4. 交互效应
        report.append("\n## 交互效应分析\n")
        
        three_way = self.results.get('interaction_effects', {}).get('three_way_interaction', {})
        if three_way:
            report.append(f"- **三向交互显著性**: {'是' if three_way.get('significant', False) else '否'}")
            report.append(f"- **模型R²**: {three_way.get('model_fit', {}).get('r_squared', 0):.3f}")
            
        synergy = self.results.get('interaction_effects', {}).get('synergy_analysis', {})
        if synergy:
            report.append(f"- **最优条件组合**: {synergy.get('optimal_condition', 'N/A')}")
            report.append(f"- **效应范围**: {synergy.get('effect_range', 0):.3f}\n")
            
        # 5. 阈值效应
        report.append("## 阈值效应检测\n")
        
        threshold_points = self.results.get('threshold_detection', {}).get('threshold_points', {})
        if threshold_points:
            report.append("### 检测到的阈值点\n")
            for context, threshold_data in threshold_points.items():
                report.append(f"- **{context}语境**: DSR = {threshold_data['threshold_value']:.3f} "
                            f"(斜率变化 = {threshold_data['slope_change']:.3f})")
                            
        # 非线性区域
        nonlinear_regions = self.results.get('threshold_detection', {}).get('nonlinear_regions', {})
        nonlinear_count = sum(1 for v in nonlinear_regions.values() if v.get('is_nonlinear', False))
        if nonlinear_count > 0:
            report.append(f"\n检测到 **{nonlinear_count}** 个非线性区域")
            
        # 6. 政策建议
        report.append("\n## 政策启示\n")
        
        implications = self.results.get('moderation_patterns', {}).get('policy_implications', [])
        if implications:
            for i, imp in enumerate(implications, 1):
                report.append(f"{i}. **{imp['area']}**: {imp['recommendation']} "
                            f"(优先级: {imp['priority']})")
                            
        # 保存报告
        report_file = self.data_path.parent / 'md' / 'moderation_analysis_report.md'
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
            
        print(f"报告已保存至: {report_file}")


def main():
    """主函数"""
    # 创建分析实例
    from pathlib import Path
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / 'output_cn' / 'data'
    
    analyzer = ModerationAnalysis(data_path)
    
    # 加载数据
    analyzer.load_data()
    
    # 运行分析
    results = analyzer.run_moderation_analysis()
    
    print("\n" + "="*60)
    print("调节效应分析完成！")
    print("="*60)
    

if __name__ == "__main__":
    main()