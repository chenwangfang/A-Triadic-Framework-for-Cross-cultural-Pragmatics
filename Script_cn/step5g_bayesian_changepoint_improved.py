#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改进的贝叶斯变点检测脚本
========================
解决原始脚本中的问题：
1. 添加烧入期（burn-in period）
2. 使用更合理的阈值
3. 更强的先验设置
4. 最小段长度约束
5. 集成多种算法的结果
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 统计分析
from scipy import stats
from scipy.special import gammaln, logsumexp
from scipy.ndimage import gaussian_filter1d

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class ImprovedBayesianChangepointDetector:
    """改进的贝叶斯变点检测器"""
    
    def __init__(self, data_path, burn_in_days=30, min_segment_length=14):
        """
        初始化检测器
        
        Parameters:
        -----------
        data_path : str
            数据路径
        burn_in_days : int
            烧入期天数（默认30天）
        min_segment_length : int
            最小段长度（默认14天）
        """
        self.data_path = Path(data_path)
        self.burn_in_days = burn_in_days
        self.min_segment_length = min_segment_length
        self.df = None
        self.time_series = None
        self.results = {
            'changepoints': {},
            'evolution_phases': {},
            'algorithm_comparison': {},
            'metadata': {
                'burn_in_days': burn_in_days,
                'min_segment_length': min_segment_length,
                'timestamp': datetime.now().isoformat()
            }
        }
        
    def load_data(self):
        """加载数据"""
        print("加载数据...")
        self.df = pd.read_csv(self.data_path / 'data' / 'data_with_metrics.csv')
        self.df['date'] = pd.to_datetime(self.df['date'])
        print(f"  加载 {len(self.df)} 条记录")
        print(f"  时间范围: {self.df['date'].min()} 至 {self.df['date'].max()}")
        
    def prepare_time_series(self):
        """准备时间序列数据"""
        print("\n准备时间序列...")
        
        # 按日期聚合
        daily_data = self.df.groupby('date').agg({
            'dsr_cognitive': 'mean',
            'tl_functional': 'mean',
            'cs_output': 'mean',
            'constitutive_index': 'mean'
        }).reset_index()
        
        # 填充缺失日期
        date_range = pd.date_range(
            start=daily_data['date'].min(),
            end=daily_data['date'].max(),
            freq='D'
        )
        daily_data = daily_data.set_index('date').reindex(date_range).reset_index()
        daily_data.rename(columns={'index': 'date'}, inplace=True)
        
        # 前向填充缺失值
        daily_data.fillna(method='ffill', inplace=True)
        daily_data.fillna(method='bfill', inplace=True)
        
        # 计算构成性综合指标
        daily_data['constitutive_strength'] = (
            daily_data['dsr_cognitive'] * 0.4 +
            daily_data['tl_functional'] * 0.3 +
            daily_data['cs_output'] * 0.3
        )
        
        # 平滑处理（使用更温和的平滑）
        daily_data['constitutive_smooth'] = gaussian_filter1d(
            daily_data['constitutive_strength'], 
            sigma=3  # 增加平滑程度
        )
        
        self.time_series = daily_data
        print(f"  生成时间序列: {len(daily_data)} 个时间点")
        
    def detect_changepoints(self):
        """执行变点检测"""
        print("\n执行变点检测...")
        
        y = self.time_series['constitutive_smooth'].values
        dates = self.time_series['date'].values
        
        # 1. 改进的在线贝叶斯变点检测
        print("  1. 在线贝叶斯变点检测...")
        online_results = self._improved_online_detection(y)
        
        # 2. 基于滑动窗口的检测
        print("  2. 滑动窗口检测...")
        window_results = self._sliding_window_detection(y)
        
        # 3. 基于理论的预期变点
        print("  3. 理论驱动的变点识别...")
        theory_results = self._theory_based_detection(y, dates)
        
        # 4. 集成结果
        print("  4. 集成多种方法结果...")
        integrated_results = self._integrate_results(
            online_results, window_results, theory_results, dates
        )
        
        self.results['changepoints'] = integrated_results
        
        # 识别演化阶段
        self._identify_evolution_phases(integrated_results, dates)
        
    def _improved_online_detection(self, data):
        """改进的在线贝叶斯变点检测"""
        n = len(data)
        
        # 跳过烧入期
        start_idx = self.burn_in_days
        if start_idx >= n:
            print(f"    警告：数据长度({n})小于烧入期({self.burn_in_days})")
            start_idx = min(30, n // 4)
        
        # 使用烧入期数据估计先验
        burn_in_data = data[:start_idx]
        prior_mean = np.mean(burn_in_data)
        prior_var = np.var(burn_in_data)
        
        # 更强的先验
        alpha = 3.0  # shape parameter
        beta = alpha * prior_var  # scale parameter
        kappa = 5.0  # 先验强度
        mu = prior_mean
        
        # 在线检测（从烧入期后开始）
        R = np.zeros((n+1, n+1))
        R[0, 0] = 1
        
        changepoint_probs = np.zeros(n)
        hazard_rate = 1 / 100  # 假设平均100天一个变点
        
        for t in range(start_idx, n):
            # 预测概率
            predprobs = np.zeros(t+1)
            
            for s in range(max(0, t - 365), t+1):  # 限制回看窗口为1年
                # 后验参数更新
                if t > s:
                    segment_data = data[s:t]
                    n_s = len(segment_data)
                    
                    kappa_n = kappa + n_s
                    mu_n = (kappa * mu + np.sum(segment_data)) / kappa_n
                    alpha_n = alpha + n_s / 2
                    beta_n = beta + 0.5 * np.sum((segment_data - mu_n)**2) + \
                             (kappa * n_s * (mu_n - mu)**2) / (2 * kappa_n)
                    
                    # 学生t分布预测
                    scale = np.sqrt(beta_n * (kappa_n + 1) / (alpha_n * kappa_n))
                    df = 2 * alpha_n
                    predprobs[s] = stats.t.pdf(data[t], df, mu_n, scale)
            
            # 增长概率（限制计算范围）
            valid_range = max(0, t - 365)
            
            # 确保索引范围正确
            if t > 0:
                # 计算每个段的增长概率
                for s in range(valid_range, t):
                    if s < t:
                        R[s+1, t] = R[s, t-1] * predprobs[s] * (1 - hazard_rate)
                
                # 变点概率
                R[0, t] = np.sum(R[valid_range:t, t-1] * hazard_rate)
                
                # 归一化
                R[:, t] = R[:, t] / (np.sum(R[:, t]) + 1e-10)
                
                changepoint_probs[t] = R[0, t]
        
        # 保存概率以供可视化
        self._online_probs = changepoint_probs
        
        return {
            'probabilities': changepoint_probs,
            'threshold': 0.1,  # 更高的阈值
            'method': 'improved_online'
        }
        
    def _sliding_window_detection(self, data, window_size=60):
        """基于滑动窗口的变点检测"""
        n = len(data)
        change_scores = np.zeros(n)
        
        for t in range(window_size, n - window_size):
            # 比较前后窗口
            before = data[t-window_size:t]
            after = data[t:t+window_size]
            
            # t检验
            t_stat, p_value = stats.ttest_ind(before, after)
            
            # KS检验
            ks_stat, ks_pvalue = stats.ks_2samp(before, after)
            
            # 综合得分
            change_scores[t] = (1 - p_value) * 0.5 + (1 - ks_pvalue) * 0.5
        
        return {
            'scores': change_scores,
            'threshold': 0.95,  # 需要很高的置信度
            'method': 'sliding_window'
        }
        
    def _theory_based_detection(self, data, dates):
        """基于理论的变点检测"""
        # 基于DSR采用理论，预期的变点时间
        theory_changepoints = []
        
        # 转换日期为年份分数
        start_date = pd.Timestamp('2021-01-01')
        years_elapsed = [(pd.Timestamp(d) - start_date).days / 365.25 for d in dates]
        
        # 理论预期：探索期结束（约1年后）
        exploration_end = 1.0
        theory_changepoints.append({
            'year_fraction': exploration_end,
            'phase_transition': 'exploration_to_integration',
            'expected_change': 'increase'
        })
        
        # 理论预期：整合期结束（约3年后）
        integration_end = 3.0
        theory_changepoints.append({
            'year_fraction': integration_end,
            'phase_transition': 'integration_to_internalization',
            'expected_change': 'stabilize'
        })
        
        # 找到最接近理论预期的实际时间点
        detected = []
        for tcp in theory_changepoints:
            target_year = tcp['year_fraction']
            
            # 找到最接近的时间索引
            distances = [abs(y - target_year) for y in years_elapsed]
            if min(distances) < 0.5:  # 6个月内
                idx = distances.index(min(distances))
                
                # 验证该时间点附近是否有实际变化
                window = 30
                if idx > window and idx < len(data) - window:
                    before = data[idx-window:idx]
                    after = data[idx:idx+window]
                    
                    # 检验变化显著性
                    t_stat, p_value = stats.ttest_ind(before, after)
                    
                    if p_value < 0.05:  # 显著变化
                        detected.append({
                            'index': idx,
                            'confidence': 1 - p_value,
                            'theory_based': True,
                            'phase_transition': tcp['phase_transition']
                        })
        
        return {
            'changepoints': detected,
            'method': 'theory_based'
        }
        
    def _integrate_results(self, online_results, window_results, theory_results, dates):
        """集成多种方法的结果"""
        integrated_changepoints = []
        
        # 1. 收集所有候选变点
        candidates = []
        
        # 从在线检测
        probs = online_results['probabilities']
        threshold = online_results['threshold']
        for i in range(len(probs)):
            if probs[i] > threshold and i > self.burn_in_days:
                candidates.append({
                    'index': i,
                    'score': probs[i],
                    'method': 'online'
                })
        
        # 从滑动窗口
        scores = window_results['scores']
        threshold = window_results['threshold']
        for i in range(len(scores)):
            if scores[i] > threshold:
                candidates.append({
                    'index': i,
                    'score': scores[i],
                    'method': 'window'
                })
        
        # 从理论驱动
        for cp in theory_results['changepoints']:
            candidates.append({
                'index': cp['index'],
                'score': cp['confidence'],
                'method': 'theory',
                'phase_transition': cp.get('phase_transition', '')
            })
        
        # 2. 合并邻近的候选点（30天内）
        candidates.sort(key=lambda x: x['index'])
        merged = []
        
        for cand in candidates:
            if not merged or cand['index'] - merged[-1]['index'] > 30:
                merged.append(cand)
            else:
                # 合并到已有变点
                if cand['score'] > merged[-1]['score']:
                    merged[-1] = cand
        
        # 3. 应用最小段长度约束
        final_changepoints = []
        last_idx = 0
        
        for cp in merged:
            if cp['index'] - last_idx >= self.min_segment_length:
                # 计算变化幅度
                before_start = max(0, cp['index'] - 30)
                before_end = cp['index']
                after_start = cp['index']
                after_end = min(len(self.time_series), cp['index'] + 30)
                
                before_mean = np.mean(self.time_series['constitutive_smooth'].iloc[before_start:before_end])
                after_mean = np.mean(self.time_series['constitutive_smooth'].iloc[after_start:after_end])
                
                final_changepoints.append({
                    'index': cp['index'],
                    'date': str(dates[cp['index']]),
                    'confidence': cp['score'],
                    'method': cp['method'],
                    'before_mean': float(before_mean),
                    'after_mean': float(after_mean),
                    'magnitude': float(after_mean - before_mean),
                    'phase_transition': cp.get('phase_transition', '')
                })
                
                last_idx = cp['index']
        
        return {
            'detected_changepoints': final_changepoints,
            'n_changepoints': len(final_changepoints),
            'methods_used': ['improved_online', 'sliding_window', 'theory_based']
        }
        
    def _identify_evolution_phases(self, changepoint_results, dates):
        """识别演化阶段"""
        changepoints = changepoint_results['detected_changepoints']
        
        phases = []
        start_idx = 0
        
        # 添加初始阶段
        if changepoints:
            phases.append({
                'phase': 'exploration',
                'start_date': str(dates[0]),
                'end_date': changepoints[0]['date'],
                'duration_days': changepoints[0]['index'],
                'characteristics': '初步探索，DSR应用逐步展开'
            })
            
            # 中间阶段
            for i in range(len(changepoints) - 1):
                start_idx = changepoints[i]['index']
                end_idx = changepoints[i + 1]['index']
                
                phase_name = 'integration' if i == 0 else 'optimization'
                
                phases.append({
                    'phase': phase_name,
                    'start_date': changepoints[i]['date'],
                    'end_date': changepoints[i + 1]['date'],
                    'duration_days': end_idx - start_idx,
                    'characteristics': '系统整合，功能扩展' if i == 0 else '优化调整'
                })
            
            # 最后阶段
            phases.append({
                'phase': 'internalization',
                'start_date': changepoints[-1]['date'],
                'end_date': str(dates[-1]),
                'duration_days': len(dates) - changepoints[-1]['index'],
                'characteristics': '内化成熟，DSR作用隐性化'
            })
        else:
            # 没有检测到变点
            phases.append({
                'phase': 'continuous',
                'start_date': str(dates[0]),
                'end_date': str(dates[-1]),
                'duration_days': len(dates),
                'characteristics': '持续演化，无明显阶段性变化'
            })
        
        self.results['evolution_phases'] = {
            'phases': phases,
            'n_phases': len(phases)
        }
        
    def visualize_results(self):
        """可视化结果"""
        print("\n生成可视化...")
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # 1. 时间序列与变点
        ax1 = axes[0]
        ts = self.time_series['constitutive_smooth']
        dates = self.time_series['date']
        
        ax1.plot(dates, ts, 'b-', linewidth=1.5, label='构成性指数')
        
        # 标记变点
        for cp in self.results['changepoints']['detected_changepoints']:
            idx = cp['index']
            ax1.axvline(x=dates.iloc[idx], color='r', linestyle='--', alpha=0.7)
            ax1.text(dates.iloc[idx], ts.iloc[idx] + 0.01, 
                    f"{cp['method'][:3]}", rotation=90, fontsize=8)
        
        # 标记烧入期
        burn_in_end = dates.iloc[self.burn_in_days] if self.burn_in_days < len(dates) else dates.iloc[-1]
        ax1.axvspan(dates.iloc[0], burn_in_end, alpha=0.2, color='gray', label='烧入期')
        
        ax1.set_title('构成性指数时间序列与检测到的变点')
        ax1.set_ylabel('构成性指数')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 变点概率
        ax2 = axes[1]
        
        # 获取在线检测的概率
        if hasattr(self, '_online_probs'):
            probs = self._online_probs
            ax2.plot(dates, probs, 'g-', linewidth=1)
            ax2.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='阈值')
            ax2.set_title('变点概率（在线贝叶斯检测）')
            ax2.set_ylabel('概率')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. 演化阶段
        ax3 = axes[2]
        
        # 绘制阶段
        colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
        phases = self.results['evolution_phases']['phases']
        
        for i, phase in enumerate(phases):
            start = pd.Timestamp(phase['start_date'])
            end = pd.Timestamp(phase['end_date'])
            ax3.axvspan(start, end, alpha=0.3, color=colors[i % len(colors)], 
                       label=phase['phase'])
            
            # 添加阶段标签
            mid_date = start + (end - start) / 2
            ax3.text(mid_date, 0.5, phase['phase'], 
                    ha='center', va='center', fontsize=10, fontweight='bold')
        
        ax3.set_title('识别的演化阶段')
        ax3.set_xlim(dates.iloc[0], dates.iloc[-1])
        ax3.set_ylim(0, 1)
        ax3.set_yticks([])
        
        plt.tight_layout()
        
        # 保存图形
        output_path = self.data_path / 'figures' / 'improved_changepoint_detection.jpg'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=1200, bbox_inches='tight', format='jpg')
        print(f"  图形已保存至: {output_path}")
        
        plt.close()  # 关闭图形而不是显示
        
    def save_results(self):
        """保存结果"""
        output_path = self.data_path / 'data' / 'improved_changepoint_results.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        print(f"\n结果已保存至: {output_path}")
        
    def generate_report(self):
        """生成分析报告"""
        print("\n生成分析报告...")
        
        report = []
        report.append("# 改进的贝叶斯变点检测报告")
        report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        report.append("\n## 方法改进")
        report.append(f"- 烧入期: {self.burn_in_days}天")
        report.append(f"- 最小段长度: {self.min_segment_length}天")
        report.append("- 集成方法: 在线贝叶斯、滑动窗口、理论驱动")
        report.append("- 更强的先验和更高的阈值")
        
        report.append("\n## 检测结果")
        cps = self.results['changepoints']['detected_changepoints']
        report.append(f"\n检测到 {len(cps)} 个主要变点：")
        
        for i, cp in enumerate(cps):
            report.append(f"\n### 变点 {i+1}")
            report.append(f"- 日期: {cp['date']}")
            report.append(f"- 置信度: {cp['confidence']:.3f}")
            report.append(f"- 检测方法: {cp['method']}")
            report.append(f"- 变化幅度: {cp['magnitude']:.3f}")
            if cp.get('phase_transition'):
                report.append(f"- 阶段转换: {cp['phase_transition']}")
        
        report.append("\n## 演化阶段")
        phases = self.results['evolution_phases']['phases']
        for phase in phases:
            report.append(f"\n### {phase['phase'].capitalize()}阶段")
            report.append(f"- 时间: {phase['start_date']} 至 {phase['end_date']}")
            report.append(f"- 持续: {phase['duration_days']}天")
            report.append(f"- 特征: {phase['characteristics']}")
        
        # 保存报告
        report_path = self.data_path / 'md' / 'improved_changepoint_report.md'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"  报告已保存至: {report_path}")


def main():
    """主函数"""
    # 设置数据路径
    data_path = Path(__file__).parent.parent / 'output_cn'
    
    # 创建检测器
    detector = ImprovedBayesianChangepointDetector(
        data_path=data_path,
        burn_in_days=30,
        min_segment_length=14
    )
    
    # 执行分析
    detector.load_data()
    detector.prepare_time_series()
    detector.detect_changepoints()
    
    # 保存结果
    detector.save_results()
    detector.generate_report()
    
    # 可视化
    detector.visualize_results()
    
    print("\n分析完成！")


if __name__ == "__main__":
    main()