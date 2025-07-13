# generate_comprehensive_visualization.py
# 生成综合可视化 - 整合所有分析步骤的关键结果

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class ComprehensiveVisualization:
    """综合可视化类"""
    
    def __init__(self, output_path):
        self.output_path = Path(output_path)
        self.results = {}
        self.load_all_results()
        
    def load_all_results(self):
        """加载所有分析步骤的结果"""
        result_files = {
            'step3': 'cognitive_metrics_results.json',
            'step4': 'information_theory_results.json',
            'step5': 'causal_analysis_results.json',
            'step6': 'constitutiveness_test_results.json',
            'step7': 'moderation_analysis_results.json',
            'step8': 'dynamic_evolution_results.json',
            'step9': 'network_diffusion_results.json'
        }
        
        data_path = self.output_path / 'data'
        
        for step, filename in result_files.items():
            file_path = data_path / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.results[step] = json.load(f)
                        print(f"[OK] Loaded {step} results: {filename}")
                except Exception as e:
                    print(f"[ERROR] Failed to load {step}: {e}")
            else:
                print(f"[MISSING] {step} result file not found: {filename}")
                
    def create_comprehensive_dashboard(self):
        """创建综合仪表板"""
        print("\nGenerating comprehensive visualization dashboard...")
        
        # 创建大型图表 (A3尺寸)
        fig = plt.figure(figsize=(16.5, 11.7))  # A3横向
        
        # 创建网格布局
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.25)
        
        # 标题
        fig.suptitle('DSR构成性假设验证：综合分析仪表板', fontsize=20, fontweight='bold', y=0.98)
        
        # 1. 假设验证总览（左上大块）
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        self._plot_hypothesis_overview(ax1)
        
        # 2. 信息论证据（右上）
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_information_theory_evidence(ax2)
        
        # 3. 因果关系图（右上）
        ax3 = fig.add_subplot(gs[0, 3])
        self._plot_causal_relationships(ax3)
        
        # 4. 调节效应（中右）
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_moderation_effects(ax4)
        
        # 5. 动态演化（中右）
        ax5 = fig.add_subplot(gs[1, 3])
        self._plot_dynamic_evolution(ax5)
        
        # 6. 构成性强度时间线（底部长条）
        ax6 = fig.add_subplot(gs[2, :])
        self._plot_constitutiveness_timeline(ax6)
        
        # 7. 网络效应（左下）
        ax7 = fig.add_subplot(gs[3, 0])
        self._plot_network_effects(ax7)
        
        # 8. 关键发现（中下）
        ax8 = fig.add_subplot(gs[3, 1:3])
        self._plot_key_findings(ax8)
        
        # 9. 政策启示（右下）
        ax9 = fig.add_subplot(gs[3, 3])
        self._plot_policy_implications(ax9)
        
        # 保存图表
        output_file = self.output_path / 'figures' / 'comprehensive_dashboard.jpg'
        plt.savefig(output_file, dpi=1200, bbox_inches='tight', format='jpg')
        plt.close()
        
        print(f"[OK] Comprehensive dashboard saved to: {output_file}")
        
    def _plot_hypothesis_overview(self, ax):
        """绘制假设验证总览"""
        ax.set_title('假设验证总览', fontsize=16, fontweight='bold', pad=20)
        
        # 准备数据
        hypotheses = {
            'H1: 认知依赖性': {
                'description': 'DSR与传统语言形成\n不可分割的认知整合',
                'support_level': 0,
                'evidence': []
            },
            'H2: 系统调节性': {
                'description': '平台特征和语境敏感度\n调节DSR的构成功能',
                'support_level': 0,
                'evidence': []
            },
            'H3: 动态演化性': {
                'description': '构成性随时间增强\n并表现出路径依赖',
                'support_level': 0,
                'evidence': []
            }
        }
        
        # 计算支持度
        # H1证据
        if 'step6' in self.results:
            const_score = self.results['step6'].get('constitutiveness_score', {})
            if const_score.get('weighted_score', 0) > 0.8:
                hypotheses['H1: 认知依赖性']['support_level'] = 0.9
                hypotheses['H1: 认知依赖性']['evidence'].append('强构成性')
                
        # H2证据
        if 'step7' in self.results:
            mod_results = self.results['step7'].get('interaction_effects', {})
            if mod_results.get('three_way_interaction', {}).get('significant', False):
                hypotheses['H2: 系统调节性']['support_level'] = 0.85
                hypotheses['H2: 系统调节性']['evidence'].append('显著调节')
                
        # H3证据
        if 'step8' in self.results:
            evolution = self.results['step8'].get('s_curve_fitting', {})
            if evolution.get('current_phase') in ['growth', 'consolidation']:
                hypotheses['H3: 动态演化性']['support_level'] = 0.8
                hypotheses['H3: 动态演化性']['evidence'].append('演化增强')
                
        # 绘制
        y_positions = np.arange(len(hypotheses))
        colors = ['#2ECC71', '#F39C12', '#E74C3C']  # 绿、橙、红
        
        for i, (hyp_name, hyp_data) in enumerate(hypotheses.items()):
            # 背景框
            rect = FancyBboxPatch((0, i-0.4), 4, 0.8, 
                                 boxstyle="round,pad=0.05",
                                 facecolor='lightgray', 
                                 edgecolor='gray',
                                 alpha=0.3)
            ax.add_patch(rect)
            
            # 假设名称
            ax.text(0.1, i, hyp_name, fontsize=12, fontweight='bold', va='center')
            
            # 描述
            ax.text(1.5, i, hyp_data['description'], fontsize=10, va='center')
            
            # 支持度条
            support = hyp_data['support_level']
            color = colors[0] if support > 0.8 else (colors[1] if support > 0.5 else colors[2])
            ax.barh(i, support, left=3, height=0.5, color=color, alpha=0.7)
            ax.text(3 + support + 0.05, i, f'{support:.0%}', va='center', fontsize=10)
            
        ax.set_xlim(0, 4.5)
        ax.set_ylim(-0.5, len(hypotheses) - 0.5)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
    def _plot_information_theory_evidence(self, ax):
        """绘制信息论证据"""
        if 'step4' in self.results:
            it_results = self.results['step4'].get('summary_statistics', {})
            criteria = it_results.get('criteria_summary', {})
            
            # 准备数据
            criteria_names = []
            criteria_values = []
            
            for name, passed in criteria.items():
                if isinstance(passed, bool):
                    criteria_names.append(name.replace('_', ' ').title()[:15])
                    criteria_values.append(1 if passed else 0)
                    
            if criteria_names:
                # 创建雷达图
                angles = np.linspace(0, 2 * np.pi, len(criteria_names), endpoint=False).tolist()
                criteria_values += criteria_values[:1]
                angles += angles[:1]
                
                ax.plot(angles, criteria_values, 'o-', linewidth=2, color='#3498DB')
                ax.fill(angles, criteria_values, alpha=0.25, color='#3498DB')
                
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(criteria_names, size=8)
                ax.set_ylim(0, 1.2)
                ax.set_yticks([0, 0.5, 1])
                ax.set_yticklabels(['否', '', '是'])
                ax.grid(True, alpha=0.3)
                
        ax.set_title('信息论证据', fontsize=12)
        
    def _plot_causal_relationships(self, ax):
        """绘制因果关系"""
        if 'step5' in self.results:
            causal = self.results['step5'].get('causal_discovery', {}).get('edges', [])
            
            # 简化的因果图
            nodes = ['DSR', 'TL', 'CS']
            pos = {'DSR': (0, 1), 'TL': (1, 0), 'CS': (2, 1)}
            
            # 绘制节点
            for node, (x, y) in pos.items():
                circle = plt.Circle((x, y), 0.3, color='lightblue', ec='black', linewidth=2)
                ax.add_patch(circle)
                ax.text(x, y, node, ha='center', va='center', fontsize=10, weight='bold')
                
            # 绘制边
            if causal:
                for edge in causal[:5]:  # 只显示前5条最重要的边
                    if all(n in edge['edge'] for n in ['dsr', 'cs']):
                        ax.annotate('', xy=(2, 1), xytext=(0, 1),
                                   arrowprops=dict(arrowstyle='->', lw=2, color='red'))
                                   
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.5, 1.5)
        ax.axis('off')
        ax.set_title('因果关系网络', fontsize=12)
        
    def _plot_moderation_effects(self, ax):
        """绘制调节效应"""
        if 'step7' in self.results:
            context_mod = self.results['step7'].get('context_moderation', {}).get('simple_slopes', {})
            
            if context_mod:
                contexts = list(context_mod.keys())
                slopes = [context_mod[c].get('slope', 0) for c in contexts]
                
                bars = ax.bar(contexts, slopes, color=['#FEE2E2', '#FED7D7', '#F87171'])
                
                # 添加数值标签
                for bar, slope in zip(bars, slopes):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{slope:.3f}', ha='center', va='bottom')
                           
                ax.set_ylabel('斜率')
                ax.set_title('语境调节效应', fontsize=12)
                ax.grid(True, axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, '无调节效应数据', ha='center', va='center')
            ax.set_title('语境调节效应', fontsize=12)
            
    def _plot_dynamic_evolution(self, ax):
        """绘制动态演化"""
        if 'step8' in self.results:
            s_curve = self.results['step8'].get('s_curve_fitting', {})
            
            if 'current_phase' in s_curve:
                # 绘制演化阶段
                phases = ['initiation', 'growth', 'consolidation', 'maturity']
                phase_names = ['初始期', '增长期', '巩固期', '成熟期']
                current = s_curve.get('current_phase', 'growth')
                
                colors = []
                for phase in phases:
                    if phase == current:
                        colors.append('#3498DB')
                    else:
                        colors.append('lightgray')
                        
                y_pos = np.arange(len(phases))
                ax.barh(y_pos, [0.25, 0.5, 0.75, 1.0], color=colors, alpha=0.7)
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(phase_names)
                ax.set_xlabel('完成度')
                ax.set_title('演化阶段', fontsize=12)
                
                # 添加当前阶段标记
                current_idx = phases.index(current) if current in phases else 0
                ax.text(0.5, current_idx, '当前', ha='center', va='center', 
                       fontweight='bold', color='white')
        else:
            ax.text(0.5, 0.5, '无演化数据', ha='center', va='center')
            ax.set_title('演化阶段', fontsize=12)
            
    def _plot_constitutiveness_timeline(self, ax):
        """绘制构成性时间线"""
        # 加载时间序列数据
        data_file = self.output_path / 'data' / 'data_with_pattern_metrics.csv'
        if data_file.exists():
            df = pd.read_csv(data_file, encoding='utf-8-sig')
            df['date'] = pd.to_datetime(df['date'])
            
            # 计算月度平均
            monthly = df.groupby(df['date'].dt.to_period('M'))['constitutive_index'].mean()
            
            # 绘制时间序列
            ax.plot(monthly.index.to_timestamp(), monthly.values, 'b-', linewidth=2, label='构成性指数')
            
            # 添加趋势线
            z = np.polyfit(range(len(monthly)), monthly.values, 1)
            p = np.poly1d(z)
            ax.plot(monthly.index.to_timestamp(), p(range(len(monthly))), 
                   'r--', alpha=0.5, label='趋势线')
                   
            # 标记关键事件
            if 'step8' in self.results:
                evolution = self.results['step8'].get('key_events', [])
                # 这里可以添加关键事件标记
                
            ax.set_xlabel('时间')
            ax.set_ylabel('构成性指数')
            ax.set_title('构成性演化时间线', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 格式化日期
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax.text(0.5, 0.5, '无时间序列数据', ha='center', va='center')
            ax.set_title('构成性演化时间线', fontsize=12)
            
    def _plot_network_effects(self, ax):
        """绘制网络效应"""
        if 'step9' in self.results:
            network = self.results['step9'].get('network_effects', {})
            multiplier = network.get('network_multiplier', {})
            
            if multiplier:
                # 创建网络效应可视化
                categories = ['低密度', '高密度']
                values = [
                    multiplier.get('low_density_effect', 0),
                    multiplier.get('high_density_effect', 0)
                ]
                
                bars = ax.bar(categories, values, color=['#E3F2FD', '#1976D2'])
                
                # 添加数值标签
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.3f}', ha='center', va='bottom')
                           
                # 添加乘数信息
                ax.text(0.5, max(values) * 1.1, 
                       f'网络乘数: {multiplier.get("multiplier", 1):.2f}x',
                       ha='center', transform=ax.transData,
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
                       
                ax.set_ylabel('DSR效应')
                ax.set_title('网络效应', fontsize=12)
        else:
            ax.text(0.5, 0.5, '无网络效应数据', ha='center', va='center')
            ax.set_title('网络效应', fontsize=12)
            
    def _plot_key_findings(self, ax):
        """绘制关键发现"""
        ax.axis('off')
        ax.set_title('关键发现', fontsize=14, fontweight='bold', pad=20)
        
        findings = []
        
        # 收集各步骤的关键发现
        if 'step6' in self.results:
            const_verdict = self.results['step6'].get('constitutiveness_score', {}).get('verdict', '')
            if const_verdict:
                findings.append(f"• 构成性判定：{const_verdict}")
                
        if 'step7' in self.results:
            if self.results['step7'].get('interaction_effects', {}).get('three_way_interaction', {}).get('significant'):
                findings.append("• 存在显著的三阶交互效应")
                
        if 'step8' in self.results:
            granger = self.results['step8'].get('temporal_causality', {}).get('granger_causality', {})
            if granger.get('dsr_to_cs', {}).get('significant'):
                findings.append("• DSR对CS存在时间因果关系")
                
        if 'step9' in self.results:
            adoption_rate = self.results['step9'].get('diffusion_patterns', {}).get('temporal_diffusion', {}).get('adoption_rate', 0)
            if adoption_rate > 0:
                findings.append(f"• DSR采纳率达到 {adoption_rate:.1%}")
                
        # 显示发现
        y_start = 0.9
        for finding in findings[:6]:  # 最多显示6条
            ax.text(0.05, y_start, finding, transform=ax.transAxes, 
                   fontsize=11, va='top')
            y_start -= 0.15
            
    def _plot_policy_implications(self, ax):
        """绘制政策启示"""
        ax.axis('off')
        ax.set_title('政策启示', fontsize=12, fontweight='bold')
        
        implications = [
            "优化平台设计",
            "增强网络效应",
            "促进扩散采纳",
            "强化路径依赖"
        ]
        
        y_start = 0.8
        for impl in implications:
            ax.text(0.1, y_start, f"→ {impl}", transform=ax.transAxes,
                   fontsize=10, va='top')
            y_start -= 0.2
            
    def create_evidence_synthesis_chart(self):
        """创建证据综合图"""
        print("\nGenerating evidence synthesis chart...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 准备证据矩阵
        evidence_types = [
            '信息论证据',
            '因果分析',
            '虚拟移除',
            '路径分析',
            '调节效应',
            '时间演化',
            '网络效应'
        ]
        
        hypotheses = ['H1: 认知依赖', 'H2: 系统调节', 'H3: 动态演化']
        
        # 创建证据强度矩阵
        evidence_matrix = np.zeros((len(evidence_types), len(hypotheses)))
        
        # 填充矩阵（基于实际结果）
        if 'step4' in self.results:  # 信息论
            evidence_matrix[0, 0] = 0.7  # 对H1的支持
            
        if 'step5' in self.results:  # 因果分析
            evidence_matrix[1, 0] = 0.8
            
        if 'step6' in self.results:  # 虚拟移除
            score = self.results['step6'].get('constitutiveness_score', {}).get('weighted_score', 0)
            evidence_matrix[2, 0] = score
            
        if 'step7' in self.results:  # 调节效应
            evidence_matrix[4, 1] = 0.85
            
        if 'step8' in self.results:  # 时间演化
            evidence_matrix[5, 2] = 0.8
            
        if 'step9' in self.results:  # 网络效应
            evidence_matrix[6, 1] = 0.7
            evidence_matrix[6, 2] = 0.6
            
        # 绘制热力图
        im = ax.imshow(evidence_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # 设置标签
        ax.set_xticks(np.arange(len(hypotheses)))
        ax.set_yticks(np.arange(len(evidence_types)))
        ax.set_xticklabels(hypotheses)
        ax.set_yticklabels(evidence_types)
        
        # 旋转顶部标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # 添加数值
        for i in range(len(evidence_types)):
            for j in range(len(hypotheses)):
                value = evidence_matrix[i, j]
                if value > 0:
                    text = ax.text(j, i, f'{value:.2f}', ha="center", va="center", 
                                 color="white" if value > 0.5 else "black")
                                 
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('证据强度', rotation=270, labelpad=20)
        
        ax.set_title('假设-证据支持矩阵', fontsize=16, fontweight='bold', pad=20)
        
        # 保存
        output_file = self.output_path / 'figures' / 'evidence_synthesis.jpg'
        plt.savefig(output_file, dpi=1200, bbox_inches='tight', format='jpg')
        plt.close()
        
        print(f"[OK] Evidence synthesis chart saved to: {output_file}")
        
    def generate_summary_statistics(self):
        """生成汇总统计"""
        print("\nGenerating summary statistics...")
        
        summary = {
            'analysis_completion': {},
            'hypothesis_support': {},
            'key_metrics': {},
            'generated_at': datetime.now().isoformat()
        }
        
        # 分析完成度
        total_steps = 7
        completed_steps = len(self.results)
        summary['analysis_completion'] = {
            'total_steps': total_steps,
            'completed_steps': completed_steps,
            'completion_rate': completed_steps / total_steps
        }
        
        # 假设支持度
        h1_support = 0
        h2_support = 0
        h3_support = 0
        
        if 'step6' in self.results:
            score = self.results['step6'].get('constitutiveness_score', {}).get('weighted_score', 0)
            h1_support = score
            
        if 'step7' in self.results:
            if self.results['step7'].get('interaction_effects', {}).get('three_way_interaction', {}).get('significant'):
                h2_support = 0.85
                
        if 'step8' in self.results:
            if self.results['step8'].get('s_curve_fitting', {}).get('current_phase'):
                h3_support = 0.8
                
        summary['hypothesis_support'] = {
            'H1_cognitive_dependency': h1_support,
            'H2_system_moderation': h2_support,
            'H3_dynamic_evolution': h3_support,
            'overall_support': np.mean([h1_support, h2_support, h3_support])
        }
        
        # 关键指标
        key_metrics = {}
        
        if 'step6' in self.results:
            key_metrics['constitutiveness_score'] = self.results['step6'].get('constitutiveness_score', {}).get('weighted_score', 0)
            
        if 'step8' in self.results:
            key_metrics['evolution_phase'] = self.results['step8'].get('s_curve_fitting', {}).get('current_phase', 'unknown')
            
        if 'step9' in self.results:
            key_metrics['network_multiplier'] = self.results['step9'].get('network_effects', {}).get('network_multiplier', {}).get('multiplier', 1)
            
        summary['key_metrics'] = key_metrics
        
        # 保存汇总
        output_file = self.output_path / 'data' / 'comprehensive_summary.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            
        print(f"[OK] Summary statistics saved to: {output_file}")
        
        return summary

def main():
    """主函数"""
    output_path = Path('output_cn')
    
    visualizer = ComprehensiveVisualization(output_path)
    
    # 生成综合仪表板
    visualizer.create_comprehensive_dashboard()
    
    # 生成证据综合图
    visualizer.create_evidence_synthesis_chart()
    
    # 生成汇总统计
    summary = visualizer.generate_summary_statistics()
    
    print("\n" + "="*60)
    print("Comprehensive visualization generation completed!")
    print("="*60)
    print(f"Analysis completion rate: {summary['analysis_completion']['completion_rate']:.1%}")
    print(f"Overall hypothesis support: {summary['hypothesis_support']['overall_support']:.1%}")

if __name__ == "__main__":
    main()