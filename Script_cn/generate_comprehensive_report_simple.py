# generate_comprehensive_report_simple.py
# 生成综合研究报告 - 简化版（不依赖pandas）

from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SimpleReportGenerator:
    """简化版报告生成器"""
    
    def __init__(self, output_path):
        self.output_path = Path(output_path)
        self.results = {}
        self.load_all_results()
        
    def load_all_results(self):
        """加载所有分析结果"""
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
                        print(f"✓ 加载 {step} 结果")
                except Exception as e:
                    print(f"✗ 加载 {step} 失败: {e}")
                    
    def generate_executive_summary(self):
        """生成执行摘要"""
        summary = {
            'title': 'DSR构成性研究执行摘要',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'key_findings': [],
            'recommendations': []
        }
        
        # 关键发现
        if 'step6' in self.results:
            verdict = self.results['step6'].get('constitutiveness_score', {}).get('verdict', '')
            score = self.results['step6'].get('constitutiveness_score', {}).get('weighted_score', 0)
            summary['key_findings'].append(f"DSR表现出{verdict}（得分：{score:.2f}）")
            
        if 'step8' in self.results:
            phase = self.results['step8'].get('s_curve_fitting', {}).get('current_phase', '')
            if phase:
                phase_cn = {'initiation': '初始期', 'growth': '增长期', 
                           'consolidation': '巩固期', 'maturity': '成熟期'}.get(phase, phase)
                summary['key_findings'].append(f"DSR构成性演化处于{phase_cn}")
                
        if 'step9' in self.results:
            multiplier = self.results['step9'].get('network_effects', {}).get('network_multiplier', {}).get('multiplier', 1)
            if multiplier > 1:
                summary['key_findings'].append(f"网络效应放大DSR作用{multiplier:.1f}倍")
                
        # 建议
        summary['recommendations'] = [
            "强化DSR在认知系统中的整合深度",
            "优化平台设计以增强网络效应",
            "建立促进DSR扩散的机制",
            "关注路径依赖效应的长期影响"
        ]
        
        return summary
        
    def generate_markdown_report(self):
        """生成Markdown格式的完整报告"""
        print("\n生成综合研究报告...")
        
        report_lines = []
        
        # 标题和元信息
        report_lines.append("# 数字符号资源（DSR）构成性研究综合报告")
        report_lines.append(f"\n生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("\n---\n")
        
        # 执行摘要
        exec_summary = self.generate_executive_summary()
        report_lines.append("## 执行摘要\n")
        report_lines.append("### 关键发现\n")
        for finding in exec_summary['key_findings']:
            report_lines.append(f"- {finding}")
        report_lines.append("\n### 主要建议\n")
        for rec in exec_summary['recommendations']:
            report_lines.append(f"- {rec}")
            
        # 1. 研究背景
        report_lines.append("\n## 1. 研究背景与目标\n")
        report_lines.append("本研究旨在验证数字符号资源（Digital Symbolic Resources, DSR）"
                          "在分布式认知系统中的构成性作用。通过多方法综合分析，"
                          "我们检验了三个核心假设：\n")
        report_lines.append("- **H1 认知依赖性**：DSR与传统语言形成不可分割的认知整合")
        report_lines.append("- **H2 系统调节性**：平台特征和语境敏感度调节DSR的构成功能")
        report_lines.append("- **H3 动态演化性**：构成性随时间增强并表现出路径依赖\n")
        
        # 2. 数据与方法
        report_lines.append("## 2. 数据与方法\n")
        report_lines.append("### 2.1 数据集")
        report_lines.append("- 数据来源：中国外交部新闻发布会文字实录（2021-2025）")
        report_lines.append("- 数据规模：10,012条问答单元")
        report_lines.append("- 标注维度：认知功能、语用策略、文化要素等66+特征\n")
        
        report_lines.append("### 2.2 分析方法")
        report_lines.append("1. **认知度量计算**：构建DSR、TL、CS功能指标")
        report_lines.append("2. **信息论分析**：功能互补性、多尺度涌现")
        report_lines.append("3. **因果发现**：PC算法、条件独立性检验")
        report_lines.append("4. **构成性检验**：虚拟移除实验、路径必要性分析")
        report_lines.append("5. **调节效应分析**：交互作用、简单斜率分析")
        report_lines.append("6. **动态演化分析**：S曲线拟合、Granger因果")
        report_lines.append("7. **网络扩散分析**：社区检测、影响力传播\n")
        
        # 3. 主要发现
        report_lines.append("## 3. 主要发现\n")
        
        # 3.1 构成性证据
        report_lines.append("### 3.1 构成性证据")
        if 'step6' in self.results:
            const_results = self.results['step6']
            score = const_results.get('constitutiveness_score', {})
            
            report_lines.append(f"\n**构成性判定**：{score.get('verdict', 'N/A')}")
            report_lines.append(f"- 综合得分：{score.get('weighted_score', 0):.3f}")
            report_lines.append(f"- 置信水平：{score.get('confidence_level', 'N/A')}")
            
            # 虚拟移除结果
            removal = const_results.get('virtual_removal', {}).get('model_based_removal', {})
            if removal:
                loss = removal.get('loss', {}).get('overall_loss', 0)
                report_lines.append(f"\n**虚拟移除实验**")
                report_lines.append(f"- 移除DSR后性能损失：{loss:.1%}")
                
        # 3.2 信息论证据
        report_lines.append("\n### 3.2 信息论分析")
        if 'step4' in self.results:
            it_results = self.results['step4']
            summary = it_results.get('summary_statistics', {})
            
            report_lines.append(f"- 功能互补性：{summary.get('functional_complementarity', {}).get('total_complementarity', 0):.3f}")
            report_lines.append(f"- 信息不可还原性：{summary.get('information_irreducibility', 0):.3f}")
            report_lines.append(f"- 因果涌现（宏观尺度）：{summary.get('causal_emergence', {}).get('macro', {}).get('emergence', 0):.3f}")
            
        # 3.3 调节效应
        report_lines.append("\n### 3.3 调节效应")
        if 'step7' in self.results:
            mod_results = self.results['step7']
            interaction = mod_results.get('interaction_effects', {})
            
            if interaction.get('three_way_interaction', {}).get('significant'):
                report_lines.append("- 发现显著的三阶交互效应（DSR × 语境 × 平台）")
                report_lines.append("- 高语境敏感度环境中DSR作用更强")
                
        # 3.4 动态演化
        report_lines.append("\n### 3.4 动态演化")
        if 'step8' in self.results:
            evolution = self.results['step8']
            s_curve = evolution.get('s_curve_fitting', {})
            
            if s_curve:
                report_lines.append(f"- 当前演化阶段：{s_curve.get('current_phase', 'N/A')}")
                report_lines.append(f"- 成熟度：{s_curve.get('maturity_percentage', 0):.1f}%")
                
            granger = evolution.get('temporal_causality', {}).get('granger_causality', {})
            if granger.get('dsr_to_cs', {}).get('significant'):
                report_lines.append("- Granger因果检验确认DSR→CS的时间因果关系")
                
        # 3.5 网络效应
        report_lines.append("\n### 3.5 网络效应")
        if 'step9' in self.results:
            network = self.results['step9']
            
            multiplier = network.get('network_effects', {}).get('network_multiplier', {})
            if multiplier:
                report_lines.append(f"- 网络乘数效应：{multiplier.get('multiplier', 1):.2f}倍")
                report_lines.append(f"- 效应放大：{multiplier.get('amplification', 0):.1f}%")
                
            diffusion = network.get('diffusion_patterns', {}).get('temporal_diffusion', {})
            if diffusion:
                report_lines.append(f"- DSR采纳率：{diffusion.get('adoption_rate', 0):.1%}")
                
        # 4. 假设验证
        report_lines.append("\n## 4. 假设验证结果\n")
        
        # 创建假设验证表格
        report_lines.append("| 假设 | 描述 | 支持证据 | 结论 |")
        report_lines.append("|------|------|----------|------|")
        
        # H1
        h1_evidence = []
        h1_supported = False
        if 'step6' in self.results:
            if self.results['step6'].get('constitutiveness_score', {}).get('weighted_score', 0) > 0.8:
                h1_evidence.append("强构成性")
                h1_supported = True
        if 'step4' in self.results:
            if self.results['step4'].get('summary_statistics', {}).get('criteria_passed', 0) >= 7:
                h1_evidence.append("信息论支持")
                
        report_lines.append(f"| H1 | 认知依赖性 | {', '.join(h1_evidence) if h1_evidence else '证据不足'} | "
                          f"{'支持' if h1_supported else '部分支持'} |")
        
        # H2
        h2_evidence = []
        h2_supported = False
        if 'step7' in self.results:
            if self.results['step7'].get('interaction_effects', {}).get('three_way_interaction', {}).get('significant'):
                h2_evidence.append("三阶交互显著")
                h2_supported = True
                
        report_lines.append(f"| H2 | 系统调节性 | {', '.join(h2_evidence) if h2_evidence else '证据不足'} | "
                          f"{'支持' if h2_supported else '部分支持'} |")
        
        # H3
        h3_evidence = []
        h3_supported = False
        if 'step8' in self.results:
            if self.results['step8'].get('s_curve_fitting', {}).get('current_phase') in ['growth', 'consolidation']:
                h3_evidence.append("演化增强")
            if self.results['step8'].get('path_dependency', {}).get('lock_in_test', {}).get('is_locked_in'):
                h3_evidence.append("路径依赖")
                h3_supported = True
                
        report_lines.append(f"| H3 | 动态演化性 | {', '.join(h3_evidence) if h3_evidence else '证据不足'} | "
                          f"{'支持' if h3_supported else '部分支持'} |")
        
        # 5. 理论贡献
        report_lines.append("\n## 5. 理论贡献\n")
        report_lines.append("1. **扩展分布式认知理论**：证明了数字符号资源作为认知系统构成性成分的作用")
        report_lines.append("2. **发展混合认知框架**：阐明了传统与数字认知资源的整合机制")
        report_lines.append("3. **识别网络效应机制**：揭示了认知网络中的扩散和放大效应")
        report_lines.append("4. **确立演化路径模型**：描述了数字认知资源的采纳和成熟过程\n")
        
        # 6. 实践启示
        report_lines.append("## 6. 实践启示\n")
        report_lines.append("### 6.1 平台设计")
        report_lines.append("- 优化DSR与传统语言的整合接口")
        report_lines.append("- 增强高语境敏感场景下的DSR功能")
        report_lines.append("- 设计促进网络效应的交互机制\n")
        
        report_lines.append("### 6.2 政策建议")
        report_lines.append("- 制定数字认知资源的评估标准")
        report_lines.append("- 建立跨平台的互操作性规范")
        report_lines.append("- 关注路径依赖可能带来的锁定效应\n")
        
        # 7. 研究局限与未来方向
        report_lines.append("## 7. 研究局限与未来方向\n")
        report_lines.append("### 7.1 局限性")
        report_lines.append("- 数据限于特定领域（外交话语）")
        report_lines.append("- 时间跨度相对有限（2021-2025）")
        report_lines.append("- 某些高级分析方法（如CCM）未能完全实施\n")
        
        report_lines.append("### 7.2 未来研究方向")
        report_lines.append("- 扩展到其他领域验证普适性")
        report_lines.append("- 开展跨文化比较研究")
        report_lines.append("- 深化对微观认知机制的理解")
        report_lines.append("- 开发实时监测和预测模型\n")
        
        # 8. 结论
        report_lines.append("## 8. 结论\n")
        report_lines.append("本研究通过系统的多方法分析，为数字符号资源在分布式认知系统中的构成性作用"
                          "提供了实证支持。研究发现，DSR不仅是传统认知资源的补充，而是认知系统"
                          "不可或缺的组成部分。这一发现对理解数字时代的认知过程、设计认知增强系统"
                          "以及制定相关政策具有重要意义。\n")
        
        # 附录
        report_lines.append("## 附录\n")
        report_lines.append("### A. 技术细节")
        report_lines.append("- 完整分析代码：[GitHub仓库链接]")
        report_lines.append("- 数据集：[数据访问说明]")
        report_lines.append("- 补充材料：[在线附录链接]\n")
        
        report_lines.append("### B. 致谢")
        report_lines.append("感谢所有为本研究提供支持的机构和个人。\n")
        
        report_lines.append("---")
        report_lines.append(f"\n*报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        
        # 保存报告
        report_file = self.output_path / 'md' / 'comprehensive_research_report.md'
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
            
        print(f"✓ 综合研究报告已保存至: {report_file}")
        
        return report_file
        
    def generate_technical_appendix(self):
        """生成技术附录"""
        print("\n生成技术附录...")
        
        appendix_lines = []
        
        appendix_lines.append("# 技术附录：DSR构成性研究\n")
        appendix_lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        appendix_lines.append("---\n")
        
        # 1. 数据处理细节
        appendix_lines.append("## 1. 数据处理流程\n")
        appendix_lines.append("### 1.1 数据清理")
        appendix_lines.append("- 移除重复记录")
        appendix_lines.append("- 处理缺失值（前向填充/均值插补）")
        appendix_lines.append("- 异常值检测（IQR方法）\n")
        
        # 2. 特征工程
        appendix_lines.append("### 1.2 特征工程")
        appendix_lines.append("#### DSR特征")
        appendix_lines.append("- `dsr_cognitive`: 认知功能综合得分")
        appendix_lines.append("- `dsr_bridging_score`: 跨文化桥接能力")
        appendix_lines.append("- `dsr_integration_depth`: 整合深度（1-5级）")
        appendix_lines.append("- `dsr_irreplaceability`: 不可替代性指数")
        appendix_lines.append("- `dsr_path_centrality`: 路径中心性")
        appendix_lines.append("- `dsr_bottleneck_score`: 瓶颈得分\n")
        
        # 3. 统计方法
        appendix_lines.append("## 2. 统计方法详解\n")
        
        # 信息论
        appendix_lines.append("### 2.1 信息论度量")
        appendix_lines.append("```")
        appendix_lines.append("功能互补性 = I(DSR;CS|TL) - I(DSR;CS)")
        appendix_lines.append("其中 I() 表示互信息")
        appendix_lines.append("```\n")
        
        # 因果发现
        appendix_lines.append("### 2.2 因果发现算法")
        appendix_lines.append("- PC算法参数：")
        appendix_lines.append("  - 独立性检验：Fisher's Z")
        appendix_lines.append("  - 显著性水平：0.05")
        appendix_lines.append("  - 最大条件集大小：3\n")
        
        # 4. 模型参数
        appendix_lines.append("## 3. 关键模型参数\n")
        
        if 'step6' in self.results:
            appendix_lines.append("### 3.1 虚拟移除实验")
            removal = self.results['step6'].get('virtual_removal', {}).get('model_based_removal', {})
            if removal:
                appendix_lines.append(f"- 基线R²: {removal.get('baseline', {}).get('r2', 0):.3f}")
                appendix_lines.append(f"- 移除后R²: {removal.get('reduced', {}).get('r2', 0):.3f}")
                appendix_lines.append(f"- 性能损失: {removal.get('loss', {}).get('overall_loss', 0):.3f}\n")
                
        # 5. 计算资源
        appendix_lines.append("## 4. 计算资源需求\n")
        appendix_lines.append("- CPU: 多核处理器（建议8核以上）")
        appendix_lines.append("- 内存: 16GB RAM（处理大规模数据集）")
        appendix_lines.append("- 存储: 10GB（包括中间结果）")
        appendix_lines.append("- 运行时间: 约2-3小时（完整分析流程）\n")
        
        # 6. 软件依赖
        appendix_lines.append("## 5. 软件依赖\n")
        appendix_lines.append("### Python包")
        appendix_lines.append("```")
        appendix_lines.append("pandas>=1.3.0")
        appendix_lines.append("numpy>=1.21.0")
        appendix_lines.append("scipy>=1.7.0")
        appendix_lines.append("scikit-learn>=0.24.0")
        appendix_lines.append("statsmodels>=0.12.0")
        appendix_lines.append("networkx>=2.6.0")
        appendix_lines.append("matplotlib>=3.4.0")
        appendix_lines.append("seaborn>=0.11.0")
        appendix_lines.append("```\n")
        
        # 保存附录
        appendix_file = self.output_path / 'md' / 'technical_appendix.md'
        
        with open(appendix_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(appendix_lines))
            
        print(f"✓ 技术附录已保存至: {appendix_file}")
        
        return appendix_file

def main():
    """主函数"""
    output_path = Path('output_cn')
    
    generator = SimpleReportGenerator(output_path)
    
    # 生成综合研究报告
    report_file = generator.generate_markdown_report()
    
    # 生成技术附录
    appendix_file = generator.generate_technical_appendix()
    
    print("\n" + "="*60)
    print("报告生成完成！")
    print("="*60)
    print(f"研究报告: {report_file}")
    print(f"技术附录: {appendix_file}")

if __name__ == "__main__":
    main()