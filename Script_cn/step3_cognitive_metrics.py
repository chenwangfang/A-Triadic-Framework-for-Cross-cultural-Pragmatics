# step3_cognitive_metrics.py
# 第三步：认知功能指标计算

import pandas as pd
import numpy as np
from pathlib import Path
import json
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class CognitiveMetricsCalculator:
    """认知功能指标计算器"""
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.df = None
        self.metrics_results = {
            'dsr_metrics': {},
            'tl_metrics': {},
            'cs_metrics': {},
            'composite_scores': {}
        }
        
    def load_data(self):
        """加载提取的数据"""
        csv_file = self.data_path / 'extracted_data.csv'
        print(f"加载数据: {csv_file}")
        
        self.df = pd.read_csv(csv_file, encoding='utf-8-sig')
        
        # 确保日期字段被正确解析
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        print(f"成功加载 {len(self.df)} 条记录")
        print(f"日期范围: {self.df['date'].min()} 至 {self.df['date'].max()}")
        
        return self.df
        
    def calculate_all_metrics(self):
        """计算所有认知功能指标"""
        print("\n开始计算认知功能指标...")
        
        # 1. 计算DSR（数字符号资源）认知功能指标
        self.calculate_dsr_metrics()
        
        # 2. 计算传统语言功能指标
        self.calculate_tl_metrics()
        
        # 3. 计算认知系统输出指标
        self.calculate_cs_metrics()
        
        # 4. 计算综合指标
        self.calculate_composite_scores()
        
        # 5. 保存结果
        self.save_metrics()
        
        return self.df
        
    def calculate_dsr_metrics(self):
        """计算数字符号资源认知功能指标"""
        print("\n1. 计算DSR认知功能指标...")
        
        # 1.1 认知桥接分数
        self.df['dsr_bridging_score'] = self._calculate_bridging_score()
        
        # 1.2 整合深度分数
        self.df['dsr_integration_depth'] = self._calculate_integration_depth()
        
        # 1.3 不可替代性分数
        self.df['dsr_irreplaceability'] = self._calculate_irreplaceability()
        
        # 1.4 路径中心性（基于系统耦合度）
        self.df['dsr_path_centrality'] = self._calculate_path_centrality()
        
        # 1.5 瓶颈分数
        self.df['dsr_bottleneck_score'] = self._calculate_bottleneck_score()
        
        # 1.6 级联影响
        self.df['dsr_cascade_impact'] = self._calculate_cascade_impact()
        
        # DSR综合认知功能指标
        dsr_components = [
            'dsr_bridging_score', 'dsr_integration_depth', 'dsr_irreplaceability',
            'dsr_path_centrality', 'dsr_bottleneck_score', 'dsr_cascade_impact'
        ]
        
        # 加权平均（核心三项权重更高）
        weights = [0.25, 0.25, 0.25, 0.1, 0.1, 0.05]
        self.df['dsr_cognitive'] = np.average(
            self.df[dsr_components].values, 
            weights=weights, 
            axis=1
        )
        
        # 统计信息
        self.metrics_results['dsr_metrics'] = {
            'mean': self.df['dsr_cognitive'].mean(),
            'std': self.df['dsr_cognitive'].std(),
            'components': {col: self.df[col].mean() for col in dsr_components}
        }
        
        print(f"  DSR认知功能均值: {self.df['dsr_cognitive'].mean():.4f}")
        
    def _calculate_bridging_score(self):
        """计算认知桥接分数（根据ACCT框架更新）"""
        # 基于DED功能类型分析
        def get_bridging_score(row):
            if pd.isna(row['ded_functions']) or row['ded_functions'] == '':
                return 0
                
            functions = row['ded_functions'].split('|')
            
            # 跨文化概念桥接（权重0.4）
            cross_cultural_functions = ['bridging', 'contextualizing', 'translating', 'mapping']
            cross_cultural = sum(1 for f in functions if f in cross_cultural_functions)
            cross_cultural_score = cross_cultural / max(len(functions), 1)
            
            # 抽象-具体转换（权重0.3）
            # 基于数字隐喻、平台术语和视觉符号
            abstract_concrete = (
                row['social_media_metaphor_count'] + 
                row['platform_terminology_count'] +
                row.get('visual_symbols_count', 0)
            ) / 15  # 调整归一化因子
            abstract_concrete_score = min(abstract_concrete, 1)
            
            # 时空连接（权重0.3）
            # 基于功能多样性和跨时空引用
            temporal_functions = ['referencing', 'comparing', 'historicizing']
            temporal_count = sum(1 for f in functions if f in temporal_functions)
            function_diversity = len(set(functions)) / 8  # 增加到8种功能类型
            temporal_linking_score = 0.5 * (temporal_count / max(len(functions), 1)) + 0.5 * function_diversity
            temporal_linking_score = min(temporal_linking_score, 1)
            
            # 根据ACCT框架的权重计算
            weights = [0.4, 0.3, 0.3]
            scores = [cross_cultural_score, abstract_concrete_score, temporal_linking_score]
            
            return np.average(scores, weights=weights)
            
        return self.df.apply(get_bridging_score, axis=1)
        
    def _calculate_integration_depth(self):
        """计算整合深度分数（基于ACCT框架的5级分类）"""
        # 基于DED平均深度和激活强度
        def get_integration_score(row):
            # 获取原始深度值（1-3映射到1-5级）
            raw_depth = row['ded_avg_depth']
            
            # 映射到5级深度系统
            if raw_depth <= 1.2:
                depth_level = 1  # Surface (表层)
            elif raw_depth <= 1.8:
                depth_level = 2  # Surface-Functional transition
            elif raw_depth <= 2.2:
                depth_level = 3  # Functional (功能性)
            elif raw_depth <= 2.6:
                depth_level = 4  # Structural (结构性)
            else:
                depth_level = 5  # Systemic (系统性)
            
            # 根据ACCT框架目标分布调整分数
            # Surface (1-2): 40-50% → 低分
            # Functional (3): 25-35% → 中分
            # Structural (4): 15-20% → 高分
            # Systemic (5): 5-10% → 最高分
            depth_weights = {1: 0.2, 2: 0.35, 3: 0.55, 4: 0.75, 5: 0.95}
            depth_score = depth_weights.get(depth_level, 0.5)
            
            # 激活强度分数（保持原有逻辑）
            intensity_score = min(row['ded_avg_intensity'] / 5, 1)
            
            # 密度因素（调整为更合理的范围）
            density_score = min(row['ded_density'] / 3, 1)  # 每百字3个为上限
            
            # 功能互补性因素（新增）
            complementarity_score = 0
            if 'ded_functions' in row and pd.notna(row['ded_functions']):
                functions = row['ded_functions'].split('|')
                # 检查是否有互补性功能
                complementary_functions = ['augmenting', 'enhancing', 'complementing', 'integrating']
                complementarity_score = sum(1 for f in functions if f in complementary_functions) / 4
            
            # 综合计算（调整权重以反映新框架）
            return (0.35 * depth_score + 
                   0.25 * intensity_score + 
                   0.20 * density_score + 
                   0.20 * complementarity_score)
            
        return self.df.apply(get_integration_score, axis=1)
        
    def _calculate_irreplaceability(self):
        """计算不可替代性"""
        def get_irreplaceability(row):
            # 独特功能（基于DED数量和功能多样性）
            if pd.isna(row['ded_functions']) or row['ded_functions'] == '':
                unique_functions = 0
            else:
                functions = row['ded_functions'].split('|')
                unique_functions = len(set(functions)) / 5  # 归一化
                
            # 替代成本（基于DED密度和响应长度比）
            substitution_cost = row['ded_density'] / 10  # 归一化
            
            # 性能损失（基于认知适应成功度和DED贡献）
            performance_loss = (row['cognitive_adaptation_success'] / 5) * \
                             (row['ded_count'] / max(row['ded_count'] + 1, 1))
            
            # 综合计算
            return (unique_functions + substitution_cost + performance_loss) / 3
            
        return self.df.apply(get_irreplaceability, axis=1)
        
    def _calculate_path_centrality(self):
        """计算路径中心性（基于ACCT框架的中介中心性概念）"""
        # 基于系统耦合强度和认知-语用方向
        def get_centrality(row):
            # 系统耦合强度（反映DSR在认知网络中的连接强度）
            coupling_score = row['system_coupling_strength'] / 5
            
            # 中介作用（DSR在TL和CS之间的桥梁作用）
            # 认知负荷越高，DSR的中介作用越重要
            mediation_score = row['cognitive_load_index'] / 5
            
            # 信息流控制（基于策略转换和DED密度）
            flow_control = min(row['strategy_transition_count'] / 5, 1) * 0.5 + \
                          min(row['ded_density'] / 3, 1) * 0.5
            
            # 功能多样性（DED功能种类越多，路径越重要）
            function_diversity = 0
            if pd.notna(row['ded_functions']) and row['ded_functions']:
                functions = row['ded_functions'].split('|')
                function_diversity = len(set(functions)) / 8  # 假设最多8种功能
            
            # 根据ACCT框架计算中介中心性
            # 强调DSR在认知网络中的不可或缺性
            return (0.35 * coupling_score + 
                   0.30 * mediation_score + 
                   0.20 * flow_control + 
                   0.15 * function_diversity)
            
        return self.df.apply(get_centrality, axis=1)
        
    def _calculate_bottleneck_score(self):
        """计算瓶颈分数"""
        # 基于DED的关键程度
        def get_bottleneck(row):
            # DED最大强度（关键节点）
            max_intensity = row['ded_max_intensity'] / 5
            
            # 情感触发（瓶颈效应）
            emotional_factor = min(row['emotional_triggers_count'] / 3, 1)
            
            # Meme潜力（传播瓶颈）
            meme_factor = row['meme_potential_score'] / 5
            
            return 0.4 * max_intensity + 0.3 * emotional_factor + 0.3 * meme_factor
            
        return self.df.apply(get_bottleneck, axis=1)
        
    def _calculate_cascade_impact(self):
        """计算级联影响"""
        # 基于DED位置和下游效应
        def get_cascade(row):
            # 病毒传播潜力
            viral_factor = row['viral_potential_high'] / max(row['soundbite_count'], 1) if row['soundbite_count'] > 0 else 0
            
            # 数字强度的整体影响
            digital_impact = row['digital_intensity_overall']
            
            # 策略密度（影响范围）
            strategy_impact = row['strategy_density'] / 10
            
            return 0.4 * viral_factor + 0.3 * digital_impact + 0.3 * strategy_impact
            
        return self.df.apply(get_cascade, axis=1)
        
    def calculate_tl_metrics(self):
        """计算传统语言功能指标（基于ACCT框架增强）"""
        print("\n2. 计算传统语言功能指标...")
        
        # 2.1 常规表达密度（包含外交套语、成语典故、政策术语）
        self.df['tl_conventional_density'] = self._calculate_conventional_density()
        
        # 2.2 结构复杂度（句法深度、词汇复杂度、语篇衔接）
        self.df['tl_structural_complexity'] = self._calculate_structural_complexity()
        
        # 2.3 语用丰富度
        self.df['tl_pragmatic_richness'] = self._calculate_pragmatic_richness()
        
        # 2.4 与DSR的交互（新增）
        self.df['tl_dsr_interaction'] = self._calculate_tl_dsr_interaction()
        
        # TL综合功能指标（更新权重）
        tl_components = [
            'tl_conventional_density', 'tl_structural_complexity', 
            'tl_pragmatic_richness', 'tl_dsr_interaction'
        ]
        
        # 根据ACCT框架调整权重
        weights = [0.3, 0.25, 0.25, 0.2]
        self.df['tl_functional'] = np.average(
            self.df[tl_components].values, 
            weights=weights, 
            axis=1
        )
        
        # 统计信息
        self.metrics_results['tl_metrics'] = {
            'mean': self.df['tl_functional'].mean(),
            'std': self.df['tl_functional'].std(),
            'components': {col: self.df[col].mean() for col in tl_components}
        }
        
        print(f"  TL功能指标均值: {self.df['tl_functional'].mean():.4f}")
        
    def _calculate_conventional_density(self):
        """计算常规表达密度"""
        def get_conventional_density(row):
            # 基于文化图式密度
            traditional_density = row['schema_density_traditional'] / 10
            
            # 立场策略（外交套语）
            diplomatic_density = (row['principle_restatement_freq'] + 
                                row['information_restriction_freq']) / 20
            
            # 关系建设密度
            relationship_density = row['relationship_building_frequency'] / 10
            
            return (traditional_density + diplomatic_density + relationship_density) / 3
            
        return self.df.apply(get_conventional_density, axis=1)
        
    def _calculate_structural_complexity(self):
        """计算结构复杂度"""
        def get_complexity(row):
            # 响应长度因素
            length_factor = min(row['response_length'] / 1000, 1)
            
            # 响应/问题长度比
            ratio_factor = min(row['response_question_ratio'] / 5, 1)
            
            # 策略多样性
            diversity_factor = row['strategy_diversity'] / 5
            
            return 0.3 * length_factor + 0.3 * ratio_factor + 0.4 * diversity_factor
            
        return self.df.apply(get_complexity, axis=1)
        
    def _calculate_pragmatic_richness(self):
        """计算语用丰富度"""
        def get_richness(row):
            # 策略多样性指数
            diversity_score = row['strategy_diversity_index']
            
            # 语境适应度
            context_score = row['contextual_adaptation'] / 5
            
            # 关系建设强度
            relationship_score = row['relationship_building_intensity'] / 5
            
            return 0.4 * diversity_score + 0.3 * context_score + 0.3 * relationship_score
            
        return self.df.apply(get_richness, axis=1)
    
    def _calculate_tl_dsr_interaction(self):
        """计算TL与DSR的交互指标（ACCT框架新增）"""
        def get_interaction_score(row):
            # DSR影响的TL使用（基于DED密度和传统图式的共现）
            dsr_influenced = 0
            if row['ded_density'] > 0 and row['schema_density_traditional'] > 0:
                # DSR和传统元素的共现程度
                co_occurrence = min(row['ded_density'] * row['schema_density_traditional'] / 25, 1)
                dsr_influenced = co_occurrence
            
            # 互补使用模式（基于策略多样性和DED功能）
            complementary_use = 0
            if row['strategy_diversity'] > 3 and row['ded_count'] > 0:
                # 策略多样性高且有DED使用表明互补
                complementary_use = min(row['strategy_diversity'] / 5 * row['ded_count'] / 5, 1)
            
            # 序列模式（基于转换频率和响应结构）
            sequential_patterns = 0
            if row['strategy_transition_count'] > 0:
                # 策略转换暗示TL-DSR序列交替
                sequential_patterns = min(row['strategy_transition_count'] / 5, 1)
            
            # 功能增强（DSR增强TL表达的程度）
            functional_enhancement = 0
            if row['ded_avg_intensity'] > 0 and row['relationship_building_frequency'] > 0:
                # DED强度和关系建设的协同
                functional_enhancement = min(
                    row['ded_avg_intensity'] / 5 * row['relationship_building_frequency'] / 10, 
                    1
                )
            
            # 综合计算TL-DSR交互分数
            return (0.25 * dsr_influenced + 
                   0.25 * complementary_use + 
                   0.25 * sequential_patterns + 
                   0.25 * functional_enhancement)
            
        return self.df.apply(get_interaction_score, axis=1)
        
    def calculate_cs_metrics(self):
        """计算认知系统输出指标（基于ACCT框架增强）"""
        print("\n3. 计算认知系统输出指标...")
        
        # 3.1 即时效果（理解准确性、认知效率、语用成功）
        self.df['cs_immediate_effects'] = self._calculate_immediate_effects()
        
        # 3.2 系统特性（适应性、稳定性、整合水平）
        self.df['cs_system_properties'] = self._calculate_system_properties()
        
        # 3.3 涌现特征（多尺度复杂性）
        self.df['cs_emergent_features'] = self._calculate_emergent_features()
        
        # 3.4 适应性指标（ACCT新增）
        self.df['cs_adaptability'] = self._calculate_adaptability()
        
        # 3.5 稳定性指标（ACCT新增）
        self.df['cs_stability'] = self._calculate_stability()
        
        # 3.6 整合水平（ACCT新增）
        self.df['cs_integration_level'] = self._calculate_integration_level()
        
        # CS综合输出指标（更新组件和权重）
        cs_components = [
            'cs_immediate_effects', 'cs_system_properties', 'cs_emergent_features',
            'cs_adaptability', 'cs_stability', 'cs_integration_level'
        ]
        
        # 根据ACCT框架调整权重
        weights = [0.25, 0.20, 0.20, 0.15, 0.10, 0.10]
        self.df['cs_output'] = np.average(
            self.df[cs_components].values, 
            weights=weights, 
            axis=1
        )
        
        # 统计信息
        self.metrics_results['cs_metrics'] = {
            'mean': self.df['cs_output'].mean(),
            'std': self.df['cs_output'].std(),
            'components': {col: self.df[col].mean() for col in cs_components}
        }
        
        print(f"  CS输出指标均值: {self.df['cs_output'].mean():.4f}")
        
    def _calculate_immediate_effects(self):
        """计算即时效果"""
        def get_immediate_effects(row):
            # 认知适应成功度
            adaptation_success = row['cognitive_adaptation_success'] / 5
            
            # 认知效率（负荷越低效率越高）
            cognitive_efficiency = 1 - (row['cognitive_load_index'] / 5)
            
            # 策略效果
            strategic_effect = row['strategic_effectiveness'] / 5
            
            return 0.4 * adaptation_success + 0.3 * cognitive_efficiency + 0.3 * strategic_effect
            
        return self.df.apply(get_immediate_effects, axis=1)
        
    def _calculate_system_properties(self):
        """计算系统特性"""
        def get_system_properties(row):
            # 灵活性（转换平滑度）
            flexibility = row['transition_smoothness'] / 5
            
            # 鲁棒性（稳定性）
            robustness = 1 if row['coupling_stability'] == 'stable' else 0.5
            
            # 创新性（创新图式比例）
            innovation = row['innovative_schemas_count'] / max(row['cultural_schemas_total'], 1)
            
            return 0.4 * flexibility + 0.3 * robustness + 0.3 * innovation
            
        return self.df.apply(get_system_properties, axis=1)
        
    def _calculate_emergent_features(self):
        """计算涌现特征"""
        def get_emergent_features(row):
            # 协同效应（DED与TL的结合）
            synergy = 1 if (row['ded_count'] > 0 and row['schema_density_total'] > 5) else 0.5
            
            # 复杂度增益
            complexity_gain = row['avg_activation_intensity'] / 5
            
            # 文化共鸣
            cultural_resonance = row['cultural_resonance'] / 5
            
            return 0.4 * synergy + 0.3 * complexity_gain + 0.3 * cultural_resonance
            
        return self.df.apply(get_emergent_features, axis=1)
    
    def _calculate_adaptability(self):
        """计算适应性指标（ACCT框架）"""
        def get_adaptability(row):
            # 策略多样性指数
            strategy_diversity = row['strategy_diversity_index']
            
            # 转换灵活性
            transition_flexibility = min(row['strategy_transition_count'] / 5, 1)
            
            # 语境响应能力
            context_responsiveness = row['contextual_adaptation'] / 5
            
            # 创新性使用
            innovation = 0
            if row['ded_count'] > 0 and row['digital_intensity_overall'] > 0:
                innovation = min(row['digital_intensity_overall'] * row['ded_count'] / 20, 1)
            
            return (0.3 * strategy_diversity + 
                   0.3 * transition_flexibility + 
                   0.2 * context_responsiveness + 
                   0.2 * innovation)
        
        return self.df.apply(get_adaptability, axis=1)
    
    def _calculate_stability(self):
        """计算稳定性指标（ACCT框架）"""
        def get_stability(row):
            # 基于认知负荷的稳定性（负荷越低越稳定）
            load_stability = 1 - (row['cognitive_load_index'] / 5)
            
            # 策略一致性（转换越少越稳定）
            strategy_consistency = 1 - min(row['strategy_transition_count'] / 10, 1)
            
            # 系统耦合的稳定性
            coupling_stability = row['system_coupling_strength'] / 5
            
            # 性能稳定性（基于认知适应成功度）
            performance_stability = row['cognitive_adaptation_success'] / 5
            
            return (0.25 * load_stability + 
                   0.25 * strategy_consistency + 
                   0.25 * coupling_stability + 
                   0.25 * performance_stability)
        
        return self.df.apply(get_stability, axis=1)
    
    def _calculate_integration_level(self):
        """计算整合水平（ACCT框架）"""
        def get_integration(row):
            # DSR-TL整合程度
            dsr_tl_integration = 0
            if row['ded_density'] > 0 and row['schema_density_total'] > 0:
                dsr_tl_integration = min(
                    row['ded_density'] * row['schema_density_total'] / 50, 
                    1
                )
            
            # 系统耦合强度
            system_coupling = row['system_coupling_strength'] / 5
            
            # 功能协同程度（基于策略效果和认知适应）
            functional_synergy = (row['strategic_effectiveness'] / 5 + 
                                row['cognitive_adaptation_success'] / 5) / 2
            
            # 语义整合（基于响应长度比和内容丰富度）
            semantic_integration = min(row['response_question_ratio'] / 5, 1)
            
            return (0.3 * dsr_tl_integration + 
                   0.3 * system_coupling + 
                   0.2 * functional_synergy + 
                   0.2 * semantic_integration)
        
        return self.df.apply(get_integration, axis=1)
        
    def calculate_composite_scores(self):
        """计算综合指标"""
        print("\n4. 计算综合指标...")
        
        # 计算交互项
        self.df['dsr_tl_interaction'] = self.df['dsr_cognitive'] * self.df['tl_functional']
        
        # 计算二次项
        self.df['dsr_cognitive_sq'] = self.df['dsr_cognitive'] ** 2
        self.df['tl_functional_sq'] = self.df['tl_functional'] ** 2
        
        # 语境调节因子
        self.df['context_moderation'] = self.df['sensitivity_code'] * self.df['dsr_cognitive']
        
        # 平台调节因子
        self.df['platform_moderation'] = self.df['media_culture_code'] * self.df['dsr_cognitive']
        
        # 时间趋势（天数）- 确保日期是datetime类型
        min_date = self.df['date'].min()
        self.df['time_trend'] = (self.df['date'] - min_date).dt.days
        
        # 构成性指数（基于ACCT框架的综合评估）
        # 权重分配：DSR认知功能(0.4) + 不可替代性(0.2) + 路径中心性(0.2) + CS输出(0.2)
        self.df['constitutive_index'] = (
            0.4 * self.df['dsr_cognitive'] +
            0.2 * self.df['dsr_irreplaceability'] +
            0.2 * self.df['dsr_path_centrality'] +
            0.2 * self.df['cs_output']
        )
        
        # 功能互补性分数（新增）
        self.df['functional_complementarity'] = self._calculate_functional_complementarity()
        
        self.metrics_results['composite_scores'] = {
            'constitutive_index_mean': self.df['constitutive_index'].mean(),
            'constitutive_index_std': self.df['constitutive_index'].std(),
            'dsr_tl_correlation': self.df['dsr_cognitive'].corr(self.df['tl_functional']),
            'dsr_cs_correlation': self.df['dsr_cognitive'].corr(self.df['cs_output'])
        }
        
        print(f"  构成性指数均值: {self.df['constitutive_index'].mean():.4f}")
        print(f"  DSR-TL相关性: {self.df['dsr_cognitive'].corr(self.df['tl_functional']):.4f}")
        print(f"  DSR-CS相关性: {self.df['dsr_cognitive'].corr(self.df['cs_output']):.4f}")
        
    def _calculate_functional_complementarity(self):
        """计算功能互补性（ACCT框架核心指标）"""
        def get_complementarity(row):
            # DSR特有功能分数
            dsr_unique = 0
            if pd.notna(row['ded_functions']) and row['ded_functions']:
                functions = row['ded_functions'].split('|')
                unique_functions = ['visualizing', 'real-time', 'interactive', 'multimodal']
                dsr_unique = sum(1 for f in functions if f in unique_functions) / 4
            
            # DSR-TL协同模式
            synergistic_patterns = 0
            if row['dsr_cognitive'] > 0.3 and row['tl_functional'] > 0.3:
                # 两者都有中等以上功能时产生协同
                synergistic_patterns = min(
                    row['dsr_cognitive'] * row['tl_functional'] * 2, 
                    1
                )
            
            # 中介作用（基于路径中心性和TL-DSR交互）
            mediation_role = row['dsr_path_centrality'] * 0.5 + row.get('tl_dsr_interaction', 0) * 0.5
            
            # 功能增值（DSR带来的额外认知价值）
            functional_gain = 0
            if row['cs_output'] > row['tl_functional']:
                # CS输出超过TL功能的部分归因于DSR
                functional_gain = min((row['cs_output'] - row['tl_functional']) * 2, 1)
            
            # 综合计算功能互补性
            return (0.25 * dsr_unique + 
                   0.25 * synergistic_patterns + 
                   0.25 * mediation_role + 
                   0.25 * functional_gain)
        
        return self.df.apply(get_complementarity, axis=1)
        
    def save_metrics(self):
        """保存计算结果"""
        # 保存增强后的数据
        output_file = self.data_path / 'data_with_metrics.csv'
        self.df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n数据已保存至: {output_file}")
        
        # 保存指标统计
        stats_file = self.data_path / 'metrics_statistics.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_results, f, ensure_ascii=False, indent=2)
            
        # 生成指标报告
        self.generate_metrics_report()
        
    def generate_metrics_report(self):
        """生成指标计算报告"""
        print("\n" + "="*60)
        print("认知功能指标计算报告")
        print("="*60)
        
        # DSR指标
        print("\n1. DSR（数字符号资源）认知功能指标:")
        for metric, value in self.metrics_results['dsr_metrics']['components'].items():
            print(f"   {metric}: {value:.4f}")
        print(f"   综合DSR认知功能: {self.metrics_results['dsr_metrics']['mean']:.4f} (±{self.metrics_results['dsr_metrics']['std']:.4f})")
        
        # TL指标
        print("\n2. 传统语言功能指标:")
        for metric, value in self.metrics_results['tl_metrics']['components'].items():
            print(f"   {metric}: {value:.4f}")
        print(f"   综合TL功能: {self.metrics_results['tl_metrics']['mean']:.4f} (±{self.metrics_results['tl_metrics']['std']:.4f})")
        
        # CS指标
        print("\n3. 认知系统输出指标:")
        for metric, value in self.metrics_results['cs_metrics']['components'].items():
            print(f"   {metric}: {value:.4f}")
        print(f"   综合CS输出: {self.metrics_results['cs_metrics']['mean']:.4f} (±{self.metrics_results['cs_metrics']['std']:.4f})")
        
        # 综合指标
        print("\n4. 综合评估:")
        print(f"   构成性指数: {self.metrics_results['composite_scores']['constitutive_index_mean']:.4f} (±{self.metrics_results['composite_scores']['constitutive_index_std']:.4f})")
        print(f"   DSR-TL相关性: {self.metrics_results['composite_scores']['dsr_tl_correlation']:.4f}")
        print(f"   DSR-CS相关性: {self.metrics_results['composite_scores']['dsr_cs_correlation']:.4f}")
        
        # 数据质量
        print(f"\n5. 数据概况:")
        print(f"   总记录数: {len(self.df)}")
        print(f"   计算的指标数: {len([col for col in self.df.columns if col.startswith(('dsr_', 'tl_', 'cs_'))])}")
        print(f"   时间跨度: {self.df['date'].min()} 至 {self.df['date'].max()}")

def main():
    """主函数"""
    # 设置数据路径
    data_path = Path('../output_cn/data')
    
    # 创建计算器
    calculator = CognitiveMetricsCalculator(data_path)
    
    # 加载数据
    df = calculator.load_data()
    
    # 计算所有指标
    df_with_metrics = calculator.calculate_all_metrics()
    
    print("\n✓ 认知功能指标计算完成！")

if __name__ == "__main__":
    main()