# step9_network_diffusion_analysis.py
# 第九步：网络效应与扩散分析 - 分析DSR在认知网络中的扩散模式和网络效应

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 统计分析
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cosine
import statsmodels.api as sm

# 网络分析
import networkx as nx
from networkx.algorithms import community
from networkx.algorithms.centrality import eigenvector_centrality, betweenness_centrality
from networkx.algorithms.link_analysis import pagerank

# 机器学习
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class NetworkDiffusionAnalysis:
    """网络效应与扩散分析类"""
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.df = None
        self.results = {
            'cognitive_network': {},
            'diffusion_patterns': {},
            'network_effects': {},
            'influence_propagation': {},
            'community_structure': {},
            'critical_nodes': {}
        }
        
    def load_data(self):
        """加载数据"""
        # 尝试多个可能的文件位置
        possible_files = [
            self.data_path / 'data_with_pattern_metrics.csv',
            self.data_path / 'data_with_metrics.csv',
            self.data_path / 'mixed_methods_data.csv',
            self.data_path / 'data_processed.csv',
            # 尝试父目录
            self.data_path.parent / 'data_with_pattern_metrics.csv',
            self.data_path.parent / 'data_with_metrics.csv',
            # 尝试绝对路径
            Path('G:/Project/Figure/output_cn/data/data_with_pattern_metrics.csv'),
            Path('G:/Project/Figure/output_cn/data/data_with_metrics.csv'),
            Path('G:/Project/Figure/output_cn/data/mixed_methods_data.csv')
        ]
        
        loaded = False
        for file_path in possible_files:
            if file_path.exists():
                try:
                    self.df = pd.read_csv(file_path, encoding='utf-8-sig')
                    loaded = True
                    print(f"  成功加载数据文件: {file_path.name}")
                    break
                except Exception as e:
                    print(f"  读取 {file_path.name} 失败: {str(e)}")
                    continue
                    
        if not loaded:
            # 列出数据目录内容帮助调试
            print(f"\n  数据目录 {self.data_path} 内容:")
            if self.data_path.exists():
                for f in self.data_path.iterdir():
                    if f.suffix == '.csv':
                        print(f"    - {f.name}")
            else:
                print(f"    目录不存在！")
                
            raise FileNotFoundError(f"未找到数据文件。请确保数据文件位于: {self.data_path.absolute()}")
            
        # 确保必要的列存在
        required_columns = ['date', 'dsr_cognitive', 'tl_functional', 'cs_output']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            print(f"  警告: 缺少列 {missing_columns}")
            # 尝试创建缺失的列
            if 'cs_output' not in self.df.columns and 'constitutive_index' in self.df.columns:
                self.df['cs_output'] = self.df['constitutive_index']
                print("  使用 constitutive_index 作为 cs_output")
                
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date')
        
        # 创建时间特征
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['quarter'] = self.df['date'].dt.quarter
        self.df['week'] = self.df['date'].dt.isocalendar().week
        
        # 确保数值列
        numeric_columns = ['dsr_cognitive', 'dsr_bridging_score', 'dsr_integration_depth',
                          'tl_functional', 'tl_pragmatic_richness', 'tl_conventional_density',
                          'cs_output', 'constitutive_index']
        
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                
        # 添加缺失的索引列（如果需要）
        if 'adaptability_index' not in self.df.columns:
            if 'constitutive_index' in self.df.columns:
                # 使用构成性指数的变化作为适应性代理
                self.df['adaptability_index'] = self.df['constitutive_index'].rolling(window=7).std()
                self.df['adaptability_index'].fillna(self.df['constitutive_index'].std(), inplace=True)
                
        if 'stability_index' not in self.df.columns:
            if 'constitutive_index' in self.df.columns:
                # 使用构成性指数的稳定性作为稳定性代理
                self.df['stability_index'] = 1 - self.df['constitutive_index'].rolling(window=7).std()
                self.df['stability_index'].fillna(0.5, inplace=True)
        
        print("="*60)
        print("第九步：网络效应与扩散分析")
        print("="*60)
        print(f"数据加载完成: {len(self.df)} 条记录")
        print(f"时间跨度: {self.df['date'].min()} 至 {self.df['date'].max()}")
        
        return self.df
        
    def run_network_diffusion_analysis(self):
        """运行所有网络扩散分析"""
        
        print("\n1. 构建认知网络")
        cognitive_network = self.build_cognitive_network()
        
        print("\n2. 扩散模式分析")
        diffusion_patterns = self.analyze_diffusion_patterns()
        
        print("\n3. 网络效应测量")
        network_effects = self.measure_network_effects()
        
        print("\n4. 影响力传播分析")
        influence_propagation = self.analyze_influence_propagation()
        
        print("\n5. 社区结构检测")
        community_structure = self.detect_community_structure()
        
        print("\n6. 关键节点识别")
        critical_nodes = self.identify_critical_nodes()
        
        # 生成可视化
        self.create_visualizations()
        
        # 保存结果
        self.save_results()
        
        # 生成综合解释
        self.generate_comprehensive_interpretation()
        
        return self.results
        
    def build_cognitive_network(self):
        """构建认知网络"""
        print("  构建认知要素网络...")
        
        results = {
            'network_properties': {},
            'node_attributes': {},
            'edge_weights': {},
            'network_evolution': []
        }
        
        # 1. 创建认知要素网络
        G = nx.Graph()
        
        # 定义核心认知要素节点
        cognitive_elements = {
            'DSR_core': {'type': 'dsr', 'label': 'DSR核心', 'weight': 1.0},
            'DSR_bridge': {'type': 'dsr', 'label': 'DSR桥接', 'weight': 0.8},
            'DSR_integrate': {'type': 'dsr', 'label': 'DSR整合', 'weight': 0.7},
            'TL_formal': {'type': 'tl', 'label': '正式语言', 'weight': 0.9},
            'TL_pragmatic': {'type': 'tl', 'label': '语用策略', 'weight': 0.7},
            'TL_cultural': {'type': 'tl', 'label': '文化惯例', 'weight': 0.6},
            'CS_output': {'type': 'cs', 'label': '认知输出', 'weight': 1.0},
            'CS_adapt': {'type': 'cs', 'label': '适应能力', 'weight': 0.8},
            'CS_stable': {'type': 'cs', 'label': '稳定性', 'weight': 0.7}
        }
        
        # 添加节点
        for node_id, attrs in cognitive_elements.items():
            G.add_node(node_id, **attrs)
            
        # 2. 基于相关性构建边
        # 计算要素间的相关性
        feature_correlations = self.df[[
            'dsr_cognitive', 'dsr_bridging_score', 'dsr_integration_depth',
            'tl_functional', 'tl_pragmatic_richness', 'tl_conventional_density',
            'cs_output', 'adaptability_index', 'stability_index'
        ]].corr()
        
        # 定义节点与数据特征的映射
        node_feature_map = {
            'DSR_core': 'dsr_cognitive',
            'DSR_bridge': 'dsr_bridging_score',
            'DSR_integrate': 'dsr_integration_depth',
            'TL_formal': 'tl_functional',
            'TL_pragmatic': 'tl_pragmatic_richness',
            'TL_cultural': 'tl_conventional_density',
            'CS_output': 'cs_output',
            'CS_adapt': 'adaptability_index',
            'CS_stable': 'stability_index'
        }
        
        # 添加边（基于相关性）
        threshold = 0.3  # 相关性阈值
        for node1, feat1 in node_feature_map.items():
            for node2, feat2 in node_feature_map.items():
                if node1 < node2 and feat1 in feature_correlations and feat2 in feature_correlations:
                    corr = feature_correlations.loc[feat1, feat2]
                    if abs(corr) > threshold:
                        G.add_edge(node1, node2, weight=abs(corr), correlation=corr)
                        
        # 3. 计算网络属性
        results['network_properties'] = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'average_clustering': nx.average_clustering(G),
            'average_path_length': nx.average_shortest_path_length(G) if nx.is_connected(G) else np.nan,
            'diameter': nx.diameter(G) if nx.is_connected(G) else np.nan,
            'assortativity': nx.degree_assortativity_coefficient(G)
        }
        
        # 4. 计算节点中心性
        centrality_measures = {
            'degree': nx.degree_centrality(G),
            'betweenness': nx.betweenness_centrality(G),
            'closeness': nx.closeness_centrality(G),
            'eigenvector': nx.eigenvector_centrality(G, max_iter=1000),
            'pagerank': nx.pagerank(G)
        }
        
        # 整合节点属性
        for node in G.nodes():
            node_attrs = {
                'degree': G.degree(node),
                'degree_centrality': centrality_measures['degree'][node],
                'betweenness_centrality': centrality_measures['betweenness'][node],
                'closeness_centrality': centrality_measures['closeness'][node],
                'eigenvector_centrality': centrality_measures['eigenvector'][node],
                'pagerank': centrality_measures['pagerank'][node]
            }
            results['node_attributes'][node] = node_attrs
            
        # 5. 时间演化分析
        # 分析网络结构随时间的变化
        time_windows = pd.date_range(self.df['date'].min(), self.df['date'].max(), freq='Q')
        
        for i in range(len(time_windows) - 1):
            window_data = self.df[(self.df['date'] >= time_windows[i]) & 
                                 (self.df['date'] < time_windows[i+1])]
            
            if len(window_data) > 10:
                window_corr = window_data[[feat for feat in node_feature_map.values() 
                                         if feat in window_data.columns]].corr()
                
                # 计算网络密度变化
                edge_count = 0
                for n1, f1 in node_feature_map.items():
                    for n2, f2 in node_feature_map.items():
                        if n1 < n2 and f1 in window_corr and f2 in window_corr:
                            if abs(window_corr.loc[f1, f2]) > threshold:
                                edge_count += 1
                                
                # 计算季度
                quarter = (time_windows[i].month - 1) // 3 + 1
                period_str = f"{time_windows[i].year}-Q{quarter}"
                
                results['network_evolution'].append({
                    'period': period_str,
                    'edge_count': edge_count,
                    'density': edge_count / (len(node_feature_map) * (len(node_feature_map) - 1) / 2)
                })
                
        # 保存网络对象
        self.cognitive_network = G
        
        print(f"    - 网络节点数: {results['network_properties']['num_nodes']}")
        print(f"    - 网络边数: {results['network_properties']['num_edges']}")
        print(f"    - 网络密度: {results['network_properties']['density']:.3f}")
        
        self.results['cognitive_network'] = results
        return results
        
    def analyze_diffusion_patterns(self):
        """分析扩散模式"""
        print("  分析DSR扩散模式...")
        
        results = {
            'temporal_diffusion': {},
            'spatial_diffusion': {},
            'adoption_curves': {},
            'diffusion_speed': {}
        }
        
        # 1. 时间扩散分析
        # 分析DSR使用随时间的扩散
        daily_adoption = self.df.groupby('date').agg({
            'dsr_cognitive': 'mean',
            'dsr_bridging_score': 'mean',
            'dsr_integration_depth': 'mean'
        }).rolling(window=7).mean()
        
        # 计算采纳率
        adoption_threshold = daily_adoption['dsr_cognitive'].quantile(0.5)
        daily_adoption['adopted'] = (daily_adoption['dsr_cognitive'] > adoption_threshold).astype(int)
        cumulative_adoption = daily_adoption['adopted'].cumsum() / len(daily_adoption)
        
        # 安全计算采纳加速度
        try:
            valid_data = cumulative_adoption.dropna()
            if len(valid_data) >= 3:
                adoption_acceleration = np.polyfit(range(len(valid_data)), valid_data.values, 2)[0]
            else:
                adoption_acceleration = 0
        except:
            adoption_acceleration = 0
            
        results['temporal_diffusion'] = {
            'adoption_rate': cumulative_adoption.iloc[-1] if len(cumulative_adoption) > 0 else 0,
            'half_adoption_date': cumulative_adoption[cumulative_adoption > 0.5].index[0].strftime('%Y-%m-%d') if any(cumulative_adoption > 0.5) else None,
            'adoption_acceleration': adoption_acceleration
        }
        
        # 2. 空间扩散分析（基于主题/类型）
        if 'topic' in self.df.columns or 'category' in self.df.columns:
            topic_col = 'topic' if 'topic' in self.df.columns else 'category'
            
            topic_adoption = self.df.groupby(topic_col).agg({
                'dsr_cognitive': ['mean', 'std', 'count'],
                'date': ['min', 'max']
            })
            
            results['spatial_diffusion'] = {
                'topic_adoption_rates': topic_adoption['dsr_cognitive']['mean'].to_dict(),
                'topic_adoption_variance': topic_adoption['dsr_cognitive']['std'].to_dict(),
                'early_adopter_topics': topic_adoption.nlargest(3, ('dsr_cognitive', 'mean')).index.tolist()
            }
            
        # 3. S型采纳曲线拟合
        from scipy.optimize import curve_fit
        
        def logistic_growth(t, L, k, t0):
            return L / (1 + np.exp(-k * (t - t0)))
            
        try:
            t_data = np.arange(len(cumulative_adoption))
            popt, pcov = curve_fit(logistic_growth, t_data, cumulative_adoption.values,
                                 p0=[1, 0.01, len(t_data)/2])
            
            results['adoption_curves'] = {
                'model': 'logistic',
                'parameters': {
                    'carrying_capacity': popt[0],
                    'growth_rate': popt[1],
                    'midpoint': popt[2]
                },
                'goodness_of_fit': 1 - np.sum((cumulative_adoption.values - logistic_growth(t_data, *popt))**2) / np.var(cumulative_adoption.values)
            }
        except:
            results['adoption_curves'] = {'model': 'failed'}
            
        # 4. 扩散速度分析
        # 计算不同时期的扩散速度
        window_size = 30  # 30天窗口
        diffusion_speed = []
        
        for i in range(window_size, len(daily_adoption)):
            window = daily_adoption.iloc[i-window_size:i]
            
            # 检查窗口数据是否有效
            if len(window) > 0 and not window['dsr_cognitive'].isna().all():
                speed = (window['dsr_cognitive'].iloc[-1] - window['dsr_cognitive'].iloc[0]) / window_size
                
                # 尝试计算加速度，如果失败则设为0
                try:
                    # 清理数据
                    y_data = window['dsr_cognitive'].values
                    x_data = np.arange(len(y_data))
                    
                    # 移除NaN值
                    valid_idx = ~np.isnan(y_data)
                    if sum(valid_idx) >= 3:  # 至少需要3个点进行二次拟合
                        acceleration = np.polyfit(x_data[valid_idx], y_data[valid_idx], 2)[0]
                    else:
                        acceleration = 0
                except:
                    acceleration = 0
                    
                diffusion_speed.append({
                    'date': window.index[-1],
                    'speed': speed,
                    'acceleration': acceleration
                })
            
        if diffusion_speed:
            speeds = [d['speed'] for d in diffusion_speed]
            
            # 安全计算速度趋势
            try:
                if len(speeds) >= 2:
                    speed_trend = np.polyfit(range(len(speeds)), speeds, 1)[0]
                else:
                    speed_trend = 0
            except:
                speed_trend = 0
                
            results['diffusion_speed'] = {
                'average_speed': np.mean(speeds),
                'max_speed': np.max(speeds),
                'speed_trend': speed_trend,
                'peak_diffusion_date': diffusion_speed[np.argmax(speeds)]['date'].strftime('%Y-%m-%d')
            }
            
        print(f"    - 总体采纳率: {results['temporal_diffusion']['adoption_rate']:.1%}")
        print(f"    - 扩散加速度: {results['temporal_diffusion']['adoption_acceleration']:.4f}")
        
        self.results['diffusion_patterns'] = results
        return results
        
    def measure_network_effects(self):
        """测量网络效应"""
        print("  测量网络效应...")
        
        results = {
            'direct_effects': {},
            'indirect_effects': {},
            'spillover_effects': {},
            'network_multiplier': {}
        }
        
        # 1. 直接网络效应
        # 分析DSR使用与认知输出的直接关系随网络密度的变化
        # 将数据按时间分组，计算每组的网络密度代理
        self.df['network_density_proxy'] = self.df.groupby('date')['dsr_cognitive'].transform('mean')
        
        # 分析不同网络密度下的效应
        density_quantiles = self.df['network_density_proxy'].quantile([0.25, 0.5, 0.75])
        
        effects_by_density = {}
        for i, (low, high) in enumerate([(0, density_quantiles[0.25]), 
                                        (density_quantiles[0.25], density_quantiles[0.75]),
                                        (density_quantiles[0.75], 1)]):
            subset = self.df[(self.df['network_density_proxy'] > low) & 
                           (self.df['network_density_proxy'] <= high)]
            
            if len(subset) > 10:
                X = subset[['dsr_cognitive', 'tl_functional']]
                y = subset['cs_output']
                
                model = sm.OLS(y, sm.add_constant(X)).fit()
                
                effects_by_density[f'density_level_{i+1}'] = {
                    'dsr_coefficient': model.params['dsr_cognitive'],
                    'sample_size': len(subset),
                    'r_squared': model.rsquared
                }
                
        results['direct_effects'] = effects_by_density
        
        # 2. 间接网络效应（通过中介变量）
        # 分析DSR通过其他认知要素的间接影响
        mediation_paths = [
            ('dsr_cognitive', 'dsr_bridging_score', 'cs_output'),
            ('dsr_cognitive', 'tl_functional', 'cs_output'),
            ('dsr_cognitive', 'dsr_integration_depth', 'cs_output')
        ]
        
        indirect_effects = {}
        for x_var, m_var, y_var in mediation_paths:
            if all(col in self.df.columns for col in [x_var, m_var, y_var]):
                # 路径a: X -> M
                model_a = sm.OLS(self.df[m_var], sm.add_constant(self.df[x_var])).fit()
                a_path = model_a.params[x_var]
                
                # 路径b: M -> Y (控制X)
                X_mb = self.df[[x_var, m_var]]
                model_b = sm.OLS(self.df[y_var], sm.add_constant(X_mb)).fit()
                b_path = model_b.params[m_var]
                
                indirect_effects[f'{x_var}_via_{m_var}'] = {
                    'indirect_effect': a_path * b_path,
                    'a_path': a_path,
                    'b_path': b_path
                }
                
        results['indirect_effects'] = indirect_effects
        
        # 3. 溢出效应
        # 分析相邻时间窗口的影响
        lag_effects = {}
        for lag in [1, 7, 30]:  # 1天、1周、1月的滞后
            if len(self.df) > lag:
                lagged_dsr = self.df['dsr_cognitive'].shift(lag)
                valid_idx = ~lagged_dsr.isna()
                
                if sum(valid_idx) > 10:
                    X = sm.add_constant(lagged_dsr[valid_idx])
                    y = self.df.loc[valid_idx, 'cs_output']
                    
                    model = sm.OLS(y, X).fit()
                    
                    lag_effects[f'lag_{lag}d'] = {
                        'coefficient': model.params.iloc[1],
                        'p_value': model.pvalues.iloc[1],
                        'significant': model.pvalues.iloc[1] < 0.05
                    }
                    
        results['spillover_effects'] = lag_effects
        
        # 4. 网络乘数效应
        # 计算网络效应的放大倍数
        if effects_by_density:
            low_density_effect = effects_by_density.get('density_level_1', {}).get('dsr_coefficient', 0)
            high_density_effect = effects_by_density.get('density_level_3', {}).get('dsr_coefficient', 0)
            
            if low_density_effect > 0:
                multiplier = high_density_effect / low_density_effect
            else:
                multiplier = 1.0
                
            results['network_multiplier'] = {
                'multiplier': multiplier,
                'low_density_effect': low_density_effect,
                'high_density_effect': high_density_effect,
                'amplification': (multiplier - 1) * 100  # 百分比增幅
            }
            
        print(f"    - 网络乘数: {results.get('network_multiplier', {}).get('multiplier', 1):.2f}")
        print(f"    - 溢出效应: {sum(1 for e in lag_effects.values() if e.get('significant', False))}/3 显著")
        
        self.results['network_effects'] = results
        return results
        
    def analyze_influence_propagation(self):
        """分析影响力传播"""
        print("  分析影响力传播机制...")
        
        results = {
            'propagation_paths': {},
            'influence_decay': {},
            'cascade_patterns': {},
            'critical_mass': {}
        }
        
        # 1. 识别传播路径
        if hasattr(self, 'cognitive_network'):
            G = self.cognitive_network
            
            # 计算所有最短路径
            all_paths = []
            dsr_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == 'dsr']
            cs_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == 'cs']
            
            for source in dsr_nodes:
                for target in cs_nodes:
                    if nx.has_path(G, source, target):
                        paths = list(nx.all_shortest_paths(G, source, target))
                        all_paths.extend(paths)
                        
            # 统计路径频率
            path_lengths = [len(p) - 1 for p in all_paths]
            
            # 计算最常见的路径长度
            if path_lengths:
                mode_result = stats.mode(path_lengths, keepdims=True)
                # 兼容新旧版本的scipy
                if hasattr(mode_result, 'mode'):
                    if isinstance(mode_result.mode, np.ndarray):
                        most_common = mode_result.mode[0] if len(mode_result.mode) > 0 else 0
                    else:
                        most_common = mode_result.mode
                else:
                    # 旧版本scipy
                    most_common = mode_result[0][0] if len(mode_result[0]) > 0 else 0
            else:
                most_common = 0
                
            results['propagation_paths'] = {
                'total_paths': len(all_paths),
                'average_path_length': np.mean(path_lengths) if path_lengths else 0,
                'path_length_distribution': dict(zip(*np.unique(path_lengths, return_counts=True))) if path_lengths else {},
                'most_common_length': most_common
            }
            
        # 2. 影响力衰减分析
        # 分析影响力随距离/时间的衰减
        time_lags = range(1, 31)  # 1-30天
        influence_by_lag = []
        
        for lag in time_lags:
            if len(self.df) > lag:
                lagged_corr = self.df['dsr_cognitive'].corr(self.df['cs_output'].shift(-lag))
                influence_by_lag.append(lagged_corr)
                
        if influence_by_lag:
            # 拟合指数衰减模型
            def exp_decay(t, a, b):
                return a * np.exp(-b * t)
                
            try:
                t_data = np.array(time_lags)
                popt, _ = curve_fit(exp_decay, t_data, np.abs(influence_by_lag))
                
                results['influence_decay'] = {
                    'decay_rate': popt[1],
                    'half_life': np.log(2) / popt[1] if popt[1] > 0 else np.inf,
                    'initial_influence': popt[0],
                    'model_fit': True
                }
            except:
                results['influence_decay'] = {
                    'decay_rate': 0,
                    'half_life': np.inf,
                    'model_fit': False
                }
                
        # 3. 级联模式分析
        # 识别影响力的级联效应
        threshold = 0.5  # 激活阈值
        
        # 创建时间窗口
        window_size = 7  # 7天窗口
        cascades = []
        
        for i in range(window_size, len(self.df)):
            window = self.df.iloc[i-window_size:i]
            
            # 检测是否发生级联
            dsr_activated = window['dsr_cognitive'] > window['dsr_cognitive'].quantile(threshold)
            if dsr_activated.sum() > window_size * 0.5:  # 超过50%激活
                # 检查后续效应
                next_window = self.df.iloc[i:i+window_size] if i+window_size <= len(self.df) else self.df.iloc[i:]
                cs_response = next_window['cs_output'].mean() - self.df['cs_output'].mean()
                
                cascades.append({
                    'start_date': window.iloc[0]['date'],
                    'activation_rate': dsr_activated.mean(),
                    'cs_response': cs_response
                })
                
        if cascades:
            results['cascade_patterns'] = {
                'num_cascades': len(cascades),
                'average_activation': np.mean([c['activation_rate'] for c in cascades]),
                'average_response': np.mean([c['cs_response'] for c in cascades]),
                'cascade_frequency': len(cascades) / (len(self.df) / window_size)
            }
            
        # 4. 临界质量分析
        # 寻找网络效应的临界点
        # 将DSR使用率分组，分析每组的网络效应
        dsr_quantiles = self.df['dsr_cognitive'].quantile(np.arange(0.1, 1.0, 0.1))
        
        effects_by_adoption = []
        for i in range(len(dsr_quantiles) - 1):
            mask = (self.df['dsr_cognitive'] >= dsr_quantiles.iloc[i]) & \
                   (self.df['dsr_cognitive'] < dsr_quantiles.iloc[i+1])
            
            if mask.sum() > 10:
                subset = self.df[mask]
                effect = subset['dsr_cognitive'].corr(subset['cs_output'])
                effects_by_adoption.append({
                    'adoption_level': (i + 1) * 0.1,
                    'effect_size': effect,
                    'sample_size': mask.sum()
                })
                
        if effects_by_adoption:
            # 寻找效应突增的点
            effects = [e['effect_size'] for e in effects_by_adoption]
            if len(effects) > 1:
                effect_changes = np.diff(effects)
                max_change_idx = np.argmax(effect_changes)
                
                results['critical_mass'] = {
                    'critical_adoption_level': effects_by_adoption[max_change_idx]['adoption_level'],
                    'effect_jump': effect_changes[max_change_idx],
                    'pre_critical_effect': effects[max_change_idx],
                    'post_critical_effect': effects[max_change_idx + 1]
                }
                
        print(f"    - 平均传播路径长度: {results.get('propagation_paths', {}).get('average_path_length', 0):.2f}")
        print(f"    - 影响力半衰期: {results.get('influence_decay', {}).get('half_life', np.inf):.1f}天")
        
        self.results['influence_propagation'] = results
        return results
        
    def detect_community_structure(self):
        """检测社区结构"""
        print("  检测认知社区结构...")
        
        results = {
            'communities': {},
            'modularity': 0,
            'community_interactions': {},
            'bridge_nodes': []
        }
        
        if hasattr(self, 'cognitive_network'):
            G = self.cognitive_network
            
            # 1. 社区检测（使用Louvain算法）
            communities = community.louvain_communities(G, seed=42)
            
            # 计算模块度
            modularity = community.modularity(G, communities)
            
            results['modularity'] = modularity
            
            # 2. 分析每个社区
            community_info = {}
            for i, comm in enumerate(communities):
                comm_nodes = list(comm)
                
                # 社区内的节点类型分布
                node_types = [G.nodes[n]['type'] for n in comm_nodes]
                type_dist = dict(zip(*np.unique(node_types, return_counts=True)))
                
                # 社区的平均中心性
                avg_centrality = np.mean([self.results['cognitive_network']['node_attributes'][n]['eigenvector_centrality'] 
                                         for n in comm_nodes])
                
                community_info[f'community_{i}'] = {
                    'size': len(comm_nodes),
                    'nodes': comm_nodes,
                    'type_distribution': type_dist,
                    'average_centrality': avg_centrality,
                    'dominant_type': max(type_dist, key=type_dist.get)
                }
                
            results['communities'] = community_info
            
            # 3. 社区间交互分析
            community_interactions = {}
            for i in range(len(communities)):
                for j in range(i+1, len(communities)):
                    inter_edges = [(u, v) for u in communities[i] for v in communities[j] 
                                  if G.has_edge(u, v)]
                    
                    if inter_edges:
                        avg_weight = np.mean([G[u][v]['weight'] for u, v in inter_edges])
                        community_interactions[f'comm_{i}_comm_{j}'] = {
                            'num_edges': len(inter_edges),
                            'average_weight': avg_weight,
                            'key_connections': inter_edges[:3]  # 前3个连接
                        }
                        
            results['community_interactions'] = community_interactions
            
            # 4. 识别桥接节点
            # 连接不同社区的关键节点
            bridge_nodes = []
            for node in G.nodes():
                neighbors = list(G.neighbors(node))
                neighbor_communities = []
                
                for neighbor in neighbors:
                    for i, comm in enumerate(communities):
                        if neighbor in comm:
                            neighbor_communities.append(i)
                            break
                            
                unique_communities = len(set(neighbor_communities))
                if unique_communities > 1:
                    bridge_nodes.append({
                        'node': node,
                        'bridges': unique_communities,
                        'betweenness': self.results['cognitive_network']['node_attributes'][node]['betweenness_centrality']
                    })
                    
            # 按介数中心性排序
            bridge_nodes.sort(key=lambda x: x['betweenness'], reverse=True)
            results['bridge_nodes'] = bridge_nodes[:5]  # 前5个桥接节点
            
        print(f"    - 社区数量: {len(results['communities'])}")
        print(f"    - 模块度: {results['modularity']:.3f}")
        print(f"    - 桥接节点数: {len(results['bridge_nodes'])}")
        
        self.results['community_structure'] = results
        return results
        
    def identify_critical_nodes(self):
        """识别关键节点"""
        print("  识别网络关键节点...")
        
        results = {
            'hub_nodes': [],
            'bottleneck_nodes': [],
            'influence_nodes': [],
            'vulnerability_assessment': {}
        }
        
        if hasattr(self, 'cognitive_network'):
            G = self.cognitive_network
            node_attrs = self.results['cognitive_network']['node_attributes']
            
            # 1. 识别枢纽节点（高度中心性）
            degree_threshold = np.percentile([attrs['degree_centrality'] 
                                            for attrs in node_attrs.values()], 75)
            
            hub_nodes = []
            for node, attrs in node_attrs.items():
                if attrs['degree_centrality'] > degree_threshold:
                    hub_nodes.append({
                        'node': node,
                        'label': G.nodes[node]['label'],
                        'type': G.nodes[node]['type'],
                        'degree_centrality': attrs['degree_centrality'],
                        'connections': G.degree(node)
                    })
                    
            hub_nodes.sort(key=lambda x: x['degree_centrality'], reverse=True)
            results['hub_nodes'] = hub_nodes
            
            # 2. 识别瓶颈节点（高介数中心性）
            betweenness_threshold = np.percentile([attrs['betweenness_centrality'] 
                                                 for attrs in node_attrs.values()], 75)
            
            bottleneck_nodes = []
            for node, attrs in node_attrs.items():
                if attrs['betweenness_centrality'] > betweenness_threshold:
                    bottleneck_nodes.append({
                        'node': node,
                        'label': G.nodes[node]['label'],
                        'type': G.nodes[node]['type'],
                        'betweenness_centrality': attrs['betweenness_centrality'],
                        'paths_controlled': int(attrs['betweenness_centrality'] * G.number_of_nodes() * (G.number_of_nodes() - 1))
                    })
                    
            bottleneck_nodes.sort(key=lambda x: x['betweenness_centrality'], reverse=True)
            results['bottleneck_nodes'] = bottleneck_nodes
            
            # 3. 识别影响力节点（综合指标）
            # 计算综合影响力得分
            influence_scores = {}
            for node, attrs in node_attrs.items():
                # 综合多个中心性指标
                influence_score = (
                    0.3 * attrs['degree_centrality'] +
                    0.3 * attrs['betweenness_centrality'] +
                    0.2 * attrs['eigenvector_centrality'] +
                    0.2 * attrs['pagerank']
                )
                influence_scores[node] = influence_score
                
            # 选择前5个影响力节点
            top_influence_nodes = sorted(influence_scores.items(), 
                                       key=lambda x: x[1], reverse=True)[:5]
            
            results['influence_nodes'] = [
                {
                    'node': node,
                    'label': G.nodes[node]['label'],
                    'type': G.nodes[node]['type'],
                    'influence_score': score,
                    'rank': i + 1
                }
                for i, (node, score) in enumerate(top_influence_nodes)
            ]
            
            # 4. 脆弱性评估
            # 模拟移除关键节点的影响
            original_connected = nx.is_connected(G)
            original_diameter = nx.diameter(G) if original_connected else np.inf
            
            vulnerability_results = {}
            for node_info in results['influence_nodes'][:3]:  # 测试前3个关键节点
                node = node_info['node']
                
                # 创建副本并移除节点
                G_test = G.copy()
                G_test.remove_node(node)
                
                # 评估影响
                still_connected = nx.is_connected(G_test)
                new_diameter = nx.diameter(G_test) if still_connected else np.inf
                
                vulnerability_results[node] = {
                    'connectivity_loss': not still_connected if original_connected else False,
                    'diameter_increase': (new_diameter - original_diameter) / original_diameter if original_diameter < np.inf else np.inf,
                    'component_increase': nx.number_connected_components(G_test) - nx.number_connected_components(G)
                }
                
            results['vulnerability_assessment'] = vulnerability_results
            
        print(f"    - 枢纽节点: {len(results['hub_nodes'])}")
        print(f"    - 瓶颈节点: {len(results['bottleneck_nodes'])}")
        print(f"    - 最关键节点: {results['influence_nodes'][0]['label'] if results['influence_nodes'] else 'None'}")
        
        self.results['critical_nodes'] = results
        return results
        
    def create_visualizations(self):
        """创建可视化"""
        print("\n生成可视化...")
        
        fig = plt.figure(figsize=(24, 20))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. 认知网络图（左上）
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_cognitive_network(ax1)
        
        # 2. 扩散曲线（中上）
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_diffusion_curves(ax2)
        
        # 3. 网络效应（右上）
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_network_effects(ax3)
        
        # 4. 影响力传播（左中）
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_influence_propagation(ax4)
        
        # 5. 社区结构（中中）
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_community_structure(ax5)
        
        # 6. 关键节点（右中）
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_critical_nodes(ax6)
        
        # 7. 时间演化热力图（整行）
        ax7 = fig.add_subplot(gs[2, :])
        self._plot_temporal_heatmap(ax7)
        
        # 8. 级联效应（左下）
        ax8 = fig.add_subplot(gs[3, 0])
        self._plot_cascade_effects(ax8)
        
        # 9. 临界质量（中下）
        ax9 = fig.add_subplot(gs[3, 1])
        self._plot_critical_mass(ax9)
        
        # 10. 网络鲁棒性（右下）
        ax10 = fig.add_subplot(gs[3, 2])
        self._plot_network_robustness(ax10)
        
        plt.suptitle('网络效应与扩散分析：DSR在认知网络中的传播模式', fontsize=20, y=0.98)
        
        # 保存图表
        output_path = self.data_path.parent / 'figures' / 'network_diffusion_analysis.jpg'
        plt.savefig(output_path, dpi=1200, bbox_inches='tight', format='jpg')
        plt.close()
        
        print(f"  可视化已保存至: {output_path}")
        
    def _plot_cognitive_network(self, ax):
        """绘制认知网络图"""
        if hasattr(self, 'cognitive_network'):
            # 临时禁用matplotlib警告
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="posx and posy should be finite values")
                
                G = self.cognitive_network
                
                # 使用更稳定的布局算法
                # 首先尝试kamada_kawai布局，它通常更稳定
                try:
                    pos = nx.kamada_kawai_layout(G)
                except:
                    # 如果失败，使用圆形布局作为备份
                    try:
                        pos = nx.circular_layout(G)
                    except:
                        # 最后的备份：手动创建位置
                        pos = {}
                        n = len(G.nodes())
                        for i, node in enumerate(G.nodes()):
                            angle = 2 * np.pi * i / n
                            pos[node] = (np.cos(angle), np.sin(angle))
                
                # 确保所有位置都是有限值
                for node in pos:
                    x, y = pos[node]
                    if not (np.isfinite(x) and np.isfinite(y)):
                        # 如果仍有无限值，设置为原点
                        pos[node] = (0, 0)
                
                # 节点颜色映射
                color_map = {'dsr': '#FF6B6B', 'tl': '#4ECDC4', 'cs': '#45B7D1'}
                node_colors = [color_map[G.nodes[n]['type']] for n in G.nodes()]
                
                # 节点大小基于度中心性
                node_sizes = [1000 * self.results['cognitive_network']['node_attributes'][n]['degree_centrality'] + 200 
                             for n in G.nodes()]
                
                # 绘制边
                edges = G.edges()
                weights = [G[u][v]['weight'] for u, v in edges]
                
                # 绘制边，使用try-except避免警告
                try:
                    nx.draw_networkx_edges(G, pos, alpha=0.3, width=[w*3 for w in weights], 
                                         ax=ax, arrows=False)
                except ValueError:
                    # 如果有问题，使用默认宽度
                    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax, arrows=False)
                
                # 绘制节点
                try:
                    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                         node_size=node_sizes, alpha=0.8, ax=ax)
                except ValueError:
                    # 如果有问题，使用默认大小
                    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                         node_size=500, alpha=0.8, ax=ax)
                
                # 添加标签
                labels = {n: G.nodes[n]['label'] for n in G.nodes()}
                try:
                    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
                except ValueError:
                    pass  # 如果标签绘制失败，跳过
                
                # 添加图例
                legend_elements = [
                    plt.scatter([], [], c='#FF6B6B', s=100, label='DSR要素'),
                    plt.scatter([], [], c='#4ECDC4', s=100, label='TL要素'),
                    plt.scatter([], [], c='#45B7D1', s=100, label='CS要素')
                ]
                ax.legend(handles=legend_elements, loc='upper right')
                
                ax.set_title('认知要素网络')
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, '无网络数据', ha='center', va='center')
            ax.set_title('认知要素网络')
            
    def _plot_diffusion_curves(self, ax):
        """绘制扩散曲线"""
        # 计算累积采纳率
        daily_mean = self.df.groupby('date')['dsr_cognitive'].mean()
        cumulative = (daily_mean > daily_mean.median()).cumsum() / len(daily_mean)
        
        # 绘制实际曲线
        ax.plot(cumulative.index, cumulative.values, 'b-', linewidth=2, label='实际采纳')
        
        # 如果有S曲线拟合结果，绘制拟合曲线
        adoption_curves = self.results.get('diffusion_patterns', {}).get('adoption_curves', {})
        if adoption_curves.get('model') == 'logistic':
            params = adoption_curves['parameters']
            
            def logistic_growth(t, L, k, t0):
                return L / (1 + np.exp(-k * (t - t0)))
                
            t_fit = np.linspace(0, len(cumulative), 100)
            y_fit = logistic_growth(t_fit, params['carrying_capacity'], 
                                  params['growth_rate'], params['midpoint'])
            
            # 将t_fit映射到日期
            date_range = pd.date_range(cumulative.index[0], cumulative.index[-1], periods=100)
            ax.plot(date_range, y_fit, 'r--', linewidth=2, label='S曲线拟合')
            
        ax.set_xlabel('时间')
        ax.set_ylabel('采纳率')
        ax.set_title('DSR扩散曲线')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 格式化日期
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
    def _plot_network_effects(self, ax):
        """绘制网络效应"""
        effects = self.results.get('network_effects', {}).get('direct_effects', {})
        
        if effects:
            # 提取不同密度水平的效应
            levels = []
            coefficients = []
            
            for level, data in sorted(effects.items()):
                levels.append(level.replace('density_level_', '密度'))
                coefficients.append(data.get('dsr_coefficient', 0))
                
            # 绘制条形图
            bars = ax.bar(levels, coefficients, color=['#FEE2E2', '#FED7D7', '#F87171'])
            
            # 添加数值标签
            for bar, coef in zip(bars, coefficients):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{coef:.3f}', ha='center', va='bottom')
                
            ax.set_ylabel('DSR系数')
            ax.set_title('网络密度对DSR效应的影响')
            
            # 添加网络乘数信息
            multiplier = self.results.get('network_effects', {}).get('network_multiplier', {})
            if multiplier:
                ax.text(0.95, 0.95, f'网络乘数: {multiplier.get("multiplier", 1):.2f}x',
                       transform=ax.transAxes, ha='right', va='top',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        else:
            ax.text(0.5, 0.5, '无网络效应数据', ha='center', va='center')
            ax.set_title('网络效应')
            
    def _plot_influence_propagation(self, ax):
        """绘制影响力传播"""
        decay = self.results.get('influence_propagation', {}).get('influence_decay', {})
        
        if decay and decay.get('model_fit'):
            # 绘制影响力衰减曲线
            t = np.linspace(1, 30, 100)
            
            def exp_decay(t, a, b):
                return a * np.exp(-b * t)
                
            y = exp_decay(t, decay['initial_influence'], decay['decay_rate'])
            
            ax.plot(t, y, 'b-', linewidth=2, label='影响力衰减')
            
            # 标记半衰期
            half_life = decay.get('half_life', np.inf)
            if half_life < 30:
                ax.axvline(x=half_life, color='red', linestyle='--', alpha=0.5)
                ax.text(half_life, max(y)*0.5, f'半衰期: {half_life:.1f}天',
                       rotation=90, va='bottom', ha='right')
                
            ax.set_xlabel('时间延迟（天）')
            ax.set_ylabel('影响力强度')
            ax.set_title('影响力传播衰减')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            # 显示传播路径统计
            paths = self.results.get('influence_propagation', {}).get('propagation_paths', {})
            if paths:
                ax.text(0.5, 0.6, f"平均路径长度: {paths.get('average_path_length', 0):.2f}",
                       ha='center', va='center', fontsize=14)
                ax.text(0.5, 0.4, f"总传播路径数: {paths.get('total_paths', 0)}",
                       ha='center', va='center', fontsize=12)
            else:
                ax.text(0.5, 0.5, '无传播数据', ha='center', va='center')
            ax.set_title('影响力传播')
            
    def _plot_community_structure(self, ax):
        """绘制社区结构"""
        communities = self.results.get('community_structure', {}).get('communities', {})
        
        if communities:
            # 创建社区大小和类型分布的可视化
            comm_names = []
            comm_sizes = []
            comm_types = []
            
            for comm_id, comm_data in communities.items():
                comm_names.append(comm_id.replace('community_', '社区'))
                comm_sizes.append(comm_data['size'])
                comm_types.append(comm_data['dominant_type'])
                
            # 颜色映射
            type_colors = {'dsr': '#FF6B6B', 'tl': '#4ECDC4', 'cs': '#45B7D1'}
            colors = [type_colors.get(t, 'gray') for t in comm_types]
            
            # 绘制条形图
            bars = ax.bar(comm_names, comm_sizes, color=colors, alpha=0.7)
            
            # 添加数值标签
            for bar, size in zip(bars, comm_sizes):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{size}', ha='center', va='bottom')
                
            ax.set_ylabel('节点数')
            ax.set_title(f'社区结构 (模块度: {self.results.get("community_structure", {}).get("modularity", 0):.3f})')
            
            # 添加图例
            legend_elements = [
                mpatches.Patch(color='#FF6B6B', label='DSR主导'),
                mpatches.Patch(color='#4ECDC4', label='TL主导'),
                mpatches.Patch(color='#45B7D1', label='CS主导')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
        else:
            ax.text(0.5, 0.5, '无社区数据', ha='center', va='center')
            ax.set_title('社区结构')
            
    def _plot_critical_nodes(self, ax):
        """绘制关键节点"""
        influence_nodes = self.results.get('critical_nodes', {}).get('influence_nodes', [])
        
        if influence_nodes:
            # 准备数据
            node_labels = [node['label'] for node in influence_nodes]
            influence_scores = [node['influence_score'] for node in influence_nodes]
            node_types = [node['type'] for node in influence_nodes]
            
            # 颜色映射
            type_colors = {'dsr': '#FF6B6B', 'tl': '#4ECDC4', 'cs': '#45B7D1'}
            colors = [type_colors.get(t, 'gray') for t in node_types]
            
            # 创建水平条形图
            y_pos = np.arange(len(node_labels))
            bars = ax.barh(y_pos, influence_scores, color=colors, alpha=0.7)
            
            # 设置标签
            ax.set_yticks(y_pos)
            ax.set_yticklabels(node_labels)
            ax.set_xlabel('影响力得分')
            ax.set_title('关键节点排名')
            
            # 添加数值标签
            for i, (bar, score) in enumerate(zip(bars, influence_scores)):
                ax.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{score:.3f}', va='center')
                       
            ax.grid(True, axis='x', alpha=0.3)
        else:
            ax.text(0.5, 0.5, '无关键节点数据', ha='center', va='center')
            ax.set_title('关键节点')
            
    def _plot_temporal_heatmap(self, ax):
        """绘制时间演化热力图"""
        # 准备月度聚合数据
        monthly_data = self.df.groupby([self.df['date'].dt.to_period('M')]).agg({
            'dsr_cognitive': 'mean',
            'dsr_bridging_score': 'mean',
            'dsr_integration_depth': 'mean',
            'tl_functional': 'mean',
            'cs_output': 'mean',
            'constitutive_index': 'mean'
        })
        
        # 归一化
        normalized_data = (monthly_data - monthly_data.min()) / (monthly_data.max() - monthly_data.min())
        
        # 创建热力图
        im = ax.imshow(normalized_data.T, aspect='auto', cmap='RdYlBu_r')
        
        # 设置标签
        ax.set_xticks(np.arange(0, len(monthly_data), max(1, len(monthly_data)//10)))
        ax.set_xticklabels([str(idx) for idx in monthly_data.index[::max(1, len(monthly_data)//10)]], 
                          rotation=45)
        ax.set_yticks(range(len(monthly_data.columns)))
        ax.set_yticklabels(['DSR认知', 'DSR桥接', 'DSR整合', '传统语言', '认知输出', '构成性指数'])
        
        ax.set_title('认知要素时间演化热力图')
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, label='归一化值')
        
    def _plot_cascade_effects(self, ax):
        """绘制级联效应"""
        cascades = self.results.get('influence_propagation', {}).get('cascade_patterns', {})
        
        if cascades and cascades.get('num_cascades', 0) > 0:
            # 创建级联效应的统计图
            labels = ['级联次数', '平均激活率', '平均响应']
            values = [
                cascades.get('num_cascades', 0),
                cascades.get('average_activation', 0) * 100,
                cascades.get('average_response', 0) * 100
            ]
            
            bars = ax.bar(labels, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            
            # 添加数值标签
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1f}', ha='center', va='bottom')
                       
            ax.set_ylabel('值')
            ax.set_title('级联效应统计')
        else:
            ax.text(0.5, 0.5, '未检测到级联效应', ha='center', va='center')
            ax.set_title('级联效应')
            
    def _plot_critical_mass(self, ax):
        """绘制临界质量"""
        critical = self.results.get('influence_propagation', {}).get('critical_mass', {})
        
        if critical:
            # 创建临界点可视化
            adoption_levels = np.linspace(0.1, 0.9, 9)
            
            # 模拟效应曲线
            effects = []
            critical_level = critical.get('critical_adoption_level', 0.5)
            pre_effect = critical.get('pre_critical_effect', 0.2)
            post_effect = critical.get('post_critical_effect', 0.6)
            
            for level in adoption_levels:
                if level < critical_level:
                    effect = pre_effect
                else:
                    effect = post_effect
                effects.append(effect)
                
            ax.plot(adoption_levels, effects, 'b-', linewidth=2, marker='o')
            
            # 标记临界点
            ax.axvline(x=critical_level, color='red', linestyle='--', alpha=0.5)
            ax.text(critical_level, (pre_effect + post_effect) / 2,
                   f'临界点: {critical_level:.1%}', rotation=90, va='bottom')
                   
            ax.set_xlabel('采纳水平')
            ax.set_ylabel('网络效应强度')
            ax.set_title('临界质量效应')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, '无临界质量数据', ha='center', va='center')
            ax.set_title('临界质量')
            
    def _plot_network_robustness(self, ax):
        """绘制网络鲁棒性"""
        vulnerability = self.results.get('critical_nodes', {}).get('vulnerability_assessment', {})
        
        if vulnerability:
            # 准备数据
            node_names = []
            diameter_increases = []
            
            for node, impact in vulnerability.items():
                if hasattr(self, 'cognitive_network'):
                    node_label = self.cognitive_network.nodes[node]['label']
                    node_names.append(node_label)
                    diameter_increases.append(impact.get('diameter_increase', 0) * 100)
                    
            if node_names:
                # 绘制影响程度
                bars = ax.bar(node_names, diameter_increases, color='#FF6B6B', alpha=0.7)
                
                # 添加数值标签
                for bar, increase in zip(bars, diameter_increases):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{increase:.1f}%', ha='center', va='bottom')
                           
                ax.set_ylabel('直径增加率 (%)')
                ax.set_title('节点移除对网络的影响')
                ax.set_xticklabels(node_names, rotation=45)
            else:
                ax.text(0.5, 0.5, '无脆弱性数据', ha='center', va='center')
        else:
            ax.text(0.5, 0.5, '无脆弱性评估', ha='center', va='center')
        ax.set_title('网络鲁棒性')
        
    def save_results(self):
        """保存分析结果"""
        output_file = self.data_path / 'network_diffusion_results.json'
        
        # 转换numpy类型为Python类型
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, dict):
                # 递归转换字典的键和值
                return {convert_numpy(k): convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                # 递归转换列表
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, tuple):
                # 递归转换元组
                return tuple(convert_numpy(item) for item in obj)
            return obj
            
        # 深度转换结果
        serializable_results = convert_numpy(self.results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            
        print(f"\n结果已保存至: {output_file}")
        
        # 生成markdown报告
        self._generate_markdown_report()
        
    def _generate_markdown_report(self):
        """生成Markdown报告"""
        report_file = self.data_path.parent / 'md' / 'network_diffusion_report.md'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 网络效应与扩散分析报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 1. 认知网络结构\n")
            network_props = self.results.get('cognitive_network', {}).get('network_properties', {})
            f.write(f"- 节点数: {network_props.get('num_nodes', 0)}\n")
            f.write(f"- 边数: {network_props.get('num_edges', 0)}\n")
            f.write(f"- 网络密度: {network_props.get('density', 0):.3f}\n")
            f.write(f"- 平均聚类系数: {network_props.get('average_clustering', 0):.3f}\n\n")
            
            f.write("## 2. 扩散模式\n")
            diffusion = self.results.get('diffusion_patterns', {})
            temporal = diffusion.get('temporal_diffusion', {})
            f.write(f"- 总体采纳率: {temporal.get('adoption_rate', 0):.1%}\n")
            f.write(f"- 半数采纳日期: {temporal.get('half_adoption_date', 'N/A')}\n")
            f.write(f"- 扩散加速度: {temporal.get('adoption_acceleration', 0):.4f}\n\n")
            
            f.write("## 3. 网络效应\n")
            multiplier = self.results.get('network_effects', {}).get('network_multiplier', {})
            f.write(f"- 网络乘数: {multiplier.get('multiplier', 1):.2f}x\n")
            f.write(f"- 效应放大: {multiplier.get('amplification', 0):.1f}%\n\n")
            
            f.write("## 4. 关键发现\n")
            # 关键节点
            influence_nodes = self.results.get('critical_nodes', {}).get('influence_nodes', [])
            if influence_nodes:
                f.write("### 最具影响力的节点:\n")
                for node in influence_nodes[:3]:
                    f.write(f"- {node['label']} (影响力得分: {node['influence_score']:.3f})\n")
                    
            # 社区结构
            f.write(f"\n### 社区结构:\n")
            f.write(f"- 模块度: {self.results.get('community_structure', {}).get('modularity', 0):.3f}\n")
            f.write(f"- 社区数量: {len(self.results.get('community_structure', {}).get('communities', {}))}\n")
            
        print(f"报告已保存至: {report_file}")
        
    def generate_comprehensive_interpretation(self):
        """生成综合解释"""
        print("\n" + "="*60)
        print("网络效应与扩散分析综合解释")
        print("="*60)
        
        # 1. 网络结构解释
        network_props = self.results.get('cognitive_network', {}).get('network_properties', {})
        if network_props:
            print("\n1. 认知网络特征：")
            density = network_props.get('density', 0)
            if density > 0.5:
                print(f"   - 高密度网络（{density:.3f}），认知要素间联系紧密")
            else:
                print(f"   - 稀疏网络（{density:.3f}），存在改进空间")
                
        # 2. 扩散特征解释
        adoption_rate = self.results.get('diffusion_patterns', {}).get('temporal_diffusion', {}).get('adoption_rate', 0)
        if adoption_rate > 0:
            print("\n2. 扩散特征：")
            if adoption_rate > 0.7:
                print(f"   - 高采纳率（{adoption_rate:.1%}），DSR已广泛扩散")
            elif adoption_rate > 0.3:
                print(f"   - 中等采纳率（{adoption_rate:.1%}），扩散进行中")
            else:
                print(f"   - 低采纳率（{adoption_rate:.1%}），扩散初期")
                
        # 3. 网络效应解释
        multiplier = self.results.get('network_effects', {}).get('network_multiplier', {}).get('multiplier', 1)
        if multiplier > 1:
            print("\n3. 网络效应：")
            print(f"   - 存在{multiplier:.2f}倍的网络放大效应")
            print("   - 高密度环境下DSR的作用显著增强")
            
        # 4. 关键节点解释
        critical_nodes = self.results.get('critical_nodes', {}).get('influence_nodes', [])
        if critical_nodes:
            print("\n4. 关键节点：")
            top_node = critical_nodes[0]
            print(f"   - 最关键节点：{top_node['label']}（{top_node['type']}类型）")
            print("   - 该节点在网络中起到枢纽作用")
            
        print("\n" + "="*60)

def main():
    """主函数"""
    # 设置数据路径 - 使用绝对路径或从当前脚本位置推断
    import os
    
    # 获取脚本所在目录
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    project_root = script_dir.parent  # Figure目录
    
    # 尝试多个可能的数据路径
    possible_paths = [
        project_root / 'output_cn' / 'data',
        Path('output_cn/data'),
        Path('G:/Project/Figure/output_cn/data'),
        Path('./output_cn/data')
    ]
    
    data_path = None
    for path in possible_paths:
        if path.exists():
            data_path = path
            break
            
    if data_path is None:
        # 使用默认路径
        data_path = project_root / 'output_cn' / 'data'
        print(f"警告: 数据目录未找到，使用默认路径: {data_path}")
    
    # 初始化分析器
    analyzer = NetworkDiffusionAnalysis(data_path)
    
    try:
        # 加载数据
        analyzer.load_data()
        
        # 运行分析
        results = analyzer.run_network_diffusion_analysis()
        
        print("\n" + "="*60)
        print("网络效应与扩散分析完成！")
        print("="*60)
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()