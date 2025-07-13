# step5g_bayesian_changepoint.py
# 贝叶斯变点检测：识别认知构成性的关键转折点

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

# 尝试导入PyMC（版本5+）
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    try:
        # 如果PyMC5不可用，尝试PyMC3（虽然已停止维护）
        import pymc3 as pm
        import arviz as az
        PYMC_AVAILABLE = True
        print("警告: 使用的是PyMC3（已停止维护），建议升级到PyMC5+")
    except ImportError:
        PYMC_AVAILABLE = False
        print("警告: PyMC未安装，将使用简化的贝叶斯方法")

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class BayesianChangepointDetector:
    """贝叶斯变点检测器"""
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.df = None
        self.results = {
            'changepoints': {},
            'regime_analysis': {},
            'probability_curves': {},
            'model_comparison': {},
            'temporal_dynamics': {}
        }
        
    def load_data(self):
        """加载数据"""
        # 尝试加载混合方法分析后的数据
        mixed_file = self.data_path / 'data_with_mixed_methods.csv'
        if mixed_file.exists():
            self.df = pd.read_csv(mixed_file, encoding='utf-8-sig')
        else:
            # 回退到基础增强数据
            self.df = pd.read_csv(self.data_path / 'data_with_pattern_metrics.csv', encoding='utf-8-sig')
            
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        print("="*60)
        print("贝叶斯变点检测分析")
        print("="*60)
        print(f"数据加载完成: {len(self.df)} 条记录")
        print(f"时间范围: {self.df['date'].min()} 至 {self.df['date'].max()}")
        
        return self.df
        
    def run_changepoint_analysis(self):
        """运行变点检测分析"""
        
        print("\n1. 数据预处理与聚合")
        self.prepare_time_series()
        
        print("\n2. 贝叶斯变点检测")
        self.detect_changepoints()
        
        print("\n3. 机制转换分析")
        self.analyze_regime_shifts()
        
        print("\n4. 构成性演化路径")
        self.analyze_constitutive_evolution()
        
        print("\n5. 预测性验证")
        self.validate_predictions()
        
        # 生成可视化
        self.create_visualizations()
        
        # 保存结果
        self.save_results()
        
        return self.results
        
    def prepare_time_series(self):
        """准备时间序列数据"""
        # 按日期聚合
        daily_data = self.df.groupby('date').agg({
            'dsr_cognitive': 'mean',
            'tl_functional': 'mean',
            'cs_output': 'mean',
            'constitutive_index': 'mean',
            'ded_functions': lambda x: x.mode()[0] if len(x) > 0 else None,
            'sensitivity_code': 'mean'
        }).reset_index()
        
        # 计算构成性综合指标
        daily_data['constitutive_strength'] = (
            daily_data['dsr_cognitive'] * 0.3 +
            daily_data['tl_functional'] * 0.3 +
            daily_data['cs_output'] * 0.4
        )
        
        # 平滑处理
        from scipy.ndimage import gaussian_filter1d
        daily_data['constitutive_smooth'] = gaussian_filter1d(
            daily_data['constitutive_strength'].fillna(method='ffill'), 
            sigma=2
        )
        
        self.time_series = daily_data
        print(f"  生成时间序列: {len(daily_data)} 个时间点")
        
    def detect_changepoints(self):
        """贝叶斯变点检测"""
        y = self.time_series['constitutive_smooth'].values
        n = len(y)
        
        print("  运行在线贝叶斯变点检测...")
        
        # 1. 在线贝叶斯变点检测 (Adams & MacKay, 2007)
        changepoint_probs = self._online_changepoint_detection(y)
        
        # 2. PELT算法作为对比
        pelt_changepoints = self._pelt_detection(y)
        
        # 3. 贝叶斯模型比较
        model_evidence = self._compare_models(y)
        
        # 识别主要变点
        threshold = 0.01  # 大幅降低阈值以检测更多变点
        major_changepoints = []
        for i in range(1, n):
            if changepoint_probs[i] > threshold:
                major_changepoints.append({
                    'index': i,
                    'date': self.time_series.iloc[i]['date'],
                    'probability': changepoint_probs[i],
                    'before_mean': np.mean(y[max(0, i-30):i]),
                    'after_mean': np.mean(y[i:min(n, i+30)]),
                    'magnitude': abs(np.mean(y[i:min(n, i+30)]) - np.mean(y[max(0, i-30):i]))
                })
        
        # 按概率排序
        major_changepoints = sorted(major_changepoints, 
                                  key=lambda x: x['probability'], 
                                  reverse=True)[:5]  # Top 5
        
        self.results['changepoints'] = {
            'major_changepoints': major_changepoints,
            'probability_curve': changepoint_probs.tolist(),
            'pelt_changepoints': pelt_changepoints,
            'model_evidence': model_evidence
        }
        
        print(f"  识别到 {len(major_changepoints)} 个主要变点")
        
    def _online_changepoint_detection(self, data, hazard_rate=1/50):
        """在线贝叶斯变点检测算法"""
        n = len(data)
        R = np.zeros((n+1, n+1))
        R[0, 0] = 1
        
        changepoint_probs = np.zeros(n)
        
        # 使用正态-逆伽马共轭先验
        alpha = 1
        beta = 1
        kappa = 1
        mu = np.mean(data[:10]) if len(data) > 10 else 0
        
        for t in range(1, n+1):
            # 预测概率
            predprobs = np.zeros(t)
            
            for s in range(t):
                # 后验参数
                n_s = t - s
                kappa_n = kappa + n_s
                mu_n = (kappa * mu + np.sum(data[s:t])) / kappa_n
                alpha_n = alpha + n_s / 2
                beta_n = beta + 0.5 * np.sum((data[s:t] - mu_n)**2) + \
                         (kappa * n_s * (mu_n - mu)**2) / (2 * kappa_n)
                
                # 学生t分布的预测概率
                scale = np.sqrt(beta_n * (kappa_n + 1) / (alpha_n * kappa_n))
                df = 2 * alpha_n
                
                if t < n:
                    predprobs[s] = stats.t.pdf(data[t], df, mu_n, scale)
            
            # 增长概率
            R[1:t+1, t] = R[:t, t-1] * predprobs * (1 - hazard_rate)
            
            # 变点概率
            R[0, t] = np.sum(R[:t, t-1] * hazard_rate)
            
            # 归一化
            R[:, t] = R[:, t] / np.sum(R[:, t])
            
            # 变点概率
            if t > 1:
                changepoint_probs[t-1] = R[0, t-1]
        
        return changepoint_probs
        
    def _pelt_detection(self, data):
        """PELT (Pruned Exact Linear Time) 算法"""
        try:
            import ruptures as rpt
            
            # 使用PELT算法
            algo = rpt.Pelt(model="rbf").fit(data)
            result = algo.predict(pen=10)
            
            # 移除最后一个点（总是数据长度）
            changepoints = result[:-1]
            
            return changepoints
        except ImportError:
            print("    警告: ruptures库未安装，跳过PELT检测")
            return []
            
    def _compare_models(self, data):
        """贝叶斯模型比较"""
        n = len(data)
        
        # 模型1: 无变点
        model1_evidence = self._calculate_model_evidence(data, [])
        
        # 模型2: 1个变点
        best_single_cp = None
        best_single_evidence = -np.inf
        
        for cp in range(int(n*0.2), int(n*0.8)):
            evidence = self._calculate_model_evidence(data, [cp])
            if evidence > best_single_evidence:
                best_single_evidence = evidence
                best_single_cp = cp
                
        # 模型3: 2个变点
        best_double_cp = None
        best_double_evidence = -np.inf
        
        # 简化搜索
        for cp1 in range(int(n*0.15), int(n*0.45), 5):
            for cp2 in range(int(n*0.55), int(n*0.85), 5):
                evidence = self._calculate_model_evidence(data, [cp1, cp2])
                if evidence > best_double_evidence:
                    best_double_evidence = evidence
                    best_double_cp = [cp1, cp2]
        
        return {
            'no_changepoint': model1_evidence,
            'single_changepoint': {
                'evidence': best_single_evidence,
                'location': best_single_cp
            },
            'double_changepoint': {
                'evidence': best_double_evidence,
                'locations': best_double_cp
            },
            'bayes_factor_single': np.exp(best_single_evidence - model1_evidence),
            'bayes_factor_double': np.exp(best_double_evidence - model1_evidence)
        }
        
    def _calculate_model_evidence(self, data, changepoints):
        """计算模型证据（边际似然）"""
        n = len(data)
        changepoints = [0] + changepoints + [n]
        
        log_evidence = 0
        
        for i in range(len(changepoints)-1):
            segment = data[changepoints[i]:changepoints[i+1]]
            if len(segment) > 1:
                # 使用贝叶斯信息准则的近似
                n_seg = len(segment)
                mean_seg = np.mean(segment)
                var_seg = np.var(segment)
                
                # 对数边际似然的近似
                log_evidence += -0.5 * n_seg * np.log(2 * np.pi * var_seg) - \
                               0.5 * n_seg - \
                               0.5 * np.log(n_seg)  # 先验惩罚
                               
        # 变点数量的先验惩罚
        k = len(changepoints) - 2
        log_evidence -= k * np.log(n)  # 类似BIC的惩罚
        
        return log_evidence
        
    def analyze_regime_shifts(self):
        """分析机制转换"""
        changepoints = self.results['changepoints']['major_changepoints']
        
        if not changepoints:
            print("  未检测到显著变点")
            # 即使没有变点，也要设置基本结果
            self.results['regime_analysis'] = {
                'segments': [],
                'transitions': [],
                'total_regimes': 1  # 只有一个状态
            }
            return
            
        # 将时间序列分段
        segments = []
        cp_indices = [0] + [cp['index'] for cp in changepoints] + [len(self.time_series)]
        
        for i in range(len(cp_indices)-1):
            start_idx = cp_indices[i]
            end_idx = cp_indices[i+1]
            
            segment_data = self.time_series.iloc[start_idx:end_idx]
            
            if len(segment_data) > 5:  # 确保段足够长
                segment_stats = {
                    'period': f"{segment_data.iloc[0]['date'].strftime('%Y-%m')} 至 "
                             f"{segment_data.iloc[-1]['date'].strftime('%Y-%m')}",
                    'duration_days': len(segment_data),
                    'mean_constitutive': segment_data['constitutive_strength'].mean(),
                    'std_constitutive': segment_data['constitutive_strength'].std(),
                    'trend': self._calculate_trend(segment_data['constitutive_strength']),
                    'dominant_function': segment_data['ded_functions'].mode()[0] if len(segment_data) > 0 else None,
                    'avg_sensitivity': segment_data['sensitivity_code'].mean()
                }
                
                # 计算机制特征
                segment_stats['mechanism_profile'] = self._profile_mechanism(segment_data)
                
                segments.append(segment_stats)
        
        # 识别机制转换模式
        regime_transitions = []
        for i in range(len(segments)-1):
            transition = {
                'from_period': segments[i]['period'],
                'to_period': segments[i+1]['period'],
                'constitutive_change': segments[i+1]['mean_constitutive'] - segments[i]['mean_constitutive'],
                'stability_change': segments[i+1]['std_constitutive'] - segments[i]['std_constitutive'],
                'mechanism_shift': self._compare_mechanisms(
                    segments[i]['mechanism_profile'],
                    segments[i+1]['mechanism_profile']
                )
            }
            regime_transitions.append(transition)
            
        self.results['regime_analysis'] = {
            'segments': segments,
            'transitions': regime_transitions,
            'total_regimes': len(segments)
        }
        
        print(f"  识别到 {len(segments)} 个不同的机制状态")
        
    def _calculate_trend(self, series):
        """计算趋势"""
        if len(series) < 3:
            return 'insufficient_data'
            
        x = np.arange(len(series))
        y = series.values
        
        # 去除NaN
        mask = ~np.isnan(y)
        if np.sum(mask) < 3:
            return 'insufficient_data'
            
        slope, _, r_value, p_value, _ = stats.linregress(x[mask], y[mask])
        
        if p_value < 0.05:
            if slope > 0.001:
                return 'increasing'
            elif slope < -0.001:
                return 'decreasing'
        return 'stable'
        
    def _profile_mechanism(self, segment_data):
        """分析段内的主导机制"""
        profile = {
            'dsr_dominance': segment_data['dsr_cognitive'].mean() / segment_data['cs_output'].mean(),
            'tl_dominance': segment_data['tl_functional'].mean() / segment_data['cs_output'].mean(),
            'integration_level': np.corrcoef(segment_data['dsr_cognitive'], 
                                           segment_data['tl_functional'])[0, 1],
            'volatility': segment_data['constitutive_strength'].std(),
            'adaptability': abs(segment_data['constitutive_strength'].diff().mean())
        }
        
        # 确定主导机制类型
        if profile['dsr_dominance'] > 0.6:
            profile['dominant_mechanism'] = 'dsr_driven'
        elif profile['tl_dominance'] > 0.6:
            profile['dominant_mechanism'] = 'tl_driven'
        elif profile['integration_level'] > 0.5:
            profile['dominant_mechanism'] = 'integrated'
        else:
            profile['dominant_mechanism'] = 'transitional'
            
        return profile
        
    def _compare_mechanisms(self, profile1, profile2):
        """比较两个机制配置"""
        changes = []
        
        # 主导机制变化
        if profile1['dominant_mechanism'] != profile2['dominant_mechanism']:
            changes.append(f"{profile1['dominant_mechanism']} → {profile2['dominant_mechanism']}")
            
        # 整合水平变化
        integration_change = profile2['integration_level'] - profile1['integration_level']
        if abs(integration_change) > 0.2:
            direction = "增强" if integration_change > 0 else "减弱"
            changes.append(f"整合水平{direction} ({integration_change:.2f})")
            
        # 稳定性变化
        volatility_change = profile2['volatility'] - profile1['volatility']
        if abs(volatility_change) > 0.05:
            direction = "增加" if volatility_change > 0 else "降低"
            changes.append(f"波动性{direction} ({volatility_change:.3f})")
            
        return changes
        
    def analyze_constitutive_evolution(self):
        """分析构成性演化路径"""
        # 使用滑动窗口分析构成性的动态演化
        window_size = 30  # 30天窗口
        
        evolution_metrics = []
        
        for i in range(window_size, len(self.time_series)):
            window_data = self.time_series.iloc[i-window_size:i]
            
            metrics = {
                'date': self.time_series.iloc[i]['date'],
                'constitutive_level': window_data['constitutive_strength'].mean(),
                'integration_strength': np.corrcoef(
                    window_data['dsr_cognitive'],
                    window_data['tl_functional']
                )[0, 1],
                'emergence_indicator': self._calculate_emergence(window_data),
                'path_dependency': self._calculate_path_dependency(window_data, i)
            }
            
            evolution_metrics.append(metrics)
            
        evolution_df = pd.DataFrame(evolution_metrics)
        
        # 识别演化阶段
        phases = self._identify_evolution_phases(evolution_df)
        
        self.results['temporal_dynamics'] = {
            'evolution_metrics': evolution_metrics[-100:],  # 最近100个点
            'evolution_phases': phases,
            'current_phase': phases[-1] if phases else None,
            'trajectory': self._classify_trajectory(evolution_df)
        }
        
        print(f"  识别到 {len(phases)} 个演化阶段")
        
    def _calculate_emergence(self, window_data):
        """计算涌现指标"""
        # 组件之和
        linear_sum = window_data['dsr_cognitive'].mean() + window_data['tl_functional'].mean()
        
        # 实际输出
        actual_output = window_data['cs_output'].mean()
        
        # 涌现 = 实际输出 / 线性和
        emergence = actual_output / linear_sum if linear_sum > 0 else 0
        
        return emergence
        
    def _calculate_path_dependency(self, window_data, current_index):
        """计算路径依赖性"""
        if current_index < 60:  # 需要足够的历史数据
            return 0
            
        # 比较当前窗口与历史窗口的相似性
        history_window = self.time_series.iloc[current_index-60:current_index-30]
        
        # 使用相关系数衡量模式相似性
        pattern_similarity = np.corrcoef(
            window_data['constitutive_strength'].values,
            history_window['constitutive_strength'].values[-30:]
        )[0, 1]
        
        return abs(pattern_similarity)
        
    def _identify_evolution_phases(self, evolution_df):
        """识别演化阶段"""
        phases = []
        
        # 使用K-means聚类识别不同阶段
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        features = ['constitutive_level', 'integration_strength', 
                   'emergence_indicator', 'path_dependency']
        X = evolution_df[features].fillna(0)
        
        if len(X) > 10:
            # 标准化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 确定最佳聚类数（2-5个阶段）
            best_k = 3
            best_score = -np.inf
            
            for k in range(2, min(6, len(X)//10)):
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(X_scaled)
                
                # 使用轮廓系数
                from sklearn.metrics import silhouette_score
                score = silhouette_score(X_scaled, labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
                    
            # 最终聚类
            kmeans = KMeans(n_clusters=best_k, random_state=42)
            evolution_df['phase'] = kmeans.fit_predict(X_scaled)
            
            # 整理阶段信息
            for phase_id in range(best_k):
                phase_data = evolution_df[evolution_df['phase'] == phase_id]
                
                if len(phase_data) > 0:
                    phase_info = {
                        'phase_id': phase_id,
                        'start_date': phase_data['date'].min(),
                        'end_date': phase_data['date'].max(),
                        'duration_days': len(phase_data),
                        'characteristics': {
                            'avg_constitutive': phase_data['constitutive_level'].mean(),
                            'avg_integration': phase_data['integration_strength'].mean(),
                            'avg_emergence': phase_data['emergence_indicator'].mean(),
                            'avg_path_dependency': phase_data['path_dependency'].mean()
                        },
                        'phase_type': self._classify_phase_type(phase_data)
                    }
                    phases.append(phase_info)
                    
        return sorted(phases, key=lambda x: x['start_date'])
        
    def _classify_phase_type(self, phase_data):
        """分类演化阶段类型"""
        chars = phase_data['characteristics'] if 'characteristics' in phase_data.columns else {
            'avg_constitutive': phase_data['constitutive_level'].mean(),
            'avg_integration': phase_data['integration_strength'].mean(),
            'avg_emergence': phase_data['emergence_indicator'].mean()
        }
        
        # 基于特征判断阶段类型
        if chars['avg_emergence'] > 0.6:
            return 'high_emergence'
        elif chars['avg_integration'] > 0.5:
            return 'integrated'
        elif chars['avg_constitutive'] > 0.5:
            return 'constitutive'
        else:
            return 'transitional'
            
    def _classify_trajectory(self, evolution_df):
        """分类整体演化轨迹"""
        # 计算趋势
        early_mean = evolution_df['constitutive_level'].iloc[:len(evolution_df)//3].mean()
        late_mean = evolution_df['constitutive_level'].iloc[-len(evolution_df)//3:].mean()
        
        change_rate = (late_mean - early_mean) / early_mean if early_mean > 0 else 0
        
        if change_rate > 0.1:
            return 'strengthening'
        elif change_rate < -0.1:
            return 'weakening'
        else:
            return 'stable'
            
    def validate_predictions(self):
        """预测性验证"""
        # 基于识别的变点和机制，进行预测验证
        
        changepoints = self.results['changepoints']['major_changepoints']
        
        if not changepoints:
            print("  无变点，跳过预测验证")
            return
            
        validation_results = []
        
        for cp in changepoints[:3]:  # 验证前3个主要变点
            # 使用变点前的数据预测变点后的趋势
            cp_idx = cp['index']
            
            if cp_idx > 30 and cp_idx < len(self.time_series) - 30:
                # 变点前数据
                pre_data = self.time_series.iloc[cp_idx-30:cp_idx]
                # 变点后实际数据
                post_actual = self.time_series.iloc[cp_idx:cp_idx+30]
                
                # 简单预测：基于变点前的趋势
                pre_trend = self._calculate_trend(pre_data['constitutive_strength'])
                
                # 基于机制转换的预测
                if 'regime_analysis' in self.results:
                    # 查找对应的机制转换
                    regime_prediction = self._predict_based_on_regime(cp_idx)
                else:
                    regime_prediction = None
                    
                validation = {
                    'changepoint_date': cp['date'],
                    'pre_trend': pre_trend,
                    'actual_change': post_actual['constitutive_strength'].mean() - 
                                   pre_data['constitutive_strength'].mean(),
                    'regime_prediction': regime_prediction,
                    'prediction_accuracy': self._evaluate_prediction(
                        pre_data, post_actual, regime_prediction
                    )
                }
                
                validation_results.append(validation)
                
        self.results['validation'] = {
            'changepoint_validations': validation_results,
            'overall_accuracy': np.mean([v['prediction_accuracy'] 
                                       for v in validation_results])
        }
        
        print(f"  预测准确率: {self.results['validation']['overall_accuracy']:.2%}")
        
    def _predict_based_on_regime(self, cp_idx):
        """基于机制转换进行预测"""
        # 查找变点对应的机制转换
        for transition in self.results['regime_analysis']['transitions']:
            # 简化的匹配逻辑
            if abs(transition['constitutive_change']) > 0.05:
                return {
                    'predicted_direction': 'increase' if transition['constitutive_change'] > 0 else 'decrease',
                    'predicted_magnitude': abs(transition['constitutive_change'])
                }
        return None
        
    def _evaluate_prediction(self, pre_data, post_data, prediction):
        """评估预测准确性"""
        actual_change = post_data['constitutive_strength'].mean() - \
                       pre_data['constitutive_strength'].mean()
        
        if prediction and 'predicted_direction' in prediction:
            # 方向预测
            direction_correct = (
                (prediction['predicted_direction'] == 'increase' and actual_change > 0) or
                (prediction['predicted_direction'] == 'decrease' and actual_change < 0)
            )
            
            # 幅度预测
            if 'predicted_magnitude' in prediction:
                magnitude_error = abs(prediction['predicted_magnitude'] - abs(actual_change))
                magnitude_accuracy = 1 - min(magnitude_error / abs(actual_change), 1)
            else:
                magnitude_accuracy = 0.5
                
            return 0.6 * direction_correct + 0.4 * magnitude_accuracy
        else:
            # 基于趋势延续的简单预测
            pre_trend_slope = np.polyfit(range(len(pre_data)), 
                                       pre_data['constitutive_strength'].values, 1)[0]
            predicted_change = pre_trend_slope * len(post_data)
            
            error = abs(predicted_change - actual_change)
            return 1 - min(error / (abs(actual_change) + 0.01), 1)
            
    def create_visualizations(self):
        """创建可视化"""
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 主标题
        fig.suptitle('贝叶斯变点检测：认知构成性的演化分析', fontsize=24, fontweight='bold')
        
        # 1. 时间序列与变点（上部整行）
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_changepoints(ax1)
        
        # 2. 变点概率曲线（中左）
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_probability_curve(ax2)
        
        # 3. 机制状态转换（中中）
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_regime_transitions(ax3)
        
        # 4. 演化阶段（中右）
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_evolution_phases(ax4)
        
        # 5. 模型比较（下左）
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_model_comparison(ax5)
        
        # 6. 预测验证（下中）
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_validation_results(ax6)
        
        # 7. 关键发现（下右）
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_key_findings(ax7)
        
        # 保存
        plt.tight_layout()
        save_path = self.data_path.parent / 'figures' / 'bayesian_changepoint_analysis.jpg'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n可视化已保存至: {save_path}")
        
    def _plot_changepoints(self, ax):
        """绘制时间序列与变点"""
        # 绘制构成性强度
        ax.plot(self.time_series['date'], 
               self.time_series['constitutive_smooth'],
               'b-', linewidth=2, label='构成性强度（平滑）')
        
        # 标记主要变点
        changepoints = self.results['changepoints']['major_changepoints']
        for i, cp in enumerate(changepoints):
            ax.axvline(x=cp['date'], color='red', linestyle='--', 
                      alpha=0.7, linewidth=2)
            ax.text(cp['date'], ax.get_ylim()[1]*0.95, 
                   f"CP{i+1}\n({cp['probability']:.2f})",
                   ha='center', fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
            
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('构成性强度', fontsize=12)
        ax.set_title('构成性演化时间序列与主要变点', fontsize=16, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
    def _plot_probability_curve(self, ax):
        """绘制变点概率曲线"""
        probs = self.results['changepoints']['probability_curve']
        
        ax.plot(probs, 'g-', linewidth=2)
        ax.axhline(y=0.01, color='red', linestyle='--', alpha=0.5, label='阈值=0.01')
        
        # 标记峰值
        peaks = np.where(np.array(probs) > 0.01)[0]
        if len(peaks) > 0:
            ax.scatter(peaks, [probs[i] for i in peaks], 
                      color='red', s=100, zorder=5)
            
        ax.set_xlabel('时间索引', fontsize=12)
        ax.set_ylabel('变点概率', fontsize=12)
        ax.set_title('贝叶斯变点概率', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_regime_transitions(self, ax):
        """绘制机制状态转换"""
        if 'regime_analysis' not in self.results or 'segments' not in self.results.get('regime_analysis', {}):
            ax.text(0.5, 0.5, '无机制转换数据', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return
            
        segments = self.results['regime_analysis']['segments']
        
        # 创建状态转换图
        states = []
        values = []
        
        for seg in segments:
            states.append(seg['period'].split(' 至 ')[0])
            values.append(seg['mean_constitutive'])
            
        x = range(len(states))
        bars = ax.bar(x, values, color=plt.cm.viridis(np.linspace(0, 1, len(states))))
        
        # 添加机制类型标签
        for i, seg in enumerate(segments):
            mechanism = seg['mechanism_profile']['dominant_mechanism']
            ax.text(i, values[i] + 0.01, mechanism, 
                   ha='center', fontsize=9, rotation=45)
            
        ax.set_xticks(x)
        ax.set_xticklabels(states, rotation=45, ha='right')
        ax.set_ylabel('平均构成性', fontsize=12)
        ax.set_title('机制状态演化', fontsize=16, fontweight='bold')
        
    def _plot_evolution_phases(self, ax):
        """绘制演化阶段"""
        if 'temporal_dynamics' not in self.results:
            ax.text(0.5, 0.5, '无演化阶段数据', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return
            
        phases = self.results['temporal_dynamics']['evolution_phases']
        
        # 创建甘特图式的阶段展示
        for i, phase in enumerate(phases):
            start = phase['start_date']
            end = phase['end_date']
            
            # 使用颜色编码阶段类型
            color_map = {
                'high_emergence': '#e74c3c',
                'integrated': '#3498db',
                'constitutive': '#2ecc71',
                'transitional': '#95a5a6'
            }
            color = color_map.get(phase['phase_type'], '#34495e')
            
            ax.barh(i, (end - start).days, 
                   left=0, height=0.8,
                   color=color, alpha=0.7,
                   label=phase['phase_type'] if i == 0 else "")
            
            # 添加特征值
            text = f"{phase['characteristics']['avg_constitutive']:.2f}"
            ax.text((end - start).days / 2, i, text,
                   ha='center', va='center', fontsize=10)
                   
        ax.set_yticks(range(len(phases)))
        ax.set_yticklabels([f"阶段{i+1}" for i in range(len(phases))])
        ax.set_xlabel('持续天数', fontsize=12)
        ax.set_title('构成性演化阶段', fontsize=16, fontweight='bold')
        
        # 添加图例
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), 
                 loc='upper right', fontsize=10)
        
    def _plot_model_comparison(self, ax):
        """绘制模型比较"""
        if 'model_evidence' not in self.results['changepoints']:
            ax.text(0.5, 0.5, '无模型比较数据', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return
            
        evidence = self.results['changepoints']['model_evidence']
        
        models = ['无变点', '单变点', '双变点']
        bayes_factors = [1, 
                        evidence['bayes_factor_single'],
                        evidence['bayes_factor_double']]
        
        bars = ax.bar(models, bayes_factors, 
                      color=['#95a5a6', '#3498db', '#e74c3c'],
                      alpha=0.8)
        
        # 添加数值标签
        for bar, bf in zip(bars, bayes_factors):
            ax.text(bar.get_x() + bar.get_width()/2, 
                   bar.get_height() + 0.1,
                   f'{bf:.2f}', ha='center', fontsize=11)
            
        ax.set_ylabel('贝叶斯因子', fontsize=12)
        ax.set_title('模型比较（贝叶斯因子）', fontsize=16, fontweight='bold')
        ax.axhline(y=3, color='red', linestyle='--', alpha=0.5, 
                  label='正面证据阈值')
        ax.legend()
        
    def _plot_validation_results(self, ax):
        """绘制预测验证结果"""
        if 'validation' not in self.results:
            ax.text(0.5, 0.5, '无验证数据', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return
            
        validations = self.results['validation']['changepoint_validations']
        
        if not validations:
            ax.text(0.5, 0.5, '无有效验证结果', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return
            
        # 绘制预测准确率
        dates = [v['changepoint_date'] for v in validations]
        accuracies = [v['prediction_accuracy'] for v in validations]
        
        bars = ax.bar(range(len(dates)), accuracies, 
                      color='green', alpha=0.7)
        
        # 添加平均线
        avg_accuracy = self.results['validation']['overall_accuracy']
        ax.axhline(y=avg_accuracy, color='red', linestyle='--', 
                  linewidth=2, label=f'平均准确率: {avg_accuracy:.2%}')
        
        ax.set_xticks(range(len(dates)))
        ax.set_xticklabels([d.strftime('%Y-%m') for d in dates], 
                          rotation=45, ha='right')
        ax.set_ylabel('预测准确率', fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_title('变点预测验证', fontsize=16, fontweight='bold')
        ax.legend()
        
    def _plot_key_findings(self, ax):
        """绘制关键发现"""
        ax.axis('off')
        ax.set_title('关键发现', fontsize=16, fontweight='bold')
        
        findings = []
        
        # 变点发现
        n_changepoints = len(self.results['changepoints']['major_changepoints'])
        findings.append(f"1. 识别到 {n_changepoints} 个主要变点")
        
        # 机制状态
        if 'regime_analysis' in self.results and 'total_regimes' in self.results['regime_analysis']:
            n_regimes = self.results['regime_analysis']['total_regimes']
            findings.append(f"2. 系统经历 {n_regimes} 个不同状态")
            
        # 演化轨迹
        if 'temporal_dynamics' in self.results:
            trajectory = self.results['temporal_dynamics']['trajectory']
            findings.append(f"3. 整体演化轨迹: {trajectory}")
            
        # 预测准确性
        if 'validation' in self.results:
            accuracy = self.results['validation']['overall_accuracy']
            findings.append(f"4. 预测准确率: {accuracy:.1%}")
            
        # 显示发现
        for i, finding in enumerate(findings):
            ax.text(0.05, 0.85 - i*0.15, finding, 
                   fontsize=12, va='top',
                   bbox=dict(boxstyle="round,pad=0.4", 
                           facecolor='#ecf0f1', alpha=0.8))
                           
    def save_results(self):
        """保存分析结果"""
        # 保存为JSON
        output_file = self.data_path / 'bayesian_changepoint_results.json'
        
        # 转换结果为可序列化格式
        def convert_to_serializable(obj):
            if isinstance(obj, (pd.Timestamp, datetime)):
                return str(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
            
        serializable_results = convert_to_serializable(self.results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            
        print(f"结果已保存至: {output_file}")
        
        # 生成报告
        self.generate_report()
        
    def generate_report(self):
        """生成分析报告"""
        report = "# 贝叶斯变点检测分析报告\n\n"
        report += f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        report += "## 1. 主要变点\n\n"
        changepoints = self.results['changepoints']['major_changepoints']
        
        if changepoints:
            report += "| 日期 | 概率 | 变化幅度 | 前均值 | 后均值 |\n"
            report += "|------|------|---------|--------|--------|\n"
            
            for cp in changepoints:
                report += f"| {cp['date'].strftime('%Y-%m-%d')} | "
                report += f"{cp['probability']:.3f} | "
                report += f"{cp['magnitude']:.3f} | "
                report += f"{cp['before_mean']:.3f} | "
                report += f"{cp['after_mean']:.3f} |\n"
        else:
            report += "未检测到显著变点。\n"
            
        report += "\n## 2. 机制状态分析\n\n"
        if 'regime_analysis' in self.results:
            segments = self.results['regime_analysis']['segments']
            
            for i, seg in enumerate(segments):
                report += f"### 状态 {i+1}: {seg['period']}\n"
                report += f"- 持续天数: {seg['duration_days']}\n"
                report += f"- 平均构成性: {seg['mean_constitutive']:.3f}\n"
                report += f"- 趋势: {seg['trend']}\n"
                report += f"- 主导机制: {seg['mechanism_profile']['dominant_mechanism']}\n\n"
                
        report += "## 3. 演化轨迹\n\n"
        if 'temporal_dynamics' in self.results:
            trajectory = self.results['temporal_dynamics']['trajectory']
            report += f"整体演化轨迹判定为: **{trajectory}**\n\n"
            
            phases = self.results['temporal_dynamics']['evolution_phases']
            if phases:
                report += "### 演化阶段:\n"
                for i, phase in enumerate(phases):
                    report += f"\n**阶段 {i+1}** ({phase['phase_type']})\n"
                    report += f"- 时间: {phase['start_date'].strftime('%Y-%m')} 至 "
                    report += f"{phase['end_date'].strftime('%Y-%m')}\n"
                    report += f"- 特征:\n"
                    for key, value in phase['characteristics'].items():
                        report += f"  - {key}: {value:.3f}\n"
                        
        report += "\n## 4. 模型比较\n\n"
        if 'model_evidence' in self.results['changepoints']:
            evidence = self.results['changepoints']['model_evidence']
            report += f"- 单变点贝叶斯因子: {evidence['bayes_factor_single']:.2f}\n"
            report += f"- 双变点贝叶斯因子: {evidence['bayes_factor_double']:.2f}\n\n"
            
            if evidence['bayes_factor_single'] > 3:
                report += "强烈支持存在至少一个变点。\n"
            elif evidence['bayes_factor_single'] > 1:
                report += "倾向于支持存在变点。\n"
            else:
                report += "证据不支持存在变点。\n"
                
        report += "\n## 5. 结论\n\n"
        report += "基于贝叶斯变点检测分析：\n\n"
        
        # 总结主要发现
        if changepoints:
            report += f"1. 系统在研究期间经历了 {len(changepoints)} 次显著转变\n"
            report += f"2. 最重要的转变发生在 {changepoints[0]['date'].strftime('%Y-%m-%d')}\n"
            
        if 'regime_analysis' in self.results:
            n_regimes = self.results['regime_analysis']['total_regimes']
            report += f"3. 识别出 {n_regimes} 个不同的运行状态\n"
            
        if 'temporal_dynamics' in self.results:
            trajectory = self.results['temporal_dynamics']['trajectory']
            report += f"4. 构成性整体呈现{trajectory}趋势\n"
            
        # 保存报告
        report_file = self.data_path / 'bayesian_changepoint_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
            
        print(f"报告已保存至: {report_file}")

def main():
    """主函数"""
    # 设置路径
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_path = project_root / 'output_cn' / 'data'
    
    # 创建检测器
    detector = BayesianChangepointDetector(data_path)
    
    # 加载数据
    detector.load_data()
    
    # 运行分析
    results = detector.run_changepoint_analysis()
    
    print("\n" + "="*60)
    print("贝叶斯变点检测完成")
    print("="*60)
    
    # 输出关键结果
    n_changepoints = len(results['changepoints']['major_changepoints'])
    print(f"\n识别到 {n_changepoints} 个主要变点")
    
    if 'regime_analysis' in results:
        n_regimes = results['regime_analysis']['total_regimes']
        print(f"系统经历 {n_regimes} 个不同状态")
        
    if 'temporal_dynamics' in results:
        trajectory = results['temporal_dynamics']['trajectory']
        print(f"整体演化轨迹: {trajectory}")
        
    print("\n✓ 分析完成！")
    print("\n查看以下文件获取详细结果:")
    print("- bayesian_changepoint_results.json")
    print("- bayesian_changepoint_analysis.jpg")
    print("- bayesian_changepoint_report.md")

if __name__ == "__main__":
    main()