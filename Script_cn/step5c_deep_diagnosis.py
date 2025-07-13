# step5c_deep_diagnosis.py
# 深度诊断与方法论重构

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

from scipy import stats, signal
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class DeepDiagnosisAnalyzer:
    """深度诊断分析器 - 识别根本问题并提出解决方案"""
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.df = None
        self.diagnosis_results = {
            'data_quality': {},
            'signal_analysis': {},
            'pattern_detection': {},
            'method_recommendations': {},
            'revised_hypothesis': {}
        }
        
    def load_and_diagnose_data(self):
        """加载数据并进行深度诊断"""
        # 加载数据
        csv_file = self.data_path / 'data_with_metrics.csv'
        self.df = pd.read_csv(csv_file, encoding='utf-8-sig')
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        print("="*60)
        print("深度诊断分析")
        print("="*60)
        
        # 1. 数据质量诊断
        self.diagnose_data_quality()
        
        # 2. 信号分析
        self.analyze_signal_structure()
        
        # 3. 模式检测
        self.detect_meaningful_patterns()
        
        # 4. 方法论建议
        self.recommend_methods()
        
        # 5. 假设修正
        self.revise_hypothesis()
        
        # 保存结果
        self.save_diagnosis_results()
        
        return self.diagnosis_results
    
    def diagnose_data_quality(self):
        """数据质量深度诊断"""
        print("\n1. 数据质量诊断")
        print("-" * 40)
        
        # 1.1 基础统计
        key_vars = ['dsr_cognitive', 'tl_functional', 'cs_output']
        quality_metrics = {}
        
        for var in key_vars:
            if var in self.df.columns:
                data = self.df[var].dropna()
                quality_metrics[var] = {
                    'n': len(data),
                    'missing_pct': (len(self.df) - len(data)) / len(self.df) * 100,
                    'mean': float(data.mean()),
                    'std': float(data.std()),
                    'cv': float(data.std() / data.mean()) if data.mean() != 0 else np.inf,
                    'skewness': float(stats.skew(data)),
                    'kurtosis': float(stats.kurtosis(data)),
                    'unique_values': len(data.unique()),
                    'zero_variance': data.std() == 0
                }
        
        # 1.2 时间序列特性
        time_diagnostics = self._diagnose_time_series()
        
        # 1.3 数据聚合级别问题
        aggregation_issues = self._check_aggregation_level()
        
        # 1.4 测量误差估计
        measurement_error = self._estimate_measurement_error()
        
        self.diagnosis_results['data_quality'] = {
            'variable_quality': quality_metrics,
            'time_series_issues': time_diagnostics,
            'aggregation_problems': aggregation_issues,
            'measurement_error': measurement_error,
            'critical_issues': self._identify_critical_issues(quality_metrics)
        }
        
        # 输出关键发现
        print(f"  变异系数过高的变量: {[k for k, v in quality_metrics.items() if v['cv'] > 0.5]}")
        print(f"  时间序列问题: {time_diagnostics['main_issue']}")
        print(f"  聚合级别: {aggregation_issues['current_level']}")
        print(f"  估计信噪比: {measurement_error['estimated_snr']:.3f}")
    
    def _diagnose_time_series(self):
        """诊断时间序列特性"""
        # 检查时间间隔
        time_diffs = self.df['date'].diff().dropna()
        
        # 自相关检查
        acf_results = {}
        for var in ['dsr_cognitive', 'tl_functional', 'cs_output']:
            if var in self.df.columns:
                data = self.df[var].dropna()
                if len(data) > 50:
                    acf_1 = data.autocorr(lag=1)
                    acf_10 = data.autocorr(lag=10)
                    acf_results[var] = {
                        'lag1': float(acf_1) if not np.isnan(acf_1) else 0,
                        'lag10': float(acf_10) if not np.isnan(acf_10) else 0
                    }
        
        # 平稳性检验（简化版）
        stationarity = {}
        for var in ['dsr_cognitive', 'tl_functional', 'cs_output']:
            if var in self.df.columns:
                data = self.df[var].dropna().values
                if len(data) > 100:
                    # 分割数据并比较均值
                    first_half = data[:len(data)//2]
                    second_half = data[len(data)//2:]
                    t_stat, p_val = stats.ttest_ind(first_half, second_half)
                    stationarity[var] = {
                        'stationary': p_val > 0.05,
                        'p_value': float(p_val)
                    }
        
        # 判断主要问题
        if all(v.get('lag1', 0) < 0.1 for v in acf_results.values()):
            main_issue = "缺乏时间依赖性 - 数据点之间相互独立"
        elif any(not v.get('stationary', True) for v in stationarity.values()):
            main_issue = "非平稳性 - 存在趋势或结构性变化"
        else:
            main_issue = "时间结构正常"
        
        return {
            'time_interval_uniform': len(time_diffs.unique()) < 10,
            'autocorrelations': acf_results,
            'stationarity': stationarity,
            'main_issue': main_issue
        }
    
    def _check_aggregation_level(self):
        """检查数据聚合级别"""
        # 检查是否为问答单元级别
        unit_level_indicators = {
            'unique_dates': len(self.df['date'].unique()),
            'records_per_date': self.df.groupby('date').size().mean(),
            'date_coverage_days': (self.df['date'].max() - self.df['date'].min()).days
        }
        
        # 判断聚合级别
        if unit_level_indicators['records_per_date'] > 5:
            current_level = "问答单元级别（过细）"
            recommendation = "考虑按日期或会话聚合"
        elif unit_level_indicators['records_per_date'] < 1.5:
            current_level = "已聚合数据"
            recommendation = "当前聚合级别可能合适"
        else:
            current_level = "混合级别"
            recommendation = "需要统一聚合标准"
        
        return {
            'current_level': current_level,
            'indicators': unit_level_indicators,
            'recommendation': recommendation
        }
    
    def _estimate_measurement_error(self):
        """估计测量误差"""
        # 使用重复测量或相似条件下的方差估计
        error_estimates = {}
        
        for var in ['dsr_cognitive', 'tl_functional', 'cs_output']:
            if var in self.df.columns:
                # 方法1：使用短时间窗口内的方差作为误差估计
                rolling_std = self.df[var].rolling(window=10, center=True).std()
                
                # 方法2：使用残差分析
                # 简单去趋势
                x = np.arange(len(self.df))
                y = self.df[var].values
                z = np.polyfit(x, y, 3)
                p = np.poly1d(z)
                residuals = y - p(x)
                
                error_estimates[var] = {
                    'local_noise': float(rolling_std.mean()),
                    'residual_std': float(np.std(residuals)),
                    'signal_std': float(self.df[var].std()),
                    'snr': float(self.df[var].std() / rolling_std.mean())
                }
        
        # 总体信噪比估计
        avg_snr = np.mean([v['snr'] for v in error_estimates.values()])
        
        return {
            'variable_errors': error_estimates,
            'estimated_snr': avg_snr,
            'noise_dominates': avg_snr < 2
        }
    
    def _identify_critical_issues(self, quality_metrics):
        """识别关键数据质量问题"""
        issues = []
        
        # 检查各种问题
        for var, metrics in quality_metrics.items():
            if metrics['cv'] > 0.5:
                issues.append(f"{var}变异系数过高({metrics['cv']:.2f})")
            if metrics['zero_variance']:
                issues.append(f"{var}零方差")
            if abs(metrics['skewness']) > 2:
                issues.append(f"{var}严重偏态(偏度={metrics['skewness']:.2f})")
            if metrics['unique_values'] < 10:
                issues.append(f"{var}独特值过少({metrics['unique_values']})")
        
        return issues
    
    def analyze_signal_structure(self):
        """分析信号结构"""
        print("\n2. 信号结构分析")
        print("-" * 40)
        
        # 2.1 主成分分析
        pca_results = self._perform_pca_analysis()
        
        # 2.2 频域分析
        frequency_results = self._analyze_frequency_domain()
        
        # 2.3 互信息结构
        mi_structure = self._analyze_mutual_information()
        
        # 2.4 非线性检测
        nonlinearity = self._detect_nonlinearity()
        
        self.diagnosis_results['signal_analysis'] = {
            'pca': pca_results,
            'frequency': frequency_results,
            'mutual_information': mi_structure,
            'nonlinearity': nonlinearity,
            'signal_type': self._classify_signal_type(pca_results, frequency_results)
        }
        
        print(f"  首个主成分解释方差: {pca_results['explained_variance_ratio'][0]:.3f}")
        print(f"  主要频率成分: {frequency_results['dominant_frequencies']}")
        print(f"  DSR-CS互信息: {mi_structure['dsr_cs_mi']:.3f}")
        print(f"  信号类型: {self.diagnosis_results['signal_analysis']['signal_type']}")
    
    def _perform_pca_analysis(self):
        """主成分分析"""
        # 准备数据
        features = ['dsr_cognitive', 'tl_functional', 'sensitivity_code']
        if all(f in self.df.columns for f in features):
            X = self.df[features].dropna()
            
            # 标准化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # PCA
            pca = PCA()
            pca.fit(X_scaled)
            
            # 计算有效维度
            cum_var = np.cumsum(pca.explained_variance_ratio_)
            effective_dim = np.argmax(cum_var > 0.9) + 1
            
            return {
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance': cum_var.tolist(),
                'effective_dimensions': int(effective_dim),
                'first_pc_loadings': pca.components_[0].tolist()
            }
        else:
            return {'error': 'Missing required features'}
    
    def _analyze_frequency_domain(self):
        """频域分析"""
        results = {}
        
        for var in ['dsr_cognitive', 'cs_output']:
            if var in self.df.columns:
                data = self.df[var].values
                
                # 去趋势
                detrended = signal.detrend(data)
                
                # FFT
                fft_vals = np.fft.fft(detrended)
                fft_freq = np.fft.fftfreq(len(detrended))
                
                # 功率谱
                power = np.abs(fft_vals) ** 2
                
                # 找到主要频率
                positive_freq_idx = fft_freq > 0
                top_freq_idx = np.argsort(power[positive_freq_idx])[-5:]
                dominant_freqs = fft_freq[positive_freq_idx][top_freq_idx]
                
                results[var] = {
                    'dominant_frequencies': dominant_freqs.tolist(),
                    'spectral_entropy': float(stats.entropy(power[positive_freq_idx])),
                    'low_freq_power_ratio': float(np.sum(power[np.abs(fft_freq) < 0.1]) / np.sum(power))
                }
        
        # 判断是否存在周期性
        periodic = any(v['spectral_entropy'] < 5 for v in results.values())
        
        return {
            'variable_spectra': results,
            'periodic_signal': periodic,
            'dominant_frequencies': "低频主导" if all(v['low_freq_power_ratio'] > 0.7 for v in results.values()) else "混合频率"
        }
    
    def _analyze_mutual_information(self):
        """分析变量间的互信息"""
        mi_matrix = {}
        
        vars_to_analyze = ['dsr_cognitive', 'tl_functional', 'cs_output']
        
        for i, var1 in enumerate(vars_to_analyze):
            for j, var2 in enumerate(vars_to_analyze):
                if i < j and var1 in self.df.columns and var2 in self.df.columns:
                    X = self.df[var1].values.reshape(-1, 1)
                    y = self.df[var2].values
                    
                    mi = mutual_info_regression(X, y, random_state=42)[0]
                    mi_matrix[f"{var1}_{var2}"] = float(mi)
        
        # 特别关注的互信息
        dsr_cs_mi = mi_matrix.get('dsr_cognitive_cs_output', 0)
        
        # 计算条件互信息的近似
        # I(DSR;CS|TL) ≈ I(DSR,TL;CS) - I(TL;CS)
        if all(v in self.df.columns for v in vars_to_analyze):
            X_joint = self.df[['dsr_cognitive', 'tl_functional']].values
            y = self.df['cs_output'].values
            
            mi_joint = mutual_info_regression(X_joint, y, random_state=42)
            X_tl = self.df[['tl_functional']].values
            mi_tl = mutual_info_regression(X_tl, y, random_state=42)[0]
            
            conditional_mi_approx = mi_joint[0] - mi_tl
        else:
            conditional_mi_approx = 0
        
        return {
            'pairwise_mi': mi_matrix,
            'dsr_cs_mi': dsr_cs_mi,
            'conditional_mi_dsr_cs_given_tl': float(conditional_mi_approx),
            'total_dependency': sum(mi_matrix.values())
        }
    
    def _detect_nonlinearity(self):
        """检测非线性关系"""
        # 比较线性和非线性模型
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score
        
        if all(v in self.df.columns for v in ['dsr_cognitive', 'tl_functional', 'cs_output']):
            X = self.df[['dsr_cognitive', 'tl_functional']].values
            y = self.df['cs_output'].values
            
            # 线性模型
            lr = LinearRegression()
            lr_scores = cross_val_score(lr, X, y, cv=5, scoring='r2')
            
            # 非线性模型
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
            
            # 差异
            nonlinearity_gain = np.mean(rf_scores) - np.mean(lr_scores)
            
            return {
                'linear_r2': float(np.mean(lr_scores)),
                'nonlinear_r2': float(np.mean(rf_scores)),
                'nonlinearity_gain': float(nonlinearity_gain),
                'significant_nonlinearity': nonlinearity_gain > 0.05
            }
        else:
            return {'error': 'Missing required features'}
    
    def _classify_signal_type(self, pca_results, frequency_results):
        """分类信号类型"""
        if pca_results.get('effective_dimensions', 3) == 1:
            return "单一主导因子"
        elif frequency_results.get('periodic_signal', False):
            return "周期性信号"
        elif pca_results.get('explained_variance_ratio', [0])[0] < 0.3:
            return "高维噪声主导"
        else:
            return "复杂多因子信号"
    
    def detect_meaningful_patterns(self):
        """检测有意义的模式"""
        print("\n3. 模式检测")
        print("-" * 40)
        
        # 3.1 聚类分析
        clustering_results = self._perform_clustering()
        
        # 3.2 异常检测
        anomaly_results = self._detect_anomalies()
        
        # 3.3 转换点检测（改进方法）
        transition_results = self._detect_transitions()
        
        # 3.4 模式周期性
        periodicity_results = self._analyze_periodicity()
        
        self.diagnosis_results['pattern_detection'] = {
            'clustering': clustering_results,
            'anomalies': anomaly_results,
            'transitions': transition_results,
            'periodicity': periodicity_results,
            'meaningful_patterns': self._identify_meaningful_patterns(clustering_results, transition_results)
        }
        
        print(f"  识别到的聚类数: {clustering_results.get('optimal_clusters', 0)}")
        print(f"  异常点比例: {anomaly_results.get('anomaly_ratio', 0):.3f}")
        print(f"  有意义的转换点: {len(transition_results.get('significant_transitions', []))}")
    
    def _perform_clustering(self):
        """聚类分析"""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        features = ['dsr_cognitive', 'tl_functional', 'cs_output']
        if all(f in self.df.columns for f in features):
            X = self.df[features].dropna()
            
            # 标准化
            scaler = RobustScaler()  # 对异常值更稳健
            X_scaled = scaler.fit_transform(X)
            
            # 寻找最优聚类数
            silhouette_scores = []
            K_range = range(2, min(10, len(X) // 50))
            
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                score = silhouette_score(X_scaled, labels)
                silhouette_scores.append(score)
            
            if silhouette_scores:
                optimal_k = K_range[np.argmax(silhouette_scores)]
                
                # 最终聚类
                kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                final_labels = kmeans.fit_predict(X_scaled)
                
                # 聚类特征
                cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
                
                return {
                    'optimal_clusters': int(optimal_k),
                    'silhouette_score': float(max(silhouette_scores)),
                    'cluster_sizes': np.bincount(final_labels).tolist(),
                    'cluster_centers': cluster_centers.tolist()
                }
            else:
                return {'error': 'Insufficient data for clustering'}
        else:
            return {'error': 'Missing required features'}
    
    def _detect_anomalies(self):
        """异常检测"""
        from sklearn.ensemble import IsolationForest
        
        features = ['dsr_cognitive', 'tl_functional', 'cs_output']
        if all(f in self.df.columns for f in features):
            X = self.df[features].dropna()
            
            # 异常检测
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            anomaly_labels = iso_forest.fit_predict(X)
            
            anomaly_indices = np.where(anomaly_labels == -1)[0]
            
            # 分析异常点特征
            if len(anomaly_indices) > 0:
                anomaly_dates = self.df.iloc[anomaly_indices]['date'].tolist()
                anomaly_contexts = self.df.iloc[anomaly_indices]['sensitivity_code'].value_counts().to_dict()
            else:
                anomaly_dates = []
                anomaly_contexts = {}
            
            return {
                'anomaly_ratio': float(len(anomaly_indices) / len(X)),
                'anomaly_count': len(anomaly_indices),
                'anomaly_dates': [str(d) for d in anomaly_dates[:10]],  # 前10个
                'anomaly_context_distribution': anomaly_contexts
            }
        else:
            return {'error': 'Missing required features'}
    
    def _detect_transitions(self):
        """改进的转换点检测"""
        # 使用CUSUM算法
        from scipy.stats import norm
        
        transitions = {}
        
        for var in ['dsr_cognitive', 'functional_complementarity']:
            if var in self.df.columns:
                data = self.df[var].values
                
                # CUSUM检测
                mean = np.mean(data)
                std = np.std(data)
                
                cusum_pos = np.zeros(len(data))
                cusum_neg = np.zeros(len(data))
                
                for i in range(1, len(data)):
                    cusum_pos[i] = max(0, cusum_pos[i-1] + data[i] - mean - 0.5*std)
                    cusum_neg[i] = max(0, cusum_neg[i-1] + mean - data[i] - 0.5*std)
                
                # 检测超过阈值的点
                threshold = 4 * std
                transitions_pos = np.where(cusum_pos > threshold)[0]
                transitions_neg = np.where(cusum_neg > threshold)[0]
                
                all_transitions = sorted(set(transitions_pos) | set(transitions_neg))
                
                # 过滤相近的转换点
                filtered_transitions = []
                min_distance = 50
                
                for t in all_transitions:
                    if not filtered_transitions or t - filtered_transitions[-1] > min_distance:
                        filtered_transitions.append(t)
                
                transitions[var] = filtered_transitions
        
        # 找到共同转换点
        if transitions:
            all_trans = []
            for trans_list in transitions.values():
                all_trans.extend(trans_list)
            
            # 聚类转换点
            if all_trans:
                from collections import Counter
                trans_counter = Counter(all_trans)
                significant_transitions = [t for t, count in trans_counter.items() if count >= len(transitions) // 2]
            else:
                significant_transitions = []
        else:
            significant_transitions = []
        
        return {
            'variable_transitions': {k: len(v) for k, v in transitions.items()},
            'significant_transitions': significant_transitions,
            'transition_dates': [str(self.df.iloc[t]['date']) for t in significant_transitions] if significant_transitions else []
        }
    
    def _analyze_periodicity(self):
        """分析周期性"""
        from scipy.signal import find_peaks, periodogram
        
        results = {}
        
        for var in ['dsr_cognitive', 'cs_output']:
            if var in self.df.columns:
                data = self.df[var].values
                
                # 自相关函数
                acf_values = [pd.Series(data).autocorr(lag=i) for i in range(1, min(100, len(data)//4))]
                acf_values = [v for v in acf_values if not np.isnan(v)]
                
                if acf_values:
                    # 找到ACF的峰值
                    peaks, _ = find_peaks(acf_values, height=0.2)
                    
                    if len(peaks) > 0:
                        # 第一个显著峰值表示主要周期
                        main_period = peaks[0] + 1
                        period_strength = acf_values[peaks[0]]
                    else:
                        main_period = None
                        period_strength = 0
                    
                    results[var] = {
                        'has_periodicity': len(peaks) > 0,
                        'main_period': main_period,
                        'period_strength': float(period_strength)
                    }
                else:
                    results[var] = {
                        'has_periodicity': False,
                        'main_period': None,
                        'period_strength': 0
                    }
        
        return results
    
    def _identify_meaningful_patterns(self, clustering, transitions):
        """识别有意义的模式"""
        patterns = []
        
        # 聚类模式
        if clustering.get('optimal_clusters', 0) > 1 and clustering.get('silhouette_score', 0) > 0.3:
            patterns.append({
                'type': 'stable_states',
                'description': f"存在{clustering['optimal_clusters']}个稳定状态",
                'evidence': f"轮廓系数={clustering['silhouette_score']:.3f}"
            })
        
        # 转换模式
        if len(transitions.get('significant_transitions', [])) > 0:
            patterns.append({
                'type': 'phase_transitions',
                'description': f"识别到{len(transitions['significant_transitions'])}个显著转换点",
                'evidence': f"日期: {', '.join(transitions['transition_dates'][:3])}"
            })
        
        # 如果没有明显模式
        if not patterns:
            patterns.append({
                'type': 'no_clear_pattern',
                'description': "未发现明显的结构化模式",
                'evidence': "数据可能主要由随机变异主导"
            })
        
        return patterns
    
    def recommend_methods(self):
        """基于诊断结果推荐方法"""
        print("\n4. 方法论建议")
        print("-" * 40)
        
        recommendations = []
        
        # 基于数据质量的建议
        if self.diagnosis_results['data_quality']['measurement_error']['noise_dominates']:
            recommendations.append({
                'issue': '高噪声水平',
                'method': '稳健统计方法',
                'specific': [
                    '使用稳健回归（Huber回归）',
                    '中位数滤波预处理',
                    '集成方法（如随机森林）提高稳定性'
                ]
            })
        
        # 基于信号结构的建议
        signal_type = self.diagnosis_results['signal_analysis']['signal_type']
        if signal_type == "高维噪声主导":
            recommendations.append({
                'issue': '信号弱/噪声强',
                'method': '降维和特征工程',
                'specific': [
                    '主成分回归（PCR）',
                    '偏最小二乘（PLS）',
                    '特征聚合（按时间窗口）'
                ]
            })
        
        # 基于时间结构的建议
        if self.diagnosis_results['data_quality']['time_series_issues']['main_issue'] == "缺乏时间依赖性":
            recommendations.append({
                'issue': '时间独立性',
                'method': '横截面分析方法',
                'specific': [
                    '混合效应模型（处理重复测量）',
                    '聚类后的组间比较',
                    '倾向得分匹配（因果推断）'
                ]
            })
        
        # 基于模式检测的建议
        if self.diagnosis_results['pattern_detection'].get('clustering', {}).get('optimal_clusters', 0) > 1:
            recommendations.append({
                'issue': '存在潜在子群体',
                'method': '分层建模',
                'specific': [
                    '有限混合模型（FMM）',
                    '潜在类别分析（LCA）',
                    '分层贝叶斯模型'
                ]
            })
        
        # 综合建议
        primary_recommendation = self._synthesize_recommendations(recommendations)
        
        self.diagnosis_results['method_recommendations'] = {
            'specific_issues': recommendations,
            'primary_approach': primary_recommendation,
            'implementation_priority': self._prioritize_methods(recommendations)
        }
        
        print(f"  主要建议: {primary_recommendation['approach']}")
        print(f"  优先实施: {', '.join(primary_recommendation['top_methods'][:3])}")
    
    def _synthesize_recommendations(self, recommendations):
        """综合推荐"""
        if not recommendations:
            return {
                'approach': '标准分析方法可能足够',
                'top_methods': ['多元回归', '时间序列分析', '因果推断'],
                'rationale': '数据质量尚可，无特殊问题'
            }
        
        # 根据问题的严重性排序
        if any('高噪声' in r['issue'] for r in recommendations):
            return {
                'approach': '稳健性优先策略',
                'top_methods': [
                    '稳健主成分分析',
                    '中位数回归',
                    'Bootstrap聚合',
                    '异常值检测与处理'
                ],
                'rationale': '数据噪声水平高，需要抗干扰方法'
            }
        elif any('潜在子群体' in r['issue'] for r in recommendations):
            return {
                'approach': '异质性建模策略',
                'top_methods': [
                    '潜在增长混合模型',
                    '多层次结构方程模型',
                    '贝叶斯混合模型'
                ],
                'rationale': '数据存在异质性，需要分层方法'
            }
        else:
            return {
                'approach': '特征工程与降维策略',
                'top_methods': [
                    '时间窗口聚合',
                    '动态因子模型',
                    '功能数据分析'
                ],
                'rationale': '需要提取更强的信号特征'
            }
    
    def _prioritize_methods(self, recommendations):
        """方法优先级排序"""
        # 提取所有具体方法
        all_methods = []
        for rec in recommendations:
            all_methods.extend(rec['specific'])
        
        # 按照实施难度和预期效果排序
        priority_scores = {
            '中位数滤波预处理': 10,
            '时间窗口聚合': 9,
            '稳健回归': 8,
            '主成分回归': 7,
            '混合效应模型': 6,
            '潜在类别分析': 5,
            '贝叶斯混合模型': 4
        }
        
        # 排序
        prioritized = sorted(all_methods, 
                           key=lambda x: priority_scores.get(x, 0), 
                           reverse=True)
        
        return prioritized[:5]  # 返回前5个
    
    def revise_hypothesis(self):
        """基于诊断修正研究假设"""
        print("\n5. 假设修正建议")
        print("-" * 40)
        
        # 原始假设评估
        original_hypotheses = {
            'H1': '数字符号资源形成与传统语言不可分离的整合',
            'H2': '平台和语境调节构成性功能',
            'H3': '构成性随时间增强并表现出路径依赖'
        }
        
        # 基于诊断的修正
        revised_hypotheses = {}
        
        # H1修正
        if self.diagnosis_results['signal_analysis']['mutual_information']['dsr_cs_mi'] < 0.1:
            revised_hypotheses['H1'] = {
                'original': original_hypotheses['H1'],
                'issue': 'DSR-CS互信息过低，直接依赖关系弱',
                'revised': '数字符号资源通过情境化中介机制影响认知系统',
                'testable_prediction': '在特定语境条件下，DSR的影响会显著增强'
            }
        else:
            revised_hypotheses['H1'] = {
                'original': original_hypotheses['H1'],
                'issue': '无需修正',
                'revised': original_hypotheses['H1'],
                'testable_prediction': 'DSR移除会导致CS性能显著下降'
            }
        
        # H2修正
        if self.diagnosis_results['pattern_detection'].get('clustering', {}).get('optimal_clusters', 0) > 1:
            revised_hypotheses['H2'] = {
                'original': original_hypotheses['H2'],
                'issue': '存在多个稳定状态，调节可能是离散的',
                'revised': '不同语境触发离散的认知模式切换',
                'testable_prediction': '语境转换伴随认知指标的非连续变化'
            }
        
        # H3修正
        if len(self.diagnosis_results['pattern_detection'].get('transitions', {}).get('significant_transitions', [])) == 0:
            revised_hypotheses['H3'] = {
                'original': original_hypotheses['H3'],
                'issue': '未检测到清晰的时间演化模式',
                'revised': '构成性表现为稳态适应而非持续增强',
                'testable_prediction': '系统快速达到稳定状态后保持恒定'
            }
        
        # 新假设建议
        new_hypotheses = []
        
        if self.diagnosis_results['data_quality']['measurement_error']['noise_dominates']:
            new_hypotheses.append({
                'H4': '认知构成性效应被测量噪声掩盖',
                'rationale': '高噪声环境下需要更精确的测量方法',
                'test': '使用多指标综合测量减少噪声影响'
            })
        
        self.diagnosis_results['revised_hypothesis'] = {
            'original_hypotheses': original_hypotheses,
            'revised_hypotheses': revised_hypotheses,
            'new_hypotheses': new_hypotheses,
            'research_direction': self._suggest_research_direction(revised_hypotheses)
        }
        
        print(f"  需要修正的假设: {len([h for h in revised_hypotheses.values() if h['issue'] != '无需修正'])}")
        print(f"  建议新增假设: {len(new_hypotheses)}")
        print(f"  研究方向: {self.diagnosis_results['revised_hypothesis']['research_direction']}")
    
    def _suggest_research_direction(self, revised_hypotheses):
        """建议研究方向"""
        issues = [h['issue'] for h in revised_hypotheses.values() if h['issue'] != '无需修正']
        
        if '直接依赖关系弱' in ' '.join(issues):
            return "探索间接因果路径和中介机制"
        elif '离散的认知模式' in ' '.join(issues):
            return "研究认知模式的转换机制和触发条件"
        elif '稳态适应' in ' '.join(issues):
            return "分析稳定状态的特征和形成过程"
        else:
            return "深化现有假设的实证检验"
    
    def save_diagnosis_results(self):
        """保存诊断结果"""
        output_file = self.data_path / 'deep_diagnosis_results.json'
        
        # 转换numpy类型
        def convert_numpy(obj):
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            elif pd.isna(obj):
                return None
            return obj
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy(self.diagnosis_results), f, ensure_ascii=False, indent=2)
        
        print(f"\n诊断结果已保存至: {output_file}")
        
        # 生成可视化
        self.create_diagnostic_plots()
    
    def create_diagnostic_plots(self):
        """创建诊断可视化"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 变量分布
        ax1 = axes[0, 0]
        vars_to_plot = ['dsr_cognitive', 'tl_functional', 'cs_output']
        for var in vars_to_plot:
            if var in self.df.columns:
                self.df[var].hist(bins=50, alpha=0.5, label=var, ax=ax1)
        ax1.set_title('变量分布')
        ax1.legend()
        
        # 2. 时间序列
        ax2 = axes[0, 1]
        if 'dsr_cognitive' in self.df.columns:
            rolling_mean = self.df['dsr_cognitive'].rolling(window=100).mean()
            ax2.plot(self.df.index, self.df['dsr_cognitive'], alpha=0.3, label='原始')
            ax2.plot(self.df.index, rolling_mean, label='移动平均')
        ax2.set_title('DSR认知指标时间序列')
        ax2.legend()
        
        # 3. 散点图矩阵
        ax3 = axes[1, 0]
        if all(v in self.df.columns for v in ['dsr_cognitive', 'cs_output']):
            ax3.scatter(self.df['dsr_cognitive'], self.df['cs_output'], alpha=0.3)
            ax3.set_xlabel('DSR认知')
            ax3.set_ylabel('CS输出')
            ax3.set_title('DSR-CS关系')
        
        # 4. 诊断总结
        ax4 = axes[1, 1]
        ax4.axis('off')
        summary_text = f"""诊断总结:
        
信噪比: {self.diagnosis_results['data_quality']['measurement_error']['estimated_snr']:.2f}
信号类型: {self.diagnosis_results['signal_analysis']['signal_type']}
主要问题: {self.diagnosis_results['data_quality']['time_series_issues']['main_issue']}
建议方法: {self.diagnosis_results['method_recommendations']['primary_approach']['approach']}
"""
        ax4.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center')
        
        plt.tight_layout()
        
        # 保存图表
        output_file = self.data_path.parent / 'figures' / 'deep_diagnosis_plot.jpg'
        output_file.parent.mkdir(exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n诊断图表已保存至: {output_file}")

def main():
    """主函数"""
    # 设置数据路径
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_path = project_root / 'output_cn' / 'data'
    
    # 创建诊断器
    diagnoser = DeepDiagnosisAnalyzer(data_path)
    
    # 运行深度诊断
    results = diagnoser.load_and_diagnose_data()
    
    print("\n" + "="*60)
    print("诊断完成 - 行动建议")
    print("="*60)
    
    print("\n基于深度诊断，建议采取以下行动：")
    print("\n1. 立即行动:")
    print("   - 实施时间窗口聚合（如按日或按会话）")
    print("   - 应用稳健标准化方法处理异常值")
    print("   - 创建综合指标减少测量噪声")
    
    print("\n2. 方法调整:")
    print("   - 放弃纯时间序列方法，转向面板数据分析")
    print("   - 使用潜在类别模型识别认知模式")
    print("   - 实施稳健回归方法")
    
    print("\n3. 假设修正:")
    print("   - 将'持续增强'修正为'稳态适应'")
    print("   - 探索间接因果路径")
    print("   - 关注语境触发的模式切换")
    
    print("\n✓ 深度诊断分析完成！")

if __name__ == "__main__":
    main()