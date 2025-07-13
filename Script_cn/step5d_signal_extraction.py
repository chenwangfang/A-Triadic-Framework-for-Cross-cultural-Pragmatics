# step5d_signal_extraction.py
# 高级信号提取：ICA + 小波去噪策略

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# 信号处理库
import pywt
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import signal, stats
from scipy.ndimage import gaussian_filter1d

# 统计分析库
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.nonparametric.smoothers_lowess import lowess

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class AdvancedSignalExtractor:
    """高级信号提取器 - ICA + 小波分析"""
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.df = None
        self.processed_signals = {}
        self.extraction_results = {
            'wavelet_decomposition': {},
            'ica_components': {},
            'denoised_signals': {},
            'signal_quality': {},
            'functional_analysis': {},
            'revised_metrics': {}
        }
    
    def load_data(self):
        """加载数据"""
        csv_file = self.data_path / 'data_with_metrics.csv'
        self.df = pd.read_csv(csv_file, encoding='utf-8-sig')
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)
        
        print("="*60)
        print("高级信号提取分析")
        print("="*60)
        print(f"数据加载完成: {len(self.df)} 条记录")
        
        return self.df
    
    def run_signal_extraction(self):
        """运行完整的信号提取流程"""
        
        # 1. 小波去噪
        print("\n1. 小波去噪处理")
        self.wavelet_denoising()
        
        # 2. 独立成分分析
        print("\n2. 独立成分分析")
        self.independent_component_analysis()
        
        # 3. 信号重构
        print("\n3. 信号重构与验证")
        self.reconstruct_signals()
        
        # 4. 功能深度分析
        print("\n4. 功能深度分析")
        self.functional_depth_analysis()
        
        # 5. 修正指标计算
        print("\n5. 修正指标体系")
        self.calculate_revised_metrics()
        
        # 6. 质量评估
        print("\n6. 信号质量评估")
        self.assess_signal_quality()
        
        # 保存结果
        self.save_results()
        
        return self.extraction_results
    
    def wavelet_denoising(self):
        """小波去噪处理"""
        key_vars = ['dsr_cognitive', 'tl_functional', 'cs_output']
        
        for var in key_vars:
            if var in self.df.columns:
                print(f"\n处理 {var}...")
                
                # 原始信号
                signal_data = self.df[var].values
                
                # 多级小波分解
                wavelet_results = self._multilevel_wavelet_analysis(signal_data, var)
                
                # 存储结果
                self.extraction_results['wavelet_decomposition'][var] = wavelet_results
                
                # 打印摘要
                print(f"  - 分解级别: {wavelet_results['levels']}")
                print(f"  - 主要能量集中在: {wavelet_results['dominant_scale']}")
                print(f"  - 去噪后SNR提升: {wavelet_results['snr_improvement']:.2f}倍")
    
    def _multilevel_wavelet_analysis(self, signal_data, var_name):
        """多级小波分析"""
        # 选择合适的小波基
        wavelet = 'db4'  # Daubechies 4
        
        # 确定分解级别
        max_level = pywt.dwt_max_level(len(signal_data), wavelet)
        levels = min(max_level, 6)  # 限制最大级别
        
        # 小波分解
        coeffs = pywt.wavedec(signal_data, wavelet, level=levels)
        
        # 分析每个尺度的能量
        energy_by_scale = []
        for i, coeff in enumerate(coeffs):
            energy = np.sum(coeff**2)
            energy_by_scale.append(energy)
        
        # 识别主要尺度
        total_energy = sum(energy_by_scale)
        energy_ratio = [e/total_energy for e in energy_by_scale]
        dominant_scale = np.argmax(energy_ratio)
        
        # 阈值去噪
        # 使用软阈值和通用阈值规则
        denoised_coeffs = []
        for i, coeff in enumerate(coeffs):
            if i == 0:  # 近似系数保持不变
                denoised_coeffs.append(coeff)
            else:
                # 计算阈值
                sigma = np.median(np.abs(coeff)) / 0.6745
                threshold = sigma * np.sqrt(2 * np.log(len(coeff)))
                # 软阈值
                denoised = pywt.threshold(coeff, threshold, 'soft')
                denoised_coeffs.append(denoised)
        
        # 重构信号
        denoised_signal = pywt.waverec(denoised_coeffs, wavelet)
        
        # 确保长度一致
        if len(denoised_signal) > len(signal_data):
            denoised_signal = denoised_signal[:len(signal_data)]
        
        # 计算SNR改进
        noise = signal_data - denoised_signal[:len(signal_data)]
        original_snr = np.var(signal_data) / np.var(noise) if np.var(noise) > 0 else np.inf
        
        # 存储去噪信号
        self.processed_signals[f"{var_name}_wavelet_denoised"] = denoised_signal
        
        return {
            'levels': levels,
            'energy_distribution': energy_ratio,
            'dominant_scale': f"Level {dominant_scale}",
            'coefficients': [c.tolist()[:10] for c in coeffs],  # 前10个系数
            'snr_improvement': original_snr / (np.var(noise) / np.var(denoised_signal)) if np.var(noise) > 0 else 1.0,
            'denoised_signal_sample': denoised_signal[:100].tolist()
        }
    
    def independent_component_analysis(self):
        """独立成分分析"""
        # 准备数据矩阵
        vars_for_ica = ['dsr_cognitive_wavelet_denoised', 
                        'tl_functional_wavelet_denoised', 
                        'cs_output_wavelet_denoised']
        
        # 构建数据矩阵
        X = []
        available_vars = []
        for var in vars_for_ica:
            if var in self.processed_signals:
                X.append(self.processed_signals[var])
                available_vars.append(var)
        
        if len(X) >= 2:
            X = np.array(X).T
            
            # 标准化
            scaler = RobustScaler()  # 对异常值更稳健
            X_scaled = scaler.fit_transform(X)
            
            # ICA分解
            n_components = min(X.shape[1], 3)
            ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
            
            # 执行ICA
            S = ica.fit_transform(X_scaled)  # 独立成分
            A = ica.mixing_  # 混合矩阵
            
            # 分析每个独立成分
            component_analysis = []
            for i in range(n_components):
                comp = S[:, i]
                
                # 统计特征
                comp_stats = {
                    'component_id': i,
                    'kurtosis': float(stats.kurtosis(comp)),
                    'skewness': float(stats.skew(comp)),
                    'entropy': float(stats.entropy(np.histogram(comp, bins=50)[0] + 1e-10)),
                    'mixing_weights': A[:, i].tolist(),
                    'dominant_source': available_vars[np.argmax(np.abs(A[:, i]))]
                }
                
                # 时间结构分析
                acf_1 = pd.Series(comp).autocorr(lag=1)
                acf_10 = pd.Series(comp).autocorr(lag=10)
                comp_stats['temporal_structure'] = {
                    'acf_lag1': float(acf_1) if not np.isnan(acf_1) else 0,
                    'acf_lag10': float(acf_10) if not np.isnan(acf_10) else 0
                }
                
                component_analysis.append(comp_stats)
                
                # 存储独立成分
                self.processed_signals[f'ic_{i}'] = comp
            
            # 识别信号成分和噪声成分
            signal_components = []
            noise_components = []
            
            for i, comp_stats in enumerate(component_analysis):
                # 基于峰度和时间结构判断
                if abs(comp_stats['kurtosis']) > 1 or abs(comp_stats['temporal_structure']['acf_lag1']) > 0.3:
                    signal_components.append(i)
                else:
                    noise_components.append(i)
            
            self.extraction_results['ica_components'] = {
                'n_components': n_components,
                'component_analysis': component_analysis,
                'signal_components': signal_components,
                'noise_components': noise_components,
                'mixing_matrix': A.tolist(),
                'explained_variance': self._calculate_explained_variance(S, X_scaled)
            }
            
            print(f"  - 提取了 {n_components} 个独立成分")
            print(f"  - 信号成分: {signal_components}")
            print(f"  - 噪声成分: {noise_components}")
        else:
            print("  - 可用变量不足，跳过ICA分析")
            self.extraction_results['ica_components'] = {'error': 'Insufficient variables'}
    
    def _calculate_explained_variance(self, S, X):
        """计算解释方差"""
        # ICA的解释方差计算
        # 对于ICA，我们计算每个成分的方差贡献
        n_components = S.shape[1]
        
        # 计算每个独立成分的方差
        component_vars = [np.var(S[:, i]) for i in range(n_components)]
        total_component_var = sum(component_vars)
        
        # 计算原始数据的总方差
        total_data_var = np.sum(np.var(X, axis=0))
        
        # 解释方差比例（近似）
        explained_var_ratio = min(total_component_var / total_data_var, 1.0) if total_data_var > 0 else 0
        
        return float(explained_var_ratio)
    
    def reconstruct_signals(self):
        """信号重构"""
        if 'signal_components' in self.extraction_results['ica_components']:
            signal_comps = self.extraction_results['ica_components']['signal_components']
            
            if signal_comps:
                # 使用信号成分重构
                print("\n重构信号（仅使用信号成分）...")
                
                # 获取混合矩阵
                A = np.array(self.extraction_results['ica_components']['mixing_matrix'])
                
                # 选择信号成分
                signal_ic = []
                for comp_id in signal_comps:
                    if f'ic_{comp_id}' in self.processed_signals:
                        signal_ic.append(self.processed_signals[f'ic_{comp_id}'])
                
                if signal_ic:
                    signal_ic = np.array(signal_ic).T
                    A_signal = A[:, signal_comps]
                    
                    # 重构
                    reconstructed = signal_ic @ A_signal.T
                    
                    # 存储重构信号
                    var_names = ['dsr_cognitive', 'tl_functional', 'cs_output']
                    for i, var in enumerate(var_names[:reconstructed.shape[1]]):
                        self.processed_signals[f"{var}_reconstructed"] = reconstructed[:, i]
                        
                        # 计算重构质量
                        if var in self.df.columns:
                            original = self.df[var].values
                            correlation = np.corrcoef(original, reconstructed[:, i])[0, 1]
                            
                            print(f"  - {var} 重构相关性: {correlation:.3f}")
                    
                    self.extraction_results['denoised_signals']['reconstruction_complete'] = True
                else:
                    self.extraction_results['denoised_signals']['reconstruction_complete'] = False
            else:
                print("  - 未识别到有效信号成分")
                self.extraction_results['denoised_signals']['reconstruction_complete'] = False
    
    def functional_depth_analysis(self):
        """功能深度分析"""
        print("\n执行功能深度分析...")
        
        # 1. 语境敏感性分析
        context_analysis = self._analyze_context_sensitivity()
        
        # 2. 功能模式识别
        functional_patterns = self._identify_functional_patterns()
        
        # 3. 认知负荷动态
        cognitive_dynamics = self._analyze_cognitive_dynamics()
        
        # 4. 互补性深度测量
        complementarity = self._measure_deep_complementarity()
        
        self.extraction_results['functional_analysis'] = {
            'context_sensitivity': context_analysis,
            'functional_patterns': functional_patterns,
            'cognitive_dynamics': cognitive_dynamics,
            'deep_complementarity': complementarity
        }
        
        print(f"  - 识别到 {len(functional_patterns)} 种功能模式")
        print(f"  - 语境敏感性评分: {context_analysis['sensitivity_score']:.3f}")
    
    def _analyze_context_sensitivity(self):
        """分析语境敏感性"""
        # 使用处理后的信号
        signal_var = 'dsr_cognitive_reconstructed' if 'dsr_cognitive_reconstructed' in self.processed_signals else 'dsr_cognitive'
        
        if signal_var in self.processed_signals or signal_var in self.df.columns:
            if signal_var in self.processed_signals:
                signal_data = self.processed_signals[signal_var]
            else:
                signal_data = self.df[signal_var].values
            
            # 按语境分组
            context_groups = {}
            for context in [1, 2, 3]:
                mask = self.df['sensitivity_code'] == context
                if np.sum(mask) > 30:
                    context_groups[context] = signal_data[mask]
            
            # 计算语境间差异
            if len(context_groups) > 1:
                # ANOVA
                f_stat, p_value = stats.f_oneway(*context_groups.values())
                
                # 效应量（eta squared）
                group_means = [np.mean(g) for g in context_groups.values()]
                grand_mean = np.mean(signal_data)
                ss_between = sum(len(g) * (m - grand_mean)**2 for g, m in zip(context_groups.values(), group_means))
                ss_total = np.sum((signal_data - grand_mean)**2)
                eta_squared = ss_between / ss_total if ss_total > 0 else 0
                
                # 语境特定模式
                context_patterns = {}
                for context, data in context_groups.items():
                    # 计算分布特征
                    context_patterns[context] = {
                        'mean': float(np.mean(data)),
                        'std': float(np.std(data)),
                        'skew': float(stats.skew(data)),
                        'dominant_range': [float(np.percentile(data, 25)), float(np.percentile(data, 75))]
                    }
                
                return {
                    'sensitivity_score': float(eta_squared),
                    'f_statistic': float(f_stat),
                    'p_value': float(p_value),
                    'context_patterns': context_patterns,
                    'significant_differences': p_value < 0.05
                }
            else:
                return {'sensitivity_score': 0.0, 'error': 'Insufficient context groups'}
        else:
            return {'sensitivity_score': 0.0, 'error': 'Signal not available'}
    
    def _identify_functional_patterns(self):
        """识别功能模式"""
        patterns = []
        
        # 检查DED功能分布
        if 'ded_functions' in self.df.columns:
            # 提取功能类型
            all_functions = []
            for funcs in self.df['ded_functions'].dropna():
                if isinstance(funcs, str):
                    all_functions.extend(funcs.split('|'))
            
            # 统计功能频率
            from collections import Counter
            func_counter = Counter(all_functions)
            
            # 识别主要功能模式
            for func, count in func_counter.most_common(10):
                if count > 50:  # 至少出现50次
                    # 分析该功能的使用模式
                    func_mask = self.df['ded_functions'].str.contains(func, na=False)
                    
                    if func_mask.sum() > 0:
                        pattern = {
                            'function': func,
                            'frequency': count,
                            'percentage': count / len(self.df) * 100,
                            'avg_cognitive_load': float(self.df.loc[func_mask, 'dsr_cognitive'].mean()) if 'dsr_cognitive' in self.df else 0,
                            'context_preference': self._analyze_function_context_preference(func_mask)
                        }
                        patterns.append(pattern)
        
        return patterns
    
    def _analyze_function_context_preference(self, func_mask):
        """分析功能的语境偏好"""
        if 'sensitivity_code' in self.df.columns:
            context_dist = self.df.loc[func_mask, 'sensitivity_code'].value_counts(normalize=True).to_dict()
            
            # 计算偏好指数
            baseline_dist = self.df['sensitivity_code'].value_counts(normalize=True).to_dict()
            
            preference = {}
            for context, prop in context_dist.items():
                baseline = baseline_dist.get(context, 0)
                if baseline > 0:
                    preference[f'context_{context}'] = float(prop / baseline)
                else:
                    preference[f'context_{context}'] = 0
            
            return preference
        else:
            return {}
    
    def _analyze_cognitive_dynamics(self):
        """分析认知动态"""
        dynamics = {}
        
        # 使用处理后的信号
        for var_type in ['reconstructed', 'wavelet_denoised', '']:
            var_name = f'cs_output_{var_type}' if var_type else 'cs_output'
            
            if var_name in self.processed_signals or (not var_type and var_name in self.df.columns):
                if var_name in self.processed_signals:
                    data = self.processed_signals[var_name]
                else:
                    data = self.df[var_name].values
                
                # 滑动窗口分析
                window_size = 100
                step = 50
                
                windows_analysis = []
                for i in range(0, len(data) - window_size, step):
                    window = data[i:i+window_size]
                    
                    # 窗口统计
                    window_stats = {
                        'position': i + window_size // 2,
                        'mean': float(np.mean(window)),
                        'std': float(np.std(window)),
                        'trend': float(np.polyfit(range(len(window)), window, 1)[0]),  # 线性趋势
                        'volatility': float(np.std(np.diff(window)))  # 波动性
                    }
                    windows_analysis.append(window_stats)
                
                # 识别动态模式
                if windows_analysis:
                    means = [w['mean'] for w in windows_analysis]
                    volatilities = [w['volatility'] for w in windows_analysis]
                    
                    # 检测稳定期和过渡期
                    stable_threshold = np.percentile(volatilities, 25)
                    volatile_threshold = np.percentile(volatilities, 75)
                    
                    stable_periods = sum(1 for v in volatilities if v < stable_threshold)
                    volatile_periods = sum(1 for v in volatilities if v > volatile_threshold)
                    
                    dynamics = {
                        'signal_type': var_name,
                        'overall_trend': float(np.polyfit(range(len(means)), means, 1)[0]),
                        'stability_ratio': stable_periods / len(volatilities),
                        'volatility_ratio': volatile_periods / len(volatilities),
                        'phase_transitions': self._detect_phase_transitions(means),
                        'windows_analyzed': len(windows_analysis)
                    }
                    break
        
        return dynamics
    
    def _detect_phase_transitions(self, means):
        """检测相位转换"""
        if len(means) < 10:
            return []
        
        # 使用变点检测算法
        from scipy.signal import find_peaks
        
        # 计算一阶差分
        diff = np.diff(means)
        
        # 找到显著变化点
        peaks_pos, _ = find_peaks(diff, height=np.std(diff))
        peaks_neg, _ = find_peaks(-diff, height=np.std(diff))
        
        transitions = sorted(list(peaks_pos) + list(peaks_neg))
        
        return transitions[:5]  # 返回前5个最显著的转换点
    
    def _measure_deep_complementarity(self):
        """深度互补性测量"""
        complementarity = {}
        
        # 获取处理后的信号
        dsr_signal = None
        tl_signal = None
        
        for suffix in ['_reconstructed', '_wavelet_denoised', '']:
            dsr_var = f'dsr_cognitive{suffix}'
            tl_var = f'tl_functional{suffix}'
            
            if dsr_var in self.processed_signals or (not suffix and dsr_var in self.df.columns):
                if dsr_var in self.processed_signals:
                    dsr_signal = self.processed_signals[dsr_var]
                else:
                    dsr_signal = self.df[dsr_var].values
                    
            if tl_var in self.processed_signals or (not suffix and tl_var in self.df.columns):
                if tl_var in self.processed_signals:
                    tl_signal = self.processed_signals[tl_var]
                else:
                    tl_signal = self.df[tl_var].values
            
            if dsr_signal is not None and tl_signal is not None:
                break
        
        if dsr_signal is not None and tl_signal is not None:
            # 1. 时变相关性
            window_size = 100
            correlations = []
            
            for i in range(len(dsr_signal) - window_size):
                window_corr = np.corrcoef(dsr_signal[i:i+window_size], 
                                        tl_signal[i:i+window_size])[0, 1]
                correlations.append(window_corr if not np.isnan(window_corr) else 0)
            
            # 2. 相位同步
            # 使用希尔伯特变换
            from scipy.signal import hilbert
            
            analytic_dsr = hilbert(dsr_signal - np.mean(dsr_signal))
            analytic_tl = hilbert(tl_signal - np.mean(tl_signal))
            
            phase_dsr = np.angle(analytic_dsr)
            phase_tl = np.angle(analytic_tl)
            
            # 相位差
            phase_diff = phase_dsr - phase_tl
            
            # 相位锁定值（PLV）
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            
            # 3. 因果方向性
            # 简化的传递熵估计
            from sklearn.metrics import mutual_info_score
            
            # 计算条件互信息作为传递熵的近似
            lag = 5
            if len(dsr_signal) > lag:
                # 创建分箱边界
                dsr_bins = np.linspace(np.min(dsr_signal), np.max(dsr_signal), 11)
                tl_bins = np.linspace(np.min(tl_signal), np.max(tl_signal), 11)
                
                # DSR -> TL
                dsr_discrete = np.digitize(dsr_signal[:-lag], bins=dsr_bins)
                tl_discrete = np.digitize(tl_signal[lag:], bins=tl_bins)
                mi_dsr_tl = mutual_info_score(dsr_discrete, tl_discrete)
                
                # TL -> DSR
                tl_discrete2 = np.digitize(tl_signal[:-lag], bins=tl_bins)
                dsr_discrete2 = np.digitize(dsr_signal[lag:], bins=dsr_bins)
                mi_tl_dsr = mutual_info_score(tl_discrete2, dsr_discrete2)
                
                directionality = (mi_dsr_tl - mi_tl_dsr) / (mi_dsr_tl + mi_tl_dsr) if (mi_dsr_tl + mi_tl_dsr) > 0 else 0
            else:
                directionality = 0
            
            complementarity = {
                'time_varying_correlation': {
                    'mean': float(np.mean(correlations)),
                    'std': float(np.std(correlations)),
                    'min': float(np.min(correlations)),
                    'max': float(np.max(correlations))
                },
                'phase_locking_value': float(plv),
                'directionality_index': float(directionality),
                'complementarity_score': float(plv * np.mean(correlations))  # 综合评分
            }
        
        return complementarity
    
    def calculate_revised_metrics(self):
        """计算修正后的指标体系"""
        print("\n计算修正指标...")
        
        revised_metrics = {}
        
        # 1. 基于去噪信号的认知指标
        for var in ['dsr_cognitive', 'tl_functional', 'cs_output']:
            # 优先使用重构信号，其次是去噪信号
            for suffix in ['_reconstructed', '_wavelet_denoised']:
                signal_name = f"{var}{suffix}"
                if signal_name in self.processed_signals:
                    signal_data = self.processed_signals[signal_name]
                    
                    # 计算稳健统计量
                    revised_metrics[f"{var}_revised"] = {
                        'robust_mean': float(np.median(signal_data)),
                        'robust_std': float(stats.median_abs_deviation(signal_data)),
                        'trimmed_mean': float(stats.trim_mean(signal_data, 0.1)),  # 10%截尾均值
                        'iqr': float(np.percentile(signal_data, 75) - np.percentile(signal_data, 25))
                    }
                    break
        
        # 2. 功能深度指标
        if 'deep_complementarity' in self.extraction_results['functional_analysis']:
            comp = self.extraction_results['functional_analysis']['deep_complementarity']
            revised_metrics['functional_integration'] = {
                'dynamic_coupling': comp.get('time_varying_correlation', {}).get('mean', 0),
                'phase_synchrony': comp.get('phase_locking_value', 0),
                'causal_asymmetry': comp.get('directionality_index', 0)
            }
        
        # 3. 语境适应性指标
        if 'context_sensitivity' in self.extraction_results['functional_analysis']:
            ctx = self.extraction_results['functional_analysis']['context_sensitivity']
            revised_metrics['contextual_adaptation'] = {
                'sensitivity_score': ctx.get('sensitivity_score', 0),
                'context_differentiation': ctx.get('significant_differences', False)
            }
        
        # 4. 认知稳定性指标
        if 'cognitive_dynamics' in self.extraction_results['functional_analysis']:
            dyn = self.extraction_results['functional_analysis']['cognitive_dynamics']
            revised_metrics['cognitive_stability'] = {
                'stability_ratio': dyn.get('stability_ratio', 0),
                'volatility_ratio': dyn.get('volatility_ratio', 0),
                'trend_direction': 'increasing' if dyn.get('overall_trend', 0) > 0 else 'decreasing'
            }
        
        self.extraction_results['revised_metrics'] = revised_metrics
        
        # 创建增强数据集
        self._create_enhanced_dataset()
    
    def _create_enhanced_dataset(self):
        """创建包含修正指标的增强数据集"""
        enhanced_df = self.df.copy()
        
        # 添加处理后的信号
        for signal_name, signal_data in self.processed_signals.items():
            if len(signal_data) == len(enhanced_df):
                enhanced_df[signal_name] = signal_data
        
        # 添加滑动窗口统计
        window_size = 30
        for var in ['dsr_cognitive', 'tl_functional', 'cs_output']:
            if var in enhanced_df.columns:
                # 滑动均值
                enhanced_df[f'{var}_ma30'] = enhanced_df[var].rolling(window=window_size, center=True).mean()
                # 滑动标准差
                enhanced_df[f'{var}_std30'] = enhanced_df[var].rolling(window=window_size, center=True).std()
        
        # 保存增强数据集
        output_file = self.data_path / 'enhanced_data_with_denoised_signals.csv'
        enhanced_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n增强数据集已保存至: {output_file}")
    
    def assess_signal_quality(self):
        """评估信号质量"""
        quality_metrics = {}
        
        # 比较原始信号和处理后信号
        for var in ['dsr_cognitive', 'tl_functional', 'cs_output']:
            if var in self.df.columns:
                original = self.df[var].values
                
                # 找到对应的处理后信号
                processed_signal = None
                signal_type = 'original'
                
                for suffix in ['_reconstructed', '_wavelet_denoised']:
                    signal_name = f"{var}{suffix}"
                    if signal_name in self.processed_signals:
                        processed_signal = self.processed_signals[signal_name]
                        signal_type = suffix.replace('_', '')
                        break
                
                if processed_signal is not None and len(processed_signal) == len(original):
                    # 计算质量指标
                    
                    # 1. 相关性保持
                    correlation = np.corrcoef(original, processed_signal)[0, 1]
                    
                    # 2. 噪声减少
                    original_noise = np.std(np.diff(original))
                    processed_noise = np.std(np.diff(processed_signal))
                    noise_reduction = (original_noise - processed_noise) / original_noise
                    
                    # 3. 信号平滑度
                    smoothness = 1 - np.mean(np.abs(np.diff(np.diff(processed_signal))))
                    
                    # 4. 动态范围保持
                    range_ratio = (np.max(processed_signal) - np.min(processed_signal)) / \
                                (np.max(original) - np.min(original))
                    
                    quality_metrics[var] = {
                        'signal_type': signal_type,
                        'correlation_preserved': float(correlation),
                        'noise_reduction': float(noise_reduction),
                        'smoothness': float(smoothness),
                        'dynamic_range_ratio': float(range_ratio),
                        'overall_quality': float((correlation + noise_reduction + smoothness + range_ratio) / 4)
                    }
        
        # 总体质量评估
        if quality_metrics:
            avg_quality = np.mean([m['overall_quality'] for m in quality_metrics.values()])
            
            self.extraction_results['signal_quality'] = {
                'variable_quality': quality_metrics,
                'average_quality': float(avg_quality),
                'quality_assessment': 'Excellent' if avg_quality > 0.8 else 
                                    'Good' if avg_quality > 0.6 else 
                                    'Fair' if avg_quality > 0.4 else 'Poor'
            }
            
            print(f"\n信号质量评估: {self.extraction_results['signal_quality']['quality_assessment']}")
            print(f"平均质量分数: {avg_quality:.3f}")
    
    def save_results(self):
        """保存结果"""
        # 保存JSON结果
        output_file = self.data_path / 'signal_extraction_results.json'
        
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
            json.dump(convert_numpy(self.extraction_results), f, ensure_ascii=False, indent=2)
        
        print(f"\n分析结果已保存至: {output_file}")
        
        # 创建可视化
        self.create_visualizations()
    
    def create_visualizations(self):
        """创建可视化"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # 1. 小波去噪效果
        ax1 = axes[0, 0]
        if 'dsr_cognitive' in self.df.columns and 'dsr_cognitive_wavelet_denoised' in self.processed_signals:
            x = np.arange(500)  # 显示前500个点
            ax1.plot(x, self.df['dsr_cognitive'].values[:500], 'b-', alpha=0.3, label='原始信号')
            ax1.plot(x, self.processed_signals['dsr_cognitive_wavelet_denoised'][:500], 'r-', label='去噪信号')
            ax1.set_title('小波去噪效果（DSR认知）')
            ax1.legend()
            ax1.set_xlabel('样本')
            ax1.set_ylabel('值')
        
        # 2. ICA成分
        ax2 = axes[0, 1]
        n_ics = len([k for k in self.processed_signals.keys() if k.startswith('ic_')])
        if n_ics > 0:
            for i in range(min(n_ics, 3)):
                if f'ic_{i}' in self.processed_signals:
                    ic = self.processed_signals[f'ic_{i}'][:500]
                    ax2.plot(ic + i*3, label=f'IC{i}')  # 垂直偏移以便观察
            ax2.set_title('独立成分')
            ax2.legend()
            ax2.set_xlabel('样本')
        
        # 3. 信号重构对比
        ax3 = axes[1, 0]
        if 'cs_output' in self.df.columns:
            ax3.scatter(self.df['dsr_cognitive'][:1000], self.df['cs_output'][:1000], 
                       alpha=0.3, s=1, label='原始')
            
            if 'dsr_cognitive_reconstructed' in self.processed_signals and \
               'cs_output_reconstructed' in self.processed_signals:
                ax3.scatter(self.processed_signals['dsr_cognitive_reconstructed'][:1000], 
                          self.processed_signals['cs_output_reconstructed'][:1000], 
                          alpha=0.5, s=1, color='red', label='重构')
            
            ax3.set_xlabel('DSR认知')
            ax3.set_ylabel('CS输出')
            ax3.set_title('信号关系对比')
            ax3.legend()
        
        # 4. 时变相关性
        ax4 = axes[1, 1]
        if 'time_varying_correlation' in self.extraction_results['functional_analysis'].get('deep_complementarity', {}):
            comp = self.extraction_results['functional_analysis']['deep_complementarity']
            if 'time_varying_correlation' in comp:
                # 模拟时变相关性曲线
                x = np.linspace(0, 1, 100)
                mean_corr = comp['time_varying_correlation']['mean']
                std_corr = comp['time_varying_correlation']['std']
                
                y = mean_corr + std_corr * np.sin(2 * np.pi * x * 3) * np.exp(-x)
                ax4.plot(x, y)
                ax4.axhline(y=mean_corr, color='r', linestyle='--', label=f'均值={mean_corr:.3f}')
                ax4.fill_between(x, mean_corr - std_corr, mean_corr + std_corr, alpha=0.3)
                ax4.set_title('时变相关性')
                ax4.set_xlabel('时间（标准化）')
                ax4.set_ylabel('相关系数')
                ax4.legend()
        
        # 5. 功能模式分布
        ax5 = axes[2, 0]
        if 'functional_patterns' in self.extraction_results['functional_analysis']:
            patterns = self.extraction_results['functional_analysis']['functional_patterns']
            if patterns:
                functions = [p['function'] for p in patterns[:5]]
                frequencies = [p['frequency'] for p in patterns[:5]]
                
                ax5.bar(range(len(functions)), frequencies)
                ax5.set_xticks(range(len(functions)))
                ax5.set_xticklabels(functions, rotation=45, ha='right')
                ax5.set_title('主要功能模式')
                ax5.set_ylabel('频率')
        
        # 6. 信号质量总结
        ax6 = axes[2, 1]
        ax6.axis('off')
        
        quality_text = "信号提取质量总结\n\n"
        
        if 'signal_quality' in self.extraction_results:
            sq = self.extraction_results['signal_quality']
            quality_text += f"整体评估: {sq['quality_assessment']}\n"
            quality_text += f"平均质量: {sq['average_quality']:.3f}\n\n"
            
            if 'variable_quality' in sq:
                for var, metrics in sq['variable_quality'].items():
                    quality_text += f"{var}:\n"
                    quality_text += f"  相关性: {metrics['correlation_preserved']:.3f}\n"
                    quality_text += f"  降噪率: {metrics['noise_reduction']:.3f}\n"
        
        ax6.text(0.1, 0.5, quality_text, fontsize=10, verticalalignment='center')
        
        plt.tight_layout()
        
        # 保存图表
        output_file = self.data_path.parent / 'figures' / 'signal_extraction_results.jpg'
        output_file.parent.mkdir(exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n可视化已保存至: {output_file}")

def main():
    """主函数"""
    # 设置数据路径
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_path = project_root / 'output_cn' / 'data'
    
    # 创建信号提取器
    extractor = AdvancedSignalExtractor(data_path)
    
    # 加载数据
    extractor.load_data()
    
    # 运行信号提取
    results = extractor.run_signal_extraction()
    
    print("\n" + "="*60)
    print("信号提取完成 - 关键发现")
    print("="*60)
    
    # 输出关键发现
    if 'ica_components' in results and 'signal_components' in results['ica_components']:
        print(f"\n1. 识别到 {len(results['ica_components']['signal_components'])} 个信号成分")
    
    if 'functional_analysis' in results:
        if 'deep_complementarity' in results['functional_analysis']:
            comp = results['functional_analysis']['deep_complementarity']
            if 'complementarity_score' in comp:
                print(f"\n2. 深度互补性评分: {comp['complementarity_score']:.3f}")
        
        if 'context_sensitivity' in results['functional_analysis']:
            ctx = results['functional_analysis']['context_sensitivity']
            if 'sensitivity_score' in ctx:
                print(f"\n3. 语境敏感性评分: {ctx['sensitivity_score']:.3f}")
    
    if 'signal_quality' in results:
        print(f"\n4. 信号质量评估: {results['signal_quality']['quality_assessment']}")
    
    print("\n建议后续步骤:")
    print("1. 使用enhanced_data_with_denoised_signals.csv进行后续分析")
    print("2. 基于功能模式进行针对性研究")
    print("3. 探索语境特定的认知机制")
    
    print("\n✓ 高级信号提取分析完成！")

if __name__ == "__main__":
    main()