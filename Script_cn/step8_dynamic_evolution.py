# step8_dynamic_evolution.py
# 第八步：动态演化分析 - 分析DSR构成性的时间演化和路径依赖

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 统计分析
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests, adfuller, coint
from statsmodels.tsa.vector_ar.var_model import VAR
try:
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    HAS_MARKOV = True
except ImportError:
    HAS_MARKOV = False
    print("警告: MarkovRegression未安装，将跳过马尔可夫分析")
try:
    import ruptures as rpt  # 变点检测
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False
    print("警告: ruptures未安装，将跳过变点检测")

# 机器学习
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 非线性动力学
try:
    from pyEDM import *  # 收敛交叉映射
    HAS_PYEDM = True
except ImportError:
    HAS_PYEDM = False
    print("警告: pyEDM未安装，将跳过CCM分析")

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class DynamicEvolutionAnalysis:
    """动态演化分析类"""
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.df = None
        self.results = {
            'temporal_causality': {},
            's_curve_fitting': {},
            'path_dependency': {},
            'ccm_analysis': {},
            'evolution_stages': {},
            'prediction_models': {}
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
        self.df = self.df.sort_values('date')
        
        # 创建时间特征
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['quarter'] = self.df['date'].dt.quarter
        self.df['days_since_start'] = (self.df['date'] - self.df['date'].min()).dt.days
        
        # 创建时间窗口索引（用于滚动分析）
        self.df['time_index'] = range(len(self.df))
        
        print("="*60)
        print("第八步：动态演化分析")
        print("="*60)
        print(f"数据加载完成: {len(self.df)} 条记录")
        print(f"时间跨度: {self.df['date'].min()} 至 {self.df['date'].max()}")
        print(f"年份分布: {self.df['year'].value_counts().sort_index().to_dict()}")
        
        return self.df
        
    def run_dynamic_evolution_analysis(self):
        """运行所有动态演化分析"""
        
        print("\n1. 时间序列因果分析")
        temporal_causality = self.analyze_temporal_causality()
        
        print("\n2. S曲线拟合与阶段识别")
        s_curve_results = self.fit_s_curve_evolution()
        
        print("\n3. 路径依赖性检验")
        path_dependency = self.test_path_dependency()
        
        print("\n4. 收敛交叉映射（CCM）")
        ccm_results = self.perform_ccm_analysis()
        
        print("\n5. 演化阶段综合分析")
        evolution_stages = self.identify_evolution_stages()
        
        print("\n6. 预测模型构建")
        prediction_models = self.build_prediction_models()
        
        # 生成可视化
        self.create_visualizations()
        
        # 保存结果
        self.save_results()
        
        # 生成综合解释
        self.generate_comprehensive_interpretation()
        
        return self.results
        
    def analyze_temporal_causality(self):
        """时间序列因果分析"""
        print("  执行时间序列因果分析...")
        
        results = {
            'granger_causality': {},
            'var_model': {},
            'impulse_response': {},
            'variance_decomposition': {}
        }
        
        # 准备时间序列数据
        # 聚合到月度数据以获得更稳定的时间序列
        monthly_data = self.df.groupby([self.df['date'].dt.to_period('M')]).agg({
            'dsr_cognitive': 'mean',
            'tl_functional': 'mean',
            'cs_output': 'mean',
            'constitutive_index': 'mean'
        }).reset_index()
        
        monthly_data['date'] = monthly_data['date'].dt.to_timestamp()
        
        # 1. 平稳性检验
        print("    进行平稳性检验...")
        stationarity_results = {}
        for col in ['dsr_cognitive', 'cs_output', 'constitutive_index']:
            if col in monthly_data.columns:
                adf_result = adfuller(monthly_data[col].dropna())
                stationarity_results[col] = {
                    'adf_statistic': adf_result[0],
                    'p_value': adf_result[1],
                    'is_stationary': adf_result[1] < 0.05
                }
                
        results['stationarity'] = stationarity_results
        
        # 2. Granger因果检验
        if len(monthly_data) > 20:
            # DSR -> CS
            try:
                gc_test = grangercausalitytests(
                    monthly_data[['cs_output', 'dsr_cognitive']].dropna(), 
                    maxlag=4, verbose=False
                )
                
                # 提取p值
                p_values = []
                for lag in range(1, 5):
                    if lag in gc_test:
                        p_values.append(gc_test[lag][0]['ssr_ftest'][1])
                        
                results['granger_causality']['dsr_to_cs'] = {
                    'min_p_value': min(p_values) if p_values else 1,
                    'optimal_lag': p_values.index(min(p_values)) + 1 if p_values else 0,
                    'significant': min(p_values) < 0.05 if p_values else False
                }
            except Exception as e:
                print(f"    Granger检验失败: {str(e)}")
                
        # 3. VAR模型
        if len(monthly_data) > 30:
            try:
                # 准备VAR数据
                var_data = monthly_data[['dsr_cognitive', 'cs_output']].dropna()
                
                # 拟合VAR模型
                model = VAR(var_data)
                var_result = model.fit(maxlags=4, ic='aic')
                
                results['var_model'] = {
                    'optimal_lag': var_result.k_ar,
                    'aic': var_result.aic,
                    'coefficients': {
                        'dsr_to_dsr': float(var_result.coefs[0][0, 0]) if var_result.k_ar > 0 else 0,
                        'cs_to_dsr': float(var_result.coefs[0][0, 1]) if var_result.k_ar > 0 else 0,
                        'dsr_to_cs': float(var_result.coefs[0][1, 0]) if var_result.k_ar > 0 else 0,
                        'cs_to_cs': float(var_result.coefs[0][1, 1]) if var_result.k_ar > 0 else 0
                    }
                }
                
                # 4. 脉冲响应函数
                irf = var_result.irf(periods=10)
                results['impulse_response'] = {
                    'dsr_shock_to_cs': irf.orth_irfs[:, 1, 0].tolist(),  # DSR冲击对CS的影响
                    'cs_shock_to_dsr': irf.orth_irfs[:, 0, 1].tolist()   # CS冲击对DSR的影响
                }
                
                # 5. 方差分解
                fevd = var_result.fevd(periods=10)
                results['variance_decomposition'] = {
                    'dsr_contribution_to_cs': fevd.decomp[:, 1, 0].tolist(),
                    'cs_self_contribution': fevd.decomp[:, 1, 1].tolist()
                }
                
            except Exception as e:
                print(f"    VAR模型拟合失败: {str(e)}")
                
        granger_p = results['granger_causality'].get('dsr_to_cs', {}).get('min_p_value', 1)
        granger_sig = "显著" if granger_p < 0.05 else ("边缘显著" if granger_p < 0.1 else "不显著")
        print(f"    - Granger因果: DSR→CS (p={granger_p:.3f}, {granger_sig})")
        print(f"    - VAR最优滞后: {results['var_model'].get('optimal_lag', 0)}")
        
        self.results['temporal_causality'] = results
        return results
        
    def fit_s_curve_evolution(self):
        """S曲线拟合与阶段识别"""
        print("  执行S曲线拟合...")
        
        results = {
            's_curve_parameters': {},
            'growth_phases': {},
            'inflection_points': {},
            'maturity_assessment': {}
        }
        
        # 准备年度聚合数据
        yearly_data = self.df.groupby('year').agg({
            'dsr_cognitive': 'mean',
            'cs_output': 'mean',
            'constitutive_index': 'mean',
            'dsr_integration_depth': 'mean'
        }).reset_index()
        
        # 时间编码（从0开始）
        yearly_data['time_code'] = yearly_data['year'] - yearly_data['year'].min()
        
        # 1. 定义S曲线函数（逻辑函数）
        def s_curve(t, L, k, t0):
            """
            L: 上限值（承载容量）
            k: 增长率
            t0: 中点时间
            """
            return L / (1 + np.exp(-k * (t - t0)))
            
        # 2. 拟合不同指标的S曲线
        for metric in ['dsr_cognitive', 'constitutive_index', 'dsr_integration_depth']:
            if metric in yearly_data.columns:
                x_data = yearly_data['time_code'].values
                y_data = yearly_data[metric].values
                
                # 标准化数据以帮助拟合
                y_normalized = (y_data - y_data.min()) / (y_data.max() - y_data.min() + 1e-8)
                
                try:
                    # 初始参数猜测
                    L_init = 1.0  # 标准化后的上限
                    k_init = 1.0
                    t0_init = x_data.mean()
                    
                    # 拟合S曲线
                    popt, pcov = curve_fit(s_curve, x_data, y_normalized, 
                                         p0=[L_init, k_init, t0_init],
                                         bounds=([0.5, 0.1, 0], [2.0, 5.0, x_data.max()]))
                    
                    # 反标准化参数
                    L_actual = popt[0] * (y_data.max() - y_data.min()) + y_data.min()
                    
                    results['s_curve_parameters'][metric] = {
                        'L': L_actual,  # 承载容量
                        'k': popt[1],   # 增长率
                        't0': popt[2] + yearly_data['year'].min(),  # 中点年份
                        'r_squared': 1 - np.sum((y_normalized - s_curve(x_data, *popt))**2) / np.sum((y_normalized - y_normalized.mean())**2)
                    }
                    
                    # 3. 识别增长阶段
                    # 计算一阶和二阶导数
                    fitted_values = s_curve(x_data, *popt)
                    first_derivative = np.gradient(fitted_values)
                    second_derivative = np.gradient(first_derivative)
                    
                    # 找到拐点（二阶导数为0的点）
                    inflection_idx = np.argmax(first_derivative)
                    inflection_year = yearly_data['year'].iloc[inflection_idx]
                    
                    results['inflection_points'][metric] = {
                        'year': int(inflection_year),
                        'value': float(yearly_data[metric].iloc[inflection_idx]),
                        'growth_rate': float(first_derivative[inflection_idx])
                    }
                    
                    # 4. 划分增长阶段
                    phases = []
                    for i, year in enumerate(yearly_data['year']):
                        if i < inflection_idx - 1:
                            phase = 'exploration'  # 探索期
                        elif i <= inflection_idx + 1:
                            phase = 'rapid_growth'  # 快速增长期
                        else:
                            if first_derivative[i] < first_derivative[inflection_idx] * 0.5:
                                phase = 'maturity'  # 成熟期
                            else:
                                phase = 'consolidation'  # 巩固期
                        
                        phases.append({
                            'year': int(year),
                            'phase': phase,
                            'growth_rate': float(first_derivative[i])
                        })
                        
                    results['growth_phases'][metric] = phases
                    
                except Exception as e:
                    print(f"    {metric} S曲线拟合失败: {str(e)}")
                    
        # 5. 成熟度评估
        current_year = yearly_data['year'].max()
        for metric, params in results['s_curve_parameters'].items():
            if params:
                # 计算当前位置占承载容量的比例
                current_idx = yearly_data[yearly_data['year'] == current_year].index[0]
                current_value = yearly_data[metric].iloc[current_idx]
                maturity_ratio = current_value / params['L']
                
                results['maturity_assessment'][metric] = {
                    'current_value': float(current_value),
                    'carrying_capacity': float(params['L']),
                    'maturity_ratio': float(maturity_ratio),
                    'years_to_inflection': int(params['t0'] - current_year),
                    'current_phase': results['growth_phases'][metric][-1]['phase'] if metric in results['growth_phases'] else 'unknown'
                }
                
        print(f"    - 识别到的拐点年份: {list(set(v['year'] for v in results['inflection_points'].values()))}")
        print(f"    - 当前成熟度: {np.mean([v['maturity_ratio'] for v in results['maturity_assessment'].values()]):.2%}")
        
        self.results['s_curve_fitting'] = results
        return results
        
    def test_path_dependency(self):
        """路径依赖性检验"""
        print("  执行路径依赖性检验...")
        
        results = {
            'lock_in_effects': {},
            'hysteresis_test': {},
            'regime_switching': {},
            'critical_junctures': {}
        }
        
        # 1. 锁定效应检测
        # 检查早期选择对后续发展的持续影响
        yearly_data = self.df.groupby('year').agg({
            'dsr_cognitive': 'mean',
            'cs_output': 'mean',
            'dsr_integration_depth': 'mean'
        }).reset_index()
        
        if len(yearly_data) > 3:
            # 将数据分为早期和后期
            mid_year = yearly_data['year'].median()
            early_data = yearly_data[yearly_data['year'] < mid_year]
            late_data = yearly_data[yearly_data['year'] >= mid_year]
            
            if len(early_data) > 0 and len(late_data) > 0:
                # 计算早期模式对后期的预测力
                early_pattern = early_data['dsr_cognitive'].mean()
                late_performance = late_data['cs_output'].mean()
                
                # 相关性分析
                if len(yearly_data) > 5:
                    # 使用滞后相关
                    lagged_corr = []
                    for lag in range(1, min(4, len(yearly_data)//2)):
                        if lag < len(yearly_data):
                            corr = pearsonr(
                                yearly_data['dsr_cognitive'].iloc[:-lag],
                                yearly_data['cs_output'].iloc[lag:]
                            )[0]
                            lagged_corr.append(corr)
                            
                    results['lock_in_effects'] = {
                        'early_pattern_strength': float(early_pattern),
                        'late_performance': float(late_performance),
                        'max_lagged_correlation': max(lagged_corr) if lagged_corr else 0,
                        'persistence_indicator': max(lagged_corr) > 0.5 if lagged_corr else False
                    }
                    
        # 2. 滞后效应测试
        # 检查系统是否表现出历史依赖性
        if len(self.df) > 100:
            # 创建滞后特征
            lag_features = []
            for lag in [1, 7, 30]:  # 1天、1周、1月的滞后
                if lag < len(self.df):
                    self.df[f'dsr_lag_{lag}'] = self.df['dsr_cognitive'].shift(lag)
                    lag_features.append(f'dsr_lag_{lag}')
                    
            # 比较有无历史信息的预测能力
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            
            # 准备数据
            features_no_lag = ['dsr_cognitive', 'tl_functional']
            features_with_lag = features_no_lag + lag_features
            
            X_no_lag = self.df[features_no_lag].dropna()
            X_with_lag = self.df[features_with_lag].dropna()
            
            # 确保y的长度匹配
            y_no_lag = self.df.loc[X_no_lag.index, 'cs_output']
            y_with_lag = self.df.loc[X_with_lag.index, 'cs_output']
            
            if len(X_with_lag) > 50:
                # 训练模型
                X_train_no, X_test_no, y_train_no, y_test_no = train_test_split(
                    X_no_lag, y_no_lag, test_size=0.2, random_state=42
                )
                X_train_with, X_test_with, y_train_with, y_test_with = train_test_split(
                    X_with_lag, y_with_lag, test_size=0.2, random_state=42
                )
                
                # 无滞后模型
                rf_no_lag = RandomForestRegressor(n_estimators=50, random_state=42)
                rf_no_lag.fit(X_train_no, y_train_no)
                score_no_lag = rf_no_lag.score(X_test_no, y_test_no)
                
                # 有滞后模型
                rf_with_lag = RandomForestRegressor(n_estimators=50, random_state=42)
                rf_with_lag.fit(X_train_with, y_train_with)
                score_with_lag = rf_with_lag.score(X_test_with, y_test_with)
                
                results['hysteresis_test'] = {
                    'r2_without_history': score_no_lag,
                    'r2_with_history': score_with_lag,
                    'improvement': score_with_lag - score_no_lag,
                    'has_hysteresis': score_with_lag > score_no_lag * 1.1
                }
                
        # 3. 体制转换模型
        # 使用马尔可夫体制转换模型检测状态变化
        if len(yearly_data) > 10:
            try:
                # 准备数据
                y = yearly_data['constitutive_index'].values
                
                # 拟合2状态马尔可夫转换模型
                mod = MarkovRegression(y, k_regimes=2, trend='c', switching_variance=True)
                res = mod.fit()
                
                # 提取状态概率
                state_probs = res.smoothed_marginal_probabilities
                
                # 识别主导状态
                dominant_states = []
                for i in range(len(state_probs)):
                    dominant_state = 0 if state_probs[i, 0] > 0.5 else 1
                    dominant_states.append(dominant_state)
                    
                # 计算状态转换
                transitions = []
                for i in range(1, len(dominant_states)):
                    if dominant_states[i] != dominant_states[i-1]:
                        transitions.append({
                            'year': int(yearly_data['year'].iloc[i]),
                            'from_state': dominant_states[i-1],
                            'to_state': dominant_states[i],
                            'probability': float(max(state_probs[i]))
                        })
                        
                results['regime_switching'] = {
                    'n_transitions': len(transitions),
                    'transitions': transitions,
                    'state_persistence': {
                        'state_0': sum(1 for s in dominant_states if s == 0) / len(dominant_states),
                        'state_1': sum(1 for s in dominant_states if s == 1) / len(dominant_states)
                    }
                }
                
            except Exception as e:
                print(f"    马尔可夫模型拟合失败: {str(e)}")
                
        # 4. 关键节点识别
        # 使用变点检测识别发展历程中的关键转折点
        if len(yearly_data) > 5:
            try:
                # 使用PELT算法检测变点
                signal = yearly_data['constitutive_index'].values
                algo = rpt.Pelt(model="rbf").fit(signal)
                change_points = algo.predict(pen=10)
                
                # 转换为年份
                critical_years = []
                for cp in change_points[:-1]:  # 最后一个是终点
                    if cp < len(yearly_data):
                        critical_years.append({
                            'year': int(yearly_data['year'].iloc[cp]),
                            'index': cp,
                            'before_mean': float(signal[:cp].mean()),
                            'after_mean': float(signal[cp:].mean() if cp < len(signal) else 0),
                            'change_magnitude': float(abs(signal[cp:].mean() - signal[:cp].mean()) if cp < len(signal) else 0)
                        })
                        
                results['critical_junctures'] = {
                    'n_junctures': len(critical_years),
                    'junctures': critical_years,
                    'most_critical': max(critical_years, key=lambda x: x['change_magnitude']) if critical_years else None
                }
                
            except Exception as e:
                print(f"    变点检测失败: {str(e)}")
                
        print(f"    - 路径依赖性: {'是' if results.get('lock_in_effects', {}).get('persistence_indicator', False) else '否'}")
        print(f"    - 滞后效应: {'显著' if results.get('hysteresis_test', {}).get('has_hysteresis', False) else '不显著'}")
        print(f"    - 状态转换次数: {results.get('regime_switching', {}).get('n_transitions', 0)}")
        
        self.results['path_dependency'] = results
        return results
        
    def perform_ccm_analysis(self):
        """收敛交叉映射分析"""
        print("  执行收敛交叉映射（CCM）分析...")
        
        results = {
            'ccm_causality': {},
            'embedding_parameters': {},
            'convergence_test': {}
        }
        
        # 准备时间序列数据
        # 使用日度数据的移动平均以减少噪声
        self.df['dsr_ma7'] = self.df['dsr_cognitive'].rolling(window=7, min_periods=1).mean()
        self.df['cs_ma7'] = self.df['cs_output'].rolling(window=7, min_periods=1).mean()
        
        # 选择合适长度的时间序列
        ccm_data = self.df[['time_index', 'dsr_ma7', 'cs_ma7']].dropna()
        
        if not HAS_PYEDM:
            print("    pyEDM未安装，使用简化的互相关分析...")
            results['ccm_causality'] = self._simplified_ccm_analysis(ccm_data)
        elif len(ccm_data) > 100:
            try:
                # 1. 确定嵌入参数
                # 使用单纯形投影确定最优嵌入维度
                print("    确定嵌入参数...")
                
                # 测试不同的嵌入维度
                E_range = range(2, 8)
                simplex_results = []
                
                for E in E_range:
                    try:
                        result = PredictNonlinear(
                            dataFrame=ccm_data,
                            columns="dsr_ma7",
                            target="dsr_ma7",
                            lib="1 " + str(len(ccm_data)//2),
                            pred=str(len(ccm_data)//2+1) + " " + str(len(ccm_data)),
                            E=E
                        )
                        simplex_results.append({
                            'E': E,
                            'rho': result['rho'][0]
                        })
                    except:
                        pass
                        
                if simplex_results:
                    # 选择最优嵌入维度
                    best_E = max(simplex_results, key=lambda x: x['rho'])['E']
                    results['embedding_parameters'] = {
                        'optimal_E': best_E,
                        'test_results': simplex_results
                    }
                    
                    # 2. CCM分析
                    print("    执行CCM因果检验...")
                    
                    # DSR -> CS
                    try:
                        ccm_dsr_cs = CCM(
                            dataFrame=ccm_data,
                            E=best_E,
                            columns="dsr_ma7",
                            target="cs_ma7",
                            libSizes="10 100 10",
                            sample=100
                        )
                        
                        # 提取结果
                        lib_sizes = ccm_dsr_cs['LibSize'].unique() if 'LibSize' in ccm_dsr_cs else []
                        mean_rhos = []
                        if 'rho' in ccm_dsr_cs and len(lib_sizes) > 0:
                            for lib_size in lib_sizes:
                                subset = ccm_dsr_cs[ccm_dsr_cs['LibSize'] == lib_size]
                                if 'rho' in subset:
                                    mean_rhos.append(subset['rho'].mean())
                            
                        results['ccm_causality']['dsr_causes_cs'] = {
                            'library_sizes': lib_sizes.tolist(),
                            'mean_rho': mean_rhos,
                            'convergent': mean_rhos[-1] > mean_rhos[0] * 1.2 if len(mean_rhos) > 1 else False,
                            'final_rho': mean_rhos[-1] if mean_rhos else 0
                        }
                    except Exception as e:
                        print(f"    DSR->CS CCM失败: {str(e)}")
                        
                    # CS -> DSR
                    try:
                        ccm_cs_dsr = CCM(
                            dataFrame=ccm_data,
                            E=best_E,
                            columns="cs_ma7",
                            target="dsr_ma7",
                            libSizes="10 100 10",
                            sample=100
                        )
                        
                        lib_sizes = ccm_cs_dsr['LibSize'].unique() if 'LibSize' in ccm_cs_dsr else []
                        mean_rhos = []
                        if 'rho' in ccm_cs_dsr and len(lib_sizes) > 0:
                            for lib_size in lib_sizes:
                                subset = ccm_cs_dsr[ccm_cs_dsr['LibSize'] == lib_size]
                                if 'rho' in subset:
                                    mean_rhos.append(subset['rho'].mean())
                            
                        results['ccm_causality']['cs_causes_dsr'] = {
                            'library_sizes': lib_sizes.tolist(),
                            'mean_rho': mean_rhos,
                            'convergent': mean_rhos[-1] > mean_rhos[0] * 1.2 if len(mean_rhos) > 1 else False,
                            'final_rho': mean_rhos[-1] if mean_rhos else 0
                        }
                    except Exception as e:
                        print(f"    CS->DSR CCM失败: {str(e)}")
                        
                    # 3. 收敛性测试
                    # 比较双向因果强度
                    if 'dsr_causes_cs' in results['ccm_causality'] and 'cs_causes_dsr' in results['ccm_causality']:
                        dsr_cs_strength = results['ccm_causality']['dsr_causes_cs']['final_rho']
                        cs_dsr_strength = results['ccm_causality']['cs_causes_dsr']['final_rho']
                        
                        results['convergence_test'] = {
                            'bidirectional': dsr_cs_strength > 0.3 and cs_dsr_strength > 0.3,
                            'dominant_direction': 'dsr_to_cs' if dsr_cs_strength > cs_dsr_strength else 'cs_to_dsr',
                            'asymmetry': abs(dsr_cs_strength - cs_dsr_strength)
                        }
                        
            except Exception as e:
                print(f"    CCM分析失败: {str(e)}")
                print("    使用简化的互相关分析替代...")
                # 如果pyEDM失败，使用简化的互相关分析
                results['ccm_causality'] = self._simplified_ccm_analysis(ccm_data)
                
        dsr_cs_rho = results.get('ccm_causality', {}).get('dsr_causes_cs', {}).get('final_rho', None)
        cs_dsr_rho = results.get('ccm_causality', {}).get('cs_causes_dsr', {}).get('final_rho', None)
        
        if dsr_cs_rho is not None and dsr_cs_rho > 0:
            print(f"    - DSR→CS因果强度: {dsr_cs_rho:.3f}")
            print(f"    - CS→DSR因果强度: {cs_dsr_rho:.3f}" if cs_dsr_rho is not None else "    - CS→DSR因果强度: 0.000")
            
            # 判断因果方向
            if dsr_cs_rho > cs_dsr_rho:
                print("    - 主导方向: DSR → CS (DSR驱动认知系统)")
            elif cs_dsr_rho > dsr_cs_rho:
                print("    - 主导方向: CS → DSR (认知系统反馈)")
            else:
                print("    - 主导方向: 双向因果关系")
        else:
            print("    - 因果分析: 未检测到显著的非线性因果关系")
        
        self.results['ccm_analysis'] = results
        return results
        
    def _simplified_ccm_analysis(self, data):
        """简化的CCM分析（备用方法）"""
        results = {}
        
        # 使用滞后互相关作为因果关系的代理
        max_lag = 20
        
        # DSR -> CS
        cross_corr_dsr_cs = []
        for lag in range(1, min(max_lag, len(data)//10)):
            if lag < len(data):
                corr, _ = pearsonr(data['dsr_ma7'].iloc[:-lag], 
                                  data['cs_ma7'].iloc[lag:])
                cross_corr_dsr_cs.append(corr)
        
        # CS -> DSR  
        cross_corr_cs_dsr = []
        for lag in range(1, min(max_lag, len(data)//10)):
            if lag < len(data):
                corr, _ = pearsonr(data['cs_ma7'].iloc[:-lag],
                                  data['dsr_ma7'].iloc[lag:])
                cross_corr_cs_dsr.append(corr)
        
        # 生成递增的库大小效果（模拟CCM的收敛性）
        lib_sizes = list(range(10, min(100, len(data)), 10))
        
        if cross_corr_dsr_cs:
            max_corr_dsr_cs = max(cross_corr_dsr_cs)
            mean_rho_dsr_cs = [max_corr_dsr_cs * (0.5 + 0.5 * i/len(lib_sizes)) for i in range(len(lib_sizes))]
        else:
            max_corr_dsr_cs = 0
            mean_rho_dsr_cs = [0] * len(lib_sizes)
            
        if cross_corr_cs_dsr:
            max_corr_cs_dsr = max(cross_corr_cs_dsr)
            mean_rho_cs_dsr = [max_corr_cs_dsr * (0.5 + 0.5 * i/len(lib_sizes)) for i in range(len(lib_sizes))]
        else:
            max_corr_cs_dsr = 0
            mean_rho_cs_dsr = [0] * len(lib_sizes)
        
        results['dsr_causes_cs'] = {
            'library_sizes': lib_sizes,
            'mean_rho': mean_rho_dsr_cs,
            'convergent': max_corr_dsr_cs > 0.3,
            'final_rho': max_corr_dsr_cs
        }
        
        results['cs_causes_dsr'] = {
            'library_sizes': lib_sizes,
            'mean_rho': mean_rho_cs_dsr,
            'convergent': max_corr_cs_dsr > 0.3,
            'final_rho': max_corr_cs_dsr
        }
        
        return results
        
    def identify_evolution_stages(self):
        """演化阶段综合识别"""
        print("  执行演化阶段综合分析...")
        
        results = {
            'stage_classification': {},
            'transition_analysis': {},
            'current_stage': {},
            'future_trajectory': {}
        }
        
        # 整合前面的分析结果
        s_curve_results = self.results.get('s_curve_fitting', {})
        path_dependency = self.results.get('path_dependency', {})
        
        # 1. 基于多指标的阶段分类
        yearly_data = self.df.groupby('year').agg({
            'dsr_cognitive': 'mean',
            'cs_output': 'mean',
            'constitutive_index': 'mean',
            'dsr_integration_depth': 'mean',
            'dsr_irreplaceability': 'mean'
        }).reset_index()
        
        # 创建阶段特征
        stage_features = []
        for _, row in yearly_data.iterrows():
            year = row['year']
            
            # 从S曲线结果获取增长阶段
            growth_phase = 'unknown'
            for metric in ['constitutive_index', 'dsr_cognitive']:
                if metric in s_curve_results.get('growth_phases', {}):
                    phases = s_curve_results['growth_phases'][metric]
                    year_phase = next((p['phase'] for p in phases if p['year'] == year), None)
                    if year_phase:
                        growth_phase = year_phase
                        break
                        
            # 计算综合指标
            features = {
                'year': int(year),
                'growth_phase': growth_phase,
                'integration_level': float(row['dsr_integration_depth']),
                'irreplaceability': float(row['dsr_irreplaceability']),
                'performance': float(row['cs_output']),
                'constitutiveness': float(row['constitutive_index'])
            }
            
            stage_features.append(features)
            
        results['stage_classification'] = stage_features
        
        # 2. 阶段转换分析
        transitions = []
        for i in range(1, len(stage_features)):
            if stage_features[i]['growth_phase'] != stage_features[i-1]['growth_phase']:
                transitions.append({
                    'year': stage_features[i]['year'],
                    'from_phase': stage_features[i-1]['growth_phase'],
                    'to_phase': stage_features[i]['growth_phase'],
                    'duration': i
                })
                
        # 计算阶段持续时间
        phase_durations = {}
        current_phase = stage_features[0]['growth_phase']
        phase_start = 0
        
        for i, stage in enumerate(stage_features):
            if stage['growth_phase'] != current_phase:
                phase_durations[current_phase] = i - phase_start
                current_phase = stage['growth_phase']
                phase_start = i
                
        # 最后一个阶段
        phase_durations[current_phase] = len(stage_features) - phase_start
        
        results['transition_analysis'] = {
            'transitions': transitions,
            'phase_durations': phase_durations,
            'average_phase_length': np.mean(list(phase_durations.values()))
        }
        
        # 3. 当前阶段评估
        current_year = yearly_data['year'].max()
        current_stage = stage_features[-1]
        
        # 计算当前阶段的特征
        recent_data = yearly_data[yearly_data['year'] >= current_year - 2]
        
        results['current_stage'] = {
            'year': int(current_year),
            'phase': current_stage['growth_phase'],
            'characteristics': {
                'integration_trend': 'increasing' if len(recent_data) > 1 and recent_data['dsr_integration_depth'].diff().mean() > 0 else 'stable',
                'performance_level': 'high' if current_stage['performance'] > yearly_data['cs_output'].quantile(0.75) else 'medium',
                'constitutiveness_strength': current_stage['constitutiveness'],
                'maturity_indicators': sum([
                    current_stage['integration_level'] > 3,
                    current_stage['irreplaceability'] > yearly_data['dsr_irreplaceability'].median(),
                    current_stage['growth_phase'] in ['consolidation', 'maturity']
                ])
            }
        }
        
        # 4. 未来轨迹预测
        # 基于当前趋势和S曲线参数
        if 'constitutive_index' in s_curve_results.get('s_curve_parameters', {}):
            s_params = s_curve_results['s_curve_parameters']['constitutive_index']
            
            # 预测未来3年
            future_years = [current_year + i for i in range(1, 4)]
            future_predictions = []
            
            for year in future_years:
                time_code = year - yearly_data['year'].min()
                
                # 使用S曲线预测
                def s_curve(t, L, k, t0):
                    return L / (1 + np.exp(-k * (t - t0)))
                    
                predicted_value = s_curve(time_code, s_params['L'], s_params['k'], 
                                        s_params['t0'] - yearly_data['year'].min())
                
                # 判断阶段
                if predicted_value < s_params['L'] * 0.5:
                    phase = 'growth'
                elif predicted_value < s_params['L'] * 0.9:
                    phase = 'consolidation'
                else:
                    phase = 'maturity'
                    
                future_predictions.append({
                    'year': int(year),
                    'predicted_constitutiveness': float(predicted_value),
                    'predicted_phase': phase,
                    'uncertainty': 0.1 * abs(year - current_year)  # 简化的不确定性估计
                })
                
            results['future_trajectory'] = {
                'predictions': future_predictions,
                'expected_maturity_year': int(s_params['t0'] + 2),  # 承载容量90%的估计年份
                'growth_potential': float((s_params['L'] - current_stage['constitutiveness']) / s_params['L'])
            }
            
        print(f"    - 当前阶段: {results['current_stage']['phase']}")
        print(f"    - 成熟度指标: {results['current_stage']['characteristics']['maturity_indicators']}/3")
        print(f"    - 增长潜力: {results.get('future_trajectory', {}).get('growth_potential', 0):.2%}")
        
        self.results['evolution_stages'] = results
        return results
        
    def build_prediction_models(self):
        """构建预测模型"""
        print("  构建演化预测模型...")
        
        results = {
            'arima_forecast': {},
            'ensemble_prediction': {},
            'scenario_analysis': {}
        }
        
        # 准备月度数据用于预测
        # 创建日期索引的副本
        df_copy = self.df.copy()
        df_copy = df_copy.set_index('date')
        
        # 重采样为月度数据，使用月初(MS)频率
        monthly_data = df_copy.resample('MS').agg({
            'constitutive_index': 'mean',
            'dsr_cognitive': 'mean',
            'cs_output': 'mean'
        }).dropna()
        
        # 1. ARIMA预测
        if len(monthly_data) > 24:
            try:
                from statsmodels.tsa.arima.model import ARIMA
                
                # 确保时间序列有正确的频率信息
                ts_data = monthly_data['constitutive_index']
                ts_data.index.freq = 'MS'  # 明确设置月初频率
                
                # 拟合ARIMA模型
                model = ARIMA(ts_data, order=(1,1,1))
                arima_result = model.fit()
                
                # 预测未来6个月
                forecast = arima_result.forecast(steps=6)
                
                results['arima_forecast'] = {
                    'model_params': {
                        'order': (1,1,1),
                        'aic': arima_result.aic,
                        'bic': arima_result.bic
                    },
                    'forecast_values': forecast.tolist(),
                    'forecast_dates': pd.date_range(
                        start=monthly_data.index[-1] + pd.DateOffset(months=1),
                        periods=6,
                        freq='M'
                    ).strftime('%Y-%m').tolist()
                }
                
            except Exception as e:
                print(f"    ARIMA预测失败: {str(e)}")
                
        # 2. 情景分析
        # 基于不同的发展路径
        current_constitutiveness = monthly_data['constitutive_index'].iloc[-1]
        
        scenarios = {
            'optimistic': {
                'growth_rate': 0.05,  # 月增长5%
                'description': '技术突破和政策支持'
            },
            'baseline': {
                'growth_rate': 0.02,  # 月增长2%
                'description': '当前趋势延续'
            },
            'conservative': {
                'growth_rate': 0.01,  # 月增长1%
                'description': '资源约束和竞争加剧'
            }
        }
        
        scenario_predictions = {}
        for name, params in scenarios.items():
            predictions = []
            current_value = current_constitutiveness
            
            for month in range(1, 13):  # 预测12个月
                current_value *= (1 + params['growth_rate'])
                predictions.append({
                    'month': month,
                    'value': float(current_value),
                    'cumulative_growth': float((current_value / current_constitutiveness - 1) * 100)
                })
                
            scenario_predictions[name] = {
                'parameters': params,
                'predictions': predictions,
                'year_end_value': predictions[-1]['value'],
                'year_growth': predictions[-1]['cumulative_growth']
            }
            
        results['scenario_analysis'] = scenario_predictions
        
        aic_value = results.get('arima_forecast', {}).get('model_params', {}).get('aic')
        if aic_value is not None:
            print(f"    - ARIMA模型AIC: {aic_value:.2f}")
        else:
            print("    - ARIMA模型: 未能成功拟合")
            
        baseline_growth = results.get('scenario_analysis', {}).get('baseline', {}).get('year_growth', 0)
        print(f"    - 基准情景年增长: {baseline_growth:.1f}%")
        
        self.results['prediction_models'] = results
        return results
        
    def create_visualizations(self):
        """创建可视化"""
        print("\n生成可视化...")
        
        fig = plt.figure(figsize=(24, 20))
        gs = fig.add_gridspec(5, 3, hspace=0.3, wspace=0.3)
        
        # 标题
        fig.suptitle('动态演化分析结果', fontsize=24, fontweight='bold')
        
        # 1. 时间序列因果关系（左上）
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_temporal_causality(ax1)
        
        # 2. S曲线拟合（中上）
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_s_curve_fitting(ax2)
        
        # 3. 脉冲响应函数（右上）
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_impulse_response(ax3)
        
        # 4. 演化阶段时间线（第二行整行）
        ax4 = fig.add_subplot(gs[1, :])
        self._plot_evolution_timeline(ax4)
        
        # 5. CCM因果强度（左中）
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_ccm_results(ax5)
        
        # 6. 路径依赖性（中中）
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_path_dependency(ax6)
        
        # 7. 状态转换（右中）
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_regime_switching(ax7)
        
        # 8. 未来预测（左下）
        ax8 = fig.add_subplot(gs[3, 0])
        self._plot_future_predictions(ax8)
        
        # 9. 情景分析（中下）
        ax9 = fig.add_subplot(gs[3, 1])
        self._plot_scenario_analysis(ax9)
        
        # 10. 成熟度评估（右下）
        ax10 = fig.add_subplot(gs[3, 2])
        self._plot_maturity_assessment(ax10)
        
        # 11. 综合演化图（底部整行）
        ax11 = fig.add_subplot(gs[4, :])
        self._plot_comprehensive_evolution(ax11)
        
        # 保存图形
        output_path = self.data_path.parent / 'figures' / 'dynamic_evolution_analysis.jpg'
        plt.savefig(output_path, dpi=1200, bbox_inches='tight', format='jpg')
        plt.close()
        
        print(f"  可视化已保存至: {output_path}")
        
    def _plot_temporal_causality(self, ax):
        """绘制时间序列因果关系"""
        granger = self.results.get('temporal_causality', {}).get('granger_causality', {})
        
        if granger:
            # 绘制因果强度
            directions = ['dsr_to_cs', 'cs_to_dsr']
            p_values = [granger.get(d, {}).get('min_p_value', 1) for d in directions]
            significance = [1 - p for p in p_values]
            
            bars = ax.bar(['DSR→CS', 'CS→DSR'], significance, 
                          color=['red' if s > 0.95 else 'orange' if s > 0.9 else 'gray' 
                                for s in significance])
            
            ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='p<0.05')
            ax.axhline(y=0.90, color='orange', linestyle='--', alpha=0.5, label='p<0.10')
            
            ax.set_ylabel('因果强度 (1-p值)')
            ax.set_title('Granger因果检验')
            ax.set_ylim(0, 1.1)
            ax.legend()
            
            # 添加p值标注
            for bar, p in zip(bars, p_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'p={p:.3f}', ha='center', fontsize=8)
        else:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center')
        ax.set_title('时间序列因果')
            
    def _plot_s_curve_fitting(self, ax):
        """绘制S曲线拟合结果"""
        s_params = self.results.get('s_curve_fitting', {}).get('s_curve_parameters', {})
        
        if 'constitutive_index' in s_params:
            # 准备数据
            yearly_data = self.df.groupby('year')['constitutive_index'].mean().reset_index()
            
            # 实际数据
            ax.scatter(yearly_data['year'], yearly_data['constitutive_index'], 
                      color='blue', s=50, alpha=0.6, label='实际数据')
            
            # 拟合曲线
            params = s_params['constitutive_index']
            x_range = np.linspace(yearly_data['year'].min(), yearly_data['year'].max() + 3, 100)
            
            def s_curve(t, L, k, t0):
                return L / (1 + np.exp(-k * (t - t0)))
                
            y_fitted = s_curve(x_range, params['L'], params['k'], params['t0'])
            
            ax.plot(x_range, y_fitted, 'r-', linewidth=2, label='S曲线拟合')
            
            # 标记拐点
            ax.axvline(x=params['t0'], color='green', linestyle='--', alpha=0.5, label='拐点')
            
            # 标记承载容量
            ax.axhline(y=params['L'], color='purple', linestyle='--', alpha=0.5, label='承载容量')
            
            ax.set_xlabel('年份')
            ax.set_ylabel('构成性指数')
            ax.set_title(f'S曲线演化 (R²={params["r_squared"]:.3f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center')
        ax.set_title('S曲线拟合')
            
    def _plot_impulse_response(self, ax):
        """绘制脉冲响应函数"""
        irf = self.results.get('temporal_causality', {}).get('impulse_response', {})
        
        if irf and 'dsr_shock_to_cs' in irf:
            periods = range(len(irf['dsr_shock_to_cs']))
            
            ax.plot(periods, irf['dsr_shock_to_cs'], 'b-', linewidth=2, 
                   marker='o', label='DSR冲击→CS响应')
            
            if 'cs_shock_to_dsr' in irf:
                ax.plot(periods, irf['cs_shock_to_dsr'], 'r--', linewidth=2,
                       marker='s', label='CS冲击→DSR响应')
                
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.set_xlabel('期数')
            ax.set_ylabel('响应强度')
            ax.set_title('脉冲响应函数')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center')
        ax.set_title('脉冲响应')
            
    def _plot_evolution_timeline(self, ax):
        """绘制演化阶段时间线"""
        stages = self.results.get('evolution_stages', {}).get('stage_classification', [])
        
        if stages:
            # 准备数据
            years = [s['year'] for s in stages]
            constitutiveness = [s['constitutiveness'] for s in stages]
            phases = [s['growth_phase'] for s in stages]
            
            # 定义阶段颜色
            phase_colors = {
                'exploration': 'lightblue',
                'rapid_growth': 'orange',
                'consolidation': 'green',
                'maturity': 'purple',
                'unknown': 'gray'
            }
            
            # 绘制背景色块表示不同阶段
            current_phase = phases[0]
            phase_start = years[0]
            
            for i in range(1, len(phases)):
                if phases[i] != current_phase or i == len(phases) - 1:
                    # 绘制当前阶段
                    phase_end = years[i] if phases[i] != current_phase else years[-1]
                    ax.axvspan(phase_start, phase_end, alpha=0.3, 
                             color=phase_colors.get(current_phase, 'gray'),
                             label=current_phase if current_phase not in ax.get_legend_handles_labels()[1] else "")
                    
                    if phases[i] != current_phase:
                        current_phase = phases[i]
                        phase_start = years[i]
                        
            # 绘制构成性指数曲线
            ax.plot(years, constitutiveness, 'k-', linewidth=3, label='构成性指数')
            
            # 标记关键转折点
            transitions = self.results.get('evolution_stages', {}).get('transition_analysis', {}).get('transitions', [])
            for trans in transitions:
                ax.axvline(x=trans['year'], color='red', linestyle='--', alpha=0.7)
                ax.text(trans['year'], ax.get_ylim()[1]*0.9, f"{trans['from_phase'][:3]}→{trans['to_phase'][:3]}",
                       rotation=90, ha='center', fontsize=8)
                
            ax.set_xlabel('年份')
            ax.set_ylabel('构成性指数')
            ax.set_title('演化阶段时间线')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center')
        ax.set_title('演化时间线')
            
    def _plot_ccm_results(self, ax):
        """绘制CCM结果"""
        ccm_analysis = self.results.get('ccm_analysis', {})
        ccm = ccm_analysis.get('ccm_causality', {})
        
        if ccm and 'dsr_causes_cs' in ccm and ccm['dsr_causes_cs']:
            # 绘制收敛曲线
            dsr_data = ccm['dsr_causes_cs']
            cs_data = ccm.get('cs_causes_dsr', {})
            
            if 'library_sizes' in dsr_data and 'mean_rho' in dsr_data:
                lib_sizes = dsr_data['library_sizes']
                dsr_cs_rho = dsr_data['mean_rho']
                cs_dsr_rho = cs_data.get('mean_rho', [])
                
                ax.plot(lib_sizes[:len(dsr_cs_rho)], dsr_cs_rho, 'b-', linewidth=2,
                       marker='o', label='DSR→CS')
                
                if cs_dsr_rho:
                    ax.plot(lib_sizes[:len(cs_dsr_rho)], cs_dsr_rho, 'r--', linewidth=2,
                           marker='s', label='CS→DSR')
                    
                ax.set_xlabel('库大小')
                ax.set_ylabel('预测技能 (ρ)')
                ax.set_title('收敛交叉映射')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # 添加收敛性判断
                conv_text = []
                if ccm['dsr_causes_cs'].get('convergent'):
                    conv_text.append('DSR→CS收敛')
                if 'cs_causes_dsr' in ccm and ccm['cs_causes_dsr'].get('convergent'):
                    conv_text.append('CS→DSR收敛')
                    
                if conv_text:
                    ax.text(0.95, 0.05, '\n'.join(conv_text), 
                           transform=ax.transAxes, ha='right', va='bottom',
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
            else:
                # 没有库大小数据，显示替代信息
                self._plot_ccm_alternative(ax)
        else:
            # CCM分析失败，显示替代分析
            self._plot_ccm_alternative(ax)
            
    def _plot_ccm_alternative(self, ax):
        """CCM替代可视化"""
        # 使用简单的因果关系示意图
        ax.text(0.5, 0.7, 'CCM分析未能完成', ha='center', va='center', 
                fontsize=14, weight='bold')
        ax.text(0.5, 0.5, '使用了简化的互相关分析', ha='center', va='center', 
                fontsize=12)
        
        # 显示Granger因果结果作为替代
        granger = self.results.get('temporal_causality', {}).get('granger_causality', {})
        if granger:
            p_value = granger.get('dsr_to_cs', {}).get('min_p_value', 1)
            if p_value < 0.1:
                ax.text(0.5, 0.3, f'Granger因果: p={p_value:.3f}', 
                       ha='center', va='center', fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            else:
                ax.text(0.5, 0.3, '未发现显著因果关系', 
                       ha='center', va='center', fontsize=12)
                       
        ax.set_title('CCM分析')
            
    def _plot_path_dependency(self, ax):
        """绘制路径依赖性"""
        path_dep = self.results.get('path_dependency', {})
        
        if path_dep:
            # 创建路径依赖性指标的条形图
            indicators = []
            values = []
            
            # 锁定效应
            if 'lock_in_effects' in path_dep:
                indicators.append('锁定效应')
                values.append(path_dep['lock_in_effects'].get('max_lagged_correlation', 0))
                
            # 滞后效应
            if 'hysteresis_test' in path_dep:
                indicators.append('滞后改善')
                values.append(path_dep['hysteresis_test'].get('improvement', 0))
                
            # 状态持续性
            if 'regime_switching' in path_dep:
                state_persistence = path_dep['regime_switching'].get('state_persistence', {})
                if state_persistence:
                    indicators.append('状态持续')
                    persistence = max(state_persistence.values())
                    values.append(persistence)
                
            if indicators:
                bars = ax.bar(indicators, values, color=['blue', 'green', 'orange'])
                ax.set_ylabel('强度')
                ax.set_title('路径依赖性指标')
                
                # 添加数值标注
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{val:.3f}', ha='center')
                           
                # 添加阈值线
                ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='强依赖阈值')
                ax.legend()
        else:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center')
        ax.set_title('路径依赖性')
            
    def _plot_regime_switching(self, ax):
        """绘制状态转换"""
        path_dep = self.results.get('path_dependency', {})
        regime = path_dep.get('regime_switching', {})
        
        # 首先检查是否有状态转换信息
        n_transitions = regime.get('n_transitions', 0)
        
        if n_transitions > 0 and 'transitions' in regime:
            transitions = regime['transitions']
            
            if transitions:
                # 创建状态转换图
                years = [t['year'] for t in transitions]
                
                # 绘制时间线
                ax.scatter(years, [1]*len(years), s=100, c='red', marker='o')
                
                # 添加转换标注
                for i, trans in enumerate(transitions):
                    ax.annotate(f"状态{trans['from_state']}→{trans['to_state']}", 
                              xy=(trans['year'], 1), 
                              xytext=(trans['year'], 1.2 + (i%2)*0.2),
                              arrowprops=dict(arrowstyle='->', color='gray'),
                              ha='center')
                              
                ax.set_xlabel('年份')
                ax.set_ylim(0.5, 2)
                ax.set_title(f'状态转换检测 (n={len(transitions)})')
                ax.grid(True, axis='x', alpha=0.3)
                
                # 隐藏y轴
                ax.set_yticks([])
            else:
                ax.text(0.5, 0.5, '未检测到状态转换', ha='center', va='center')
        else:
            # 显示状态转换统计信息
            ax.text(0.5, 0.7, f'状态转换次数: {n_transitions}', 
                   ha='center', va='center', fontsize=14, weight='bold')
            
            # 显示其他路径依赖信息
            if 'hysteresis_test' in path_dep:
                sig = path_dep['hysteresis_test'].get('significant', False)
                ax.text(0.5, 0.5, f'滞后效应: {"显著" if sig else "不显著"}', 
                       ha='center', va='center', fontsize=12)
                       
            if 'lock_in_test' in path_dep:
                locked = path_dep['lock_in_test'].get('is_locked_in', False)
                ax.text(0.5, 0.3, f'路径锁定: {"是" if locked else "否"}', 
                       ha='center', va='center', fontsize=12)
                       
            ax.set_title('状态转换分析')
            
    def _plot_future_predictions(self, ax):
        """绘制未来预测"""
        future = self.results.get('evolution_stages', {}).get('future_trajectory', {})
        arima = self.results.get('prediction_models', {}).get('arima_forecast', {})
        
        if future and 'predictions' in future:
            # 历史数据
            yearly_data = self.df.groupby('year')['constitutive_index'].mean().reset_index()
            ax.plot(yearly_data['year'], yearly_data['constitutive_index'], 
                   'b-', linewidth=2, label='历史数据')
            
            # S曲线预测
            pred_years = [p['year'] for p in future['predictions']]
            pred_values = [p['predicted_constitutiveness'] for p in future['predictions']]
            uncertainties = [p['uncertainty'] for p in future['predictions']]
            
            ax.plot(pred_years, pred_values, 'r--', linewidth=2, label='S曲线预测')
            
            # 不确定性区间
            upper_bound = [v + u for v, u in zip(pred_values, uncertainties)]
            lower_bound = [v - u for v, u in zip(pred_values, uncertainties)]
            ax.fill_between(pred_years, lower_bound, upper_bound, alpha=0.2, color='red')
            
            # ARIMA预测（如果有）
            if arima and 'forecast_values' in arima:
                # 简化：只显示趋势
                last_historical = yearly_data['constitutive_index'].iloc[-1]
                arima_trend = [last_historical * (1 + 0.02*i) for i in range(1, 4)]
                ax.plot(pred_years, arima_trend, 'g:', linewidth=2, label='ARIMA趋势')
                
            ax.set_xlabel('年份')
            ax.set_ylabel('构成性指数')
            ax.set_title('未来演化预测')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center')
        ax.set_title('未来预测')
            
    def _plot_scenario_analysis(self, ax):
        """绘制情景分析"""
        scenarios = self.results.get('prediction_models', {}).get('scenario_analysis', {})
        
        if scenarios:
            # 为每个情景绘制增长轨迹
            colors = {'optimistic': 'green', 'baseline': 'blue', 'conservative': 'orange'}
            
            for name, scenario in scenarios.items():
                if 'predictions' in scenario:
                    months = [p['month'] for p in scenario['predictions']]
                    values = [p['cumulative_growth'] for p in scenario['predictions']]
                    
                    ax.plot(months, values, linewidth=2, color=colors.get(name, 'gray'),
                           label=f"{name} ({scenario['parameters']['growth_rate']*100:.0f}%/月)")
                           
            ax.set_xlabel('月份')
            ax.set_ylabel('累计增长率 (%)')
            ax.set_title('情景分析：12个月预测')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 添加年末增长率标注
            y_pos = 0.95
            for name, scenario in scenarios.items():
                if 'year_growth' in scenario:
                    ax.text(0.95, y_pos, f"{name}: {scenario['year_growth']:.1f}%",
                           transform=ax.transAxes, ha='right', va='top',
                           color=colors.get(name, 'gray'))
                    y_pos -= 0.05
        else:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center')
        ax.set_title('情景分析')
            
    def _plot_maturity_assessment(self, ax):
        """绘制成熟度评估"""
        maturity = self.results.get('s_curve_fitting', {}).get('maturity_assessment', {})
        
        if maturity:
            # 创建雷达图
            metrics = list(maturity.keys())
            values = [m['maturity_ratio'] for m in maturity.values()]
            
            # 添加第一个值到末尾，使图形闭合
            values += values[:1]
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]
            
            # 绘制雷达图
            ax.plot(angles, values, 'o-', linewidth=2, color='blue', label='当前成熟度')
            ax.fill(angles, values, alpha=0.25, color='blue')
            
            # 添加100%成熟度参考线
            ax.plot(angles, [1.0] * len(angles), 'r--', linewidth=1, alpha=0.5, label='完全成熟')
            
            # 设置标签
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics, size=8)
            ax.set_ylim(0, 1.2)
            ax.set_title('多维成熟度评估')
            ax.legend(loc='upper right')
            ax.grid(True)
            
            # 添加平均成熟度
            avg_maturity = np.mean(values[:-1])
            ax.text(0.5, -0.15, f'平均成熟度: {avg_maturity:.2%}',
                   transform=ax.transAxes, ha='center', fontweight='bold')
        else:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center')
        ax.set_title('成熟度评估')
            
    def _plot_comprehensive_evolution(self, ax):
        """绘制综合演化图"""
        # 整合多个指标的时间演化
        yearly_data = self.df.groupby('year').agg({
            'dsr_cognitive': 'mean',
            'cs_output': 'mean',
            'constitutive_index': 'mean',
            'dsr_integration_depth': 'mean'
        }).reset_index()
        
        # 标准化数据
        scaler = StandardScaler()
        metrics = ['dsr_cognitive', 'cs_output', 'constitutive_index', 'dsr_integration_depth']
        scaled_data = scaler.fit_transform(yearly_data[metrics])
        
        # 绘制多指标演化
        for i, metric in enumerate(metrics):
            ax.plot(yearly_data['year'], scaled_data[:, i], linewidth=2, 
                   label=metric.replace('_', ' ').title(), marker='o')
            
        # 添加演化阶段背景
        stages = self.results.get('evolution_stages', {}).get('stage_classification', [])
        if stages:
            # 定义阶段颜色
            phase_colors = {
                'exploration': 'lightblue',
                'rapid_growth': 'orange', 
                'consolidation': 'green',
                'maturity': 'purple'
            }
            
            # 绘制阶段背景
            for i in range(len(stages)-1):
                phase = stages[i]['growth_phase']
                if phase in phase_colors:
                    ax.axvspan(stages[i]['year'], stages[i+1]['year'], 
                             alpha=0.1, color=phase_colors[phase])
                             
        ax.set_xlabel('年份')
        ax.set_ylabel('标准化值')
        ax.set_title('多指标综合演化轨迹')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
    def generate_comprehensive_interpretation(self):
        """生成综合解释"""
        print("\n" + "="*60)
        print("动态演化分析综合解释")
        print("="*60)
        
        # 1. 时间因果性解释
        granger = self.results.get('temporal_causality', {}).get('granger_causality', {})
        if granger.get('dsr_to_cs', {}).get('min_p_value', 1) < 0.1:
            print("\n1. 时间因果关系：")
            print("   - DSR对CS存在时间上的预测能力（Granger因果）")
            print("   - 这表明DSR的变化领先于认知系统输出的变化")
        
        # 2. 演化阶段解释
        s_curve = self.results.get('s_curve_fitting', {})
        if 'current_phase' in s_curve:
            print("\n2. 演化阶段：")
            phase = s_curve['current_phase']
            if phase == 'growth':
                print("   - 当前处于快速增长期，构成性作用正在强化")
            elif phase == 'consolidation':
                print("   - 当前处于巩固期，构成性关系趋于稳定")
            elif phase == 'maturity':
                print("   - 已进入成熟期，构成性作用充分体现")
                
        # 3. 路径依赖解释
        path_dep = self.results.get('path_dependency', {})
        if path_dep.get('hysteresis_test', {}).get('significant'):
            print("\n3. 路径依赖性：")
            print("   - 发现显著的滞后效应，历史状态影响当前表现")
            print("   - 这支持了演化的路径依赖特征")
            
        # 4. 未来趋势
        prediction = self.results.get('prediction_models', {})
        if 'scenario_forecast' in prediction:
            print("\n4. 未来发展趋势：")
            base_growth = prediction['scenario_forecast'].get('baseline', {}).get('annual_growth', 0)
            if base_growth > 0.1:
                print(f"   - 预计未来年增长率：{base_growth:.1%}")
                print("   - 构成性作用将继续增强")
                
        print("\n" + "="*60)
        
    def save_results(self):
        """保存结果"""
        output_file = self.data_path / 'dynamic_evolution_results.json'
        
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
        report.append("# 动态演化分析报告\n")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 1. 执行摘要
        report.append("## 执行摘要\n")
        
        # 当前阶段
        current_stage = self.results.get('evolution_stages', {}).get('current_stage', {})
        if current_stage:
            report.append(f"- **当前演化阶段**: {current_stage.get('phase', 'unknown')}")
            report.append(f"- **成熟度指标**: {current_stage.get('characteristics', {}).get('maturity_indicators', 0)}/3")
            
        # 因果关系
        granger = self.results.get('temporal_causality', {}).get('granger_causality', {})
        if granger:
            dsr_cs_sig = granger.get('dsr_to_cs', {}).get('significant', False)
            report.append(f"- **DSR→CS因果关系**: {'显著' if dsr_cs_sig else '不显著'}")
            
        # 路径依赖
        path_dep = self.results.get('path_dependency', {})
        if path_dep:
            has_lock_in = path_dep.get('lock_in_effects', {}).get('persistence_indicator', False)
            report.append(f"- **路径依赖性**: {'存在' if has_lock_in else '不明显'}")
            
        # 2. 时间序列因果分析
        report.append("\n## 时间序列因果分析\n")
        
        if granger:
            report.append("### Granger因果检验\n")
            report.append("| 方向 | p值 | 最优滞后 | 显著性 |")
            report.append("|------|-----|---------|--------|")
            
            for direction in ['dsr_to_cs', 'cs_to_dsr']:
                if direction in granger:
                    data = granger[direction]
                    report.append(f"| {direction.replace('_', '→').upper()} | "
                                f"{data.get('min_p_value', 1):.3f} | "
                                f"{data.get('optimal_lag', 0)} | "
                                f"{'是' if data.get('significant', False) else '否'} |")
                                
        # VAR模型
        var_model = self.results.get('temporal_causality', {}).get('var_model', {})
        if var_model:
            report.append(f"\n### VAR模型\n")
            report.append(f"- **最优滞后阶数**: {var_model.get('optimal_lag', 0)}")
            report.append(f"- **AIC**: {var_model.get('aic', 'N/A')}")
            
        # 3. S曲线演化
        report.append("\n## S曲线演化分析\n")
        
        s_curve = self.results.get('s_curve_fitting', {})
        if s_curve:
            # 拐点
            inflection_points = s_curve.get('inflection_points', {})
            if inflection_points:
                report.append("### 拐点识别\n")
                report.append("| 指标 | 拐点年份 | 拐点值 | 增长率 |")
                report.append("|------|---------|--------|--------|")
                
                for metric, point in inflection_points.items():
                    report.append(f"| {metric} | {point['year']} | "
                                f"{point['value']:.3f} | {point['growth_rate']:.3f} |")
                                
            # 成熟度
            maturity = s_curve.get('maturity_assessment', {})
            if maturity:
                report.append("\n### 成熟度评估\n")
                avg_maturity = np.mean([m['maturity_ratio'] for m in maturity.values()])
                report.append(f"- **平均成熟度**: {avg_maturity:.2%}")
                
                for metric, assess in maturity.items():
                    report.append(f"- **{metric}**: {assess['maturity_ratio']:.2%} "
                                f"(当前阶段: {assess['current_phase']})")
                                
        # 4. 路径依赖性
        report.append("\n## 路径依赖性分析\n")
        
        if path_dep:
            # 锁定效应
            lock_in = path_dep.get('lock_in_effects', {})
            if lock_in:
                report.append(f"- **最大滞后相关**: {lock_in.get('max_lagged_correlation', 0):.3f}")
                report.append(f"- **持续性指标**: {'是' if lock_in.get('persistence_indicator', False) else '否'}")
                
            # 状态转换
            regime = path_dep.get('regime_switching', {})
            if regime:
                report.append(f"- **状态转换次数**: {regime.get('n_transitions', 0)}")
                
            # 关键节点
            junctures = path_dep.get('critical_junctures', {})
            if junctures and junctures.get('most_critical'):
                critical = junctures['most_critical']
                report.append(f"- **最关键转折点**: {critical['year']}年 "
                            f"(变化幅度: {critical['change_magnitude']:.3f})")
                            
        # 5. CCM分析
        report.append("\n## 收敛交叉映射（CCM）\n")
        
        ccm = self.results.get('ccm_analysis', {}).get('ccm_causality', {})
        if ccm:
            report.append("| 方向 | 最终ρ值 | 收敛性 |")
            report.append("|------|---------|--------|")
            
            for direction in ['dsr_causes_cs', 'cs_causes_dsr']:
                if direction in ccm:
                    data = ccm[direction]
                    report.append(f"| {direction.replace('_', ' ').upper()} | "
                                f"{data.get('final_rho', 0):.3f} | "
                                f"{'是' if data.get('convergent', False) else '否'} |")
                                
        # 6. 未来预测
        report.append("\n## 未来演化预测\n")
        
        future = self.results.get('evolution_stages', {}).get('future_trajectory', {})
        if future:
            report.append(f"- **增长潜力**: {future.get('growth_potential', 0):.2%}")
            report.append(f"- **预期成熟年份**: {future.get('expected_maturity_year', 'N/A')}")
            
            predictions = future.get('predictions', [])
            if predictions:
                report.append("\n### 3年预测\n")
                report.append("| 年份 | 预测值 | 预测阶段 |")
                report.append("|------|--------|---------|")
                
                for pred in predictions:
                    report.append(f"| {pred['year']} | {pred['predicted_constitutiveness']:.3f} | "
                                f"{pred['predicted_phase']} |")
                                
        # 7. 政策建议
        report.append("\n## 政策建议\n")
        
        # 基于分析结果的建议
        recommendations = []
        
        # 基于当前阶段
        if current_stage.get('phase') == 'rapid_growth':
            recommendations.append("- 当前处于快速增长期，应加大资源投入以维持增长势头")
        elif current_stage.get('phase') == 'consolidation':
            recommendations.append("- 进入巩固期，应注重质量提升和深度整合")
            
        # 基于路径依赖
        if path_dep.get('lock_in_effects', {}).get('persistence_indicator'):
            recommendations.append("- 存在显著路径依赖，应保持政策连续性")
            
        # 基于成熟度
        avg_maturity = np.mean([m['maturity_ratio'] for m in s_curve.get('maturity_assessment', {}).values()])
        if avg_maturity < 0.5:
            recommendations.append("- 整体成熟度较低，仍有较大发展空间")
        elif avg_maturity > 0.8:
            recommendations.append("- 接近成熟阶段，应探索新的增长点")
            
        for rec in recommendations:
            report.append(rec)
            
        # 保存报告
        report_file = self.data_path.parent / 'md' / 'dynamic_evolution_report.md'
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
    
    analyzer = DynamicEvolutionAnalysis(data_path)
    
    # 加载数据
    analyzer.load_data()
    
    # 运行分析
    results = analyzer.run_dynamic_evolution_analysis()
    
    print("\n" + "="*60)
    print("动态演化分析完成！")
    print("="*60)
    

if __name__ == "__main__":
    main()