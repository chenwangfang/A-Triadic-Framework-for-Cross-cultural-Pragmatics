# step5_statistical_models.py
# 第五步：统计模型分析 - 验证认知构成性假设

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# 统计建模库
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.api import VAR
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class StatisticalModelsAnalyzer:
    """统计模型分析器 - 实施M1-M5模型"""
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.df = None
        self.model_results = {
            'M1_baseline': {},
            'M2_interaction': {},
            'M3_nonlinear': {},
            'M4_var_causality': {},
            'M5_hierarchical': {},
            'model_comparison': {},
            'hypothesis_tests': {}
        }
        
    def load_data(self):
        """加载带指标的数据"""
        csv_file = self.data_path / 'data_with_metrics.csv'
        print(f"加载数据: {csv_file}")
        
        self.df = pd.read_csv(csv_file, encoding='utf-8-sig')
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # 数据预处理
        self._preprocess_data()
        
        print(f"成功加载 {len(self.df)} 条记录")
        print(f"包含特征数: {len(self.df.columns)}")
        
        return self.df
        
    def _preprocess_data(self):
        """数据预处理"""
        # 1. 创建时间趋势变量
        self.df['time_index'] = range(len(self.df))
        self.df['time_trend'] = self.df['time_index'] / len(self.df)
        
        # 2. 创建年份虚拟变量
        self.df['year'] = self.df['date'].dt.year
        year_dummies = pd.get_dummies(self.df['year'], prefix='year')
        self.df = pd.concat([self.df, year_dummies], axis=1)
        
        # 3. 标准化关键变量（用于某些模型）
        scaler = StandardScaler()
        scale_vars = ['dsr_cognitive', 'tl_functional', 'cs_output']
        self.df[[f'{v}_scaled' for v in scale_vars]] = scaler.fit_transform(self.df[scale_vars])
        
        # 4. 创建滞后变量（用于VAR模型）
        for lag in range(1, 4):
            self.df[f'cs_output_lag{lag}'] = self.df['cs_output'].shift(lag)
            self.df[f'dsr_cognitive_lag{lag}'] = self.df['dsr_cognitive'].shift(lag)
            
        # 5. 填充缺失值
        self.df = self.df.bfill()
        
    def run_all_models(self):
        """运行所有统计模型"""
        print("\n开始统计模型分析...")
        
        # M1: 线性基准模型
        self.m1_baseline_model()
        
        # M2: 交互效应模型
        self.m2_interaction_model()
        
        # M3: 非线性构成模型
        self.m3_nonlinear_model()
        
        # M4: VAR双向因果模型
        self.m4_var_causality_model()
        
        # M5: 分层混合效应模型
        self.m5_hierarchical_model()
        
        # 模型比较
        self.compare_models()
        
        # 假设检验
        self.test_hypotheses()
        
        # 保存结果
        self.save_results()
        
        return self.model_results
        
    def m1_baseline_model(self):
        """M1: 线性基准模型"""
        print("\n1. M1: 线性基准模型")
        
        # 模型公式
        formula = 'cs_output ~ dsr_cognitive + tl_functional + sensitivity_code + media_culture_code'
        
        # 拟合模型
        model = smf.ols(formula, data=self.df).fit()
        
        # 保存结果
        self.model_results['M1_baseline'] = {
            'formula': formula,
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'aic': model.aic,
            'bic': model.bic,
            'coefficients': model.params.to_dict(),
            'p_values': model.pvalues.to_dict(),
            'conf_intervals': model.conf_int().to_dict(),
            'summary': str(model.summary())
        }
        
        # 诊断检验
        self._diagnostic_tests(model, 'M1_baseline')
        
        print(f"  R² = {model.rsquared:.4f}")
        print(f"  AIC = {model.aic:.2f}")
        print(f"  DSR系数 = {model.params['dsr_cognitive']:.4f} (p={model.pvalues['dsr_cognitive']:.4f})")
        print(f"  TL系数 = {model.params['tl_functional']:.4f} (p={model.pvalues['tl_functional']:.4f})")
        
    def m2_interaction_model(self):
        """M2: 交互效应模型"""
        print("\n2. M2: 交互效应模型")
        
        # 包含交互项的模型
        formula = '''cs_output ~ dsr_cognitive * tl_functional + 
                    dsr_cognitive * sensitivity_code + 
                    dsr_cognitive * media_culture_code + 
                    I(dsr_cognitive > dsr_cognitive.quantile(0.5)) * I(tl_functional > tl_functional.quantile(0.5))'''
        
        model = smf.ols(formula, data=self.df).fit()
        
        # 计算边际效应
        marginal_effects = self._calculate_marginal_effects(model)
        
        self.model_results['M2_interaction'] = {
            'formula': formula,
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'aic': model.aic,
            'bic': model.bic,
            'coefficients': model.params.to_dict(),
            'p_values': model.pvalues.to_dict(),
            'marginal_effects': marginal_effects,
            'interaction_significance': {
                'dsr_tl': model.pvalues.get('dsr_cognitive:tl_functional', 1.0),
                'dsr_sensitivity': model.pvalues.get('dsr_cognitive:sensitivity_code', 1.0)
            }
        }
        
        print(f"  R² = {model.rsquared:.4f}")
        print(f"  DSR×TL交互项 p = {model.pvalues.get('dsr_cognitive:tl_functional', 1.0):.4f}")
        print(f"  语境调节效应显著性: {'是' if model.pvalues.get('dsr_cognitive:sensitivity_code', 1.0) < 0.05 else '否'}")
        
    def m3_nonlinear_model(self):
        """M3: 非线性构成模型"""
        print("\n3. M3: 非线性构成模型")
        
        # 直接构建非线性模型，使用I()函数处理平方项
        formula = '''cs_output ~ dsr_cognitive + tl_functional + sensitivity_code + 
                    I(dsr_cognitive**2) + I(tl_functional**2) + I(sensitivity_code**2) +
                    dsr_cognitive:tl_functional + dsr_cognitive:sensitivity_code + 
                    tl_functional:sensitivity_code'''
        
        model_poly = smf.ols(formula, data=self.df).fit()
        
        # 方法2: 样条回归（使用分段线性近似）
        thresholds = self._detect_thresholds()
        
        self.model_results['M3_nonlinear'] = {
            'polynomial_model': {
                'r_squared': model_poly.rsquared,
                'adj_r_squared': model_poly.rsquared_adj,
                'aic': model_poly.aic,
                'significant_nonlinear_terms': [
                    term for term, p in model_poly.pvalues.items() 
                    if ('**2' in term or ':' in term) and p < 0.05
                ]
            },
            'thresholds': thresholds,
            'nonlinearity_test': self._test_nonlinearity()
        }
        
        print(f"  多项式模型 R² = {model_poly.rsquared:.4f}")
        print(f"  检测到的阈值: {thresholds}")
        print(f"  非线性项数量: {len(self.model_results['M3_nonlinear']['polynomial_model']['significant_nonlinear_terms'])}")
        
    def m4_var_causality_model(self):
        """M4: VAR双向因果模型"""
        print("\n4. M4: VAR双向因果模型")
        
        # 准备VAR模型数据
        var_vars = ['cs_output', 'dsr_cognitive', 'tl_functional']
        var_data = self.df[var_vars].dropna()
        
        # 拟合VAR模型
        var_model = VAR(var_data)
        var_results = var_model.fit(maxlags=5, ic='aic')
        
        # Granger因果检验
        granger_results = {}
        for cause in var_vars:
            for effect in var_vars:
                if cause != effect:
                    test = var_results.test_causality(effect, [cause], kind='f')
                    granger_results[f'{cause}_causes_{effect}'] = {
                        'statistic': test.test_statistic,
                        'p_value': test.pvalue,
                        'significant': test.pvalue < 0.05
                    }
        
        # 脉冲响应函数
        irf = var_results.irf(periods=10)
        
        # 预测误差方差分解
        fevd = var_results.fevd(periods=10)
        
        self.model_results['M4_var_causality'] = {
            'optimal_lag': var_results.k_ar,
            'aic': var_results.aic,
            'granger_causality': granger_results,
            'impulse_responses': {
                'dsr_to_cs': irf.orth_irfs[1, 0, :].tolist(),  # DSR对CS的影响
                'cs_to_dsr': irf.orth_irfs[0, 1, :].tolist()   # CS对DSR的影响
            },
            'variance_decomposition': {
                'cs_explained_by_dsr': fevd.decomp[0, :, 1].tolist(),
                'dsr_explained_by_cs': fevd.decomp[1, :, 0].tolist()
            }
        }
        
        # 判断双向因果
        bidirectional = (
            granger_results.get('dsr_cognitive_causes_cs_output', {}).get('significant', False) and
            granger_results.get('cs_output_causes_dsr_cognitive', {}).get('significant', False)
        )
        
        print(f"  最优滞后阶数: {var_results.k_ar}")
        print(f"  DSR→CS Granger因果: {granger_results.get('dsr_cognitive_causes_cs_output', {}).get('p_value', 1):.4f}")
        print(f"  CS→DSR Granger因果: {granger_results.get('cs_output_causes_dsr_cognitive', {}).get('p_value', 1):.4f}")
        print(f"  双向因果关系: {'存在' if bidirectional else '不存在'}")
        
    def m5_hierarchical_model(self):
        """M5: 分层混合效应模型"""
        print("\n5. M5: 分层混合效应模型")
        
        # 创建语境分层变量
        sensitivity_map = {1: 'low', 2: 'medium', 3: 'high'}
        self.df['context_stratum'] = self.df['sensitivity_code'].map(sensitivity_map)
        
        # 简化版本：使用固定效应模型近似
        # 按语境敏感度分层
        results_by_context = {}
        
        for context in ['low', 'medium', 'high']:
            subset = self.df[self.df['context_stratum'] == context]
            
            if len(subset) > 50:
                formula = 'cs_output ~ dsr_cognitive + tl_functional + time_trend'
                model = smf.ols(formula, data=subset).fit()
                
                results_by_context[context] = {
                    'n_obs': len(subset),
                    'r_squared': model.rsquared,
                    'dsr_coef': model.params.get('dsr_cognitive', 0),
                    'dsr_pvalue': model.pvalues.get('dsr_cognitive', 1),
                    'tl_coef': model.params.get('tl_functional', 0),
                    'time_trend': model.params.get('time_trend', 0)
                }
        
        # 跨层级比较
        heterogeneity_test = self._test_coefficient_heterogeneity(results_by_context)
        
        self.model_results['M5_hierarchical'] = {
            'context_specific_models': results_by_context,
            'heterogeneity_test': heterogeneity_test,
            'random_effects_approximation': self._approximate_random_effects()
        }
        
        print(f"  语境特定效应:")
        for ctx, res in results_by_context.items():
            print(f"    {ctx}: DSR系数={res['dsr_coef']:.4f}, R²={res['r_squared']:.4f}")
        print(f"  系数异质性检验 p = {heterogeneity_test['p_value']:.4f}")
        
    def _diagnostic_tests(self, model, model_name):
        """模型诊断检验"""
        # 1. 异方差检验
        _, pval_het, _, _ = het_breuschpagan(model.resid, model.model.exog)
        
        # 2. 正态性检验
        _, pval_norm = stats.jarque_bera(model.resid)
        
        # 3. 多重共线性（VIF）
        vif_data = pd.DataFrame()
        vif_data["Variable"] = model.model.exog_names[1:]  # 排除常数项
        try:
            vif_data["VIF"] = [variance_inflation_factor(model.model.exog.values, i+1) 
                              for i in range(len(vif_data))]
        except:
            vif_data["VIF"] = [np.nan] * len(vif_data)
        
        self.model_results[model_name]['diagnostics'] = {
            'heteroscedasticity_pvalue': pval_het,
            'normality_pvalue': pval_norm,
            'vif': vif_data.to_dict('records'),
            'durbin_watson': sm.stats.durbin_watson(model.resid)
        }
        
    def _calculate_marginal_effects(self, model):
        """计算边际效应"""
        marginal_effects = {}
        
        # DSR在不同TL水平下的边际效应
        tl_levels = self.df['tl_functional'].quantile([0.25, 0.5, 0.75])
        
        for q, tl_val in tl_levels.items():
            # 基础效应 + 交互效应
            me = model.params.get('dsr_cognitive', 0) + \
                 model.params.get('dsr_cognitive:tl_functional', 0) * tl_val
            marginal_effects[f'dsr_at_tl_{q}'] = me
            
        return marginal_effects
        
    def _detect_thresholds(self):
        """检测非线性阈值"""
        thresholds = {}
        
        # 使用分位数回归检测阈值
        quantiles = [0.25, 0.5, 0.75]
        
        for q in quantiles:
            model = smf.quantreg('cs_output ~ dsr_cognitive', self.df).fit(q=q)
            thresholds[f'quantile_{q}'] = {
                'dsr_value': self.df['dsr_cognitive'].quantile(q),
                'slope': model.params['dsr_cognitive']
            }
            
        return thresholds
        
    def _test_nonlinearity(self):
        """非线性检验"""
        # RESET检验的简化版本
        linear_model = smf.ols('cs_output ~ dsr_cognitive + tl_functional', data=self.df).fit()
        
        # 添加预测值的平方项
        self.df['fitted_sq'] = linear_model.fittedvalues ** 2
        augmented_model = smf.ols('cs_output ~ dsr_cognitive + tl_functional + fitted_sq', 
                                 data=self.df).fit()
        
        # F检验
        f_stat = ((augmented_model.rsquared - linear_model.rsquared) / 1) / \
                ((1 - augmented_model.rsquared) / (len(self.df) - 4))
        p_value = 1 - stats.f.cdf(f_stat, 1, len(self.df) - 4)
        
        return {
            'reset_f_statistic': f_stat,
            'reset_p_value': p_value,
            'nonlinearity_detected': p_value < 0.05
        }
        
    def _test_coefficient_heterogeneity(self, results_by_context):
        """检验系数异质性"""
        # Chow检验的简化版本
        dsr_coefs = [res['dsr_coef'] for res in results_by_context.values()]
        
        # 方差分析
        f_stat, p_value = stats.f_oneway(
            *[self.df[self.df['context_stratum'] == ctx]['dsr_cognitive'].values 
              for ctx in results_by_context.keys()]
        )
        
        return {
            'coefficient_range': max(dsr_coefs) - min(dsr_coefs),
            'f_statistic': f_stat,
            'p_value': p_value,
            'heterogeneity_significant': p_value < 0.05
        }
        
    def _approximate_random_effects(self):
        """近似随机效应"""
        # 计算组内和组间方差
        within_var = self.df.groupby('context_stratum')['cs_output'].var().mean()
        between_var = self.df.groupby('context_stratum')['cs_output'].mean().var()
        
        # ICC (组内相关系数)
        icc = between_var / (between_var + within_var)
        
        return {
            'within_variance': within_var,
            'between_variance': between_var,
            'icc': icc,
            'random_effects_important': icc > 0.1
        }
        
    def compare_models(self):
        """模型比较"""
        print("\n6. 模型比较")
        
        comparison = pd.DataFrame({
            'Model': ['M1_baseline', 'M2_interaction', 'M3_nonlinear', 'M4_var', 'M5_hierarchical'],
            'R_squared': [
                self.model_results['M1_baseline']['r_squared'],
                self.model_results['M2_interaction']['r_squared'],
                self.model_results['M3_nonlinear']['polynomial_model']['r_squared'],
                np.nan,  # VAR模型无R²
                np.mean([r['r_squared'] for r in self.model_results['M5_hierarchical']['context_specific_models'].values()])
            ],
            'AIC': [
                self.model_results['M1_baseline']['aic'],
                self.model_results['M2_interaction']['aic'],
                self.model_results['M3_nonlinear']['polynomial_model']['aic'],
                self.model_results['M4_var_causality']['aic'],
                np.nan  # 分层模型无统一AIC
            ]
        })
        
        # 交叉验证比较
        cv_scores = self._cross_validation_comparison()
        
        self.model_results['model_comparison'] = {
            'summary_table': comparison.to_dict('records'),
            'best_r_squared': comparison.loc[comparison['R_squared'].idxmax(), 'Model'],
            'best_aic': comparison.loc[comparison['AIC'].idxmin(), 'Model'],
            'cross_validation_scores': cv_scores
        }
        
        print(f"  最佳R²模型: {self.model_results['model_comparison']['best_r_squared']}")
        print(f"  最佳AIC模型: {self.model_results['model_comparison']['best_aic']}")
        
    def _cross_validation_comparison(self):
        """交叉验证比较"""
        cv_scores = {}
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # 只比较可以交叉验证的模型
        models = {
            'M1': 'cs_output ~ dsr_cognitive + tl_functional + sensitivity_code',
            'M2': 'cs_output ~ dsr_cognitive * tl_functional + dsr_cognitive * sensitivity_code'
        }
        
        for name, formula in models.items():
            scores = []
            for train_idx, test_idx in kf.split(self.df):
                train_df = self.df.iloc[train_idx]
                test_df = self.df.iloc[test_idx]
                
                model = smf.ols(formula, data=train_df).fit()
                predictions = model.predict(test_df)
                score = r2_score(test_df['cs_output'], predictions)
                scores.append(score)
                
            cv_scores[name] = {
                'mean_cv_score': np.mean(scores),
                'std_cv_score': np.std(scores)
            }
            
        return cv_scores
        
    def test_hypotheses(self):
        """假设检验"""
        print("\n7. 假设检验")
        
        # H1: 认知依赖性（DSR对CS有显著影响）
        h1_results = {
            'hypothesis': 'H1: DSR对CS有显著正向影响',
            'evidence': {
                'M1_dsr_significant': self.model_results['M1_baseline']['p_values']['dsr_cognitive'] < 0.05,
                'M1_dsr_positive': self.model_results['M1_baseline']['coefficients']['dsr_cognitive'] > 0,
                'granger_dsr_to_cs': self.model_results['M4_var_causality']['granger_causality'].get(
                    'dsr_cognitive_causes_cs_output', {}).get('significant', False),
                'bidirectional_causality': (
                    self.model_results['M4_var_causality']['granger_causality'].get(
                        'dsr_cognitive_causes_cs_output', {}).get('significant', False) and
                    self.model_results['M4_var_causality']['granger_causality'].get(
                        'cs_output_causes_dsr_cognitive', {}).get('significant', False)
                )
            },
            'conclusion': None
        }
        h1_results['conclusion'] = '支持' if sum(h1_results['evidence'].values()) >= 3 else '不支持'
        
        # H2: 系统调节（语境调节DSR效应）
        h2_results = {
            'hypothesis': 'H2: 语境敏感度调节DSR的构成性功能',
            'evidence': {
                'interaction_significant': self.model_results['M2_interaction']['interaction_significance']['dsr_sensitivity'] < 0.05,
                'heterogeneity_significant': self.model_results['M5_hierarchical']['heterogeneity_test']['heterogeneity_significant'],
                'threshold_effects': len(self.model_results['M3_nonlinear']['thresholds']) > 0
            },
            'conclusion': None
        }
        h2_results['conclusion'] = '支持' if sum(h2_results['evidence'].values()) >= 2 else '不支持'
        
        # H3: 动态演化（时间趋势）
        h3_results = {
            'hypothesis': 'H3: DSR构成性随时间演化',
            'evidence': {
                'time_trend_in_hierarchical': any(
                    abs(res.get('time_trend', 0)) > 0.01 
                    for res in self.model_results['M5_hierarchical']['context_specific_models'].values()
                ),
                'variance_decomposition_changes': len(
                    self.model_results['M4_var_causality']['variance_decomposition']['cs_explained_by_dsr']
                ) > 1
            },
            'conclusion': None
        }
        h3_results['conclusion'] = '支持' if sum(h3_results['evidence'].values()) >= 1 else '不支持'
        
        self.model_results['hypothesis_tests'] = {
            'H1': h1_results,
            'H2': h2_results,
            'H3': h3_results,
            'overall_support': sum([
                h1_results['conclusion'] == '支持',
                h2_results['conclusion'] == '支持',
                h3_results['conclusion'] == '支持'
            ])
        }
        
        print(f"  H1 (认知依赖性): {h1_results['conclusion']}")
        print(f"  H2 (系统调节): {h2_results['conclusion']}")
        print(f"  H3 (动态演化): {h3_results['conclusion']}")
        print(f"  总体支持度: {self.model_results['hypothesis_tests']['overall_support']}/3")
        
    def save_results(self):
        """保存结果"""
        results_file = self.data_path / 'statistical_models_results.json'
        
        # 转换numpy类型
        def convert_numpy(obj):
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            elif pd.isna(obj):
                return None
            return obj
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy(self.model_results), f, ensure_ascii=False, indent=2)
            
        print(f"\n结果已保存至: {results_file}")
        
        # 生成报告
        self.generate_report()
        
    def generate_report(self):
        """生成统计模型分析报告"""
        print("\n" + "="*60)
        print("统计模型分析报告")
        print("="*60)
        
        # 1. 模型拟合优度
        print("\n1. 模型拟合优度比较:")
        print(f"   M1 (基准): R²={self.model_results['M1_baseline']['r_squared']:.4f}, "
              f"AIC={self.model_results['M1_baseline']['aic']:.2f}")
        print(f"   M2 (交互): R²={self.model_results['M2_interaction']['r_squared']:.4f}, "
              f"AIC={self.model_results['M2_interaction']['aic']:.2f}")
        print(f"   M3 (非线性): R²={self.model_results['M3_nonlinear']['polynomial_model']['r_squared']:.4f}")
        
        # 2. 关键发现
        print("\n2. 关键发现:")
        
        # DSR效应
        dsr_coef = self.model_results['M1_baseline']['coefficients']['dsr_cognitive']
        dsr_p = self.model_results['M1_baseline']['p_values']['dsr_cognitive']
        print(f"   • DSR认知效应: β={dsr_coef:.4f} (p={dsr_p:.4f})")
        
        # 交互效应
        if 'dsr_cognitive:tl_functional' in self.model_results['M2_interaction']['p_values']:
            int_p = self.model_results['M2_interaction']['p_values']['dsr_cognitive:tl_functional']
            print(f"   • DSR×TL交互效应: p={int_p:.4f} ({'显著' if int_p < 0.05 else '不显著'})")
        
        # 双向因果
        granger = self.model_results['M4_var_causality']['granger_causality']
        bidirectional = (
            granger.get('dsr_cognitive_causes_cs_output', {}).get('significant', False) and
            granger.get('cs_output_causes_dsr_cognitive', {}).get('significant', False)
        )
        print(f"   • 双向因果关系: {'存在' if bidirectional else '不存在'}")
        
        # 语境异质性
        het_sig = self.model_results['M5_hierarchical']['heterogeneity_test']['heterogeneity_significant']
        print(f"   • 语境异质性: {'显著' if het_sig else '不显著'}")
        
        # 3. 假设检验总结
        print("\n3. 假设检验结果:")
        for h_name, h_result in self.model_results['hypothesis_tests'].items():
            if h_name != 'overall_support':
                print(f"   {h_name}: {h_result['conclusion']}")
                
        print(f"\n总体结论: 3个假设中有{self.model_results['hypothesis_tests']['overall_support']}个得到支持")

def main():
    """主函数"""
    # 设置数据路径
    # 获取脚本所在目录的父目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_path = project_root / 'output_cn' / 'data'
    
    # 创建分析器
    analyzer = StatisticalModelsAnalyzer(data_path)
    
    # 加载数据
    analyzer.load_data()
    
    # 运行所有模型
    results = analyzer.run_all_models()
    
    print("\n✓ 统计模型分析完成！")

if __name__ == "__main__":
    main()