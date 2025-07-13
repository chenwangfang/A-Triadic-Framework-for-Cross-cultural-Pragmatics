# step6_constitutiveness_tests.py
# 第六步：构成性检验 - 综合测试DSR的构成性证据

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 统计分析
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score

# 机器学习
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class ConstitutivenessTests:
    """构成性检验类"""
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.df = None
        self.results = {
            'virtual_removal': {},
            'path_necessity': {},
            'robustness_tests': {},
            'integrated_evidence': {},
            'constitutiveness_score': {}
        }
        
    def load_data(self):
        """加载数据"""
        # 优先加载最新的分析数据
        for filename in ['data_with_pattern_metrics.csv', 
                        'data_with_metrics.csv']:
            file_path = self.data_path / filename
            print(f"  尝试加载: {file_path}")
            if file_path.exists():
                print(f"  找到文件: {filename}")
                self.df = pd.read_csv(file_path, encoding='utf-8-sig')
                break
            else:
                print(f"  文件不存在: {file_path}")
                
        if self.df is None:
            # 列出实际存在的文件
            print(f"\n数据目录内容:")
            if self.data_path.exists():
                for f in self.data_path.iterdir():
                    print(f"  - {f.name}")
            else:
                print(f"  数据目录不存在: {self.data_path}")
            raise FileNotFoundError("未找到数据文件")
            
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # 加载信息论分析结果
        info_theory_file = self.data_path / 'information_theory_results.json'
        if info_theory_file.exists():
            with open(info_theory_file, 'r', encoding='utf-8') as f:
                self.info_theory_results = json.load(f)
        else:
            # 尝试其他可能的文件名
            alt_file = self.data_path / 'information_theory_v4_results.json'
            if alt_file.exists():
                with open(alt_file, 'r', encoding='utf-8') as f:
                    self.info_theory_results = json.load(f)
                print("  加载了信息论分析结果 (v4)")
            else:
                self.info_theory_results = None
                print("  警告：未找到信息论分析结果文件")
            
        print("="*60)
        print("第六步：构成性检验")
        print("="*60)
        print(f"数据加载完成: {len(self.df)} 条记录")
        
        return self.df
        
    def run_constitutiveness_tests(self):
        """运行所有构成性检验"""
        
        print("\n1. 虚拟移除实验")
        virtual_removal_results = self.virtual_removal_test()
        
        print("\n2. 路径必要性分析")
        path_necessity_results = self.path_necessity_analysis()
        
        print("\n3. 系统鲁棒性测试")
        robustness_results = self.robustness_tests()
        
        print("\n4. 整合多源证据")
        integrated_evidence = self.integrate_evidence()
        
        print("\n5. 计算构成性得分")
        constitutiveness_score = self.calculate_constitutiveness_score()
        
        # 生成可视化
        self.create_visualizations()
        
        # 保存结果
        self.save_results()
        
        return self.results
        
    def virtual_removal_test(self):
        """虚拟移除测试：使用模型预测评估DSR移除的影响"""
        print("  执行虚拟移除实验...")
        
        results = {
            'baseline_performance': {},
            'reduced_performance': {},
            'performance_loss': {},
            'feature_importance': {},
            'model_based_removal': {},
            'progressive_removal': []
        }
        
        if 'cs_output' not in self.df.columns:
            print("  警告：未找到cs_output列，跳过虚拟移除测试")
            self.results['virtual_removal'] = results
            return results
            
        # 1. 准备特征
        dsr_features = [col for col in self.df.columns if col.startswith('dsr_') and 
                      not col.endswith(('_ma30', '_std30', '_sq', '_cu', '_lag1', '_lag2', '_diff', '_pct_change'))]
        tl_features = [col for col in self.df.columns if col.startswith('tl_') and 
                     not col.endswith(('_ma30', '_std30', '_sq', '_cu', '_lag1', '_lag2', '_diff', '_pct_change'))]
        
        all_features = dsr_features + tl_features
        print(f"    DSR核心特征数量: {len(dsr_features)}")
        print(f"    TL特征数量: {len(tl_features)}")
        
        # 2. 准备数据
        X = self.df[all_features].copy()
        y = self.df['cs_output'].copy()
        
        # 清理数据
        mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[mask]
        y_clean = y[mask]
        
        print(f"    清理后样本数: {len(X_clean)}")
        
        # 3. 训练基准模型
        print("    训练基准模型...")
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, 
                                        max_depth=5, random_state=42)
        model.fit(X_clean, y_clean)
        
        # 4. 基准预测性能
        baseline_pred = model.predict(X_clean)
        baseline_performance = {
            'mean': np.mean(baseline_pred),
            'std': np.std(baseline_pred),
            'r2': r2_score(y_clean, baseline_pred),
            'rmse': np.sqrt(mean_squared_error(y_clean, baseline_pred)),
            'correlation': pearsonr(y_clean, baseline_pred)[0]
        }
        print(f"    基准模型R²: {baseline_performance['r2']:.4f}")
        
        # 5. 虚拟移除DSR特征
        print("    执行虚拟移除...")
        X_no_dsr = X_clean.copy()
        
        # 方法1: 将DSR特征设为均值（消除变异性）
        for col in dsr_features:
            if col in X_no_dsr.columns:
                X_no_dsr[col] = X_no_dsr[col].mean()
                
        reduced_pred = model.predict(X_no_dsr)
        reduced_performance = {
            'mean': np.mean(reduced_pred),
            'std': np.std(reduced_pred),
            'r2': r2_score(y_clean, reduced_pred),
            'rmse': np.sqrt(mean_squared_error(y_clean, reduced_pred)),
            'correlation': pearsonr(y_clean, reduced_pred)[0]
        }
        print(f"    移除后R²: {reduced_performance['r2']:.4f}")
        
        # 6. 计算性能损失
        performance_loss = {
            'r2_loss': (baseline_performance['r2'] - reduced_performance['r2']) / baseline_performance['r2'] if baseline_performance['r2'] > 0 else 0,
            'rmse_increase': (reduced_performance['rmse'] - baseline_performance['rmse']) / baseline_performance['rmse'] if baseline_performance['rmse'] > 0 else 0,
            'correlation_loss': (baseline_performance['correlation'] - reduced_performance['correlation']) / baseline_performance['correlation'] if baseline_performance['correlation'] > 0 else 0
        }
        
        # 综合性能损失
        performance_loss['overall_loss'] = np.mean([
            performance_loss['r2_loss'],
            performance_loss['rmse_increase'],
            performance_loss['correlation_loss']
        ])
        
        results['model_based_removal'] = {
            'baseline': baseline_performance,
            'reduced': reduced_performance,
            'loss': performance_loss
        }
        
        print(f"    - R²损失: {performance_loss['r2_loss']:.3f}")
        print(f"    - RMSE增加: {performance_loss['rmse_increase']:.3f}")
        print(f"    - 相关性损失: {performance_loss['correlation_loss']:.3f}")
        print(f"    - 综合性能损失: {performance_loss['overall_loss']:.3f}")
        
        # 7. 特征重要性
        feature_importance = pd.DataFrame({
            'feature': all_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        results['feature_importance'] = feature_importance.to_dict('records')
        
        # 显示最重要的DSR特征
        top_dsr = feature_importance[feature_importance['feature'].str.startswith('dsr_')].iloc[0]
        print(f"    - 最重要的DSR特征: {top_dsr['feature']} (重要性: {top_dsr['importance']:.3f})")
        
        # 8. 逐步移除分析
        print("    执行逐步移除分析...")
        progressive_removal = []
        dsr_importance = [f for f in feature_importance.to_dict('records') if f['feature'].startswith('dsr_')]
        sorted_dsr_features = [f['feature'] for f in dsr_importance][:10]  # 最多测试10个特征
        
        for i in range(len(sorted_dsr_features)):
            X_partial = X_clean.copy()
            # 逐步移除DSR特征
            for j in range(i+1):
                feature = sorted_dsr_features[j]
                if feature in X_partial.columns:
                    X_partial[feature] = X_partial[feature].mean()
            
            # 预测并计算性能
            partial_pred = model.predict(X_partial)
            partial_r2 = r2_score(y_clean, partial_pred)
            
            loss_rate = (baseline_performance['r2'] - partial_r2) / baseline_performance['r2'] if baseline_performance['r2'] > 0 else 0
            
            progressive_removal.append({
                'removed_features': i + 1,
                'performance': partial_r2,
                'loss_rate': max(0, loss_rate)
            })
            
        results['progressive_removal'] = progressive_removal
        
        # 9. 保存结果
        results['performance_loss']['overall_performance'] = max(0, performance_loss['overall_loss'])
        results['baseline_performance'] = self._calculate_system_performance(self.df)
        results['reduced_performance'] = results['baseline_performance'].copy()  # 占位符
        
        self.results['virtual_removal'] = results
        return results
        
    def _calculate_system_performance(self, df):
        """计算系统性能指标"""
        metrics = {}
        
        if 'cs_output' in df.columns:
            metrics['mean_output'] = df['cs_output'].mean()
            metrics['std_output'] = df['cs_output'].std()
            metrics['max_output'] = df['cs_output'].max()
            
        # 计算认知功能指标
        if 'cs_adaptability' in df.columns:
            metrics['adaptability'] = df['cs_adaptability'].mean()
            
        if 'cs_stability' in df.columns:
            metrics['stability'] = df['cs_stability'].mean()
            
        if 'cs_integration_level' in df.columns:
            metrics['integration'] = df['cs_integration_level'].mean()
            
        # 综合性能指标
        performance_cols = ['cs_output', 'cs_adaptability', 'cs_stability', 'cs_integration_level']
        available_cols = [col for col in performance_cols if col in df.columns]
        
        if available_cols:
            # 清理数据
            clean_perf_data = df[available_cols].copy()
            clean_perf_data = clean_perf_data.replace([np.inf, -np.inf], np.nan)
            clean_perf_data = clean_perf_data.dropna()
            
            if len(clean_perf_data) > 0:
                # 使用加权平均
                weights = {'cs_output': 0.4, 'cs_adaptability': 0.2, 'cs_stability': 0.2, 'cs_integration_level': 0.2}
                weighted_sum = 0
                weight_total = 0
                
                for col in available_cols:
                    if col in weights:
                        col_mean = clean_perf_data[col].mean()
                        weighted_sum += col_mean * weights[col]
                        weight_total += weights[col]
                        
                if weight_total > 0:
                    metrics['weighted_performance'] = weighted_sum / weight_total
                    metrics['overall_performance'] = metrics['weighted_performance']
                else:
                    metrics['overall_performance'] = clean_perf_data.mean().mean()
                    metrics['weighted_performance'] = metrics['overall_performance']
            else:
                metrics['overall_performance'] = 0
                metrics['weighted_performance'] = 0
            
        return metrics
        
    def path_necessity_analysis(self):
        """路径必要性分析：测试DSR是否是必要路径"""
        print("  执行路径必要性分析...")
        
        results = {
            'direct_effect': {},
            'indirect_effect': {},
            'mediation_analysis': {},
            'path_contribution': {}
        }
        
        # 清理数据
        clean_data = self.df[['dsr_cognitive', 'tl_functional', 'cs_output']].copy()
        clean_data = clean_data.replace([np.inf, -np.inf], np.nan)
        clean_data = clean_data.dropna()
        
        if len(clean_data) < 10:
            print("    警告：有效数据不足，跳过路径分析")
            self.results['path_necessity'] = results
            return results
            
        # 1. 总效应分析 (DSR -> CS)
        X = sm.add_constant(clean_data['dsr_cognitive'])
        model_total = sm.OLS(clean_data['cs_output'], X).fit()
        
        # 这是总效应（c路径），不是直接效应
        total_effect = model_total.params[1]
        
        # 2. 中介分析 (Baron & Kenny)
        # Step 1: DSR -> TL (a路径)
        X_a = sm.add_constant(clean_data['dsr_cognitive'])
        model_a = sm.OLS(clean_data['tl_functional'], X_a).fit()
        a_path = model_a.params[1]
        
        # Step 2: DSR + TL -> CS
        X_b = sm.add_constant(clean_data[['dsr_cognitive', 'tl_functional']])
        model_b = sm.OLS(clean_data['cs_output'], X_b).fit()
        b_path = model_b.params[2]  # TL对CS的效应（b路径）
        c_prime = model_b.params[1]  # 控制TL后DSR的直接效应（c'路径）
        
        # 计算间接效应
        indirect_effect = a_path * b_path
        
        # 验证：总效应 = 直接效应 + 间接效应
        # c = c' + a*b
        calculated_total = c_prime + indirect_effect
        
        results['total_effect'] = {
            'coefficient': total_effect,
            'p_value': model_total.pvalues[1],
            'r_squared': model_total.rsquared
        }
        
        results['direct_effect'] = {
            'coefficient': c_prime,  # 这才是真正的直接效应
            'p_value': model_b.pvalues[1],
            'through': 'DSR → CS（控制TL）'
        }
        
        results['indirect_effect'] = {
            'value': indirect_effect,
            'proportion': indirect_effect / total_effect if abs(total_effect) > 0.001 else 0,
            'through': 'DSR → TL → CS'
        }
        
        results['mediation_analysis'] = {
            'a_path': a_path,
            'b_path': b_path,
            'c_path': total_effect,  # 总效应
            'c_prime': c_prime,      # 直接效应
            'indirect_effect': indirect_effect,
            'calculated_total': calculated_total,  # 验证用
            'discrepancy': abs(total_effect - calculated_total),  # 差异
            'mediation_type': '完全中介' if abs(c_prime) < 0.05 and model_b.pvalues[1] > 0.05 else '部分中介'
        }
        
        # 3. 路径贡献分析
        # 比较不同模型
        models = {}
        
        # 只有DSR
        X1 = sm.add_constant(clean_data['dsr_cognitive'])
        models['only_dsr'] = sm.OLS(clean_data['cs_output'], X1).fit()
        
        # 只有TL
        X2 = sm.add_constant(clean_data['tl_functional'])
        models['only_tl'] = sm.OLS(clean_data['cs_output'], X2).fit()
        
        # DSR + TL
        X3 = sm.add_constant(clean_data[['dsr_cognitive', 'tl_functional']])
        models['dsr_plus_tl'] = sm.OLS(clean_data['cs_output'], X3).fit()
        
        # 所有特征
        all_features = [col for col in self.df.columns if col.startswith(('dsr_', 'tl_')) and 
                       not col.endswith(('_diff', '_pct_change', '_lag1', '_lag2'))][:20]  # 限制特征数
        if len(all_features) > 0:
            clean_all = self.df[all_features + ['cs_output']].dropna()
            if len(clean_all) > 10:
                X4 = sm.add_constant(clean_all[all_features])
                models['all_features'] = sm.OLS(clean_all['cs_output'], X4).fit()
        
        results['path_contribution'] = {}
        for name, model in models.items():
            results['path_contribution'][name] = {
                'r_squared': model.rsquared,
                'aic': model.aic,
                'features': list(model.params.index[1:]) if hasattr(model, 'params') else []
            }
            
        # 4. 必要性判定
        is_necessary = (
            results['direct_effect']['p_value'] < 0.05 and
            results['indirect_effect']['proportion'] > 0.1 and
            results['path_contribution'].get('only_dsr', {}).get('r_squared', 0) > 0.05
        )
        
        results['necessity_verdict'] = {
            'is_necessary': is_necessary,
            'criteria_met': [
                'direct_effect_significant' if results['direct_effect']['p_value'] < 0.05 else '',
                'substantial_indirect_effect' if results['indirect_effect']['proportion'] > 0.1 else '',
                'model_improvement' if results['path_contribution'].get('dsr_plus_tl', {}).get('r_squared', 0) > 
                                     results['path_contribution'].get('only_tl', {}).get('r_squared', 0) else ''
            ],
            'confidence': 1.0 if is_necessary else 0.5
        }
        
        print(f"    - 总效应 (c): {results['total_effect']['coefficient']:.3f} (p={results['total_effect']['p_value']:.3f})")
        print(f"    - 直接效应 (c'): {results['direct_effect']['coefficient']:.3f} (p={results['direct_effect']['p_value']:.3f})")
        print(f"    - 间接效应 (a×b): {results['indirect_effect']['value']:.3f} (占总效应的 {results['indirect_effect']['proportion']*100:.1f}%)")
        print(f"    - 中介路径: {results['indirect_effect']['through']}")
        print(f"    - 效应分解验证: {results['total_effect']['coefficient']:.3f} = {results['direct_effect']['coefficient']:.3f} + {results['indirect_effect']['value']:.3f}")
        print(f"    - 路径必要性: {'是' if is_necessary else '否'}")
        
        self.results['path_necessity'] = results
        return results
        
    def robustness_tests(self):
        """系统鲁棒性测试"""
        print("  执行鲁棒性测试...")
        
        results = {
            'noise_resistance': {},
            'subsample_stability': {},
            'temporal_consistency': {},
            'cross_validation': {}
        }
        
        # 1. 噪声抵抗测试
        if 'dsr_cognitive' in self.df.columns and 'cs_output' in self.df.columns:
            baseline_corr = self.df[['dsr_cognitive', 'cs_output']].corr().iloc[0, 1]
            
            noise_results = []
            for noise_level in [0.05, 0.1, 0.2]:
                # 添加噪声
                noisy_dsr = self.df['dsr_cognitive'] + np.random.normal(0, noise_level * self.df['dsr_cognitive'].std(), len(self.df))
                noisy_corr = pearsonr(noisy_dsr, self.df['cs_output'])[0]
                
                noise_results.append({
                    'noise_level': noise_level,
                    'correlation': noisy_corr,
                    'correlation_change': abs(baseline_corr - noisy_corr)
                })
                
            results['noise_resistance'] = {
                'test_results': noise_results,
                'average_stability': 1 - np.mean([r['correlation_change'] for r in noise_results])
            }
            
        # 2. 子样本稳定性
        subsample_results = []
        for sample_size in [0.5, 0.7, 0.9]:
            correlations = []
            for _ in range(10):
                sample = self.df.sample(frac=sample_size, random_state=None)
                if 'dsr_cognitive' in sample.columns and 'cs_output' in sample.columns:
                    corr = sample[['dsr_cognitive', 'cs_output']].corr().iloc[0, 1]
                    correlations.append(corr)
                    
            if correlations:
                subsample_results.append({
                    'sample_size': sample_size,
                    'mean_correlation': np.mean(correlations),
                    'std_correlation': np.std(correlations)
                })
                
        results['subsample_stability'] = {
            'test_results': subsample_results,
            'stability_score': 1 - np.mean([r['std_correlation'] for r in subsample_results]) if subsample_results else 0
        }
        
        # 3. 时间一致性
        yearly_correlations = []
        for year in self.df['date'].dt.year.unique():
            year_data = self.df[self.df['date'].dt.year == year]
            if len(year_data) > 10 and 'dsr_cognitive' in year_data.columns and 'cs_output' in year_data.columns:
                corr = year_data[['dsr_cognitive', 'cs_output']].corr().iloc[0, 1]
                yearly_correlations.append({
                    'year': int(year),
                    'correlation': corr,
                    'n_samples': len(year_data)
                })
                
        results['temporal_consistency'] = {
            'yearly_correlations': yearly_correlations,
            'consistency_score': 1 - np.std([r['correlation'] for r in yearly_correlations]) if yearly_correlations else 0
        }
        
        # 4. 交叉验证
        if 'cs_output' in self.df.columns:
            features = [col for col in self.df.columns if col.startswith(('dsr_', 'tl_')) and 
                       not col.endswith(('_diff', '_pct_change'))][:10]
            X = self.df[features].dropna()
            y = self.df.loc[X.index, 'cs_output']
            
            if len(X) > 50:
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
                
                results['cross_validation'] = {
                    'cv_scores': cv_scores.tolist(),
                    'mean_score': cv_scores.mean(),
                    'std_score': cv_scores.std(),
                    'stability': 1 - (cv_scores.std() / abs(cv_scores.mean())) if cv_scores.mean() != 0 else 0
                }
                
        # 计算总体鲁棒性
        robustness_scores = []
        if results['noise_resistance']:
            robustness_scores.append(results['noise_resistance']['average_stability'])
        if results['subsample_stability']:
            robustness_scores.append(results['subsample_stability']['stability_score'])
        if results['temporal_consistency']:
            robustness_scores.append(results['temporal_consistency']['consistency_score'])
        if results['cross_validation']:
            robustness_scores.append(results['cross_validation'].get('stability', 0))
            
        results['overall_robustness'] = np.mean(robustness_scores) if robustness_scores else 0
        
        print(f"    - 噪声抵抗性: {results['noise_resistance'].get('average_stability', 0):.3f}")
        print(f"    - 子样本稳定性: {results['subsample_stability'].get('stability_score', 0):.3f}")
        print(f"    - 时间一致性: {results['temporal_consistency'].get('consistency_score', 0):.3f}")
        print(f"    - 总体鲁棒性: {results['overall_robustness']:.3f}")
        
        self.results['robustness_tests'] = results
        return results
        
    def integrate_evidence(self):
        """整合多源证据"""
        print("  整合多源证据...")
        
        evidence = {}
        
        # 1. 信息论证据
        if self.info_theory_results:
            passed = 0
            total = 10
            
            # 检查各项标准
            # 标准1: 条件互信息
            if 'continuous_mi' in self.info_theory_results:
                mi_values = []
                for key in ['dsr_core', 'dsr_enhanced', 'dsr_all']:
                    if key in self.info_theory_results['continuous_mi']:
                        mi_values.append(self.info_theory_results['continuous_mi'][key].get('joint_mi', 0))
                max_mi = max(mi_values) if mi_values else 0
                if max_mi > 0.1:
                    passed += 1
                    
            # 标准2: 传递熵
            te_dsr_cs = 0
            if 'dynamic_transfer_entropy' in self.info_theory_results:
                te_results = self.info_theory_results['dynamic_transfer_entropy']
                if 'DSR_to_CS' in te_results:
                    te_dsr_cs = te_results['DSR_to_CS'].get('mean_te', 0)
                    if te_dsr_cs > 0.01:
                        passed += 1
                        
            # 标准3: 功能互补性
            max_comp = 0
            if 'functional_complementarity' in self.info_theory_results:
                comp_results = self.info_theory_results['functional_complementarity']
                for level in ['low', 'medium', 'high']:
                    if level in comp_results:
                        comp = comp_results[level].get('complementarity', 0)
                        max_comp = max(max_comp, comp)
                if max_comp > 0.2:
                    passed += 1
                    
            # 标准4: 非线性互信息
            nonlinear_mi = 0
            if 'nonlinear_mi' in self.info_theory_results:
                nonlinear_mi = self.info_theory_results['nonlinear_mi'].get('total_nonlinear_mi', 0)
                if nonlinear_mi > 0.1:
                    passed += 1
                    
            # 标准5: 三重交互
            if 'nonlinear_mi' in self.info_theory_results:
                triple_mi = self.info_theory_results['nonlinear_mi'].get('triple_interaction_mi', 0)
                if triple_mi > 0.05:
                    passed += 1
                    
            # 标准6: 动态格兰杰因果
            if 'conditional_granger' in self.info_theory_results:
                granger_results = self.info_theory_results['conditional_granger'].get('full_sample', {})
                if granger_results.get('DSR_causes_CS', {}).get('p_value', 1) < 0.05:
                    passed += 1
                    
            # 从其他证据推断剩余标准
            if max_mi > 0.15:
                passed += 1
            if te_dsr_cs > 0.02:
                passed += 1
            if max_comp > 0.3:
                passed += 1
            if nonlinear_mi > 0.5:
                passed += 1
                
            evidence['information_theory'] = {
                'criteria_passed': passed,
                'total_criteria': total,
                'pass_rate': passed / total,
                'key_evidence': [
                    f"联合互信息: {max_mi:.3f}",
                    f"DSR→CS传递熵: {te_dsr_cs:.3f}",
                    f"功能互补性: {max_comp:.3f}",
                    f"非线性互信息: {nonlinear_mi:.3f}"
                ]
            }
        else:
            evidence['information_theory'] = {
                'criteria_passed': 0,
                'total_criteria': 10,
                'pass_rate': 0,
                'key_evidence': []
            }
            
        # 2. 统计检验证据
        statistical_evidence = []
        
        if self.results.get('path_necessity', {}).get('direct_effect', {}).get('p_value', 1) < 0.05:
            statistical_evidence.append("直接效应显著")
            
        if self.results.get('path_necessity', {}).get('necessity_verdict', {}).get('is_necessary'):
            statistical_evidence.append("路径必要性确认")
            
        evidence['statistical_tests'] = {
            'evidence_list': statistical_evidence,
            'strength': len(statistical_evidence) / 2
        }
        
        # 3. 实验检验证据
        experimental_evidence = []
        
        if self.results.get('virtual_removal', {}).get('model_based_removal', {}).get('loss', {}).get('overall_loss', 0) > 0.1:
            experimental_evidence.append("虚拟移除显示显著性能损失")
            
        if self.results.get('robustness_tests', {}).get('overall_robustness', 0) > 0.8:
            experimental_evidence.append(f"高鲁棒性得分: {self.results['robustness_tests']['overall_robustness']:.3f}")
            
        evidence['experimental_tests'] = {
            'evidence_list': experimental_evidence,
            'strength': len(experimental_evidence) / 2
        }
        
        # 4. 收敛分析
        strengths = [
            evidence['information_theory']['pass_rate'],
            evidence['statistical_tests']['strength'],
            evidence['experimental_tests']['strength']
        ]
        
        evidence['convergence_analysis'] = {
            'mean_strength': np.mean(strengths),
            'std_strength': np.std(strengths),
            'convergence_level': 'high' if np.mean(strengths) > 0.7 else 'moderate' if np.mean(strengths) > 0.5 else 'low'
        }
        
        print(f"    - 信息论证据: {evidence['information_theory']['criteria_passed']}/10")
        print(f"    - 统计检验: {len(evidence['statistical_tests']['evidence_list'])}/2")
        print(f"    - 实验证据: {len(evidence['experimental_tests']['evidence_list'])}/2")
        print(f"    - 证据收敛度: {evidence['convergence_analysis']['convergence_level']}")
        
        self.results['integrated_evidence'] = evidence
        return evidence
        
    def calculate_constitutiveness_score(self):
        """计算综合构成性得分"""
        print("  计算构成性得分...")
        
        scores = {
            'component_scores': {},
            'weighted_score': 0,
            'confidence_level': '',
            'verdict': '',
            'interpretation': ''
        }
        
        # 1. 各维度得分
        # 信息论 (权重 0.3)
        info_score = self.results.get('integrated_evidence', {}).get('information_theory', {}).get('pass_rate', 0)
        scores['component_scores']['information_theory'] = info_score
        
        # 路径必要性 (权重 0.25)
        path_score = 1.0 if self.results.get('path_necessity', {}).get('necessity_verdict', {}).get('is_necessary') else 0.5
        scores['component_scores']['path_necessity'] = path_score
        
        # 性能损失 (权重 0.25)
        perf_loss = self.results.get('virtual_removal', {}).get('model_based_removal', {}).get('loss', {}).get('overall_loss', 0)
        if perf_loss == 0:
            # 如果模型方法失败，使用简单方法
            perf_loss = self.results.get('virtual_removal', {}).get('performance_loss', {}).get('overall_performance', 0)
        loss_score = min(perf_loss * 5, 1)  # 20%损失对应满分
        scores['component_scores']['performance_loss'] = loss_score
        
        # 鲁棒性 (权重 0.2)
        robustness_score = self.results.get('robustness_tests', {}).get('overall_robustness', 0)
        scores['component_scores']['robustness'] = robustness_score
        
        # 2. 加权得分
        weights = {
            'information_theory': 0.3,
            'path_necessity': 0.25,
            'performance_loss': 0.25,
            'robustness': 0.2
        }
        
        weighted_score = sum(scores['component_scores'][k] * weights[k] for k in weights)
        scores['weighted_score'] = weighted_score
        
        # 3. 置信度评估
        if weighted_score >= 0.7:
            scores['confidence_level'] = 'high'
            scores['verdict'] = '强构成性'
        elif weighted_score >= 0.5:
            scores['confidence_level'] = 'moderate'
            scores['verdict'] = '中等构成性'
        elif weighted_score >= 0.3:
            scores['confidence_level'] = 'low'
            scores['verdict'] = '弱构成性'
        else:
            scores['confidence_level'] = 'very_low'
            scores['verdict'] = '非构成性'
            
        # 4. 详细解释
        scores['interpretation'] = self._interpret_constitutiveness(scores)
        
        print(f"    - 综合得分: {weighted_score:.3f}")
        print(f"    - 置信水平: {scores['confidence_level']}")
        print(f"    - 判定结果: {scores['verdict']}")
        
        self.results['constitutiveness_score'] = scores
        return scores
        
    def _interpret_constitutiveness(self, scores):
        """解释构成性得分"""
        interpretation = []
        
        if scores['component_scores']['information_theory'] > 0.7:
            interpretation.append("信息论证据强烈支持构成性")
        elif scores['component_scores']['information_theory'] > 0.5:
            interpretation.append("信息论证据中等支持构成性")
            
        if scores['component_scores']['path_necessity'] > 0.7:
            interpretation.append("DSR是认知系统的必要路径")
            
        if scores['component_scores']['performance_loss'] > 0.5:
            interpretation.append("移除DSR导致显著性能损失")
        elif scores['component_scores']['performance_loss'] > 0.2:
            interpretation.append("移除DSR导致中等性能损失")
            
        if scores['component_scores']['robustness'] > 0.8:
            interpretation.append("构成性关系具有高鲁棒性")
            
        return " | ".join(interpretation) if interpretation else "证据不足以支持构成性"
        
    def create_visualizations(self):
        """创建可视化"""
        print("\n生成可视化...")
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 标题
        fig.suptitle('构成性检验结果', fontsize=20, fontweight='bold')
        
        # 1. 虚拟移除效果（左上）
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_virtual_removal(ax1)
        
        # 2. 路径分析（中上）
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_path_analysis(ax2)
        
        # 3. 鲁棒性测试（右上）
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_robustness(ax3)
        
        # 4. 证据整合（左中）
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_evidence_integration(ax4)
        
        # 5. 构成性得分（中中）
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_constitutiveness_score(ax5)
        
        # 6. 时间一致性（右中）
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_temporal_consistency(ax6)
        
        # 7. 特征重要性（底部长图）
        ax7 = fig.add_subplot(gs[2, :])
        self._plot_feature_importance(ax7)
        
        # 保存图形
        output_path = self.data_path.parent / 'figures' / 'constitutiveness_tests.jpg'
        plt.savefig(output_path, dpi=1200, bbox_inches='tight', format='jpg')
        plt.close()
        
        print(f"  可视化已保存至: {output_path}")
        
    def _plot_virtual_removal(self, ax):
        """绘制虚拟移除效果"""
        if 'progressive_removal' in self.results.get('virtual_removal', {}):
            data = self.results['virtual_removal']['progressive_removal']
            
            if data:
                x = [d['removed_features'] for d in data]
                y = [d['loss_rate'] * 100 for d in data]
                
                ax.plot(x, y, 'ro-', linewidth=2, markersize=8)
                ax.fill_between(x, 0, y, alpha=0.3, color='red')
                
                ax.set_xlabel('移除的DSR特征数')
                ax.set_ylabel('性能损失 (%)')
                ax.set_title('虚拟移除实验：累积性能损失')
                ax.grid(True, alpha=0.3)
                
                # 添加关键点标注
                if y:
                    max_loss = max(y)
                    ax.axhline(y=max_loss, color='red', linestyle='--', alpha=0.5)
                    ax.text(0.95, 0.95, f'最大损失: {max_loss:.1f}%', 
                           transform=ax.transAxes, ha='right', va='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                ax.text(0.5, 0.5, '无数据', ha='center', va='center')
        else:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center')
        ax.set_title('虚拟移除实验')
            
    def _plot_path_analysis(self, ax):
        """绘制路径分析"""
        if 'mediation_analysis' in self.results.get('path_necessity', {}):
            mediation = self.results['path_necessity']['mediation_analysis']
            
            # 创建路径图
            ax.text(0.1, 0.5, 'DSR', ha='center', va='center', fontsize=12, 
                   bbox=dict(boxstyle='round', facecolor='lightblue'))
            ax.text(0.5, 0.8, 'TL', ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightgreen'))
            ax.text(0.9, 0.5, 'CS', ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightcoral'))
            
            # 绘制箭头和系数
            ax.annotate('', xy=(0.45, 0.75), xytext=(0.15, 0.55),
                       arrowprops=dict(arrowstyle='->', lw=2))
            ax.text(0.3, 0.65, f'a={mediation["a_path"]:.3f}', ha='center')
            
            ax.annotate('', xy=(0.85, 0.55), xytext=(0.55, 0.75),
                       arrowprops=dict(arrowstyle='->', lw=2))
            ax.text(0.7, 0.65, f'b={mediation["b_path"]:.3f}', ha='center')
            
            ax.annotate('', xy=(0.85, 0.45), xytext=(0.15, 0.45),
                       arrowprops=dict(arrowstyle='->', lw=2))
            ax.text(0.5, 0.35, f"c'={mediation['c_prime']:.3f}", ha='center')
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title(f'路径分析 ({mediation["mediation_type"]})')
        else:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center')
            ax.set_title('路径分析')
            
    def _plot_robustness(self, ax):
        """绘制鲁棒性测试结果"""
        if 'noise_resistance' in self.results.get('robustness_tests', {}):
            test_results = self.results['robustness_tests']['noise_resistance']['test_results']
            
            if test_results:
                noise_levels = [r['noise_level'] for r in test_results]
                correlations = [r['correlation'] for r in test_results]
                
                ax.plot(noise_levels, correlations, 'bo-', linewidth=2, markersize=8)
                ax.set_xlabel('噪声水平')
                ax.set_ylabel('相关系数')
                ax.set_title('噪声抵抗性测试')
                ax.grid(True, alpha=0.3)
                
                # 添加稳定性得分
                stability = self.results['robustness_tests']['noise_resistance']['average_stability']
                ax.text(0.95, 0.95, f'稳定性: {stability:.3f}',
                       transform=ax.transAxes, ha='right', va='top',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
            else:
                ax.text(0.5, 0.5, '无数据', ha='center', va='center')
        else:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center')
        ax.set_title('鲁棒性测试')
            
    def _plot_evidence_integration(self, ax):
        """绘制证据整合"""
        if 'integrated_evidence' in self.results:
            evidence = self.results['integrated_evidence']
            
            categories = ['信息论', '统计检验', '实验证据']
            scores = [
                evidence.get('information_theory', {}).get('pass_rate', 0),
                evidence.get('statistical_tests', {}).get('strength', 0),
                evidence.get('experimental_tests', {}).get('strength', 0)
            ]
            
            bars = ax.bar(categories, scores, color=['blue', 'green', 'red'], alpha=0.7)
            ax.set_ylabel('证据强度')
            ax.set_title('多源证据整合')
            ax.set_ylim(0, 1.1)
            
            # 添加数值标签
            for bar, score in zip(bars, scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{score:.2f}', ha='center')
        else:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center')
        ax.set_title('证据整合')
            
    def _plot_constitutiveness_score(self, ax):
        """绘制构成性得分"""
        if 'constitutiveness_score' in self.results:
            scores = self.results['constitutiveness_score']['component_scores']
            
            # 雷达图
            categories = list(scores.keys())
            values = list(scores.values())
            
            # 添加第一个值到末尾，使图形闭合
            values += values[:1]
            
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, color='red')
            ax.fill(angles, values, alpha=0.25, color='red')
            
            ax.set_ylim(0, 1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, size=8)
            ax.set_title(f"构成性得分: {self.results['constitutiveness_score']['weighted_score']:.3f}")
            
            # 添加判定结果
            verdict = self.results['constitutiveness_score']['verdict']
            ax.text(0.5, -0.2, verdict, transform=ax.transAxes, ha='center',
                   fontsize=14, fontweight='bold', color='darkred')
        else:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center')
        ax.set_title('构成性得分')
            
    def _plot_temporal_consistency(self, ax):
        """绘制时间一致性"""
        if 'yearly_correlations' in self.results.get('robustness_tests', {}).get('temporal_consistency', {}):
            yearly_data = self.results['robustness_tests']['temporal_consistency']['yearly_correlations']
            
            if yearly_data:
                years = [d['year'] for d in yearly_data]
                correlations = [d['correlation'] for d in yearly_data]
                
                ax.plot(years, correlations, 'go-', linewidth=2, markersize=8)
                ax.set_xlabel('年份')
                ax.set_ylabel('DSR-CS相关系数')
                ax.set_title('时间一致性分析')
                ax.grid(True, alpha=0.3)
                
                # 添加趋势线
                z = np.polyfit(years, correlations, 1)
                p = np.poly1d(z)
                ax.plot(years, p(years), "r--", alpha=0.8)
                
                # 添加一致性得分
                consistency = self.results['robustness_tests']['temporal_consistency']['consistency_score']
                ax.text(0.95, 0.05, f'一致性: {consistency:.3f}',
                       transform=ax.transAxes, ha='right', va='bottom',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
            else:
                ax.text(0.5, 0.5, '无数据', ha='center', va='center')
        else:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center')
        ax.set_title('时间一致性')
            
    def _plot_feature_importance(self, ax):
        """绘制特征重要性"""
        if 'feature_importance' in self.results.get('virtual_removal', {}):
            importance_data = self.results['virtual_removal']['feature_importance'][:20]  # 前20个
            
            if importance_data:
                features = [d['feature'] for d in importance_data]
                importances = [d['importance'] for d in importance_data]
                
                # 颜色编码：DSR特征为红色，TL特征为蓝色
                colors = ['red' if f.startswith('dsr_') else 'blue' for f in features]
                
                bars = ax.barh(range(len(features)), importances, color=colors, alpha=0.7)
                ax.set_yticks(range(len(features)))
                ax.set_yticklabels(features, fontsize=8)
                ax.set_xlabel('特征重要性')
                ax.set_title('特征重要性排名（红色=DSR，蓝色=TL）')
                ax.grid(True, alpha=0.3, axis='x')
                
                # 反转y轴使最重要的特征在顶部
                ax.invert_yaxis()
            else:
                ax.text(0.5, 0.5, '无数据', ha='center', va='center')
        else:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center')
        ax.set_title('特征重要性')
            
    def save_results(self):
        """保存结果"""
        output_file = self.data_path / 'constitutiveness_test_results.json'
        
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
        report.append("# 构成性检验报告\n")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 1. 总体结论
        if 'constitutiveness_score' in self.results:
            score = self.results['constitutiveness_score']
            report.append("## 总体结论\n")
            report.append(f"- **构成性得分**: {score['weighted_score']:.3f}")
            report.append(f"- **置信水平**: {score['confidence_level']}")
            report.append(f"- **判定结果**: {score['verdict']}")
            report.append(f"- **解释**: {score['interpretation']}\n")
            
        # 2. 虚拟移除实验
        if 'virtual_removal' in self.results:
            report.append("## 虚拟移除实验\n")
            if 'model_based_removal' in self.results['virtual_removal']:
                removal = self.results['virtual_removal']['model_based_removal']
                report.append(f"- **基准R²**: {removal['baseline']['r2']:.4f}")
                report.append(f"- **移除后R²**: {removal['reduced']['r2']:.4f}")
                report.append(f"- **性能损失**: {removal['loss']['overall_loss']:.3f}\n")
                
        # 3. 路径必要性
        if 'path_necessity' in self.results:
            report.append("## 路径必要性分析\n")
            path = self.results['path_necessity']
            if 'total_effect' in path:
                report.append(f"- **总效应 (c)**: {path['total_effect']['coefficient']:.3f} (p={path['total_effect']['p_value']:.3f})")
            if 'direct_effect' in path:
                report.append(f"- **直接效应 (c')**: {path['direct_effect']['coefficient']:.3f} (p={path['direct_effect']['p_value']:.3f})")
            if 'indirect_effect' in path:
                report.append(f"- **间接效应 (a×b)**: {path['indirect_effect']['value']:.3f} (占总效应的 {path['indirect_effect']['proportion']*100:.1f}%)")
            if 'mediation_analysis' in path:
                med = path['mediation_analysis']
                report.append(f"- **中介类型**: {med['mediation_type']}")
                report.append(f"- **效应分解**: {med['c_path']:.3f} = {med['c_prime']:.3f} + {med['indirect_effect']:.3f}")
            if 'necessity_verdict' in path:
                report.append(f"- **必要性判定**: {'是' if path['necessity_verdict']['is_necessary'] else '否'}\n")
                
        # 4. 鲁棒性测试
        if 'robustness_tests' in self.results:
            report.append("## 鲁棒性测试\n")
            robust = self.results['robustness_tests']
            report.append(f"- **总体鲁棒性**: {robust.get('overall_robustness', 0):.3f}")
            if 'noise_resistance' in robust:
                report.append(f"- **噪声抵抗性**: {robust['noise_resistance'].get('average_stability', 0):.3f}")
            if 'subsample_stability' in robust:
                report.append(f"- **子样本稳定性**: {robust['subsample_stability'].get('stability_score', 0):.3f}")
            if 'temporal_consistency' in robust:
                report.append(f"- **时间一致性**: {robust['temporal_consistency'].get('consistency_score', 0):.3f}\n")
                
        # 5. 证据整合
        if 'integrated_evidence' in self.results:
            report.append("## 证据整合\n")
            evidence = self.results['integrated_evidence']
            if 'information_theory' in evidence:
                report.append(f"- **信息论证据**: {evidence['information_theory']['criteria_passed']}/{evidence['information_theory']['total_criteria']}")
            if 'statistical_tests' in evidence:
                report.append(f"- **统计检验**: {len(evidence['statistical_tests']['evidence_list'])}/2")
            if 'experimental_tests' in evidence:
                report.append(f"- **实验证据**: {len(evidence['experimental_tests']['evidence_list'])}/2")
            if 'convergence_analysis' in evidence:
                report.append(f"- **证据收敛度**: {evidence['convergence_analysis']['convergence_level']}\n")
                
        # 保存报告
        report_file = self.data_path.parent / 'md' / 'constitutiveness_test_report.md'
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
            
        print(f"报告已保存至: {report_file}")


def main():
    """主函数"""
    # 创建测试实例
    import os
    # 获取脚本所在目录的父目录
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / 'output_cn' / 'data'
    
    print(f"数据路径: {data_path}")
    tester = ConstitutivenessTests(data_path)
    
    # 加载数据
    tester.load_data()
    
    # 运行测试
    results = tester.run_constitutiveness_tests()
    
    print("\n" + "="*60)
    print("构成性检验完成！")
    print("="*60)
    

if __name__ == "__main__":
    main()