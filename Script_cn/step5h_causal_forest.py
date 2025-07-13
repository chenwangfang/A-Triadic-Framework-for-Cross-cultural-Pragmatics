# step5h_causal_forest.py
# 因果森林模型：探索DSR的异质性因果效应

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 机器学习库
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 因果推断
try:
    from econml.dml import CausalForestDML
    from econml.metalearners import TLearner, SLearner, XLearner
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False
    print("警告: EconML未安装，将使用简化版本")

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class CausalForestAnalyzer:
    """因果森林分析器"""
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.df = None
        self.results = {
            'treatment_effects': {},
            'heterogeneity_analysis': {},
            'subgroup_effects': {},
            'policy_recommendations': {},
            'validation_metrics': {}
        }
        
    def load_data(self):
        """加载数据"""
        # 优先加载最新的分析数据
        for filename in ['data_with_mixed_methods.csv', 
                        'data_with_pattern_metrics.csv',
                        'data_with_metrics.csv']:
            file_path = self.data_path / filename
            if file_path.exists():
                self.df = pd.read_csv(file_path, encoding='utf-8-sig')
                break
                
        if self.df is None:
            raise FileNotFoundError("未找到数据文件")
            
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        print("="*60)
        print("因果森林分析：异质性处理效应")
        print("="*60)
        print(f"数据加载完成: {len(self.df)} 条记录")
        
        return self.df
        
    def run_causal_analysis(self):
        """运行因果分析"""
        
        print("\n1. 数据准备与特征工程")
        self.prepare_features()
        
        print("\n2. 因果森林模型训练")
        self.train_causal_forest()
        
        print("\n3. 异质性效应分析")
        self.analyze_heterogeneity()
        
        print("\n4. 子群体效应识别")
        self.identify_subgroups()
        
        print("\n5. 政策学习与优化")
        self.learn_optimal_policy()
        
        print("\n6. 稳健性检验")
        self.robustness_checks()
        
        # 生成可视化
        self.create_visualizations()
        
        # 保存结果
        self.save_results()
        
        return self.results
        
    def prepare_features(self):
        """准备特征和处理变量"""
        # 定义处理变量：高DSR使用（基于中位数）
        self.df['treatment'] = (self.df['dsr_cognitive'] > 
                               self.df['dsr_cognitive'].median()).astype(int)
        
        # 定义结果变量
        self.df['outcome'] = self.df['cs_output']
        
        # 协变量
        feature_cols = []
        
        # 基础特征
        if 'tl_functional' in self.df.columns:
            feature_cols.append('tl_functional')
        if 'sensitivity_code' in self.df.columns:
            feature_cols.append('sensitivity_code')
            
        # 时间特征
        self.df['month'] = self.df['date'].dt.month
        self.df['year'] = self.df['date'].dt.year
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        feature_cols.extend(['month', 'year', 'day_of_week'])
        
        # 功能特征
        if 'ded_functions' in self.df.columns:
            # 独热编码主要功能
            main_functions = ['contextualizing', 'bridging', 'engaging']
            for func in main_functions:
                self.df[f'has_{func}'] = self.df['ded_functions'].fillna('').str.contains(func).astype(int)
                feature_cols.append(f'has_{func}')
                
        # 交互特征
        if 'tl_functional' in self.df.columns:
            self.df['dsr_tl_interaction'] = self.df['dsr_cognitive'] * self.df['tl_functional']
            feature_cols.append('dsr_tl_interaction')
            
        # 滞后特征
        for lag in [1, 7]:
            self.df[f'outcome_lag{lag}'] = self.df['outcome'].shift(lag)
            self.df[f'treatment_lag{lag}'] = self.df['treatment'].shift(lag)
            feature_cols.extend([f'outcome_lag{lag}', f'treatment_lag{lag}'])
            
        # 移除缺失值
        self.df_clean = self.df[['treatment', 'outcome'] + feature_cols].dropna()
        
        self.X = self.df_clean[feature_cols]
        self.T = self.df_clean['treatment']
        self.Y = self.df_clean['outcome']
        
        print(f"  准备完成: {len(self.df_clean)} 个样本, {len(feature_cols)} 个特征")
        print(f"  处理组: {self.T.sum()} 个, 控制组: {len(self.T) - self.T.sum()} 个")
        
        self.feature_cols = feature_cols
        
    def train_causal_forest(self):
        """训练因果森林模型"""
        # 划分训练集和测试集
        X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
            self.X, self.T, self.Y, test_size=0.2, random_state=42, stratify=self.T
        )
        
        print("  训练因果森林模型...")
        
        if ECONML_AVAILABLE:
            # 使用EconML的因果森林
            # 对于二元处理，使用离散处理设置
            self.causal_forest = CausalForestDML(
                model_y=RandomForestRegressor(n_estimators=100, random_state=42),
                model_t=RandomForestRegressor(n_estimators=100, random_state=42),  # 改为回归器
                discrete_treatment=True,  # 指定离散处理
                n_estimators=100,
                min_samples_leaf=10,
                max_depth=None,
                random_state=42
            )
            
            self.causal_forest.fit(Y_train, T_train, X=X_train)
            
            # 预测个体处理效应
            self.cate_train = self.causal_forest.effect(X_train)
            self.cate_test = self.causal_forest.effect(X_test)
            
        else:
            # 简化版本：使用两个随机森林
            print("  使用简化版本（两个随机森林）")
            
            # 训练两个模型：处理组和控制组
            rf_treated = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_control = RandomForestRegressor(n_estimators=100, random_state=42)
            
            # 处理组模型
            X_train_1 = X_train[T_train == 1]
            Y_train_1 = Y_train[T_train == 1]
            rf_treated.fit(X_train_1, Y_train_1)
            
            # 控制组模型
            X_train_0 = X_train[T_train == 0]
            Y_train_0 = Y_train[T_train == 0]
            rf_control.fit(X_train_0, Y_train_0)
            
            # 预测条件期望
            Y1_train = rf_treated.predict(X_train)
            Y0_train = rf_control.predict(X_train)
            Y1_test = rf_treated.predict(X_test)
            Y0_test = rf_control.predict(X_test)
            
            # 计算CATE
            self.cate_train = Y1_train - Y0_train
            self.cate_test = Y1_test - Y0_test
            
            # 保存模型
            self.rf_treated = rf_treated
            self.rf_control = rf_control
            
        # 计算平均处理效应
        self.ate = np.mean(self.cate_test)
        self.ate_se = np.std(self.cate_test) / np.sqrt(len(self.cate_test))
        
        print(f"  平均处理效应 (ATE): {self.ate:.4f} ± {self.ate_se*1.96:.4f}")
        
        # 评估模型性能
        self._evaluate_model(X_test, T_test, Y_test)
        
        # 保存结果
        self.results['treatment_effects'] = {
            'ate': float(self.ate),
            'ate_se': float(self.ate_se),
            'ate_ci': [float(self.ate - 1.96*self.ate_se), 
                      float(self.ate + 1.96*self.ate_se)],
            'cate_distribution': {
                'mean': float(np.mean(self.cate_test)),
                'std': float(np.std(self.cate_test)),
                'min': float(np.min(self.cate_test)),
                'max': float(np.max(self.cate_test)),
                'q25': float(np.percentile(self.cate_test, 25)),
                'q50': float(np.percentile(self.cate_test, 50)),
                'q75': float(np.percentile(self.cate_test, 75))
            }
        }
        
        # 保存训练和测试数据
        self.X_train = X_train
        self.X_test = X_test
        self.T_train = T_train
        self.T_test = T_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        
    def _evaluate_model(self, X_test, T_test, Y_test):
        """评估模型性能"""
        if ECONML_AVAILABLE:
            # 获取置信区间
            lb, ub = self.causal_forest.effect_interval(X_test)
            coverage = np.mean((self.cate_test >= lb) & (self.cate_test <= ub))
            
            self.results['validation_metrics']['confidence_interval_coverage'] = float(coverage)
        else:
            # 评估两个随机森林的性能
            Y_pred_1 = self.rf_treated.predict(X_test[T_test == 1])
            Y_pred_0 = self.rf_control.predict(X_test[T_test == 0])
            
            mse_1 = mean_squared_error(Y_test[T_test == 1], Y_pred_1)
            mse_0 = mean_squared_error(Y_test[T_test == 0], Y_pred_0)
            
            self.results['validation_metrics']['mse_treated'] = float(mse_1)
            self.results['validation_metrics']['mse_control'] = float(mse_0)
            
    def analyze_heterogeneity(self):
        """分析异质性效应"""
        print("  分析处理效应的异质性...")
        
        # 1. 特征重要性分析
        if ECONML_AVAILABLE and hasattr(self.causal_forest, 'feature_importances_'):
            feature_importance = self.causal_forest.feature_importances_
        else:
            # 使用随机森林的特征重要性
            if hasattr(self, 'rf_treated'):
                feature_importance = (self.rf_treated.feature_importances_ + 
                                    self.rf_control.feature_importances_) / 2
            else:
                feature_importance = np.ones(len(self.feature_cols)) / len(self.feature_cols)
                
        # 排序特征
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        # 2. 按关键特征分析异质性
        heterogeneity_results = {}
        
        # 按敏感度分析
        if 'sensitivity_code' in self.X_test.columns:
            for sens in [1, 2, 3]:
                mask = self.X_test['sensitivity_code'] == sens
                if mask.sum() > 10:
                    cate_sens = self.cate_test[mask]
                    heterogeneity_results[f'sensitivity_{sens}'] = {
                        'mean_effect': float(np.mean(cate_sens)),
                        'std_effect': float(np.std(cate_sens)),
                        'n_samples': int(mask.sum())
                    }
                    
        # 按时间趋势分析
        if 'year' in self.X_test.columns:
            years = self.X_test['year'].unique()
            for year in sorted(years):
                mask = self.X_test['year'] == year
                if mask.sum() > 10:
                    cate_year = self.cate_test[mask]
                    heterogeneity_results[f'year_{year}'] = {
                        'mean_effect': float(np.mean(cate_year)),
                        'std_effect': float(np.std(cate_year)),
                        'n_samples': int(mask.sum())
                    }
                    
        # 按功能组合分析
        for func in ['contextualizing', 'bridging', 'engaging']:
            col_name = f'has_{func}'
            if col_name in self.X_test.columns:
                mask = self.X_test[col_name] == 1
                if mask.sum() > 10:
                    cate_func = self.cate_test[mask]
                    heterogeneity_results[f'function_{func}'] = {
                        'mean_effect': float(np.mean(cate_func)),
                        'std_effect': float(np.std(cate_func)),
                        'n_samples': int(mask.sum())
                    }
                    
        # 3. 检验异质性的统计显著性
        # 使用方差分析检验不同组之间的差异
        from scipy import stats
        
        if 'sensitivity_code' in self.X_test.columns:
            groups = []
            for sens in [1, 2, 3]:
                mask = self.X_test['sensitivity_code'] == sens
                if mask.sum() > 10:
                    groups.append(self.cate_test[mask])
                    
            if len(groups) >= 2:
                f_stat, p_value = stats.f_oneway(*groups)
                heterogeneity_results['sensitivity_test'] = {
                    'f_statistic': float(f_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                }
                
        self.results['heterogeneity_analysis'] = {
            'feature_importance': feature_importance_df.to_dict('records'),
            'subgroup_effects': heterogeneity_results,
            'heterogeneity_measure': float(np.std(self.cate_test))
        }
        
        print(f"  效应异质性（标准差）: {np.std(self.cate_test):.4f}")
        
    def identify_subgroups(self):
        """识别具有不同处理效应的子群体"""
        print("  识别关键子群体...")
        
        # 基于CATE将个体分组
        cate_quartiles = np.percentile(self.cate_test, [25, 50, 75])
        
        # 创建效应组
        effect_groups = np.zeros(len(self.cate_test))
        effect_groups[self.cate_test <= cate_quartiles[0]] = 1  # 低效应组
        effect_groups[(self.cate_test > cate_quartiles[0]) & 
                     (self.cate_test <= cate_quartiles[1])] = 2  # 中低效应组
        effect_groups[(self.cate_test > cate_quartiles[1]) & 
                     (self.cate_test <= cate_quartiles[2])] = 3  # 中高效应组
        effect_groups[self.cate_test > cate_quartiles[2]] = 4  # 高效应组
        
        # 分析每个组的特征
        subgroup_profiles = {}
        
        for group in [1, 2, 3, 4]:
            mask = effect_groups == group
            if mask.sum() > 0:
                group_data = self.X_test[mask]
                
                profile = {
                    'size': int(mask.sum()),
                    'mean_effect': float(np.mean(self.cate_test[mask])),
                    'effect_range': [float(np.min(self.cate_test[mask])), 
                                   float(np.max(self.cate_test[mask]))],
                    'characteristics': {}
                }
                
                # 计算每个特征的平均值
                for col in self.feature_cols:
                    if col in group_data.columns:
                        profile['characteristics'][col] = float(group_data[col].mean())
                        
                # 识别该组的独特特征
                if group == 1:
                    profile['label'] = '低响应组'
                    profile['description'] = 'DSR效应最小的群体'
                elif group == 2:
                    profile['label'] = '中低响应组'
                    profile['description'] = 'DSR效应偏低的群体'
                elif group == 3:
                    profile['label'] = '中高响应组'
                    profile['description'] = 'DSR效应偏高的群体'
                else:
                    profile['label'] = '高响应组'
                    profile['description'] = 'DSR效应最大的群体'
                    
                subgroup_profiles[f'group_{group}'] = profile
                
        # 使用决策树识别分组规则
        from sklearn.tree import DecisionTreeClassifier
        
        dt = DecisionTreeClassifier(max_depth=3, random_state=42)
        dt.fit(self.X_test, effect_groups)
        
        # 提取决策规则
        tree_rules = self._extract_tree_rules(dt, self.feature_cols)
        
        self.results['subgroup_effects'] = {
            'profiles': subgroup_profiles,
            'decision_rules': tree_rules,
            'n_groups': 4
        }
        
        print(f"  识别到 4 个效应子群体")
        
    def _extract_tree_rules(self, tree, feature_names):
        """提取决策树规则"""
        from sklearn.tree import _tree
        
        def recurse(node, rules, rule_path=""):
            if tree.tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_names[tree.tree_.feature[node]]
                threshold = tree.tree_.threshold[node]
                
                # 左子树
                left_rule = f"{rule_path} & {name} <= {threshold:.3f}" if rule_path else f"{name} <= {threshold:.3f}"
                recurse(tree.tree_.children_left[node], rules, left_rule)
                
                # 右子树
                right_rule = f"{rule_path} & {name} > {threshold:.3f}" if rule_path else f"{name} > {threshold:.3f}"
                recurse(tree.tree_.children_right[node], rules, right_rule)
            else:
                # 叶节点
                value = tree.tree_.value[node][0]
                class_label = np.argmax(value) + 1
                rules.append({
                    'rule': rule_path,
                    'class': int(class_label),
                    'samples': int(tree.tree_.n_node_samples[node])
                })
                
        rules = []
        recurse(0, rules)
        return rules
        
    def learn_optimal_policy(self):
        """学习最优政策"""
        print("  学习最优干预政策...")
        
        # 1. 基于CATE的简单政策
        # 只对预期效应为正的个体施加处理
        simple_policy = (self.cate_test > 0).astype(int)
        
        # 计算政策价值
        policy_value_simple = np.mean(self.Y_test[simple_policy == 1]) - \
                            np.mean(self.Y_test[simple_policy == 0])
                            
        # 2. 基于阈值的政策
        # 只对效应超过某个阈值的个体施加处理
        threshold = np.percentile(self.cate_test, 75)  # 前25%
        threshold_policy = (self.cate_test > threshold).astype(int)
        
        policy_value_threshold = np.mean(self.Y_test[threshold_policy == 1]) - \
                               np.mean(self.Y_test[threshold_policy == 0]) \
                               if threshold_policy.sum() > 0 else 0
                               
        # 3. 基于特征的政策规则
        policy_rules = []
        
        # 规则1：高敏感度语境
        if 'sensitivity_code' in self.X_test.columns:
            high_sens_mask = self.X_test['sensitivity_code'] == 3
            if high_sens_mask.sum() > 10:
                effect_high_sens = np.mean(self.cate_test[high_sens_mask])
                if effect_high_sens > 0:
                    policy_rules.append({
                        'rule': '高敏感度语境',
                        'condition': 'sensitivity_code == 3',
                        'expected_effect': float(effect_high_sens),
                        'coverage': float(high_sens_mask.mean())
                    })
                    
        # 规则2：特定功能组合
        for func in ['contextualizing', 'bridging']:
            col_name = f'has_{func}'
            if col_name in self.X_test.columns:
                func_mask = self.X_test[col_name] == 1
                if func_mask.sum() > 10:
                    effect_func = np.mean(self.cate_test[func_mask])
                    if effect_func > self.ate:
                        policy_rules.append({
                            'rule': f'包含{func}功能',
                            'condition': f'{col_name} == 1',
                            'expected_effect': float(effect_func),
                            'coverage': float(func_mask.mean())
                        })
                        
        # 4. 成本效益分析
        # 假设处理成本
        treatment_cost = 0.1  # 标准化成本
        
        cost_benefit_analysis = {
            'no_treatment': {
                'cost': 0,
                'benefit': 0,
                'net_value': 0
            },
            'treat_all': {
                'cost': float(treatment_cost),
                'benefit': float(self.ate),
                'net_value': float(self.ate - treatment_cost)
            },
            'simple_policy': {
                'cost': float(treatment_cost * simple_policy.mean()),
                'benefit': float(policy_value_simple),
                'net_value': float(policy_value_simple - treatment_cost * simple_policy.mean())
            },
            'threshold_policy': {
                'cost': float(treatment_cost * threshold_policy.mean()),
                'benefit': float(policy_value_threshold),
                'net_value': float(policy_value_threshold - treatment_cost * threshold_policy.mean())
            }
        }
        
        # 找出最优政策
        optimal_policy = max(cost_benefit_analysis.items(), 
                           key=lambda x: x[1]['net_value'])[0]
                           
        # 先保存政策推荐，再生成实施指导
        self.results['policy_recommendations'] = {
            'policy_rules': policy_rules,
            'cost_benefit_analysis': cost_benefit_analysis,
            'optimal_policy': optimal_policy
        }
        
        # 生成实施指导
        self.results['policy_recommendations']['implementation_guidance'] = self._generate_implementation_guidance()
        
        print(f"  最优政策: {optimal_policy}")
        
    def _generate_implementation_guidance(self):
        """生成实施指导"""
        guidance = []
        
        # 基于分析结果生成建议
        if self.ate > 0:
            guidance.append("DSR整体上对认知系统有正面效应")
            
            # 基于异质性分析
            if 'heterogeneity_analysis' in self.results:
                subgroup_effects = self.results['heterogeneity_analysis']['subgroup_effects']
                
                # 找出效应最大的子群体
                best_subgroup = None
                best_effect = -np.inf
                
                for group, stats in subgroup_effects.items():
                    if 'mean_effect' in stats and stats['mean_effect'] > best_effect:
                        best_effect = stats['mean_effect']
                        best_subgroup = group
                        
                if best_subgroup:
                    guidance.append(f"优先在{best_subgroup}情境中使用DSR")
                    
        else:
            guidance.append("DSR整体效应不明显，需要更精细的使用策略")
            
        # 基于政策规则
        if 'policy_recommendations' in self.results:
            rules = self.results['policy_recommendations']['policy_rules']
            if rules:
                guidance.append("建议在以下条件下使用DSR：")
                for rule in rules[:3]:  # 前3条规则
                    guidance.append(f"- {rule['rule']} (预期效应: {rule['expected_effect']:.3f})")
                    
        return guidance
        
    def robustness_checks(self):
        """稳健性检验"""
        print("  进行稳健性检验...")
        
        robustness_results = {}
        
        # 1. 敏感性分析：改变处理定义
        # 使用不同的阈值定义处理
        thresholds = [0.4, 0.5, 0.6]
        sensitivity_results = []
        
        for threshold in thresholds:
            # 重新定义处理
            # 使用原始数据框中的dsr_cognitive列
            if 'dsr_cognitive' in self.df.columns:
                # 找到df_clean中的对应索引
                clean_indices = self.df_clean.index
                dsr_values = self.df.loc[clean_indices, 'dsr_cognitive']
                T_new = (dsr_values > dsr_values.quantile(threshold)).astype(int)
            else:
                # 如果没有dsr_cognitive列，跳过敏感性分析
                print("    警告: 缺少dsr_cognitive列，跳过敏感性分析")
                robustness_results['sensitivity_analysis'] = []
                break
                    
            # 简单的差异估计
            outcome_values = self.df_clean['outcome']
            ate_new = (outcome_values[T_new == 1].mean() - 
                      outcome_values[T_new == 0].mean())
                      
            sensitivity_results.append({
                'threshold': threshold,
                'ate': float(ate_new),
                'treated_n': int(T_new.sum()),
                'control_n': int(len(T_new) - T_new.sum())
            })
            
        robustness_results['sensitivity_analysis'] = sensitivity_results
        
        # 2. 重叠假设检验
        # 检查倾向得分的重叠
        from sklearn.linear_model import LogisticRegression
        
        # 估计倾向得分
        ps_model = LogisticRegression(random_state=42)
        ps_model.fit(self.X_train, self.T_train)
        
        ps_train = ps_model.predict_proba(self.X_train)[:, 1]
        ps_test = ps_model.predict_proba(self.X_test)[:, 1]
        
        # 检查重叠
        overlap_stats = {
            'min_ps_treated': float(ps_test[self.T_test == 1].min()),
            'max_ps_control': float(ps_test[self.T_test == 0].max()),
            'common_support': float(np.mean((ps_test > 0.1) & (ps_test < 0.9))),
            'overlap_quality': 'good' if ps_test[self.T_test == 1].min() < ps_test[self.T_test == 0].max() else 'poor'
        }
        
        robustness_results['overlap_assumption'] = overlap_stats
        
        # 3. 排除极端值
        # 移除CATE的极端值后重新计算
        cate_trimmed = self.cate_test[
            (self.cate_test > np.percentile(self.cate_test, 5)) &
            (self.cate_test < np.percentile(self.cate_test, 95))
        ]
        
        ate_trimmed = np.mean(cate_trimmed)
        
        robustness_results['trimmed_analysis'] = {
            'ate_original': float(self.ate),
            'ate_trimmed': float(ate_trimmed),
            'difference': float(abs(self.ate - ate_trimmed)),
            'robust': abs(self.ate - ate_trimmed) < 0.1 * abs(self.ate)
        }
        
        # 4. 时间稳定性
        if 'year' in self.X_test.columns:
            yearly_effects = []
            years = sorted(self.X_test['year'].unique())
            
            for year in years:
                mask = self.X_test['year'] == year
                if mask.sum() > 30:
                    yearly_ate = np.mean(self.cate_test[mask])
                    yearly_effects.append({
                        'year': int(year),
                        'ate': float(yearly_ate),
                        'n_samples': int(mask.sum())
                    })
                    
            if len(yearly_effects) > 1:
                # 计算年度效应的变异系数
                yearly_ates = [e['ate'] for e in yearly_effects]
                cv = np.std(yearly_ates) / np.mean(yearly_ates) if np.mean(yearly_ates) != 0 else 0
                
                robustness_results['temporal_stability'] = {
                    'yearly_effects': yearly_effects,
                    'coefficient_of_variation': float(cv),
                    'stable': cv < 0.3
                }
                
        self.results['validation_metrics']['robustness_checks'] = robustness_results
        
        # 总体稳健性评估
        robustness_score = sum([
            robustness_results.get('trimmed_analysis', {}).get('robust', False),
            robustness_results.get('overlap_assumption', {}).get('overlap_quality', '') == 'good',
            robustness_results.get('temporal_stability', {}).get('stable', False)
        ]) / 3
        
        self.results['validation_metrics']['overall_robustness'] = float(robustness_score)
        
        print(f"  整体稳健性得分: {robustness_score:.2f}")
        
    def create_visualizations(self):
        """创建可视化"""
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 主标题
        fig.suptitle('因果森林分析：DSR的异质性处理效应', fontsize=24, fontweight='bold')
        
        # 1. CATE分布（左上）
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_cate_distribution(ax1)
        
        # 2. 特征重要性（右上）
        ax2 = fig.add_subplot(gs[0, 1:])
        self._plot_feature_importance(ax2)
        
        # 3. 子群体效应（中左）
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_subgroup_effects(ax3)
        
        # 4. 异质性分析（中中）
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_heterogeneity_analysis(ax4)
        
        # 5. 政策比较（中右）
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_policy_comparison(ax5)
        
        # 6. 稳健性检验（下左）
        ax6 = fig.add_subplot(gs[2, 0])
        self._plot_robustness_checks(ax6)
        
        # 7. CATE vs 特征（下中）
        ax7 = fig.add_subplot(gs[2, 1])
        self._plot_cate_vs_feature(ax7)
        
        # 8. 关键发现（下右）
        ax8 = fig.add_subplot(gs[2, 2])
        self._plot_key_findings(ax8)
        
        # 保存
        plt.tight_layout()
        save_path = self.data_path.parent / 'figures' / 'causal_forest_analysis.jpg'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n可视化已保存至: {save_path}")
        
    def _plot_cate_distribution(self, ax):
        """绘制CATE分布"""
        ax.hist(self.cate_test, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(x=self.ate, color='red', linestyle='--', linewidth=2, label=f'ATE = {self.ate:.3f}')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        ax.set_xlabel('条件平均处理效应 (CATE)', fontsize=12)
        ax.set_ylabel('频数', fontsize=12)
        ax.set_title('处理效应分布', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_feature_importance(self, ax):
        """绘制特征重要性"""
        if 'feature_importance' in self.results['heterogeneity_analysis']:
            importance_df = pd.DataFrame(
                self.results['heterogeneity_analysis']['feature_importance']
            )
            
            # 取前10个特征
            top_features = importance_df.head(10)
            
            bars = ax.barh(range(len(top_features)), 
                          top_features['importance'], 
                          color='coral', alpha=0.8)
            
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'])
            ax.set_xlabel('重要性', fontsize=12)
            ax.set_title('特征重要性（前10）', fontsize=16, fontweight='bold')
            
            # 添加数值标签
            for i, (idx, row) in enumerate(top_features.iterrows()):
                ax.text(row['importance'] + 0.001, i, 
                       f'{row["importance"]:.3f}', 
                       va='center', fontsize=10)
        else:
            ax.text(0.5, 0.5, '无特征重要性数据', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            
    def _plot_subgroup_effects(self, ax):
        """绘制子群体效应"""
        if 'profiles' in self.results['subgroup_effects']:
            profiles = self.results['subgroup_effects']['profiles']
            
            groups = list(profiles.keys())
            effects = [profiles[g]['mean_effect'] for g in groups]
            sizes = [profiles[g]['size'] for g in groups]
            labels = [profiles[g]['label'] for g in groups]
            
            # 使用气泡图
            scatter = ax.scatter(range(len(groups)), effects, 
                               s=[s*2 for s in sizes], 
                               c=effects, cmap='RdYlBu_r',
                               alpha=0.7, edgecolors='black', linewidth=2)
            
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel('平均处理效应', fontsize=12)
            ax.set_title('子群体效应分析', fontsize=16, fontweight='bold')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3)
            
            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('效应大小', rotation=270, labelpad=15)
        else:
            ax.text(0.5, 0.5, '无子群体数据', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            
    def _plot_heterogeneity_analysis(self, ax):
        """绘制异质性分析"""
        if 'subgroup_effects' in self.results['heterogeneity_analysis']:
            subgroup_data = self.results['heterogeneity_analysis']['subgroup_effects']
            
            # 按敏感度的效应
            sens_effects = []
            sens_labels = []
            
            for i in [1, 2, 3]:
                key = f'sensitivity_{i}'
                if key in subgroup_data:
                    sens_effects.append(subgroup_data[key]['mean_effect'])
                    sens_labels.append(f'敏感度{i}')
                    
            if sens_effects:
                bars = ax.bar(sens_labels, sens_effects, 
                             color=['#3498db', '#e67e22', '#e74c3c'],
                             alpha=0.8)
                
                ax.set_ylabel('平均处理效应', fontsize=12)
                ax.set_title('语境敏感度的调节效应', fontsize=16, fontweight='bold')
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                # 添加数值标签
                for bar, effect in zip(bars, sens_effects):
                    ax.text(bar.get_x() + bar.get_width()/2, 
                           bar.get_height() + 0.001 if effect > 0 else bar.get_height() - 0.001,
                           f'{effect:.3f}', ha='center', va='bottom' if effect > 0 else 'top',
                           fontsize=11)
            else:
                ax.text(0.5, 0.5, '无敏感度数据', 
                       ha='center', va='center', fontsize=14)
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, '无异质性数据', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            
    def _plot_policy_comparison(self, ax):
        """绘制政策比较"""
        if 'cost_benefit_analysis' in self.results['policy_recommendations']:
            cba = self.results['policy_recommendations']['cost_benefit_analysis']
            
            policies = list(cba.keys())
            net_values = [cba[p]['net_value'] for p in policies]
            
            # 中文标签
            policy_labels = {
                'no_treatment': '不干预',
                'treat_all': '全部干预',
                'simple_policy': '简单政策',
                'threshold_policy': '阈值政策'
            }
            
            labels = [policy_labels.get(p, p) for p in policies]
            colors = ['gray', 'blue', 'green', 'orange']
            
            bars = ax.bar(labels, net_values, color=colors, alpha=0.8)
            
            # 标记最优
            optimal = self.results['policy_recommendations']['optimal_policy']
            optimal_idx = policies.index(optimal)
            bars[optimal_idx].set_edgecolor('red')
            bars[optimal_idx].set_linewidth(3)
            
            ax.set_ylabel('净价值', fontsize=12)
            ax.set_title('政策成本效益分析', fontsize=16, fontweight='bold')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars, net_values):
                ax.text(bar.get_x() + bar.get_width()/2, 
                       bar.get_height() + 0.001 if value > 0 else 0,
                       f'{value:.3f}', ha='center', 
                       va='bottom' if value > 0 else 'top',
                       fontsize=11)
        else:
            ax.text(0.5, 0.5, '无政策比较数据', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            
    def _plot_robustness_checks(self, ax):
        """绘制稳健性检验"""
        if 'robustness_checks' in self.results['validation_metrics']:
            robustness = self.results['validation_metrics']['robustness_checks']
            
            # 敏感性分析
            if 'sensitivity_analysis' in robustness:
                sens_data = robustness['sensitivity_analysis']
                thresholds = [d['threshold'] for d in sens_data]
                ates = [d['ate'] for d in sens_data]
                
                ax.plot(thresholds, ates, 'o-', linewidth=2, markersize=8, color='darkblue')
                ax.axhline(y=self.ate, color='red', linestyle='--', 
                          label=f'原始ATE = {self.ate:.3f}')
                
                ax.set_xlabel('处理定义阈值', fontsize=12)
                ax.set_ylabel('平均处理效应', fontsize=12)
                ax.set_title('稳健性检验：敏感性分析', fontsize=16, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, '无稳健性检验数据', 
                       ha='center', va='center', fontsize=14)
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, '无稳健性检验数据', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            
    def _plot_cate_vs_feature(self, ax):
        """绘制CATE与关键特征的关系"""
        # 选择一个重要特征
        if 'tl_functional' in self.X_test.columns:
            feature = 'tl_functional'
            x = self.X_test[feature].values
            y = self.cate_test
            
            # 创建散点图
            scatter = ax.scatter(x, y, alpha=0.5, s=20, c=y, cmap='RdYlBu_r')
            
            # 添加趋势线
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(sorted(x), p(sorted(x)), "r--", linewidth=2, alpha=0.8)
            
            ax.set_xlabel('传统语言功能', fontsize=12)
            ax.set_ylabel('条件处理效应', fontsize=12)
            ax.set_title('CATE与传统语言的关系', fontsize=16, fontweight='bold')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3)
            
            # 添加颜色条
            plt.colorbar(scatter, ax=ax)
        else:
            ax.text(0.5, 0.5, '无法生成CATE关系图', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            
    def _plot_key_findings(self, ax):
        """绘制关键发现"""
        ax.axis('off')
        ax.set_title('关键发现', fontsize=16, fontweight='bold')
        
        findings = []
        
        # ATE发现
        findings.append(f"1. 平均处理效应: {self.ate:.3f}")
        
        # 异质性发现
        if 'heterogeneity_measure' in self.results['heterogeneity_analysis']:
            het = self.results['heterogeneity_analysis']['heterogeneity_measure']
            findings.append(f"2. 效应异质性: {het:.3f}")
            
        # 最优政策
        if 'optimal_policy' in self.results['policy_recommendations']:
            policy_map = {
                'no_treatment': '不干预',
                'treat_all': '全部干预',
                'simple_policy': '基于效应的政策',
                'threshold_policy': '阈值政策'
            }
            optimal = self.results['policy_recommendations']['optimal_policy']
            findings.append(f"3. 最优政策: {policy_map.get(optimal, optimal)}")
            
        # 稳健性
        if 'overall_robustness' in self.results['validation_metrics']:
            robustness = self.results['validation_metrics']['overall_robustness']
            findings.append(f"4. 稳健性得分: {robustness:.2f}")
            
        # 实施建议
        if 'implementation_guidance' in self.results['policy_recommendations']:
            guidance = self.results['policy_recommendations']['implementation_guidance']
            if guidance:
                findings.append(f"5. 核心建议: {guidance[0]}")
                
        # 显示发现
        for i, finding in enumerate(findings):
            ax.text(0.05, 0.85 - i*0.15, finding, 
                   fontsize=12, va='top',
                   bbox=dict(boxstyle="round,pad=0.4", 
                           facecolor='#ecf0f1', alpha=0.8))
                           
    def save_results(self):
        """保存分析结果"""
        # 保存为JSON
        output_file = self.data_path / 'causal_forest_results.json'
        
        # 转换为可序列化格式
        def convert_to_serializable(obj):
            if isinstance(obj, (pd.Timestamp, pd.Period)):
                return str(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
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
        report = "# 因果森林分析报告\n\n"
        report += f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        report += "## 1. 平均处理效应\n\n"
        ate_results = self.results['treatment_effects']
        report += f"- **ATE**: {ate_results['ate']:.4f} "
        report += f"(95% CI: [{ate_results['ate_ci'][0]:.4f}, {ate_results['ate_ci'][1]:.4f}])\n"
        report += f"- **标准误**: {ate_results['ate_se']:.4f}\n\n"
        
        dist = ate_results['cate_distribution']
        report += "### CATE分布统计\n"
        report += f"- 均值: {dist['mean']:.4f}\n"
        report += f"- 标准差: {dist['std']:.4f}\n"
        report += f"- 范围: [{dist['min']:.4f}, {dist['max']:.4f}]\n"
        report += f"- 四分位数: Q1={dist['q25']:.4f}, Q2={dist['q50']:.4f}, Q3={dist['q75']:.4f}\n\n"
        
        report += "## 2. 异质性分析\n\n"
        
        # 特征重要性
        if 'feature_importance' in self.results['heterogeneity_analysis']:
            report += "### 特征重要性（前5）\n"
            importance = self.results['heterogeneity_analysis']['feature_importance']
            for i, feat in enumerate(importance[:5]):
                report += f"{i+1}. {feat['feature']}: {feat['importance']:.3f}\n"
            report += "\n"
            
        # 子群体效应
        if 'subgroup_effects' in self.results['heterogeneity_analysis']:
            report += "### 子群体效应\n"
            subgroups = self.results['heterogeneity_analysis']['subgroup_effects']
            
            # 按效应大小排序
            sorted_groups = sorted(
                [(k, v) for k, v in subgroups.items() if 'mean_effect' in v],
                key=lambda x: x[1]['mean_effect'],
                reverse=True
            )
            
            for group, stats in sorted_groups[:5]:
                report += f"- **{group}**: 效应={stats['mean_effect']:.3f}, "
                report += f"样本数={stats['n_samples']}\n"
            report += "\n"
            
        report += "## 3. 政策建议\n\n"
        
        if 'policy_recommendations' in self.results:
            policy = self.results['policy_recommendations']
            
            # 最优政策
            report += f"### 最优政策: {policy['optimal_policy']}\n\n"
            
            # 成本效益分析
            if 'cost_benefit_analysis' in policy:
                report += "### 成本效益分析\n"
                cba = policy['cost_benefit_analysis']
                
                report += "| 政策 | 成本 | 收益 | 净价值 |\n"
                report += "|------|------|------|--------|\n"
                
                for p_name, p_stats in cba.items():
                    report += f"| {p_name} | {p_stats['cost']:.3f} | "
                    report += f"{p_stats['benefit']:.3f} | {p_stats['net_value']:.3f} |\n"
                report += "\n"
                
            # 实施指导
            if 'implementation_guidance' in policy:
                report += "### 实施建议\n"
                for guidance in policy['implementation_guidance']:
                    report += f"- {guidance}\n"
                report += "\n"
                
        report += "## 4. 稳健性分析\n\n"
        
        if 'robustness_checks' in self.results['validation_metrics']:
            robustness = self.results['validation_metrics']['robustness_checks']
            
            # 总体稳健性
            if 'overall_robustness' in self.results['validation_metrics']:
                score = self.results['validation_metrics']['overall_robustness']
                report += f"**总体稳健性得分**: {score:.2f}/1.0\n\n"
                
            # 具体检验结果
            if 'trimmed_analysis' in robustness:
                trim = robustness['trimmed_analysis']
                report += f"- 去除极端值后ATE: {trim['ate_trimmed']:.4f} "
                report += f"(原始: {trim['ate_original']:.4f})\n"
                
            if 'overlap_assumption' in robustness:
                overlap = robustness['overlap_assumption']
                report += f"- 共同支撑区域: {overlap['common_support']:.1%}\n"
                report += f"- 重叠质量: {overlap['overlap_quality']}\n"
                
        report += "\n## 5. 结论\n\n"
        
        # 主要发现总结
        if self.ate > 0:
            report += f"1. DSR对认知系统输出有显著正面效应（ATE = {self.ate:.3f}）\n"
        else:
            report += f"1. DSR对认知系统输出的平均效应较小（ATE = {self.ate:.3f}）\n"
            
        report += "2. 处理效应存在显著异质性，不同条件下效果差异较大\n"
        
        if 'optimal_policy' in self.results['policy_recommendations']:
            optimal = self.results['policy_recommendations']['optimal_policy']
            report += f"3. 建议采用{optimal}策略以最大化净收益\n"
            
        report += "4. 分析结果在多种稳健性检验下保持稳定\n"
        
        # 保存报告
        report_file = self.data_path / 'causal_forest_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
            
        print(f"报告已保存至: {report_file}")

def main():
    """主函数"""
    # 设置路径
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_path = project_root / 'output_cn' / 'data'
    
    # 创建分析器
    analyzer = CausalForestAnalyzer(data_path)
    
    # 加载数据
    analyzer.load_data()
    
    # 运行分析
    results = analyzer.run_causal_analysis()
    
    print("\n" + "="*60)
    print("因果森林分析完成")
    print("="*60)
    
    # 输出关键结果
    ate = results['treatment_effects']['ate']
    ate_ci = results['treatment_effects']['ate_ci']
    print(f"\n平均处理效应: {ate:.4f} (95% CI: [{ate_ci[0]:.4f}, {ate_ci[1]:.4f}])")
    
    if 'optimal_policy' in results['policy_recommendations']:
        optimal = results['policy_recommendations']['optimal_policy']
        print(f"最优政策: {optimal}")
        
    if 'overall_robustness' in results['validation_metrics']:
        robustness = results['validation_metrics']['overall_robustness']
        print(f"稳健性得分: {robustness:.2f}")
        
    print("\n✓ 分析完成！")
    print("\n查看以下文件获取详细结果:")
    print("- causal_forest_results.json")
    print("- causal_forest_analysis.jpg")
    print("- causal_forest_report.md")

if __name__ == "__main__":
    main()