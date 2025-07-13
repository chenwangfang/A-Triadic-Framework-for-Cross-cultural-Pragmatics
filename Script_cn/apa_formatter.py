#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APA格式化工具函数
符合APA第7版的统计量格式化标准
"""

def format_p_value(p_value):
    """
    格式化p值符合APA第7版标准
    
    参数:
        p_value: float, p值
        
    返回:
        str: 格式化后的p值字符串（使用*表示斜体）
    """
    if p_value < 0.001:
        return "*p* < .001"
    elif p_value < 0.01:
        return f"*p* = {p_value:.3f}"
    else:
        return f"*p* = {p_value:.2f}"

def format_correlation(r_value, p_value=None, n=None):
    """
    格式化相关系数
    
    参数:
        r_value: float, 相关系数
        p_value: float, p值（可选）
        n: int, 样本量（可选）
        
    返回:
        str: 格式化后的相关系数字符串
    """
    if n is not None:
        r_str = f"*r*({n-2}) = {r_value:.3f}"
    else:
        r_str = f"*r* = {r_value:.3f}"
    
    if p_value is not None:
        r_str += f", {format_p_value(p_value)}"
    
    return r_str

def format_t_test(t_value, df, p_value):
    """
    格式化t检验结果
    
    参数:
        t_value: float, t统计量
        df: int, 自由度
        p_value: float, p值
        
    返回:
        str: 格式化后的t检验结果
    """
    return f"*t*({df}) = {t_value:.2f}, {format_p_value(p_value)}"

def format_f_test(f_value, df1, df2, p_value):
    """
    格式化F检验结果
    
    参数:
        f_value: float, F统计量
        df1: int, 组间自由度
        df2: int, 组内自由度
        p_value: float, p值
        
    返回:
        str: 格式化后的F检验结果
    """
    return f"*F*({df1}, {df2}) = {f_value:.2f}, {format_p_value(p_value)}"

def format_chi_square(chi2_value, df, n, p_value):
    """
    格式化卡方检验结果
    
    参数:
        chi2_value: float, 卡方统计量
        df: int, 自由度
        n: int, 样本量
        p_value: float, p值
        
    返回:
        str: 格式化后的卡方检验结果
    """
    return f"χ²({df}, *N* = {n}) = {chi2_value:.2f}, {format_p_value(p_value)}"

def format_mean_sd(mean, sd):
    """
    格式化均值和标准差
    
    参数:
        mean: float, 均值
        sd: float, 标准差
        
    返回:
        str: 格式化后的均值和标准差
    """
    return f"*M* = {mean:.2f}, *SD* = {sd:.2f}"

def format_effect_size(effect_type, value, ci_lower=None, ci_upper=None):
    """
    格式化效应量
    
    参数:
        effect_type: str, 效应量类型 ('d', 'r', 'eta2', 'eta2p')
        value: float, 效应量值
        ci_lower: float, 置信区间下限（可选）
        ci_upper: float, 置信区间上限（可选）
        
    返回:
        str: 格式化后的效应量
    """
    if effect_type == 'd':
        result = f"Cohen's *d* = {value:.2f}"
    elif effect_type == 'r':
        result = f"*r* = {value:.3f}"
    elif effect_type == 'eta2':
        result = f"η² = {value:.3f}"
    elif effect_type == 'eta2p':
        result = f"ηp² = {value:.3f}"
    else:
        result = f"{effect_type} = {value:.3f}"
    
    if ci_lower is not None and ci_upper is not None:
        result += f", 95% CI [{ci_lower:.2f}, {ci_upper:.2f}]"
    
    return result

def format_regression(r2, f_value, df1, df2, p_value):
    """
    格式化回归分析整体模型结果
    
    参数:
        r2: float, R²值
        f_value: float, F统计量
        df1: int, 回归自由度
        df2: int, 残差自由度
        p_value: float, p值
        
    返回:
        str: 格式化后的回归结果
    """
    return f"*R*² = {r2:.3f}, *F*({df1}, {df2}) = {f_value:.2f}, {format_p_value(p_value)}"

def format_beta(beta, se, t_value, p_value):
    """
    格式化回归系数
    
    参数:
        beta: float, 标准化回归系数
        se: float, 标准误
        t_value: float, t统计量
        p_value: float, p值
        
    返回:
        str: 格式化后的回归系数
    """
    return f"β = {beta:.3f}, *SE* = {se:.3f}, *t* = {t_value:.2f}, {format_p_value(p_value)}"

# 测试函数
if __name__ == "__main__":
    print("APA格式化示例：")
    print(format_p_value(0.0001))  # *p* < .001
    print(format_p_value(0.003))   # *p* = .003
    print(format_p_value(0.045))   # *p* = .04
    print(format_correlation(0.356, 0.001, 100))  # *r*(98) = .356, *p* < .001
    print(format_t_test(2.45, 58, 0.017))  # *t*(58) = 2.45, *p* = .02
    print(format_f_test(5.32, 2, 147, 0.006))  # *F*(2, 147) = 5.32, *p* = .006
    print(format_mean_sd(4.55, 0.65))  # *M* = 4.55, *SD* = 0.65
    print(format_effect_size('d', 0.85, 0.45, 1.25))  # Cohen's *d* = 0.85, 95% CI [0.45, 1.25]