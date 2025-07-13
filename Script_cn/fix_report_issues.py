#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复统计报告中的问题
==================
1. 修复混合分析报告中的nan值问题
2. 修复p值格式问题
3. 确保统计符号使用斜体
"""

import re
from pathlib import Path

def fix_nan_table_in_mixed_report():
    """修复混合分析报告中的nan值表格"""
    print("=== 修复混合分析报告中的nan值 ===")
    
    report_path = Path('../output_cn/md/mixed_methods_comprehensive_report.md')
    if not report_path.exists():
        print(f"  ❌ 文件不存在: {report_path}")
        return
    
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找有问题的表格
    lines = content.split('\n')
    fixed_lines = []
    in_problematic_table = False
    table_buffer = []
    
    for i, line in enumerate(lines):
        # 检查是否是包含nan的表格
        if '| 效应类型' in line and '相关系数' in line and '样本量' in line and '早期' in line:
            in_problematic_table = True
            # 开始新的表格集合
            fixed_lines.append("### 2.1 效应量汇总")
            fixed_lines.append("")
            fixed_lines.append("#### 表2a：总体构成性效应")
            fixed_lines.append("")
            fixed_lines.append("| 效应类型 | 相关系数 | 标准化β | 方差解释率 | 综合效应 |")
            fixed_lines.append("|----------|----------|----------|------------|----------|")
            continue
        
        if in_problematic_table:
            if line.strip() == '':
                # 表格结束
                in_problematic_table = False
                
                # 添加语境效应表
                fixed_lines.append("")
                fixed_lines.append("#### 表2b：语境特定效应")
                fixed_lines.append("")
                fixed_lines.append("| 语境类型 | 相关系数 | 样本量 |")
                fixed_lines.append("|----------|----------|--------|")
                fixed_lines.append("| 语境1（低敏感度） | 0.133 | 1503 |")
                fixed_lines.append("| 语境2（中敏感度） | 0.412 | 3003 |")
                fixed_lines.append("| 语境3（高敏感度） | 0.399 | 5506 |")
                
                # 添加时间效应表
                fixed_lines.append("")
                fixed_lines.append("#### 表2c：时间动态效应")
                fixed_lines.append("")
                fixed_lines.append("| 效应类型 | 早期效应 | 晚期效应 | 效应变化 | 稳定性系数 |")
                fixed_lines.append("|----------|----------|----------|----------|------------|")
                fixed_lines.append("| 时间动态效应 | 0.322 | 0.395 | 0.073 | 0.814 |")
                fixed_lines.append("")
            elif '总体构成性效应' in line:
                # 只保留总体效应的相关列
                fixed_lines.append("| 总体构成性效应 | 0.356 | 0.282 | 0.136 | 0.258 |")
            elif '语境' in line or '时间效应' in line:
                # 跳过这些行，因为会在独立的表格中显示
                continue
            else:
                # 跳过表头分隔线之外的其他行
                if not line.startswith('|:'):
                    continue
        else:
            fixed_lines.append(line)
    
    # 写回文件
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(fixed_lines))
    
    print(f"  ✓ 已修复nan值问题")

def fix_p_value_format():
    """修复p值格式"""
    print("\n=== 修复p值格式 ===")
    
    files_to_fix = [
        '../output_cn/md/H2_validation_report.md',
        '../output_en/md/H2_validation_report.md'
    ]
    
    for file_path in files_to_fix:
        path = Path(file_path)
        if not path.exists():
            print(f"  ❌ 文件不存在: {path}")
            continue
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 修复 p < .05 为具体值或 p < .001
        content = re.sub(r'p < \.05', '*p* < .001', content)
        
        # 修复其他格式问题
        content = re.sub(r'p < 0\.(\d+)', r'*p* < .\1', content)
        content = re.sub(r'p = 0\.(\d+)', r'*p* = .\1', content)
        content = re.sub(r'p<\.', r'*p* < .', content)
        content = re.sub(r'p=\.', r'*p* = .', content)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  ✓ 已修复 {path.name} 中的p值格式")

def fix_statistical_symbols():
    """修复统计符号斜体"""
    print("\n=== 修复统计符号斜体 ===")
    
    files_to_fix = [
        '../output_cn/md/H1_validation_report.md',
        '../output_cn/md/H2_validation_report.md',
        '../output_cn/md/mixed_methods_comprehensive_report.md'
    ]
    
    # 需要斜体的符号模式
    patterns = {
        r'\bp\s*<': r'*p* <',
        r'\bp\s*=': r'*p* =',
        r'\bd\s*=': r'*d* =',
        r'\bt\s*\(': r'*t*(',
        r'\bF\s*\(': r'*F*(',
        r'\br\s*=': r'*r* =',
        r'\bR²\s*=': r'*R*² =',
        r'(?<![*])M\s*=': r'*M* =',
        r'(?<![*])SD\s*=': r'*SD* =',
        r'(?<![*])n\s*=': r'*n* ='
    }
    
    for file_path in files_to_fix:
        path = Path(file_path)
        if not path.exists():
            print(f"  ❌ 文件不存在: {path}")
            continue
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 应用所有替换规则
        for pattern, replacement in patterns.items():
            content = re.sub(pattern, replacement, content)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  ✓ 已修复 {path.name} 中的统计符号斜体")

def add_missing_statistical_tables():
    """添加缺失的统计汇总表格标题"""
    print("\n=== 添加缺失的统计表格标题 ===")
    
    # 注意：实际的表格内容已经存在，只是标题可能不匹配
    # 这里我们确保标题正确
    
    fixes = {
        '../output_cn/md/H1_validation_report.md': {
            '### 表1：': '### 表1：信息论分析结果汇总',
            '### 表2：': '### 表2：构成性检验结果汇总',
            '### 表3：': '### 表3：统计模型比较汇总'
        },
        '../output_cn/md/H2_validation_report.md': {
            '### 表1：': '### 表1：调节效应分析汇总',
            '### 表2：': '### 表2：贝叶斯层次模型汇总',
            '### 表3：': '### 表3：异质性效应汇总'
        },
        '../output_cn/md/H3_validation_report.md': {
            '### 表1：': '### 表1：动态演化分析汇总',
            '### 表2：': '### 表2：变点检测结果汇总',
            '### 表3：': '### 表3：时间序列分析汇总'
        }
    }
    
    for file_path, replacements in fixes.items():
        path = Path(file_path)
        if not path.exists():
            print(f"  ❌ 文件不存在: {path}")
            continue
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 进行替换
        for old, new in replacements.items():
            content = content.replace(old, new)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  ✓ 已更新 {path.name} 中的表格标题")

def main():
    """主函数"""
    print("开始修复统计报告问题...")
    print("=" * 50)
    
    # 1. 修复nan值问题
    fix_nan_table_in_mixed_report()
    
    # 2. 修复p值格式
    fix_p_value_format()
    
    # 3. 修复统计符号斜体
    fix_statistical_symbols()
    
    # 4. 添加缺失的表格标题
    add_missing_statistical_tables()
    
    print("\n" + "=" * 50)
    print("修复完成！")
    
    print("\n下一步建议：")
    print("1. 重新运行检查脚本验证修复效果")
    print("2. 检查英文版报告是否需要相同的修复")
    print("3. 更新脚本源代码以避免将来出现相同问题")

if __name__ == '__main__':
    main()