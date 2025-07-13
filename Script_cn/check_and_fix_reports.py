#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查和修复统计报告问题
====================
1. 检查混合分析报告中的nan值问题
2. 验证所有脚本的p值格式是否符合APA标准
3. 确认所有报告包含完整的统计汇总表格
"""

import os
import re
from pathlib import Path

def check_nan_values_in_reports():
    """检查报告中的nan值"""
    print("=== 检查报告中的nan值 ===")
    
    report_files = [
        'output_cn/md/mixed_methods_comprehensive_report.md',
        'output_en/md/mixed_methods_comprehensive_report.md'
    ]
    
    for report_file in report_files:
        file_path = Path('..') / report_file
        if file_path.exists():
            print(f"\n检查文件: {report_file}")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 查找nan值
            nan_count = len(re.findall(r'\bnan\b', content, re.IGNORECASE))
            if nan_count > 0:
                print(f"  ⚠️ 发现 {nan_count} 个nan值")
                
                # 查找包含nan的表格行
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'nan' in line.lower() and '|' in line:
                        print(f"    行 {i+1}: {line[:100]}...")
            else:
                print(f"  ✓ 未发现nan值")
        else:
            print(f"  ❌ 文件不存在: {file_path}")

def check_p_value_format():
    """检查p值格式是否符合APA标准"""
    print("\n=== 检查p值格式 ===")
    
    # 正确的p值格式示例
    correct_patterns = [
        r'\*p\* < \.001',
        r'\*p\* = \.\d{2,3}',
        r'p < \.001',  # 在代码中
        r'p = \.\d{2,3}'  # 在代码中
    ]
    
    # 错误的p值格式
    incorrect_patterns = [
        r'p < 0\.001',  # 应该去掉前导0
        r'p = 0\.\d+',  # 应该去掉前导0
        r'p<\.001',     # 缺少空格
        r'p=\.\d+',     # 缺少空格
        r'P < \.001',   # P应该小写
        r'p < \.05',    # 应该报告精确值
        r'p > \.05'     # 应该报告精确值
    ]
    
    report_files = [
        'output_cn/md/H1_validation_report.md',
        'output_cn/md/H2_validation_report.md',
        'output_cn/md/H3_validation_report.md',
        'output_cn/md/mixed_methods_comprehensive_report.md',
        'output_en/md/H1_validation_report.md',
        'output_en/md/H2_validation_report.md',
        'output_en/md/H3_validation_report.md',
        'output_en/md/mixed_methods_comprehensive_report.md'
    ]
    
    for report_file in report_files:
        file_path = Path('..') / report_file
        if file_path.exists():
            print(f"\n检查文件: {report_file}")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查错误格式
            errors_found = False
            for pattern in incorrect_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    errors_found = True
                    print(f"  ⚠️ 发现错误格式 '{pattern}': {len(matches)} 处")
                    for match in matches[:3]:  # 只显示前3个
                        print(f"     - {match}")
            
            if not errors_found:
                print(f"  ✓ p值格式正确")

def check_statistical_tables():
    """检查统计汇总表格的完整性"""
    print("\n=== 检查统计汇总表格 ===")
    
    required_tables = {
        'H1': ['信息论分析结果汇总', '构成性检验结果汇总', '统计模型比较汇总'],
        'H2': ['调节效应分析汇总', '贝叶斯层次模型汇总', '异质性效应汇总'],
        'H3': ['动态演化分析汇总', '变点检测结果汇总', '时间序列分析汇总'],
        'mixed': ['质性模式识别结果汇总', '量化验证统计结果', '三角验证结果汇总']
    }
    
    report_mapping = {
        'H1': ['output_cn/md/H1_validation_report.md', 'output_en/md/H1_validation_report.md'],
        'H2': ['output_cn/md/H2_validation_report.md', 'output_en/md/H2_validation_report.md'],
        'H3': ['output_cn/md/H3_validation_report.md', 'output_en/md/H3_validation_report.md'],
        'mixed': ['output_cn/md/mixed_methods_comprehensive_report.md', 'output_en/md/mixed_methods_comprehensive_report.md']
    }
    
    for report_type, files in report_mapping.items():
        print(f"\n{report_type} 报告:")
        for file_path in files:
            path = Path('..') / file_path
            if path.exists():
                print(f"  检查文件: {file_path}")
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 检查必需的表格
                missing_tables = []
                for table_name in required_tables[report_type]:
                    # 对于英文版，使用相应的英文表名
                    if 'output_en' in file_path:
                        # 简单的中英文映射
                        en_table_name = table_name.replace('汇总', 'Summary').replace('结果', 'Results')
                    else:
                        en_table_name = table_name
                    
                    if table_name not in content and en_table_name not in content:
                        missing_tables.append(table_name)
                
                if missing_tables:
                    print(f"    ⚠️ 缺少表格: {', '.join(missing_tables)}")
                else:
                    print(f"    ✓ 包含所有必需表格")

def check_statistical_symbols():
    """检查统计符号是否使用斜体"""
    print("\n=== 检查统计符号斜体 ===")
    
    # 应该使用斜体的统计符号
    italic_required = ['M', 'SD', 'SE', 'N', 'n', 't', 'F', 'p', 'r', 'R', 'd', 'df']
    
    report_files = [
        'output_cn/md/H1_validation_report.md',
        'output_cn/md/H2_validation_report.md', 
        'output_cn/md/H3_validation_report.md',
        'output_cn/md/mixed_methods_comprehensive_report.md'
    ]
    
    for report_file in report_files:
        file_path = Path('..') / report_file
        if file_path.exists():
            print(f"\n检查文件: {report_file}")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查是否使用了斜体标记
            non_italic_found = False
            for symbol in italic_required:
                # 查找未使用斜体的情况（排除代码块和已经斜体的）
                pattern = rf'(?<![*`])\b{symbol}\b(?![*`])\s*[=<>]'
                matches = re.findall(pattern, content)
                if matches:
                    non_italic_found = True
                    print(f"  ⚠️ '{symbol}' 未使用斜体: {len(matches)} 处")
            
            if not non_italic_found:
                print(f"  ✓ 统计符号格式正确")

def main():
    """主函数"""
    print("开始检查统计报告问题...")
    print("=" * 50)
    
    # 1. 检查nan值
    check_nan_values_in_reports()
    
    # 2. 检查p值格式
    check_p_value_format()
    
    # 3. 检查统计表格
    check_statistical_tables()
    
    # 4. 检查统计符号
    check_statistical_symbols()
    
    print("\n" + "=" * 50)
    print("检查完成！")
    
    print("\n建议修复方案:")
    print("1. 对于nan值问题：")
    print("   - 修改混合分析脚本，将效应量汇总表拆分为独立的表格")
    print("   - 总体效应、语境效应、时间效应应分别显示")
    print("\n2. 对于p值格式问题：")
    print("   - 确保所有p值使用format_p_value函数格式化")
    print("   - 去掉前导零，使用正确的间距")
    print("\n3. 对于缺失的统计表格：")
    print("   - 检查相应的_add_statistical_tables方法")
    print("   - 确保所有必需的表格都被生成")
    print("\n4. 对于统计符号斜体：")
    print("   - 在Markdown中使用*符号*格式")
    print("   - 或在生成时就加入斜体标记")

if __name__ == '__main__':
    main()