# step1_data_validation.py
# 第一步：数据验证与结构分析

import os
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
from datetime import datetime
from collections import defaultdict
import json

class CorpusValidator:
    """XML语料库验证器"""
    
    def __init__(self, corpus_path):
        self.corpus_path = Path(corpus_path)
        self.validation_results = {
            'files_found': [],
            'total_units': 0,
            'field_coverage': defaultdict(int),
            'missing_fields': defaultdict(list),
            'data_quality': {},
            'errors': []
        }
        
    def validate_all_files(self):
        """验证所有XML文件"""
        print("开始验证XML语料库...")
        
        # 查找所有XML文件
        xml_files = list(self.corpus_path.glob("*.xml"))
        self.validation_results['files_found'] = [f.name for f in xml_files]
        
        print(f"找到 {len(xml_files)} 个XML文件")
        
        all_units = []
        
        for xml_file in xml_files:
            print(f"\n正在验证: {xml_file.name}")
            units = self.validate_file(xml_file)
            all_units.extend(units)
            
        self.validation_results['total_units'] = len(all_units)
        
        # 计算数据质量统计
        self.calculate_data_quality(all_units)
        
        # 生成验证报告
        self.generate_validation_report()
        
        return all_units
        
    def validate_file(self, file_path):
        """验证单个XML文件"""
        units = []
        
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # 查找所有unit节点
            for unit in root.findall('.//unit'):
                unit_data = self.validate_unit(unit, file_path.name)
                if unit_data:
                    units.append(unit_data)
                    
        except Exception as e:
            self.validation_results['errors'].append({
                'file': file_path.name,
                'error': str(e)
            })
            print(f"  错误: {e}")
            
        print(f"  验证了 {len(units)} 个问答单元")
        return units
        
    def validate_unit(self, unit_elem, filename):
        """验证单个问答单元"""
        unit_id = unit_elem.get('id', 'MISSING')
        
        # 必需字段定义
        required_fields = {
            # 基础元数据
            'unit_id': lambda: unit_elem.get('id'),
            'date': lambda: unit_elem.get('date'),
            'sequence': lambda: unit_elem.get('sequence'),
            
            # metadata部分
            'spokesperson': lambda: unit_elem.findtext('.//spokesperson'),
            'media_source': lambda: unit_elem.findtext('.//media_source'),
            'media_culture': lambda: unit_elem.find('.//media_source').get('culture') if unit_elem.find('.//media_source') is not None else None,
            'topic': lambda: unit_elem.findtext('.//topic'),
            'topic_category': lambda: unit_elem.find('.//topic').get('category') if unit_elem.find('.//topic') is not None else None,
            'topic_sensitivity': lambda: unit_elem.find('.//topic').get('sensitivity') if unit_elem.find('.//topic') is not None else None,
            
            # 问答文本
            'question_text': lambda: unit_elem.findtext('.//question/text'),
            'response_text': lambda: unit_elem.findtext('.//response/text'),
            
            # 认知标注
            'cognitive_adaptation_success': lambda: unit_elem.findtext('.//cognitive_adaptation_success'),
            'cognitive_load_index': lambda: unit_elem.findtext('.//cognitive_load_index/value'),
            'system_coupling_strength': lambda: self._get_system_coupling_strength(unit_elem),
            
            # 语用策略
            'pragmatic_strategies_count': lambda: len(unit_elem.findall('.//pragmatic_strategies/strategy')),
            'relationship_building_score': lambda: unit_elem.findtext('.//relationship_building/composite_score'),
            
            # 文化图式
            'cultural_schemas_count': lambda: len(unit_elem.findall('.//cultural_schemas/schema')),
            'schema_density_total': lambda: unit_elem.findtext('.//schema_density/total'),
            
            # 数字媒体适应
            'digital_intensity': lambda: unit_elem.findtext('.//digital_intensity/overall_intensity'),
            'ded_schemas_count': lambda: len(unit_elem.findall('.//cultural_schemas/schema[@type="DED"]')),
        }
        
        unit_data = {'filename': filename, 'unit_id': unit_id}
        
        # 验证每个字段
        for field_name, field_getter in required_fields.items():
            try:
                value = field_getter()
                if value is not None:
                    unit_data[field_name] = value
                    self.validation_results['field_coverage'][field_name] += 1
                else:
                    self.validation_results['missing_fields'][field_name].append(unit_id)
            except Exception as e:
                self.validation_results['missing_fields'][field_name].append(unit_id)
                
        return unit_data
        
    def _get_system_coupling_strength(self, unit_elem):
        """获取系统耦合强度"""
        # 尝试多个可能的路径
        coupling = unit_elem.find('.//system_coupling/cognitive_pragmatic')
        if coupling is not None:
            return coupling.get('strength')
        return None
        
    def calculate_data_quality(self, all_units):
        """计算数据质量统计"""
        total_units = len(all_units)
        
        if total_units == 0:
            return
            
        # 计算每个字段的覆盖率
        field_coverage_rates = {}
        for field, count in self.validation_results['field_coverage'].items():
            coverage_rate = (count / total_units) * 100
            field_coverage_rates[field] = f"{coverage_rate:.2f}%"
            
        self.validation_results['data_quality'] = {
            'total_units': total_units,
            'field_coverage_rates': field_coverage_rates,
            'average_coverage': f"{sum(self.validation_results['field_coverage'].values()) / (len(self.validation_results['field_coverage']) * total_units) * 100:.2f}%"
        }
        
    def generate_validation_report(self):
        """生成验证报告"""
        report = {
            '1_总体情况': {
                '文件数量': len(self.validation_results['files_found']),
                '文件列表': self.validation_results['files_found'],
                '问答单元总数': self.validation_results['total_units'],
                '错误数量': len(self.validation_results['errors'])
            },
            '2_字段覆盖率': self.validation_results['data_quality'].get('field_coverage_rates', {}),
            '3_缺失情况': {
                field: f"{len(units)}个单元缺失" 
                for field, units in self.validation_results['missing_fields'].items()
                if len(units) > 0
            },
            '4_数据质量': {
                '平均字段覆盖率': self.validation_results['data_quality'].get('average_coverage', 'N/A'),
                '关键字段完整性': {
                    'DED标记覆盖率': self.validation_results['data_quality'].get('field_coverage_rates', {}).get('ded_schemas_count', 'N/A'),
                    '认知适应度覆盖率': self.validation_results['data_quality'].get('field_coverage_rates', {}).get('cognitive_adaptation_success', 'N/A'),
                    '系统耦合度覆盖率': self.validation_results['data_quality'].get('field_coverage_rates', {}).get('system_coupling_strength', 'N/A')
                }
            },
            '5_错误详情': self.validation_results['errors']
        }
        
        # 保存报告
        output_dir = Path('../output_cn/data')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'validation_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            
        # 打印报告摘要
        print("\n" + "="*60)
        print("验证报告摘要")
        print("="*60)
        print(f"文件数量: {report['1_总体情况']['文件数量']}")
        print(f"问答单元总数: {report['1_总体情况']['问答单元总数']}")
        print(f"平均字段覆盖率: {report['4_数据质量']['平均字段覆盖率']}")
        print(f"DED标记覆盖率: {report['4_数据质量']['关键字段完整性']['DED标记覆盖率']}")
        print(f"\n详细报告已保存至: {output_dir / 'validation_report.json'}")
        
        return report

def main():
    """主函数"""
    # 设置语料库路径
    corpus_path = r"G:\Project\Figure\Corpus"
    
    # 创建验证器
    validator = CorpusValidator(corpus_path)
    
    # 执行验证
    all_units = validator.validate_all_files()
    
    # 判断是否满足后续分析要求
    print("\n" + "="*60)
    print("数据完整性判定")
    print("="*60)
    
    if validator.validation_results['total_units'] > 0:
        avg_coverage = float(validator.validation_results['data_quality']['average_coverage'].strip('%'))
        
        if avg_coverage >= 80:
            print("✓ 数据质量良好，可以进行后续分析")
        elif avg_coverage >= 60:
            print("! 数据质量中等，建议检查缺失字段但可以继续")
        else:
            print("✗ 数据质量较差，建议先补充缺失数据")
            
        # 检查关键字段
        critical_fields = ['ded_schemas_count', 'cognitive_adaptation_success', 'system_coupling_strength']
        print("\n关键字段检查:")
        for field in critical_fields:
            coverage = validator.validation_results['data_quality']['field_coverage_rates'].get(field, '0%')
            print(f"  - {field}: {coverage}")
    else:
        print("✗ 未找到有效数据，请检查语料库路径")

if __name__ == "__main__":
    main()