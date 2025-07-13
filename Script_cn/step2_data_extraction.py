# step2_data_extraction.py
# 第二步：数据提取与预处理

import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
from datetime import datetime
import numpy as np
import re
from collections import defaultdict
import json

class DataExtractor:
    """XML数据提取器"""
    
    def __init__(self, corpus_path):
        self.corpus_path = Path(corpus_path)
        self.data_records = []
        self.extraction_stats = {
            'total_units': 0,
            'ded_elements_found': 0,
            'avg_ded_per_unit': 0,
            'media_distribution': defaultdict(int),
            'topic_distribution': defaultdict(int),
            'sensitivity_distribution': defaultdict(int)
        }
        
    def extract_all_files(self):
        """从所有XML文件提取数据"""
        print("开始提取数据...")
        
        xml_files = sorted(self.corpus_path.glob("*.xml"))
        
        for xml_file in xml_files:
            print(f"\n处理文件: {xml_file.name}")
            self.extract_file(xml_file)
            
        # 计算统计信息
        self.calculate_statistics()
        
        # 检查是否有数据
        if not self.data_records:
            print("警告：没有提取到任何数据！")
            return pd.DataFrame()
        
        # 创建DataFrame
        df = pd.DataFrame(self.data_records)
        print(f"\n成功创建DataFrame，包含 {len(df)} 条记录")
        
        # 数据预处理
        df = self.preprocess_data(df)
        
        # 保存数据
        self.save_data(df)
        
        return df
        
    def extract_file(self, file_path):
        """从单个XML文件提取数据"""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            units_extracted = 0
            
            for unit in root.findall('.//unit'):
                unit_data = self.extract_unit(unit)
                if unit_data:
                    self.data_records.append(unit_data)
                    units_extracted += 1
                    
            print(f"  提取了 {units_extracted} 个单元")
            
        except Exception as e:
            print(f"  解析文件 {file_path} 时出错: {e}")
        
    def extract_unit(self, unit_elem):
        """提取单个问答单元的所有数据"""
        try:
            # 基础信息
            unit_data = {
                'unit_id': unit_elem.get('id'),
                'date': unit_elem.get('date'),
                'sequence': int(unit_elem.get('sequence', 0)),
                
                # metadata
                'spokesperson': unit_elem.findtext('.//spokesperson'),
                'media_source': unit_elem.findtext('.//media_source'),
                'media_culture': unit_elem.find('.//media_source').get('culture') if unit_elem.find('.//media_source') is not None else None,
                'topic': unit_elem.findtext('.//topic'),
                'topic_category': unit_elem.find('.//topic').get('category') if unit_elem.find('.//topic') is not None else None,
                'topic_sensitivity': unit_elem.find('.//topic').get('sensitivity') if unit_elem.find('.//topic') is not None else None,
                
                # 文本内容
                'question_text': unit_elem.findtext('.//question/text', ''),
                'response_text': unit_elem.findtext('.//response/text', ''),
                
                # 文本长度
                'question_length': len(unit_elem.findtext('.//question/text', '')),
                'response_length': len(unit_elem.findtext('.//response/text', '')),
            }
            
            # 提取认知指标
            cognitive_data = self.extract_cognitive_indicators(unit_elem)
            unit_data.update(cognitive_data)
            
            # 提取语用策略
            pragmatic_data = self.extract_pragmatic_strategies(unit_elem)
            unit_data.update(pragmatic_data)
            
            # 提取文化图式
            cultural_data = self.extract_cultural_schemas(unit_elem)
            unit_data.update(cultural_data)
            
            # 提取数字媒体适应
            digital_data = self.extract_digital_adaptation(unit_elem)
            unit_data.update(digital_data)
            
            # 提取DED（数字符号资源）
            ded_data = self.extract_ded_elements(unit_elem)
            unit_data.update(ded_data)
            
            # 统计信息更新
            self.extraction_stats['total_units'] += 1
            if unit_data['media_culture']:
                self.extraction_stats['media_distribution'][unit_data['media_culture']] += 1
            if unit_data['topic_category']:
                self.extraction_stats['topic_distribution'][unit_data['topic_category']] += 1
            if unit_data['topic_sensitivity']:
                self.extraction_stats['sensitivity_distribution'][unit_data['topic_sensitivity']] += 1
            
            return unit_data
            
        except Exception as e:
            print(f"  提取单元 {unit_elem.get('id')} 时出错: {e}")
            return None
            
    def extract_cognitive_indicators(self, unit_elem):
        """提取认知功能指标"""
        cognitive_data = {}
        
        # 认知适应成功度
        cas_elem = unit_elem.find('.//cognitive_adaptation_success')
        cognitive_data['cognitive_adaptation_success'] = float(cas_elem.text) if cas_elem is not None and cas_elem.text else 0
        
        # 认知负荷指数
        cli_elem = unit_elem.find('.//cognitive_load_index/value')
        cognitive_data['cognitive_load_index'] = float(cli_elem.text) if cli_elem is not None and cli_elem.text else 0
        
        # 策略转换
        stc_elem = unit_elem.find('.//strategy_transition_count')
        cognitive_data['strategy_transition_count'] = int(stc_elem.text) if stc_elem is not None and stc_elem.text else 0
        
        ts_elem = unit_elem.find('.//transition_smoothness')
        cognitive_data['transition_smoothness'] = float(ts_elem.text) if ts_elem is not None and ts_elem.text else 0
        
        # 系统耦合
        sc_elem = unit_elem.find('.//system_coupling/cognitive_pragmatic')
        if sc_elem is not None:
            cognitive_data['system_coupling_strength'] = float(sc_elem.get('strength', 0))
        else:
            cognitive_data['system_coupling_strength'] = 0
            
        # 耦合动态
        cognitive_data['coupling_phase'] = unit_elem.findtext('.//coupling_dynamics/phase', '')
        cognitive_data['coupling_stability'] = unit_elem.findtext('.//coupling_dynamics/coupling_stability', '')
        cognitive_data['adaptation_speed'] = unit_elem.findtext('.//coupling_dynamics/adaptation_speed', '')
        
        # 整体评估
        cr_elem = unit_elem.find('.//overall_assessment/cultural_resonance')
        cognitive_data['cultural_resonance'] = float(cr_elem.text) if cr_elem is not None and cr_elem.text else 0
        
        se_elem = unit_elem.find('.//overall_assessment/strategic_effectiveness')
        cognitive_data['strategic_effectiveness'] = float(se_elem.text) if se_elem is not None and se_elem.text else 0
        
        return cognitive_data
        
    def extract_pragmatic_strategies(self, unit_elem):
        """提取语用策略数据"""
        pragmatic_data = {}
        
        strategies = unit_elem.findall('.//pragmatic_strategies/strategy')
        
        # 策略类型统计
        strategy_types = [s.get('type', '') for s in strategies]
        strategy_effectiveness = [s.get('effectiveness', '') for s in strategies]
        
        pragmatic_data['pragmatic_strategies_count'] = len(strategies)
        pragmatic_data['strategy_types'] = '|'.join(strategy_types) if strategy_types else ''
        pragmatic_data['strategy_effectiveness_scores'] = '|'.join(strategy_effectiveness) if strategy_effectiveness else ''
        
        # 策略模式
        pattern_elem = unit_elem.find('.//strategy_pattern/pattern_type')
        pragmatic_data['dominant_strategy_pattern'] = pattern_elem.text if pattern_elem is not None else ''
        
        # 关系建设指标
        rb_elem = unit_elem.find('.//relationship_building')
        if rb_elem is not None:
            pragmatic_data['relationship_building_frequency'] = float(rb_elem.findtext('usage_frequency', 0))
            pragmatic_data['relationship_building_intensity'] = float(rb_elem.findtext('usage_intensity', 0))
            pragmatic_data['strategy_diversity_index'] = float(rb_elem.findtext('diversity_index', 0))
            pragmatic_data['strategy_density'] = float(rb_elem.findtext('strategy_density', 0))
            pragmatic_data['contextual_adaptation'] = float(rb_elem.findtext('contextual_adaptation', 0))
            pragmatic_data['relationship_building_score'] = float(rb_elem.findtext('composite_score', 0))
        else:
            pragmatic_data.update({
                'relationship_building_frequency': 0,
                'relationship_building_intensity': 0,
                'strategy_diversity_index': 0,
                'strategy_density': 0,
                'contextual_adaptation': 0,
                'relationship_building_score': 0
            })
            
        # 立场策略
        pr_elem = unit_elem.find('.//position_strategies/principle_restatement')
        pragmatic_data['principle_restatement_freq'] = int(pr_elem.get('frequency', 0)) if pr_elem is not None else 0
        
        ir_elem = unit_elem.find('.//position_strategies/information_restriction')
        pragmatic_data['information_restriction_freq'] = int(ir_elem.get('frequency', 0)) if ir_elem is not None else 0
        
        return pragmatic_data
        
    def extract_cultural_schemas(self, unit_elem):
        """提取文化图式数据"""
        cultural_data = {}
        
        schemas = unit_elem.findall('.//cultural_schemas/schema')
        
        # 分类统计
        traditional_schemas = [s for s in schemas if s.get('type') not in ['DED', 'MTE']]
        innovative_schemas = [s for s in schemas if s.get('type') == 'MTE']
        
        cultural_data['cultural_schemas_total'] = len(schemas)
        cultural_data['traditional_schemas_count'] = len(traditional_schemas)
        cultural_data['innovative_schemas_count'] = len(innovative_schemas)
        
        # 图式密度
        density_elem = unit_elem.find('.//schema_density')
        if density_elem is not None:
            cultural_data['schema_density_traditional'] = float(density_elem.findtext('traditional', 0))
            cultural_data['schema_density_innovative'] = float(density_elem.findtext('innovative', 0))
            cultural_data['schema_density_total'] = float(density_elem.findtext('total', 0))
        else:
            cultural_data.update({
                'schema_density_traditional': 0,
                'schema_density_innovative': 0,
                'schema_density_total': 0
            })
            
        # 桥接效果
        be_elem = unit_elem.find('.//bridging_effectiveness')
        if be_elem is not None:
            cultural_data['bridging_clarity'] = float(be_elem.findtext('clarity', 0))
            cultural_data['cultural_accessibility'] = float(be_elem.findtext('cultural_accessibility', 0))
        else:
            cultural_data['bridging_clarity'] = 0
            cultural_data['cultural_accessibility'] = 0
            
        # 激活强度统计
        intensities = []
        for s in schemas:
            ai_elem = s.find('activation_intensity')
            if ai_elem is not None and ai_elem.text:
                intensities.append(float(ai_elem.text))
                
        cultural_data['avg_activation_intensity'] = np.mean(intensities) if intensities else 0
        
        return cultural_data
        
    def extract_digital_adaptation(self, unit_elem):
        """提取数字媒体适应数据"""
        digital_data = {}
        
        # 声咬设计
        soundbites = unit_elem.findall('.//soundbite_design')
        digital_data['soundbite_count'] = len(soundbites)
        
        memorabilities = []
        viral_high = 0
        for s in soundbites:
            mem = s.get('memorability')
            if mem:
                memorabilities.append(float(mem))
            vp_elem = s.find('viral_potential')
            if vp_elem is not None and vp_elem.text == 'high':
                viral_high += 1
                
        digital_data['avg_memorability'] = np.mean(memorabilities) if memorabilities else 0
        digital_data['viral_potential_high'] = viral_high
        
        # 数字强度
        di_elem = unit_elem.find('.//digital_intensity')
        if di_elem is not None:
            digital_data['social_media_metaphor_count'] = int(di_elem.findtext('social_media_metaphor_count', 0))
            digital_data['platform_terminology_count'] = int(di_elem.findtext('platform_terminology_count', 0))
            digital_data['digital_intensity_overall'] = float(di_elem.findtext('overall_intensity', 0))
        else:
            digital_data.update({
                'social_media_metaphor_count': 0,
                'platform_terminology_count': 0,
                'digital_intensity_overall': 0
            })
            
        # 情感触发
        triggers = unit_elem.findall('.//emotional_triggers/trigger')
        digital_data['emotional_triggers_count'] = len(triggers)
        
        # Meme潜力
        mp_elem = unit_elem.find('.//meme_potential')
        digital_data['meme_potential_score'] = float(mp_elem.get('score', 0)) if mp_elem is not None else 0
        
        return digital_data
        
    def extract_ded_elements(self, unit_elem):
        """提取DED（数字符号资源）元素"""
        ded_data = {}
        
        ded_schemas = unit_elem.findall('.//cultural_schemas/schema[@type="DED"]')
        
        self.extraction_stats['ded_elements_found'] += len(ded_schemas)
        
        # 提取DED详细信息
        ded_functions = []
        ded_depths = []
        ded_intensities = []
        ded_markers = []
        
        for ded in ded_schemas:
            # 功能
            func = ded.get('function', '')
            if func:
                ded_functions.append(func)
                
            # 深度
            marker_elem = ded.find('marker')
            if marker_elem is not None:
                # 标记文本
                marker_text = marker_elem.text if marker_elem.text else ''
                ded_markers.append(marker_text)
                
                # 深度
                depth = marker_elem.get('depth', '')
                if depth:
                    depth_map = {'surface': 1, 'middle': 2, 'deep': 3}
                    if depth in depth_map:
                        ded_depths.append(depth_map[depth])
                    elif depth.isdigit():
                        ded_depths.append(int(depth))
                    else:
                        ded_depths.append(2)
                else:
                    ded_depths.append(2)
            else:
                ded_depths.append(2)
                
            # 激活强度
            intensity = ded.get('activation_intensity')
            if intensity:
                ded_intensities.append(float(intensity))
            else:
                ded_intensities.append(0)
                
        ded_data['ded_count'] = len(ded_schemas)
        ded_data['ded_functions'] = '|'.join(ded_functions) if ded_functions else ''
        ded_data['ded_avg_depth'] = np.mean(ded_depths) if ded_depths else 0
        ded_data['ded_avg_intensity'] = np.mean(ded_intensities) if ded_intensities else 0
        ded_data['ded_max_intensity'] = max(ded_intensities) if ded_intensities else 0
        ded_data['ded_markers'] = '|'.join(ded_markers) if ded_markers else ''
        
        return ded_data
        
    def preprocess_data(self, df):
        """数据预处理"""
        print("\n执行数据预处理...")
        
        # 检查DataFrame是否为空
        if df.empty:
            print("警告：DataFrame为空，跳过预处理")
            return df
            
        # 1. 日期处理
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['quarter'] = df['date'].dt.to_period('Q')
        df['month'] = df['date'].dt.month
        
        # 2. 类别编码
        # 媒体文化类型
        media_culture_map = {'Western': 0, 'CN': 1, 'Other': 2}
        df['media_culture_code'] = df['media_culture'].map(media_culture_map).fillna(2)
        
        # 话题敏感度
        sensitivity_map = {'low': 1, 'medium': 2, 'high': 3}
        df['sensitivity_code'] = df['topic_sensitivity'].map(sensitivity_map).fillna(2)
        
        # 3. 计算派生指标
        # 文本比例
        df['response_question_ratio'] = df['response_length'] / (df['question_length'] + 1)
        
        # DED密度（每百字）
        df['ded_density'] = (df['ded_count'] / (df['response_length'] + 1)) * 100
        
        # 策略多样性
        df['strategy_diversity'] = df['strategy_types'].apply(
            lambda x: len(set(x.split('|'))) if pd.notna(x) and x else 0
        )
        
        # 4. 处理缺失值
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        # 5. 排序
        df = df.sort_values(['date', 'sequence']).reset_index(drop=True)
        
        print(f"预处理完成，数据形状: {df.shape}")
        
        return df
        
    def calculate_statistics(self):
        """计算提取统计信息"""
        if self.extraction_stats['total_units'] > 0:
            self.extraction_stats['avg_ded_per_unit'] = (
                self.extraction_stats['ded_elements_found'] / 
                self.extraction_stats['total_units']
            )
            
    def save_data(self, df):
        """保存提取的数据"""
        # 保存为CSV
        output_path = Path('../output_cn/data')
        output_path.mkdir(parents=True, exist_ok=True)
        
        csv_file = output_path / 'extracted_data.csv'
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"\n数据已保存至: {csv_file}")
        
        # 保存统计信息
        stats_file = output_path / 'extraction_stats.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.extraction_stats, f, ensure_ascii=False, indent=2)
            
        # 打印统计摘要
        print("\n" + "="*60)
        print("数据提取统计")
        print("="*60)
        print(f"总单元数: {self.extraction_stats['total_units']}")
        print(f"DED元素总数: {self.extraction_stats['ded_elements_found']}")
        print(f"平均每单元DED数: {self.extraction_stats['avg_ded_per_unit']:.2f}")
        print(f"\n媒体分布:")
        for media, count in self.extraction_stats['media_distribution'].items():
            if self.extraction_stats['total_units'] > 0:
                percentage = count/self.extraction_stats['total_units']*100
                print(f"  {media}: {count} ({percentage:.1f}%)")
        print(f"\n话题敏感度分布:")
        for sensitivity, count in self.extraction_stats['sensitivity_distribution'].items():
            if self.extraction_stats['total_units'] > 0:
                percentage = count/self.extraction_stats['total_units']*100
                print(f"  {sensitivity}: {count} ({percentage:.1f}%)")
            
        # 数据质量检查
        if not df.empty:
            print(f"\n数据质量检查:")
            print(f"  空值比例: {df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100:.2f}%")
            print(f"  数值列数: {len(df.select_dtypes(include=[np.number]).columns)}")
            print(f"  文本列数: {len(df.select_dtypes(include=['object']).columns)}")

def main():
    """主函数"""
    # 设置语料库路径
    corpus_path = r"G:\Project\Figure\Corpus"
    
    # 创建提取器
    extractor = DataExtractor(corpus_path)
    
    # 执行数据提取
    df = extractor.extract_all_files()
    
    if not df.empty:
        print(f"\n✓ 数据提取完成！共提取 {len(df)} 条记录")
        print(f"数据包含 {df.shape[1]} 个特征")
        
        # 保存数据
        output_dir = Path('../output_cn/data')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / 'extracted_data.csv'
        
        # 保存CSV文件，使用utf-8-sig编码以支持Excel
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n数据已保存至: {output_file}")
        
        # 显示数据样例
        print("\n数据样例（前5条）:")
        display_cols = ['unit_id', 'date', 'spokesperson', 'ded_count', 'cognitive_adaptation_success']
        # 确保显示列存在
        available_cols = [col for col in display_cols if col in df.columns]
        if available_cols:
            print(df[available_cols].head())
    else:
        print("\n✗ 数据提取失败，请检查XML文件格式")

if __name__ == "__main__":
    main()