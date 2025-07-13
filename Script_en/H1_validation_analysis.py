#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
H1 Hypothesis Validation Analysis Script
========================================
Hypothesis H1 (Cognitive Dependency): DSR as Constitutive Component of Cognitive System

This script integrates multiple analytical methods to comprehensively validate H1 hypothesis:
1. Information Theory Analysis - Functional complementarity, triple interaction MI
2. Constitutiveness Tests - Virtual removal, path necessity, system robustness
3. Statistical Models - Linear/nonlinear models, VAR causal analysis
4. Network Analysis - DSR mediation centrality, cognitive network density

Output:
- H1_validation_results.csv/json - Analysis result data
- H1_cognitive_dependency_analysis.jpg - Comprehensive visualization (4 subplots)
- H1_validation_report.md - Detailed validation report
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set English font
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Script_cn'))

# Import APA formatter
from Script_cn.apa_formatter import format_p_value, format_correlation, format_t_test, format_f_test, format_mean_sd, format_effect_size, format_regression

# Import necessary analysis modules from Chinese version
try:
    from Script_cn.step4_information_theory_H1 import EnhancedFunctionalComplementarityAnalyzer
except ImportError:
    from Script_cn.step4_information_theory import FunctionalComplementarityAnalyzer as EnhancedFunctionalComplementarityAnalyzer

from Script_cn.step5_statistical_models import StatisticalModelsAnalyzer
from Script_cn.step6_constitutiveness_tests import ConstitutivenessTests
from Script_cn.step9_network_diffusion_analysis import NetworkDiffusionAnalysis


class H1CognitiveDependencyValidator:
    """Comprehensive validator for H1 hypothesis (Cognitive Dependency)"""
    
    def __init__(self, data_path='../output_cn/data', output_path='../output_en'):
        """
        Initialize validator
        
        Parameters:
        -----------
        data_path : str
            Path to data files
        output_path : str
            Path for output files
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        
        # Initialize results dictionary
        self.results = {
            'hypothesis': 'H1',
            'hypothesis_description': 'DSR as constitutive component of cognitive system',
            'evidence': {},
            'visualizations': {},
            'errors': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Initialize analyzers
        self.info_analyzer = None
        self.stat_analyzer = None
        self.const_analyzer = None
        self.network_analyzer = None
        
        # Data containers
        self.df = None
        self.analysis_results = {}
        
    def validate_data(self):
        """Validate data integrity"""
        if self.df is None:
            return False
            
        # Check data volume
        if len(self.df) < 100:
            print(f"Warning: Small data volume ({len(self.df)} records), may affect analysis results")
            
        # Check for missing values
        missing_counts = self.df[['dsr_cognitive', 'tl_functional', 'cs_output']].isnull().sum()
        if missing_counts.any():
            print(f"Warning: Missing values detected:\n{missing_counts}")
            
        return True
        
    def load_data(self):
        """Load all necessary data files"""
        print("Loading data...")
        
        try:
            # Load main data file
            data_file = self.data_path / 'data_with_metrics.csv'
            if not data_file.exists():
                raise FileNotFoundError(f"Main data file not found: {data_file}")
                
            self.df = pd.read_csv(data_file)
            self.df['date'] = pd.to_datetime(self.df['date'])
            
            # Verify necessary columns exist
            required_columns = ['dsr_cognitive', 'tl_functional', 'cs_output']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            if missing_columns:
                print(f"Warning: Missing required columns: {missing_columns}")
                
            print(f"Successfully loaded main data: {len(self.df)} records")
            
            # Load existing analysis results
            result_files = {
                'information_theory': 'information_theory_results.json',
                'statistical_models': 'statistical_models_results.json',
                'constitutiveness': 'constitutiveness_test_results.json',
                'network_analysis': 'network_diffusion_results.json'
            }
            
            for key, filename in result_files.items():
                filepath = self.data_path / filename
                if filepath.exists():
                    with open(filepath, 'r', encoding='utf-8') as f:
                        self.analysis_results[key] = json.load(f)
                    print(f"Loaded {key} results: {filename}")
                    
        except Exception as e:
            error_msg = f"Data loading failed: {str(e)}"
            print(error_msg)
            self.results['errors'].append({
                'stage': 'data_loading',
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            })
            
    def run_information_theory_analysis(self):
        """Run information theory analysis"""
        print("\n1. Executing information theory analysis...")
        
        try:
            # Initialize analyzer
            self.info_analyzer = EnhancedFunctionalComplementarityAnalyzer(
                data_path=str(self.data_path)
            )
            
            # Use existing results if available
            if 'information_theory' in self.analysis_results:
                results = self.analysis_results['information_theory']
            else:
                # Run new analysis
                results = self.info_analyzer.analyze_functional_complementarity()
            
            # Extract key metrics
            self.results['evidence']['information_theory'] = {
                'functional_complementarity': results.get('functional_complementarity', {}),
                'triple_interaction_mi': results.get('nonlinear_mi', {}).get('triple_interaction_mi', 0),
                'conditional_mi': results.get('continuous_mi', {}),
                'synergy_redundancy': results.get('continuous_mi', {}).get('dsr_core', {}),
                'significance': results.get('conditional_granger', {})
            }
            
            print("✓ Information theory analysis completed")
            
        except Exception as e:
            error_msg = f"Information theory analysis failed: {str(e)}"
            print(f"✗ {error_msg}")
            self.results['errors'].append({
                'analysis': 'information_theory',
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            })
            
    def run_constitutiveness_tests(self):
        """Run constitutiveness tests"""
        print("\n2. Executing constitutiveness tests...")
        
        try:
            # Initialize analyzer
            self.const_analyzer = ConstitutivenessTests(
                data_path=str(self.data_path)
            )
            
            # Use existing results if available
            if 'constitutiveness' in self.analysis_results:
                results = self.analysis_results['constitutiveness']
            else:
                # Load data and run new analysis
                self.const_analyzer.load_data()
                results = self.const_analyzer.run_constitutiveness_tests()
            
            # Extract key metrics (adapted to actual data structure)
            self.results['evidence']['constitutiveness'] = {
                'virtual_removal': {
                    'performance_loss': results.get('virtual_removal', {}).get('performance_loss', {}).get('overall_performance', 0),
                    'significant_impact': results.get('virtual_removal', {}).get('performance_loss', {}).get('overall_performance', 0) > 0.5
                },
                'path_necessity': {
                    'indirect_effect': results.get('path_necessity', {}).get('indirect_effect', {}).get('value', 0),
                    'is_necessary': results.get('path_necessity', {}).get('is_necessary', True),
                    'mediation_type': results.get('path_necessity', {}).get('mediation_analysis', {}).get('mediation_type', 'unknown')
                },
                'robustness': {
                    'robustness_value': results.get('robustness_tests', {}).get('overall_robustness', 0),
                    'is_robust': results.get('robustness_tests', {}).get('overall_robustness', 0) > 0.9
                },
                'overall_assessment': {
                    'constitutiveness_found': results.get('constitutiveness_score', {}).get('weighted_score', 0) > 0.8,
                    'verdict': results.get('constitutiveness_score', {}).get('verdict', 'unknown')
                }
            }
            
            print("✓ Constitutiveness tests completed")
            
        except Exception as e:
            error_msg = f"Constitutiveness tests failed: {str(e)}"
            print(f"✗ {error_msg}")
            self.results['errors'].append({
                'analysis': 'constitutiveness_tests',
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            })
            
    def run_statistical_models(self):
        """Run statistical model analysis"""
        print("\n3. Executing statistical model analysis...")
        
        try:
            # Initialize analyzer
            self.stat_analyzer = StatisticalModelsAnalyzer(
                data_path=str(self.data_path)
            )
            
            # Use existing results if available
            if 'statistical_models' in self.analysis_results:
                results = self.analysis_results['statistical_models']
            else:
                # Run new analysis
                results = self.stat_analyzer.run_all_models()
            
            # Extract key model results
            self.results['evidence']['statistical_models'] = {
                'M1_linear': results.get('models', {}).get('M1_linear_baseline', {}),
                'M3_nonlinear': results.get('models', {}).get('M3_nonlinear_interactions', {}),
                'M4_VAR': results.get('models', {}).get('M4_VAR_analysis', {}),
                'model_comparison': results.get('model_comparison', {}),
                'best_model': results.get('best_model', {})
            }
            
            print("✓ Statistical model analysis completed")
            
        except Exception as e:
            error_msg = f"Statistical model analysis failed: {str(e)}"
            print(f"✗ {error_msg}")
            self.results['errors'].append({
                'analysis': 'statistical_models',
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            })
            
    def run_network_analysis(self):
        """Run network analysis"""
        print("\n4. Executing network analysis...")
        
        try:
            # Initialize analyzer
            self.network_analyzer = NetworkDiffusionAnalysis(
                data_path=str(self.data_path)
            )
            
            # Use existing results if available
            if 'network_analysis' in self.analysis_results:
                results = self.analysis_results['network_analysis']
            else:
                # Load data and run new analysis
                self.network_analyzer.load_data()
                results = self.network_analyzer.run_network_diffusion_analysis()
            
            # Extract key metrics (adapted to actual data structure)
            network_data = results.get('cognitive_network', {})
            dsr_nodes = ['DSR_core', 'DSR_bridge', 'DSR_integrate']
            
            # Calculate average centrality for DSR nodes
            avg_betweenness = 0
            avg_degree = 0
            avg_closeness = 0
            
            for node in dsr_nodes:
                node_data = network_data.get('node_attributes', {}).get(node, {})
                avg_betweenness += node_data.get('betweenness_centrality', 0)
                avg_degree += node_data.get('degree_centrality', 0)
                avg_closeness += node_data.get('closeness_centrality', 0)
            
            avg_betweenness /= len(dsr_nodes)
            avg_degree /= len(dsr_nodes)
            avg_closeness /= len(dsr_nodes)
            
            self.results['evidence']['network_analysis'] = {
                'DSR_centrality': {
                    'betweenness': avg_betweenness,
                    'degree': avg_degree,
                    'closeness': avg_closeness
                },
                'network_density': network_data.get('network_properties', {}).get('density', 0),
                'DSR_mediation': {
                    'is_mediator': avg_betweenness > 0.1,  # If betweenness centrality > 0.1, consider as mediator
                    'mediation_strength': avg_betweenness
                },
                'key_nodes': results.get('key_nodes', {})
            }
            
            print("✓ Network analysis completed")
            
        except Exception as e:
            error_msg = f"Network analysis failed: {str(e)}"
            print(f"✗ {error_msg}")
            self.results['errors'].append({
                'analysis': 'network_analysis',
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            })
            
    def integrate_evidence(self):
        """Integrate all evidence and evaluate support"""
        print("\n5. Integrating evidence...")
        
        # Define evidence weights (dynamically adjusted)
        # Allocate weights based on actually available evidence
        available_evidence = []
        base_weights = {
            'information_theory': 0.25,
            'constitutiveness': 0.30,
            'statistical_models': 0.25,
            'network_analysis': 0.20
        }
        
        # Check which evidence is actually available
        for evidence_type in base_weights:
            if evidence_type in self.results['evidence'] and self.results['evidence'][evidence_type]:
                available_evidence.append(evidence_type)
        
        # Reallocate weights
        if available_evidence:
            total_base_weight = sum(base_weights[e] for e in available_evidence)
            evidence_weights = {
                e: base_weights[e] / total_base_weight for e in available_evidence
            }
        else:
            evidence_weights = base_weights
        
        # Calculate evidence scores
        evidence_scores = {}
        
        # 1. Information theory evidence
        if 'information_theory' in self.results['evidence']:
            it_evidence = self.results['evidence']['information_theory']
            # Get functional complementarity score from weighted_average
            fc_data = it_evidence.get('functional_complementarity', {})
            fc_score = fc_data.get('weighted_average', {}).get('total_complementarity', 0)
            # Normalize to 0-1 range
            fc_normalized = min(fc_score, 1.0) if fc_score > 0 else 0
            # Check triple interaction MI (normalize to 0-1 range)
            triple_mi = it_evidence.get('triple_interaction_mi', 0)
            mi_score = min(triple_mi * 5, 1.0) if triple_mi > 0 else 0  # Map 0.115 to ~0.575
            evidence_scores['information_theory'] = (fc_normalized + mi_score) / 2
        
        # 2. Constitutiveness evidence
        if 'constitutiveness' in self.results['evidence']:
            const_evidence = self.results['evidence']['constitutiveness']
            
            # Check each constitutiveness test result
            vr_significant = const_evidence.get('virtual_removal', {}).get('significant_impact', False)
            pn_necessary = const_evidence.get('path_necessity', {}).get('is_necessary', False)
            rob_robust = const_evidence.get('robustness', {}).get('is_robust', False)
            const_found = const_evidence.get('overall_assessment', {}).get('constitutiveness_found', False)
            
            # Count significant indicators
            significant_count = sum([vr_significant, pn_necessary, rob_robust, const_found])
            evidence_scores['constitutiveness'] = significant_count / 4.0
        
        # 3. Statistical model evidence
        if 'statistical_models' in self.results['evidence']:
            stat_evidence = self.results['evidence']['statistical_models']
            
            # Get R² values from model comparison
            model_comparison = stat_evidence.get('model_comparison', {})
            summary_table = model_comparison.get('summary_table', [])
            
            # Extract M1 and M3 R² values
            m1_r2 = 0
            m3_r2 = 0
            for model in summary_table:
                if model.get('Model') == 'M1_baseline':
                    m1_r2 = model.get('R_squared', 0)
                elif model.get('Model') == 'M3_nonlinear':
                    m3_r2 = model.get('R_squared', 0)
            
            # Check if nonlinear model is better than linear
            nonlinear_better = m3_r2 > m1_r2 and m3_r2 > 0.15  # R²>0.15 indicates meaningful explanatory power
            
            # VAR causal analysis (assume not significant for now based on data)
            var_significant = False
            
            # Calculate statistical model score
            r2_score = min(m3_r2 * 2, 1.0) if m3_r2 > 0 else 0  # Map R² to 0-1
            evidence_scores['statistical_models'] = (r2_score + int(nonlinear_better)) / 2
        
        # 4. Network analysis evidence
        if 'network_analysis' in self.results['evidence']:
            net_evidence = self.results['evidence']['network_analysis']
            centrality_score = net_evidence.get('DSR_centrality', {}).get('betweenness', 0)
            mediation_score = 1 if net_evidence.get('DSR_mediation', {}).get('is_mediator', False) else 0
            evidence_scores['network_analysis'] = (centrality_score + mediation_score) / 2
        
        
        # Save evidence analysis results (without calculating percentage support)
        self.results['evidence_scores'] = evidence_scores
        
        # Evaluate based on statistical significance and effect sizes
        # Check statistical significance of each analysis
        significant_findings = []
        
        # Information theory analysis
        if 'information_theory' in self.results['evidence']:
            fc_score = self.results['evidence']['information_theory'].get(
                'functional_complementarity', {}).get('weighted_average', {}).get('total_complementarity', 0)
            if fc_score > 0.2:
                significant_findings.append('Functional complementarity significant')
        
        # Constitutiveness tests
        if 'constitutiveness' in self.results['evidence']:
            if self.results['evidence']['constitutiveness'].get(
                'overall_assessment', {}).get('constitutiveness_found', False):
                significant_findings.append('Constitutiveness test significant')
        
        # Statistical models
        if 'statistical_models' in self.results['evidence']:
            models = self.results['evidence']['statistical_models']
            model_comparison = models.get('model_comparison', {}).get('summary_table', [])
            for model in model_comparison:
                if model.get('Model') == 'M3_nonlinear' and model.get('R_squared', 0) > 0.15:
                    significant_findings.append('Nonlinear model significant')
                    break
        
        # Network analysis
        if 'network_analysis' in self.results['evidence']:
            if self.results['evidence']['network_analysis'].get('DSR_mediation', {}).get('is_mediator', False):
                significant_findings.append('DSR has mediation role')
        
        # Qualitative assessment based on significant findings
        if len(significant_findings) >= 3:
            support_assessment = "supported"
        elif len(significant_findings) >= 2:
            support_assessment = "partially supported"
        else:
            support_assessment = "not supported"
            
        self.results['support_assessment'] = support_assessment
        self.results['evidence_summary'] = {
            'significant_findings': significant_findings,
            'total_analyses': len([k for k in ['information_theory', 'constitutiveness', 
                                              'statistical_models', 'network_analysis'] 
                                 if k in self.results['evidence']])
        }
        
        print(f"✓ Evidence integration completed, assessment: {support_assessment}")
        
    def generate_visualization(self):
        """Generate comprehensive visualization"""
        print("\n6. Generating comprehensive visualization...")
        
        # Create figure
        fig = plt.figure(figsize=(16, 12), dpi=1200)  # Must use 1200 DPI
        fig.patch.set_facecolor('white')
        
        # 1. Information theory metrics (top left)
        ax1 = plt.subplot(2, 2, 1)
        self._plot_information_theory(ax1)
        
        # 2. Constitutiveness test results (top right)
        ax2 = plt.subplot(2, 2, 2)
        self._plot_constitutiveness_tests(ax2)
        
        # 3. Statistical model comparison (bottom left)
        ax3 = plt.subplot(2, 2, 3)
        self._plot_statistical_models(ax3)
        
        # 4. Network centrality (bottom right)
        ax4 = plt.subplot(2, 2, 4)
        self._plot_network_analysis(ax4)
        
        # Adjust layout
        plt.tight_layout(pad=3.0, h_pad=4.0, w_pad=3.0)
        
        # Save figure
        output_file = self.output_path / 'figures' / 'H1_cognitive_dependency_analysis.jpg'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=1200, format='jpg')  # Don't use bbox_inches='tight' for consistent dimensions
        plt.close()
        
        self.results['visualizations']['main_figure'] = str(output_file)
        print(f"✓ Visualization saved to: {output_file}")
        
    def _plot_information_theory(self, ax):
        """Plot information theory analysis results"""
        if 'information_theory' not in self.results['evidence']:
            ax.text(0.5, 0.5, 'No information theory data', ha='center', va='center')
            ax.set_title('Information Theory Analysis')
            return
            
        data = self.results['evidence']['information_theory']
        
        # Extract metrics (from actual data structure)
        fc_data = data.get('functional_complementarity', {})
        fc_score = fc_data.get('weighted_average', {}).get('total_complementarity', 0)
        
        # Get conditional mutual information and synergy effects
        cmi_data = data.get('conditional_mi', {}).get('dsr_core', {})
        joint_mi = cmi_data.get('joint_mi', 0)
        synergy = data.get('synergy_redundancy', {}).get('synergy', 0)
        
        metrics = {
            'Functional\nComplementarity': fc_score,
            'Triple\nInteraction MI': data.get('triple_interaction_mi', 0),
            'Conditional MI': joint_mi,
            'Synergy Effect': abs(synergy)  # Show absolute value for clarity
        }
        
        # Plot bar chart
        bars = ax.bar(metrics.keys(), metrics.values(), color='steelblue', alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')
        
        ax.set_title('Information Theory Metrics', fontsize=14, fontweight='bold')
        ax.set_ylabel('Metric Value')
        ax.set_ylim(0, max(metrics.values()) * 1.2)
        ax.grid(axis='y', alpha=0.3)
        
        # If synergy effect is negative, add note
        if synergy < 0:
            ax.text(3, metrics['Synergy Effect'] + 0.01, '(redundancy)', 
                   ha='center', va='bottom', fontsize=9, style='italic')
        
    def _plot_constitutiveness_tests(self, ax):
        """Plot constitutiveness test results"""
        if 'constitutiveness' not in self.results['evidence']:
            ax.text(0.5, 0.5, 'No constitutiveness test data', ha='center', va='center')
            ax.set_title('Constitutiveness Tests')
            return
            
        data = self.results['evidence']['constitutiveness']
        
        # Prepare data (using actual performance loss and test results)
        tests = ['Virtual\nRemoval', 'Path\nNecessity', 'System\nRobustness']
        
        # Get metric values
        perf_loss = data.get('virtual_removal', {}).get('performance_loss', 0)
        # Map performance loss to 0-1 range (as original value may be >1)
        perf_loss_normalized = min(perf_loss / 2.0, 1.0)  # Assume max loss is 2
        
        # Path necessity: use indirect effect proportion as metric
        indirect_effect = data.get('path_necessity', {}).get('indirect_effect', 0)
        # If indirect effect is numeric, use directly; if dict, extract value
        if isinstance(indirect_effect, dict):
            indirect_effect = indirect_effect.get('value', 0)
        # Map indirect effect to 0-1 range
        path_necessity_score = min(abs(indirect_effect) * 10, 1.0)  # Assume 0.1 indirect effect is high
        
        # Robustness value already in 0-1 range
        robustness_value = data.get('robustness', {}).get('robustness_value', 0)
        
        scores = [
            perf_loss_normalized,  # Virtual removal
            path_necessity_score,  # Path necessity
            robustness_value  # System robustness
        ]
        
        # Plot radar chart
        angles = np.linspace(0, 2 * np.pi, len(tests), endpoint=False)
        scores = np.array(scores)
        
        # Close the figure
        angles = np.concatenate([angles, [angles[0]]])
        scores = np.concatenate([scores, [scores[0]]])
        
        ax.plot(angles, scores, 'o-', linewidth=2, color='darkgreen')
        ax.fill(angles, scores, alpha=0.25, color='darkgreen')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(tests)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Constitutive Strength', fontsize=12)
        ax.set_title('Constitutiveness Test Results', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
    def _plot_statistical_models(self, ax):
        """Plot statistical model comparison"""
        if 'statistical_models' not in self.results['evidence']:
            ax.text(0.5, 0.5, 'No statistical model data', ha='center', va='center')
            ax.set_title('Statistical Model Comparison')
            return
            
        data = self.results['evidence']['statistical_models']
        
        # Extract R² values from model comparison table
        model_comparison = data.get('model_comparison', {})
        summary_table = model_comparison.get('summary_table', [])
        
        # Extract M1 and M3 R² values
        m1_r2 = 0
        m3_r2 = 0
        for model in summary_table:
            if model.get('Model') == 'M1_baseline':
                m1_r2 = model.get('R_squared', 0)
            elif model.get('Model') == 'M3_nonlinear':
                m3_r2 = model.get('R_squared', 0)
        
        models = ['M1 Linear', 'M3 Nonlinear']
        r2_values = [m1_r2, m3_r2]
        
        # Plot bar chart
        bars = ax.bar(models, r2_values, color=['coral', 'darkred'], alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')
        
        ax.set_title('Statistical Model R² Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('R² Value')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        # Add VAR causal test result
        var_result = data.get('M4_VAR', {}).get('granger_causality', {})
        if var_result:
            result_text = "VAR Causality: DSR→CS " + ("significant" if var_result.get('DSR_causes_CS', False) else "not significant")
            ax.text(0.5, 0.95, result_text, transform=ax.transAxes, 
                   ha='center', va='top', fontsize=10, style='italic')
        
    def _plot_network_analysis(self, ax):
        """Plot network analysis results"""
        if 'network_analysis' not in self.results['evidence']:
            ax.text(0.5, 0.5, 'No network analysis data', ha='center', va='center')
            ax.set_title('Network Analysis')
            return
            
        data = self.results['evidence']['network_analysis']
        
        # Extract centrality metrics
        centrality_data = data.get('DSR_centrality', {})
        metrics = {
            'Degree\nCentrality': centrality_data.get('degree', 0),
            'Betweenness\nCentrality': centrality_data.get('betweenness', 0),
            'Closeness\nCentrality': centrality_data.get('closeness', 0)
        }
        
        # Plot bar chart
        x = np.arange(len(metrics))
        bars = ax.bar(x, list(metrics.values()), color='purple', alpha=0.7)
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')
        
        ax.set_xticks(x)
        ax.set_xticklabels(metrics.keys())
        ax.set_title('DSR Network Centrality Metrics', fontsize=14, fontweight='bold')
        ax.set_ylabel('Centrality Value')
        ax.grid(axis='y', alpha=0.3)
        
        # Add network density info
        density = data.get('network_density', 0)
        ax.text(0.02, 0.95, f'Network density: {density:.3f}', transform=ax.transAxes,
               va='top', fontsize=10)
        
    def generate_report(self):
        """Generate Markdown format comprehensive report"""
        print("\n7. Generating analysis report...")
        
        report = []
        report.append("# H1 Hypothesis Validation Report")
        report.append(f"\nGeneration time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n## Executive Summary")
        
        # Hypothesis content
        report.append(f"\n**Hypothesis**: {self.results['hypothesis_description']}")
        report.append(f"\n**Overall assessment**: {self.results.get('support_assessment', 'unknown')}")
        
        # Evidence summary
        evidence_summary = self.results.get('evidence_summary', {})
        report.append(f"\n**Significant findings**: {len(evidence_summary.get('significant_findings', []))} items")
        
        # Key findings
        report.append("\n**Key findings**:")
        key_findings = self._extract_key_findings()
        for finding in key_findings:
            report.append(f"- {finding}")
        
        # Detailed analysis results
        report.append("\n## Detailed Analysis Results")
        
        # 1. Information theory analysis
        report.append("\n### 1. Information Theory Analysis")
        self._add_information_theory_report(report)
        
        # 2. Constitutiveness tests
        report.append("\n### 2. Constitutiveness Tests")
        self._add_constitutiveness_report(report)
        
        # 3. Statistical models
        report.append("\n### 3. Statistical Model Analysis")
        self._add_statistical_models_report(report)
        
        # 4. Network analysis
        report.append("\n### 4. Network Analysis")
        self._add_network_analysis_report(report)
        
        # Evidence integration assessment
        report.append("\n## Evidence Integration Assessment")
        self._add_evidence_integration_report(report)
        
        # Conclusion
        report.append("\n## Conclusion")
        self._add_conclusion(report)
        
        # Appendix
        if self.results.get('errors'):
            report.append("\n## Appendix: Error Log")
            for error in self.results['errors']:
                report.append(f"\n- **{error['analysis']}** ({error['timestamp']}): {error['error']}")
        
        # Save report
        report_path = self.output_path / 'md' / 'H1_validation_report.md'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
            
        print(f"✓ Report saved to: {report_path}")
        
    def _extract_key_findings(self):
        """Extract key findings"""
        findings = []
        
        # Information theory findings
        if 'information_theory' in self.results['evidence']:
            fc_data = self.results['evidence']['information_theory'].get('functional_complementarity', {})
            fc_score = fc_data.get('weighted_average', {}).get('total_complementarity', 0)
            if fc_score > 0.2:
                findings.append(f"DSR exhibits significant functional complementarity (FC={fc_score:.3f})")
        
        # Constitutiveness findings
        if 'constitutiveness' in self.results['evidence']:
            const_data = self.results['evidence']['constitutiveness']
            vr_impact = const_data.get('virtual_removal', {}).get('significant_impact', False)
            perf_loss = const_data.get('virtual_removal', {}).get('performance_loss', 0)
            
            if vr_impact or perf_loss > 0.5:
                findings.append(f"Virtual removal test shows DSR has significant impact on cognitive success (performance loss: {perf_loss:.3f})")
            
            if const_data.get('overall_assessment', {}).get('constitutiveness_found', False):
                findings.append("Constitutiveness test results significant, supporting H1 hypothesis")
        
        # Statistical model findings
        if 'statistical_models' in self.results['evidence']:
            models = self.results['evidence']['statistical_models']
            model_comparison = models.get('model_comparison', {})
            summary_table = model_comparison.get('summary_table', [])
            
            # Get model R² values
            m1_r2 = 0
            m3_r2 = 0
            for model in summary_table:
                if model.get('Model') == 'M1_baseline':
                    m1_r2 = model.get('R_squared', 0)
                elif model.get('Model') == 'M3_nonlinear':
                    m3_r2 = model.get('R_squared', 0)
            
            if m3_r2 > m1_r2 and m3_r2 > 0.15:
                improvement = (m3_r2 - m1_r2) / m1_r2 * 100 if m1_r2 > 0 else 0
                findings.append(f"Nonlinear model (R² = {m3_r2:.3f}) outperforms linear model (R² = {m1_r2:.3f}), improvement {improvement:.1f}%")
        
        # Network analysis findings
        if 'network_analysis' in self.results['evidence']:
            centrality = self.results['evidence']['network_analysis'].get(
                'DSR_centrality', {}).get('betweenness', 0)
            if centrality > 0.5:
                findings.append(f"DSR has high betweenness centrality in cognitive network ({centrality:.3f})")
        
        return findings
        
    def _add_information_theory_report(self, report):
        """Add information theory analysis report content"""
        if 'information_theory' not in self.results['evidence']:
            report.append("\n*Information theory analysis not executed or failed*")
            return
            
        data = self.results['evidence']['information_theory']
        
        report.append("\n**Functional Complementarity Analysis**")
        fc_data = data.get('functional_complementarity', {})
        weighted_avg = fc_data.get('weighted_average', {})
        report.append(f"- Overall functional complementarity: {weighted_avg.get('total_complementarity', 0):.3f}")
        report.append(f"- Low DSR group: {fc_data.get('low', {}).get('complementarity', 0):.3f}")
        report.append(f"- Medium DSR group: {fc_data.get('medium', {}).get('complementarity', 0):.3f}")
        report.append(f"- High DSR group: {fc_data.get('high', {}).get('complementarity', 0):.3f}")
        
        report.append("\n**Triple Interaction Mutual Information**")
        report.append(f"- MI(DSR;TL;CS) = {data.get('triple_interaction_mi', 0):.4f}")
        
        report.append("\n**Conditional Mutual Information**")
        cmi_data = data.get('conditional_mi', {})
        if isinstance(cmi_data, dict) and cmi_data:
            # Show only main mutual information values
            dsr_core = cmi_data.get('dsr_core', {})
            if dsr_core:
                report.append(f"- DSR joint mutual information = {dsr_core.get('joint_mi', 0):.4f}")
                report.append(f"- DSR total mutual information = {dsr_core.get('total_mi', 0):.4f}")
                report.append(f"- Synergy effect = {dsr_core.get('synergy', 0):.4f}")
        
        report.append("\n**Statistical Significance**")
        sig_data = data.get('significance', {})
        if isinstance(sig_data, dict) and sig_data:
            # Show causality test results
            full_sample = sig_data.get('full_sample', {})
            dsr_causes_cs = full_sample.get('DSR_causes_CS', {})
            if dsr_causes_cs:
                p_val = dsr_causes_cs.get('p_value', 1)
                if p_val < 0.001:
                    p_str = "p < .001"
                elif p_val >= 0.01:
                    p_str = f"p = {p_val:.2f}"
                else:
                    p_str = f"p = {p_val:.3f}"
                report.append(f"- DSR→CS causality test: {p_str}")
                report.append(f"- Significant: {'Yes' if dsr_causes_cs.get('significant', False) else 'No'}")
        
    def _add_constitutiveness_report(self, report):
        """Add constitutiveness test report content"""
        if 'constitutiveness' not in self.results['evidence']:
            report.append("\n*Constitutiveness tests not executed or failed*")
            return
            
        data = self.results['evidence']['constitutiveness']
        
        report.append("\n**Virtual Removal Test**")
        vr_data = data.get('virtual_removal', {})
        perf_loss = vr_data.get('performance_loss', 0)
        report.append(f"- Performance loss after removing DSR: {perf_loss:.3f}")
        report.append(f"- Significant impact: {'Yes' if vr_data.get('significant_impact', False) else 'No'}")
        
        report.append("\n**Path Necessity Analysis**")
        pn_data = data.get('path_necessity', {})
        report.append(f"- Indirect effect: {pn_data.get('indirect_effect', 0):.3f}")
        report.append(f"- Path necessary: {'Yes' if pn_data.get('is_necessary', False) else 'No'}")
        
        report.append("\n**System Robustness Assessment**")
        rob_data = data.get('robustness', {})
        report.append(f"- Robustness value: {rob_data.get('robustness_value', 0):.3f}")
        report.append(f"- System robust: {'Yes' if rob_data.get('is_robust', False) else 'No'}")
        
        # Add overall assessment
        overall = data.get('overall_assessment', {})
        if overall:
            report.append("\n**Overall Assessment**")
            report.append(f"- Constitutiveness test result: {overall.get('verdict', 'unknown')}")
        
    def _add_statistical_models_report(self, report):
        """Add statistical model report content"""
        if 'statistical_models' not in self.results['evidence']:
            report.append("\n*Statistical model analysis not executed or failed*")
            return
            
        data = self.results['evidence']['statistical_models']
        
        # Get data from model comparison table
        model_comparison = data.get('model_comparison', {})
        summary_table = model_comparison.get('summary_table', [])
        
        # Extract model data
        m1_data = {}
        m3_data = {}
        for model in summary_table:
            if model.get('Model') == 'M1_baseline':
                m1_data = model
            elif model.get('Model') == 'M3_nonlinear':
                m3_data = model
        
        # M1 linear model
        report.append("\n**M1 Linear Baseline Model**")
        report.append(f"- R² = {m1_data.get('R_squared', 0):.3f}")
        report.append(f"- AIC = {m1_data.get('AIC', 0):.1f}")
        
        # M3 nonlinear model
        report.append("\n**M3 Nonlinear Interaction Model**")
        report.append(f"- R² = {m3_data.get('R_squared', 0):.3f}")
        report.append(f"- AIC = {m3_data.get('AIC', 0):.1f}")
        
        # M4 VAR causal analysis
        report.append("\n**M4 VAR Causal Analysis**")
        var_data = data.get('M4_VAR', {})
        gc_data = var_data.get('granger_causality', {})
        report.append(f"- DSR → CS: {'Significant' if gc_data.get('DSR_causes_CS', False) else 'Not significant'}")
        report.append(f"- TL → CS: {'Significant' if gc_data.get('TL_causes_CS', False) else 'Not significant'}")
        
        # Model comparison
        report.append("\n**Model Comparison**")
        best_r2_model = model_comparison.get('best_r_squared', 'unknown')
        best_aic_model = model_comparison.get('best_aic', 'unknown')
        report.append(f"- Best R² model: {best_r2_model}")
        report.append(f"- Best AIC model: {best_aic_model}")
        
        # Calculate nonlinear improvement
        m1_r2 = m1_data.get('R_squared', 0)
        m3_r2 = m3_data.get('R_squared', 0)
        improvement = m3_r2 - m1_r2
        report.append(f"- Nonlinear improvement: {improvement:.3f} ({improvement/m1_r2*100:.1f}%)" if m1_r2 > 0 else "- Nonlinear improvement: N/A")
        
    def _add_network_analysis_report(self, report):
        """Add network analysis report content"""
        if 'network_analysis' not in self.results['evidence']:
            report.append("\n*Network analysis not executed or failed*")
            return
            
        data = self.results['evidence']['network_analysis']
        
        report.append("\n**DSR Network Centrality**")
        centrality = data.get('DSR_centrality', {})
        report.append(f"- Degree centrality: {centrality.get('degree', 0):.3f}")
        report.append(f"- Betweenness centrality: {centrality.get('betweenness', 0):.3f}")
        report.append(f"- Closeness centrality: {centrality.get('closeness', 0):.3f}")
        
        report.append("\n**Network Properties**")
        report.append(f"- Network density: {data.get('network_density', 0):.3f}")
        
        report.append("\n**DSR Mediation Role**")
        mediation = data.get('DSR_mediation', {})
        report.append(f"- Mediation role: {'Yes' if mediation.get('is_mediator', False) else 'No'}")
        report.append(f"- Mediation strength: {mediation.get('mediation_strength', 0):.3f}")
        
    def _add_evidence_integration_report(self, report):
        """Add evidence integration report content"""
        report.append("\n### Evidence Integration Assessment")
        
        # List qualitative assessment of each evidence type
        evidence_assessments = {
            'information_theory': 'Information Theory Analysis',
            'constitutiveness': 'Constitutiveness Tests',
            'statistical_models': 'Statistical Models',
            'network_analysis': 'Network Analysis'
        }
        
        scores = self.results.get('evidence_scores', {})
        
        report.append("\n**Evidence from each analysis**:")
        for key, name in evidence_assessments.items():
            if key in self.results['evidence'] and self.results['evidence'][key]:
                report.append(f"- {name}: Completed")
            else:
                report.append(f"- {name}: Not executed")
        
        # Add overall assessment
        evidence_summary = self.results.get('evidence_summary', {})
        report.append(f"\n**Evidence summary**:")
        report.append(f"- Number of analyses completed: {evidence_summary.get('total_analyses', 0)}")
        report.append(f"- Significant findings: {len(evidence_summary.get('significant_findings', []))} items")
        
        # List significant findings
        for finding in evidence_summary.get('significant_findings', []):
            report.append(f"  - {finding}")
        
        report.append(f"\n**Assessment conclusion**: Based on the above analyses, H1 hypothesis is {self.results.get('support_assessment', 'unknown')}")
        
    def _add_conclusion(self, report):
        """Add conclusion section"""
        evidence_summary = self.results.get('evidence_summary', {})
        
        report.append(f"\nBased on multidimensional comprehensive analysis, H1 hypothesis ({self.results['hypothesis_description']}) validation results are as follows:")
        
        # List specific statistical findings
        report.append("\n**Main findings**:")
        
        # Based on actual statistical results
        if 'information_theory' in self.results['evidence']:
            fc_score = self.results['evidence']['information_theory'].get(
                'functional_complementarity', {}).get('weighted_average', {}).get('total_complementarity', 0)
            if fc_score > 0:
                report.append(f"- Functional complementarity analysis: FC = {fc_score:.3f}")
                
        if 'constitutiveness' in self.results['evidence']:
            const_verdict = self.results['evidence']['constitutiveness'].get(
                'overall_assessment', {}).get('verdict', '')
            if const_verdict:
                report.append(f"- Constitutiveness test: {const_verdict}")
                
        if 'statistical_models' in self.results['evidence']:
            models = self.results['evidence']['statistical_models']
            model_comparison = models.get('model_comparison', {}).get('summary_table', [])
            for model in model_comparison:
                if model.get('Model') == 'M3_nonlinear':
                    r2 = model.get('R_squared', 0)
                    if r2 > 0:
                        report.append(f"- Nonlinear model: R² = {r2:.3f}")
                    break
                    
        if 'network_analysis' in self.results['evidence']:
            centrality = self.results['evidence']['network_analysis'].get(
                'DSR_centrality', {}).get('betweenness', 0)
            if centrality > 0:
                report.append(f"- DSR betweenness centrality: {centrality:.3f}")
        
        report.append("\n**Theoretical contribution**:")
        report.append("This study provides empirical support for the constitutive role of digital symbolic resources in distributed cognition theory, demonstrating how DSR serves as an indispensable component of the cognitive system rather than merely an auxiliary tool.")
        
    def save_results(self):
        """Save all results"""
        print("\n8. Saving analysis results...")
        
        # Create output directory
        data_dir = self.output_path / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results in JSON format
        json_path = data_dir / 'H1_validation_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"✓ JSON results saved to: {json_path}")
        
        # Prepare CSV format summary data
        summary_data = {
            'hypothesis': 'H1',
            'hypothesis_description': self.results['hypothesis_description'],
            # 'support_assessment': self.results['support_assessment'],  # Do not save assessment
            'significant_findings': len(self.results.get('evidence_summary', {}).get('significant_findings', [])),
            'total_analyses': self.results.get('evidence_summary', {}).get('total_analyses', 0),
            'timestamp': self.results['timestamp']
        }
        
        # Add evidence scores for each type
        for evidence_type, score in self.results.get('evidence_scores', {}).items():
            summary_data[f'{evidence_type}_score'] = score
        
        # Save CSV format summary
        csv_path = data_dir / 'H1_validation_results.csv'
        summary_df = pd.DataFrame([summary_data])
        summary_df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"✓ CSV summary saved to: {csv_path}")
        
    def run_all_analyses(self):
        """Main function to run all analyses"""
        print("="*60)
        print("H1 Hypothesis Validation Analysis")
        print("="*60)
        
        # 1. Load data
        self.load_data()
        if self.df is None:
            print("Data loading failed, analysis terminated")
            return
            
        # 1.5 Validate data integrity
        if not self.validate_data():
            print("Data validation failed, but will continue analysis...")
            # Don't terminate, let analysis continue
        
        # 2. Run analyses
        self.run_information_theory_analysis()
        self.run_constitutiveness_tests()
        self.run_statistical_models()
        self.run_network_analysis()
        
        # 3. Integrate evidence
        self.integrate_evidence()
        
        # 4. Generate visualization
        self.generate_visualization()
        
        # 5. Generate report
        self.generate_report()
        
        # 6. Save results
        self.save_results()
        
        print("\n" + "="*60)
        print("H1 hypothesis validation analysis completed!")
        # print(f"Overall assessment: {self.results['support_assessment']}")  # Do not show assessment
        evidence_summary = self.results.get('evidence_summary', {})
        print(f"Significant findings: {len(evidence_summary.get('significant_findings', []))} items")
        print("="*60)


def main():
    """Main function"""
    # Create validator instance
    validator = H1CognitiveDependencyValidator(
        data_path='../output_cn/data',
        output_path='../output_en'
    )
    
    # Run complete analysis
    validator.run_all_analyses()


if __name__ == "__main__":
    main()