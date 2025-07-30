# A Triadic Framework for Cross-cultural Pragmatics: Digital Symbolic Resources as Constitutive Components of Distributed Cognition

<p align="center">
  <img src="https://img.shields.io/badge/Status-In%20Progress-brightgreen" alt="Status">
  <img src="https://img.shields.io/badge/Data%20Coverage-2021--2025-blue" alt="Data Coverage">
  <img src="https://img.shields.io/badge/Methods-Mixed%20Methods-orange" alt="Methods">
  <img src="https://img.shields.io/badge/Language-中文/English-red" alt="Language">
</p>

## Project Overview

This project analyzes Chinese Foreign Ministry press conferences (2021-2025) to demonstrate that **Digital Symbolic Resources (DSR) are constitutive components of distributed cognition**. The research employs mixed methods combining statistical analysis, information theory, and network analysis to systematically validate three core hypotheses.

### Core Hypotheses

- **H1 Cognitive Dependency**: DSR serves as constitutive component of cognitive system
- **H2 System Moderation**: Contextual factors systematically moderate DSR's cognitive role  
- **H3 Dynamic Evolution**: DSR-cognition relationship shows structured evolution patterns

### Theoretical Contributions

1. **Cross-cultural Pragmatics**: Proposes triadic framework (digital tools-cognitive processes-cultural context)
2. **Distributed Cognition Theory**: Establishes constitutive status of digital symbolic resources
3. **Methodological Innovation**: Develops comprehensive 66+ feature analysis framework

## Quick Start

### Requirements

```bash
# Python 3.8+
pip install -r requirements.txt
```

### Data Processing Pipeline

```bash
# 1. Data validation and preprocessing
cd Script_en
python step1_data_validation.py      # Validate XML corpus
python step2_data_extraction.py      # Extract 66+ features
python step3_cognitive_metrics.py    # Calculate DSR, TL, CS metrics

# 2. Hypothesis validation
python H1_validation_analysis.py     # H1 cognitive dependency
python H2_validation_analysis.py     # H2 system moderation
python H3_validation_analysis.py     # H3 dynamic evolution

# 3. Generate comprehensive reports
python generate_comprehensive_report.py
```

## Project Structure

```
A-Triadic-Framework-for-Cross-cultural-Pragmatics/
├── Corpus/                    # XML corpus files (2021-2025)
├── Script_cn/                 # Chinese analysis scripts
├── Script_en/                 # English analysis scripts
│   ├── step1-9_*.py          # Sequential analysis
│   ├── H[1-3]_validation_*.py # Hypothesis validation
│   └── mixed_methods_*.py     # Mixed methods analysis
├── output_en/                 # English outputs
│   ├── data/                 # CSV and JSON data
│   ├── figures/              # High-res visualizations (1200 DPI)
│   └── md/                   # Markdown reports
├── data/                      # Core data files
├── context/                   # Background materials
│   ├── 2021年3月-2025年5月/   # Time period documentation
│   ├── 外交话语标注方案/       # Annotation scheme
│   └── 标注语料实例/          # Annotation examples
└── README_EN.md              # This document
```

## Core Metrics

### Cognitive Measures

- **DSR (Digital Symbolic Resources)**: Digital resource utilization
  - Measures digital tool engagement in cognitive processes
  - Range: 0-1, higher values indicate more digital tool use

- **TL (Task Load)**: Cognitive task demands
  - Quantifies complexity and requirements of cognitive tasks
  - Considers question complexity, time pressure, etc.

- **CS (Cognitive Success)**: Cognitive performance outcomes
  - Evaluates effectiveness of cognitive performance
  - Based on completeness, accuracy, and multiple dimensions

### Analysis Dimensions (66+ Features)

1. **Linguistic Features**: Lexical density, syntactic complexity, modality use
2. **Pragmatic Strategies**: Usage patterns of 10 relational construction strategies
3. **Cognitive Markers**: Uncertainty expressions, cognitive load indicators
4. **Interaction Features**: Turn-taking, question-answer correspondence
5. **Temporal Dynamics**: Quarterly trends, evolution phases

## Key Findings

### H1 Cognitive Dependency (Validated)

- Information theory analysis reveals significant functional complementarity (FC = 0.303, *p* < .001)
- Virtual removal tests show strong necessity (λ = 0.905)
- Statistical models demonstrate robust relationships (*R*² = 0.187, *p* < .001)

### H2 System Moderation (Validated)

- Strongest moderation effects in medium contexts (*β* = 0.521, *p* < .001)
- Bayesian hierarchical models show *R*² variation across contexts
- Causal forest analysis identifies high-sensitivity subgroups

### H3 Dynamic Evolution (Validated)

- S-curve fitting confirms three-phase evolution pattern
- Changepoint detection identifies structural shift in Q1 2023
- Network density shows quarterly growth trends

## Technology Stack

### Data Processing
- pandas, numpy - Data manipulation
- xml.etree - XML parsing
- json - Data storage

### Statistical Analysis
- scipy, statsmodels - Traditional statistics
- scikit-learn - Machine learning
- pymc3 - Bayesian modeling

### Visualization
- matplotlib, seaborn - Chart generation
- networkx - Network visualization

## Usage Guide

### Running Full Analysis

```bash
# Use provided batch script
./run_full_analysis.sh

# Or run step by step
python step1_data_validation.py
python step2_data_extraction.py
# ... continue sequentially
```

### Custom Analysis

```python
# Import core analyzers
from step4_information_theory import InformationTheoryAnalyzer
from step5_statistical_models import StatisticalModeler

# Create analysis instance
analyzer = InformationTheoryAnalyzer()
results = analyzer.run_analysis(data_path='your_data.csv')
```

### Generating Visualizations

All figures use standardized 1200 DPI high-resolution output:

```python
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
fig = plt.figure(figsize=(20, 16), dpi=1200)
# Plotting code
plt.savefig('output.jpg', dpi=1200, format='jpg')
```

## Citation

If you use this project's code or data, please cite:

```bibtex
@misc{chen2025triadic,
  title={A Triadic Framework for Cross-cultural Pragmatics: 
         Digital Symbolic Resources as Constitutive Components 
         of Distributed Cognition},
  author={To be provided after review},
  year={2025},
  howpublished={\url{https://github.com/chenwangfang/A-Triadic-Framework-for-Cross-cultural-Pragmatics}}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

## Contact

- Principal Investigators: To be provided after review
- Email: To be provided after review
- Project Homepage: [GitHub Repository](https://github.com/chenwangfang/A-Triadic-Framework-for-Cross-cultural-Pragmatics)

## Acknowledgments

We thank all colleagues and reviewers who provided support and suggestions for this project. Special thanks to the Ministry of Foreign Affairs for providing public press conference corpus resources.

---

<p align="center">
  <i>Advancing Cross-cultural Pragmatics Research · Deepening Distributed Cognition Theory</i>
</p>