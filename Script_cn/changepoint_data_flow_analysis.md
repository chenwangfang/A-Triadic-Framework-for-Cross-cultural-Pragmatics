# Bayesian Changepoint Detection Data Flow Analysis

## Executive Summary

After examining the complete data flow from corpus to changepoint detection, I've identified why all changepoints are concentrated in March 2021 with low confidence. The issue stems from a combination of data characteristics and algorithm initialization.

## 1. Source Corpus Structure and Format

### XML Structure
The corpus consists of 5 XML files (2021-2025), each containing press conference Q&A units with this structure:

```xml
<unit id="2021-03-19-Q01" date="2021-03-19" sequence="1">
  <metadata>
    <spokesperson>赵立坚</spokesperson>
    <media_source culture="Western">美国有线电视新闻网</media_source>
    <topic category="BR" sensitivity="high">全球治理</topic>
  </metadata>
  <question><text>...</text></question>
  <response>
    <text>...</text>
    <pragmatic_strategies>...</pragmatic_strategies>
    <cultural_schemas>
      <schema type="DED" function="contextualizing">
        <marker depth="2">网上发布</marker>
        <activation_intensity>3.0</activation_intensity>
      </schema>
    </cultural_schemas>
  </response>
</unit>
```

Key elements:
- **DED (Digital Engagement Devices)** schemas marked with `type="DED"`
- Cognitive indicators: adaptation success, load index, coupling strength
- Temporal data: date and sequence within each day

## 2. Data Extraction Process (step1 & step2)

### Step 1: Data Validation
- Validates XML structure and field completeness
- Checks coverage of required fields
- Reports missing data

### Step 2: Feature Extraction
Extracts 66+ features including:
- **DED elements**: count, functions, depth, intensity
- **Cognitive metrics**: adaptation success, load index, coupling strength
- **Pragmatic strategies**: types, effectiveness, patterns
- **Cultural schemas**: traditional vs innovative
- **Digital adaptation**: soundbites, viral potential, meme scores

The extraction produces `extracted_data.csv` with one row per Q&A unit.

## 3. Cognitive Metrics Calculation (step3)

Calculates three main metric categories:

### DSR (Digital Symbolic Resources) Metrics
- **Bridging score**: Cross-cultural conceptual bridging capability
- **Integration depth**: 5-level depth classification
- **Irreplaceability**: Uniqueness and substitution cost
- **Path centrality**: Mediation role in cognitive network
- **Bottleneck score**: Critical node identification
- **Cascade impact**: Downstream influence

### TL (Traditional Language) Metrics  
- **Conventional density**: Diplomatic language, idioms
- **Structural complexity**: Syntax, vocabulary
- **Pragmatic richness**: Strategy diversity
- **DSR interaction**: TL-DSR complementarity

### CS (Cognitive Success) Metrics
- **Immediate effects**: Understanding accuracy, efficiency
- **System properties**: Adaptability, stability
- **Emergent features**: Synergy, complexity gains
- **Integration level**: DSR-TL integration degree

The output `data_with_metrics.csv` contains all original features plus calculated metrics.

## 4. Bayesian Changepoint Detection Process (step5g)

### Data Aggregation
1. Groups data by date (daily aggregation)
2. Calculates mean values for key metrics
3. Computes `constitutive_strength = 0.3*DSR + 0.3*TL + 0.4*CS`
4. Applies Gaussian smoothing (sigma=2)

### Online Changepoint Detection Algorithm
Uses Adams & MacKay (2007) algorithm with:
- **Hazard rate**: 1/50 (expects changepoint every 50 days)
- **Prior**: Normal-Inverse Gamma conjugate prior
- **Threshold**: 0.01 (very low, causing oversensitivity)

### The Problem
The algorithm detects all changepoints in the first few days (March 19-26, 2021) because:

1. **Initialization bias**: The algorithm needs time to establish baseline statistics
2. **Low threshold**: 0.01 is too sensitive for early fluctuations
3. **Prior mismatch**: Default priors don't match data characteristics
4. **No burn-in period**: Early estimates are unstable

## 5. Root Cause Analysis

### Why March 2021?

1. **Data characteristics at start**:
   - First days have high variance as system initializes
   - Limited historical data for stable estimates
   - Natural fluctuations interpreted as changepoints

2. **Algorithm parameters**:
   ```python
   threshold = 0.01  # Too low - detects minor fluctuations
   hazard_rate = 1/50  # Assumes frequent changes
   # No minimum segment length constraint
   # No burn-in period to stabilize estimates
   ```

3. **Prior specification**:
   ```python
   alpha = 1  # Weak prior
   beta = 1   # Weak prior
   kappa = 1  # Low confidence in initial mean
   mu = np.mean(data[:10])  # Only 10 points for initial estimate
   ```

## 6. Recommendations

### Immediate Fixes

1. **Increase threshold**:
   ```python
   threshold = 0.1  # 10x higher to reduce false positives
   ```

2. **Add burn-in period**:
   ```python
   burn_in = 30  # Skip first 30 days
   changepoints = [cp for cp in changepoints if cp['index'] > burn_in]
   ```

3. **Minimum segment length**:
   ```python
   min_segment = 14  # At least 2 weeks between changepoints
   ```

4. **Adjust priors based on data**:
   ```python
   # Use more informative priors
   alpha = 2  # Stronger shape prior
   beta = np.var(data[:30])  # Data-driven scale
   kappa = 5  # Higher confidence in mean
   ```

### Algorithmic Improvements

1. **Use retrospective analysis**: Run algorithm forward and backward, combine results
2. **Ensemble methods**: Combine multiple algorithms (PELT, BOCPD, etc.)
3. **Cross-validation**: Test different parameters on held-out data
4. **Domain constraints**: Incorporate knowledge about press conference patterns

### Data Preprocessing

1. **Outlier removal**: Clean extreme values before analysis
2. **Seasonal adjustment**: Account for known patterns
3. **Feature engineering**: Use more stable composite metrics

## 7. Expected Results After Fixes

With proper parameters, changepoints should:
- Distribute across the entire time period
- Align with known events (policy changes, crises)
- Have higher confidence scores (>0.5)
- Show meaningful magnitude changes (>0.1)

## Conclusion

The concentration of changepoints in March 2021 is an artifact of algorithm initialization rather than a real phenomenon. The combination of weak priors, low threshold, and lack of burn-in period causes the algorithm to interpret early data fluctuations as significant changes. Implementing the recommended fixes will produce more meaningful and distributed changepoints throughout the 2021-2025 period.