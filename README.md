# Leveraging Machine Learning to Identify Potential Biological Indicators of ME/CFS

**Authors:** Cang Liang, Manali Raut, Nicolas Wirth

---

## Overview

This project uses machine learning algorithms to analyze medical biomarker data and identify potential biological indicators (proteins, genes, metabolites) that could enable early detection and diagnosis of Myalgic Encephalomyelitis/Chronic Fatigue Syndrome (ME/CFS).

### The Problem

- **Prevalence:** ME/CFS affects up to 3.3 million Americans (~1% of population)
- **Economic Impact:** $18-51 billion annually in medical costs and lost productivity
- **Diagnostic Gap:** 90% of ME/CFS cases remain undiagnosed (CDC estimate)
- **Current Challenge:** No objective biomarker exists; diagnosis relies solely on clinical symptoms requiring 6+ months observation

### The Solution

We employ a multi-stage machine learning pipeline to:
1. Identify statistically significant biomarkers using logistic regression
2. Reduce dimensionality while preserving variance using PCA
3. Discover patient subgroups using K-Means clustering
4. Build predictive models using Random Forest classification
5. Validate findings with clinical domain experts

---

## Data Sources

We leverage biomarker datasets from the [mapMECFS database](https://www.mapme-cfs.org/), including:

| Study | Biomarker Type | # Indicators | # Participants | Source |
|-------|---|---|---|---|
| Helliwell et al #1 | DNA Methylation | 146,576 | 20 (10/10) | Clin Epigenet (2020) |
| Helliwell et al #2 | DNA Methylation | 75,452 | 20 (10/10) | Clin Epigenet (2020) |
| Germain et al | Auto-Immune Antibody Profile | 1,134 | 103 (59/44) | Int. J. Mol. Sci. (2025) |
| Germain et al | Plasma Metabolomics | 832 | 51 (19/32) | Metabolites (2018) |
| Germain et al | Lipidomics | 1,022 | 52 (26/26) | Metabolites (2020) |
| Mandarano et al | Cytokine & T-Cell Metabolism | 44 | 75 (37/38) | J. Clin. Invest. (2020) |

**Note:** We selected Germain et al's Autoimmune Antibody Profile study due to it being the study with the largest number of participants and also due to the focus of the study being the profiling of ME/CFS patients' Autoimmune Antibody profile (our literature review suggested ME/CFS is an autoimmune disorder).

---

## Project Structure

```
MeCfsBiology/
├── data_german/
│                         # Original datasets from mapMECFS
│   │   ├── augmenta-oncimmune_assay_data_export_for_rti.tsv
│   │   ├── augmenta-oncimmune_phenotype_export_for_rti.tsv
|   |   ├── GermainHanson_AutoAntibodyImmuneProfile.pdf
│
├── experiments/
│   ├── data_cleaning.ipynb        # Load, merge, preprocess biomarker data, Dimensionality reduction, Identify significant biomarkers
│   ├── datacleaning_Kmeans_RFC.ipynb   # Patient subgroup discovery
│   └── ml-project-cang-raut-wirth (1).ipynb        # Final classification model
│
├── mecfs_requirements.txt                  # Python dependencies
├── README.md                         # This file
└── LICENSE                           # MIT License

```

---

## Installation & Setup

### Prerequisites
- Python 3.11+
- pip or conda
- 4GB+ RAM (for PCA on large feature sets)
- Git

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/me-cfs-ml-project.git
   cd me-cfs-ml-project
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download datasets**
   - Request access to mapMECFS biomarker data via [mapme-cfs.org](https://www.mapme-cfs.org/)
   - Place raw TSV/CSV files in `data/raw/`

5. **Run Jupyter notebooks**
   ```bash
   jupyter notebook
   ```

---

## Methodology

### Stage 1: Logistic Regression
**Purpose:** Identify statistically significant biomarkers  
**Process:**
- Standardize biomarker values (zero mean, unit variance)
- Check for multicollinearity (VIF < 5)
- Fit logistic regression with disease/control labels
- Perform k-fold cross-validation (k=5)
- Extract coefficients and p-values

**Output:** Top 10-20 significant biomarkers with odds ratios

### Stage 2: Principal Component Analysis (PCA)
**Purpose:** Reduce dimensionality from thousands to ~40 components  
**Process:**
- Fit PCA on training data to retain 95%+ variance
- Transform training & test data
- Visualize PC1 vs PC2 colored by phenotype

**Output:** PCA-transformed feature matrix, scree plot, variance explained

### Stage 3: K-Means Clustering
**Purpose:** Discover patient subgroups (e.g., immune-dominant vs. metabolic subtypes)  
**Process:**
- Apply k-means (k=2-5) on PCA components
- Evaluate with silhouette score and Davies-Bouldin index
- Characterize clusters by mean biomarker profiles

**Output:** Patient cluster assignments, cluster centroids, subgroup profiles

### Stage 4: Random Forest Classification
**Purpose:** Build final predictive model capturing non-linear interactions  
**Process:**
- Train random forest (100 trees) on PCA features + selected biomarkers
- Perform nested cross-validation
- Extract feature importance scores
- Evaluate: accuracy, sensitivity, specificity, AUC-ROC

**Output:** Trained model, feature importance ranking, confusion matrix

---

## Key Results (Preliminary)

### Data Preprocessing
- **Dataset:** Germain et al. antibody profiles (103 samples, 1,134 features)
- **Train/Test Split:** 80/20 stratified by phenotype
- **Class Balance:** 43 ME/CFS, 60 healthy controls (training set)

### PCA Analysis

### Model Performance (Preliminary)

---

## Usage Examples

### Running Data Cleaning
```python
from src.preprocessing import load_and_merge_data, standardize_features

# Load raw data
X, y = load_and_merge_data(data_path='data/raw/')

# Standardize
X_scaled = standardize_features(X)
```

### Training Models
```python
from src.model_training import train_pipeline
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

models = train_pipeline(X_train, y_train, X_test, y_test)
```

### Evaluating Results
```python
from src.evaluation import evaluate_model

results = evaluate_model(models['random_forest'], X_test, y_test)
print(f"Accuracy: {results['accuracy']:.3f}")
print(f"AUC-ROC: {results['auc']:.3f}")
```

---

## Limitations & Future Work

### Current Limitations
1. **Small cohort sizes** (20-103 participants per study) → overfitting risk
2. **Heterogeneous datasets** (different biomarker types) → requires careful integration
3. **Imbalanced classes** in some studies → affects model performance
4. **Lack of prospective validation** → needs external cohorts

### Future Directions
- **Scale to UK Biobank** (1,455 ME/CFS vs. 131,000 controls)
- **Multi-omics integration** (combine all biomarker types in unified framework)
- **Interpretability** (SHAP/LIME explainability for clinical decision support)
- **Prospective validation** on new patient cohorts
- **Develop diagnostic tool** for clinical deployment (web app, lab integration)

---


## References

1. CDC. "ME/CFS Basics." *Center for Disease Control & Prevention*, May 2024. https://www.cdc.gov/me-cfs/about/index.html

2. Grach et al. "Diagnosis and Management of Myalgic Encephalomyelitis/Chronic Fatigue Syndrome." *Mayo Clin Proc*, 98(10), Oct 2023.

3. Helliwell et al. "Changes in DNA methylation profiles of ME/CFS patients." *Clin Epigenet*, 12:167, 2020.

4. Germain et al. "An In-Depth Exploration of the Autoantibody Immune Profile in ME/CFS." *Int. J. Mol. Sci.*, 26(6):2799, 2025.

5. Mandarano et al. "Myalgic encephalomyelitis/chronic fatigue syndrome patients exhibit altered T cell metabolism." *J. Clin. Invest.*, 2020.

---

## Team

- **Cang Liang** - Machine Learning Engineering, Model Development
- **Manali Raut** - Data Processing, Statistical Analysis
- **Nicolas Wirth** - Project Coordination, Clinical Integration

---

## License

MIT License - see LICENSE file for details

---

## Acknowledgments

- **mapMECFS Database** for providing biomarker datasets
- **CDC** for epidemiological data and diagnostic criteria
- **Duke & Jackson Labs** for comparative analysis frameworks
- **Open Medicine Foundation** for patient advocacy and funding support
