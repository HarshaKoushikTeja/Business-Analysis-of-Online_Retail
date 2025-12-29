<div align="center">

# 🛒 Online Retail Analytics
## Comprehensive Data Science Project

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-Educational-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

**📊 Analyzing 541,909 transactions | 🎯 4,338 customers | 🌍 38 countries | 📈 5+ ML models**

---

</div>

## 📊 Project Overview

This project demonstrates a **comprehensive end-to-end data science analysis** of a large-scale online retail transaction dataset. Working with **541,909 initial records** (22.6 MB dataset), I performed extensive data cleaning, exploratory data analysis, statistical hypothesis testing, customer segmentation, and predictive modeling to derive actionable business insights.

**Business Theme:** Inventory Optimization & Customer Retention Strategy

---

## 🎯 Dataset Overview

<div align="center">

### 📦 Dataset Statistics at a Glance

```
    📊 DATASET METRICS        

  Initial Records:    541,909 transactions    
  File Size:          22.6 MB                 
  Time Period:        Dec 2010 - Dec 2011     
  Geographic Scope:   38 countries            
  Product Catalog:    4,070 unique items      
  Clean Records:      392,692 transactions    
  Unique Customers:   4,338 customers         
```

</div>

- **Initial Dataset Size**: 541,909 transactions
- **File Size**: 22.6 MB (Excel format)
- **Data Period**: December 2010 - December 2011
- **Geographic Coverage**: 38 countries
- **Products**: 4,070 unique stock codes
- **Final Clean Dataset**: 392,692 transactions from 4,338 customers
- **Data Quality Challenges Handled**:
  - 135,080 missing CustomerIDs (24.93%)
  - 1,454 missing product descriptions
  - 10,624 return transactions (negative quantities)
  - Duplicate records removed

---

## 🏗️ Project Architecture


### Analysis Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                      📊 RAW DATA                           │
│           541,909 transactions | 22.6 MB                    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│        🧹 CLEANING & FEATURE ENGINEERING                   │
│  • Missing value handling (24.93% CustomerIDs)              │
│  • Duplicate removal & data validation                      │
│  • Feature creation (Date, Month, Year, DayOfWeek)          │
│  • Final dataset: 392,692 clean transactions                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         📈 EDA & HYPOTHESIS TESTING                        │
│  • Geographic revenue analysis (38 countries)               │
│  • Time series analysis & seasonality                       │
│  • 5 Statistical hypothesis tests (T-test, ANOVA, Chi²)     │
│  • Data visualization & pattern identification              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              🎯 RFM SEGMENTATION                           │
│  • Recency, Frequency, Monetary analysis                    │
│  • 5 customer segments (4,338 customers)                    │
│  • Champions, Loyalists, At Risk, Hibernating, etc.         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  🔄 CLUSTERING                             │
│  • Optimal K selection (K=2, Silhouette: 0.4328)            │
│  • Model comparison (K-Means, Hierarchical, GMM)            │
│  • 3D cluster visualization in RFM space                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            🤖 PREDICTIVE MODELING                          │
│  • Customer retention prediction                            │
│  • Time-series split (holdout method)                       │
│  • 3 ML models (Logistic Regression, Decision Tree, RF)     │
│  • Best model: Logistic Regression (ROC-AUC: 0.7221)        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔍 Key Features & Analysis

### 1. **Large-Scale Data Cleaning & Quality Assessment**
- Processed over **half a million records** with comprehensive data quality reporting
- Handled missing values, duplicates, and data inconsistencies
- Feature engineering: date parsing, temporal features (month, year, day of week)
- Created clean sales dataset with proper data types and validations

### 2. **Geographic Revenue Analysis**
- Interactive choropleth maps using Plotly (38 countries analyzed)
- Country-level revenue distribution and customer segmentation
- Top 10 countries revenue visualization

### 3. **Time Series Analysis**
- Daily revenue trends over 12-month period
- 30-day moving average calculations
- Monthly revenue patterns and seasonality detection
- Temporal pattern identification

### 4. **RFM Customer Segmentation**
- Analyzed **4,338 customers** using Recency, Frequency, Monetary metrics
- Customer categorization:
  - **Champions**: 1,677 customers (38.7%)
  - **Potential Loyalists**: 935 customers (21.6%)
  - **Hibernating/Lost**: 806 customers (18.6%)
  - **At Risk**: 485 customers (11.2%)
  - **Loyal Customers**: 435 customers (10.0%)

### 5. **Statistical Hypothesis Testing (5 Hypotheses)**
- **H1**: High-frequency vs Low-frequency monetary value (T-test) - **REJECTED** (p < 0.001)
  - High-frequency mean: £2,907.99 | Low-frequency mean: £411.25
  - T-statistic: 8.7708, P-value: 2.51e-18
- **H2**: Repeat vs Single purchase average order value (T-test) - **Not significant** (p = 0.4992)
  - Repeat customers: £57.47 (n=2,845) | Single customers: £89.18 (n=1,493)
  - T-statistic: -0.6758, P-value: 0.4992
- **H3**: Weekly transaction patterns (ANOVA) - **REJECTED** (p < 0.001)
  - F-statistic: 17.1932, P-value: 4.74e-17
- **H4**: UK vs International order values (T-test) - **REJECTED** (p < 0.001)
  - International mean: £36.84 | UK mean: £20.86
  - T-statistic: 22.9634, P-value: 1.45e-116
- **H5**: Country-level return rates (Chi-square) - **REJECTED** (p < 0.001)
  - Chi-square: 53.6785, P-value: 5.57e-08
  - Countries analyzed: 11

### 6. **Customer Clustering Analysis**
- Optimal K selection using Silhouette Score (tested K=2 to K=5)
  - K=2: Silhouette = 0.4328 (Optimal)
  - K=3: Silhouette = 0.3365
  - K=4: Silhouette = 0.3375
  - K=5: Silhouette = 0.3162
- Model comparison across three algorithms (K=2):
  - **K-Means**: Silhouette Score = 0.4328, Davies-Bouldin = 0.8925 (Best)
  - **Hierarchical Clustering**: Silhouette Score = 0.4040, Davies-Bouldin = 0.9405
  - **Gaussian Mixture Model**: Silhouette Score = 0.2855, Davies-Bouldin = 1.0670
- 3D cluster visualization in RFM space

### 7. **Predictive Modeling - Customer Retention**
- **Objective**: Predict which customers will return in the next 3 months
- **Methodology**: Time-based holdout split  
                    - Training data: transactions before 2011-09-10  
                    - Test data: subsequent 90-day period  
                    - Prevents temporal data leakage
- **Models Evaluated** (with 5-fold Cross-Validation):
  - **Logistic Regression**: CV ROC-AUC = 0.7377 (±0.0179), Test Accuracy = 0.6702, Test ROC-AUC = 0.7221 (Best)
  - **Decision Tree**: CV ROC-AUC = 0.7256 (±0.0313), Test Accuracy = 0.6536, Test ROC-AUC = 0.7109
  - **Random Forest**: CV ROC-AUC = 0.7243 (±0.0174), Test Accuracy = 0.6572, Test ROC-AUC = 0.7219
- Feature importance analysis

---

## 📁 Project Files

```
- `final.ipynb` — End-to-end analysis (cleaning → modeling)
- `figures/` — Saved visualizations and interactive plots
- `requirements.txt` — Python dependencies
- `README.md` — Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+** (tested with Python 3.11)
- **Jupyter Notebook** or **JupyterLab**
- **8GB+ RAM** recommended for processing large dataset

### Installation

1. **Clone this repository:**
```bash
git clone <your-repo-url>
cd <repository-name>
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
source .venv/bin/activate  # On macOS/Linux
```

3. **Install required packages:**
```bash
pip install -r requirements.txt
```

### Running the Analysis

1. Open `final.ipynb` in Jupyter Notebook
2. The dataset `Online_Retail.xlsx` is included in the repository
3. Run all cells sequentially to execute the complete analysis
4. Generated visualizations will be saved in the `figures/` directory

**Note**: The complete analysis may take 10-15 minutes to run due to the large dataset size.

---

## 📈 Technical Highlights

### Data Processing Scale
- **Initial Records**: 541,909 transactions
- **Processing**: Handled 24.93% missing CustomerIDs
- **Data Cleaning**: Removed duplicates and invalid records
- **Final Dataset**: 392,692 clean transactions

### Machine Learning Models
- **Clustering**: 3 algorithms compared on 4,338 customers
- **Classification**: 3 models with cross-validation
- **Feature Engineering**: 6 engineered features for prediction

### Statistical Analysis
- **5 Hypothesis Tests**: T-tests, ANOVA, Chi-square tests
- **Significance Level**: α = 0.05
- **4 out of 5 hypotheses rejected** with strong statistical evidence

---

## 📊 Key Results & Insights

<div align="center">

### 🎯 Project Metrics

| Metric | Value | Badge |
|--------|-------|-------|
| **Dataset Size** | 541,909 records | ![Records](https://img.shields.io/badge/Records-541K-blue) |
| **Customers** | 4,338 unique | ![Customers](https://img.shields.io/badge/Customers-4.3K-green) |
| **Countries** | 38 countries | ![Countries](https://img.shields.io/badge/Countries-38-orange) |
| **Products** | 4,070 unique | ![Products](https://img.shields.io/badge/Products-4K-purple) |
| **ML Models** | 6 models tested | ![Models](https://img.shields.io/badge/Models-6-red) |
| **Hypothesis Tests** | 5 tests | ![Tests](https://img.shields.io/badge/Tests-5-yellow) |

</div>

### Dataset Statistics
- **Total Revenue**: Analyzed across 38 countries
- **Customer Base**: 4,338 unique customers (after data cleaning)
- **Product Catalog**: 4,070 unique products
- **Time Period**: 12 months of transaction data (Dec 2010 - Dec 2011)

### Business Insights
1. **High-frequency customers** spend significantly more (£2,907.99 vs £411.25) - **Statistically significant**
2. **International customers** have higher average order values (£36.84 vs £20.86) - **Statistically significant**
3. **Transaction patterns** vary significantly by day of week (F-statistic: 17.19) - **Statistically significant**
4. **Return rates** differ significantly across countries (Chi-square: 53.68) - **Statistically significant**
5. **Repeat vs Single customers**: No significant difference in average order value (£57.47 vs £89.18) - **Not significant**
6. **Optimal customer segmentation** achieved with 2 clusters (Silhouette Score: 0.4328)

### Model Performance

<div align="center">

| Model Type | Algorithm | Performance | Status |
|------------|-----------|-------------|--------|
| **Clustering** | K-Means | Silhouette: 0.4328, Davies-Bouldin: 0.8925 | ✅ Best |
| **Clustering** | Hierarchical | Silhouette: 0.4040, Davies-Bouldin: 0.9405 | ⚠️ Second |
| **Clustering** | Gaussian Mixture | Silhouette: 0.2855, Davies-Bouldin: 1.0670 | ⚠️ Third |
| **Classification** | Logistic Regression | Test ROC-AUC: 0.7221, CV ROC-AUC: 0.7377 | ✅ Best |
| **Classification** | Random Forest | Test ROC-AUC: 0.7219, CV ROC-AUC: 0.7243 | ⚠️ Close Second |
| **Classification** | Decision Tree | Test ROC-AUC: 0.7109, CV ROC-AUC: 0.7256 | ⚠️ Third |
| **Validation** | 5-fold CV | Consistent | ✅ Stable |


</div>

---

## 🛠️ Technologies & Libraries

<div align="center">

### 🎯 Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557C?logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-3776AB?logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.14+-3F4F75?logo=plotly&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-1.10+-8CAAE6?logo=scipy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

</div>

### Core Data Science Stack
- **Python 3.11**: Primary programming language
- **Pandas**: Large-scale data manipulation and analysis
- **NumPy**: Numerical computing and array operations

### Visualization
- **Matplotlib**: Static visualizations
- **Seaborn**: Statistical visualizations
- **Plotly**: Interactive visualizations (HTML exports)

### Machine Learning
- **Scikit-learn**: 
  - Clustering (K-Means, Hierarchical, Gaussian Mixture)
  - Classification (Logistic Regression, Decision Tree, Random Forest)
  - Model evaluation and cross-validation
  - Feature scaling and preprocessing

### Statistical Analysis
- **SciPy**: Statistical hypothesis testing (T-tests, ANOVA, Chi-square)

### Data I/O
- **OpenPyXL**: Excel file reading

---

## 🔁 Reproducibility Notes

- Random seeds are fixed where applicable
- Time-based split ensures no data leakage
- Results may vary slightly depending on library versions

---

## 📄 Output Files

All visualizations are automatically saved in the `figures/` directory:
- **Interactive HTML maps**: Geographic revenue distribution
- **High-resolution PNGs**: All static visualizations (300 DPI)
- **3D Interactive plots**: Cluster visualizations

---

## 💡 Key Learnings & Challenges

### Working with Large Datasets
- Efficient memory management when processing 500K+ records
- Optimized data cleaning pipelines for performance
- Handling missing data at scale (24.93% missing CustomerIDs)

### Statistical Rigor
- Proper hypothesis testing methodology
- Multiple comparison corrections
- Time-series validation for predictive models

### Business Application
- Translating technical analysis into actionable insights
- Customer segmentation for targeted marketing
- Predictive modeling for retention strategies

---

## 📝 Dataset Information

### Dataset Access
The dataset is publicly available from the UCI Machine Learning Repository.
Please download `Online_Retail.xlsx` and place it in the project root directory.

**Dataset Characteristics:**
- Real-world e-commerce transaction data
- Contains both sales and returns
- International customer base
- 12-month transaction history

---

## ⚠️ Limitations & Future Work

### Current Limitations
- **Dataset Scope**: Limited to one retailer's transaction data
- **Feature Availability**: No external demographic features (age, gender, income, etc.)
- **Temporal Scope**: Analysis covers only 12 months of data
- **Product Information**: Limited product attributes beyond basic descriptions

### Future Work & Enhancements
- **Survival Analysis**: Implement Cox proportional hazards model to predict customer lifetime duration
- **Customer Lifetime Value (LTV) Prediction**: Develop models to estimate long-term customer value
- **Deep Learning Models**: Explore neural networks (LSTM, Transformer) for time-series customer behavior prediction
- **Real-time Recommendations**: Build recommendation systems based on customer segments
- **Multi-retailer Analysis**: Extend analysis to compare patterns across multiple retailers
- **External Data Integration**: Incorporate demographic, economic, and seasonal data for richer insights
- **Advanced Clustering**: Experiment with DBSCAN, HDBSCAN, or other density-based clustering methods
- **Ensemble Methods**: Combine multiple models for improved prediction accuracy

---

## 👤 Author

<div align="center">

**👨‍💻 Harsha Koushik Teja Aila**<br>
**📧 harshaus33@gmail.com**

![GitHub](https://img.shields.io/badge/GitHub-Profile-black?logo=github)
![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)

**Project Type**: ![Academic](https://img.shields.io/badge/Academic-Project-blue) ![Portfolio](https://img.shields.io/badge/Portfolio-Project-green)  
**Course**: DSE501 by Prof. Rong Pan - Data Science & Engineering  
**Institution**: Arizona State University(ASU)

</div>

---

## 📜 License

This project is for educational and portfolio purposes. The dataset is publicly available from the UCI Machine Learning Repository.

---

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for providing the Online Retail dataset
- **Open-source community** for the excellent Python data science libraries
- **Instructors and peers** for feedback and guidance

---

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please feel free to reach out!

---

---

<div align="center">

### ⭐ Star this repo if you find it helpful!

```
🚀 Transforming raw data into actionable insights
📊 541,909 transactions → Business intelligence
🎯 Machine Learning + Statistics = Better decisions
```

**Note**: This project demonstrates proficiency in handling large-scale datasets, statistical analysis, machine learning, and data visualization - essential skills for a data science career.

---

![Made with](https://img.shields.io/badge/Made%20with-Python-blue?logo=python&logoColor=white)
![Powered by](https://img.shields.io/badge/Powered%20by-Jupyter-orange?logo=jupyter&logoColor=white)
![Data](https://img.shields.io/badge/Data-UCI%20ML%20Repository-green)

**📈 From Data to Decisions** | **🔬 Science-Driven Analytics** | **💼 Business-Focused Results**

</div>
