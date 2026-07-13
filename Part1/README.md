# Part 1 – Data Acquisition, Cleaning, and Exploratory Data Analysis

## Student Information

**Course:** Applied AI & ML Essentials – Capstone Project

**Part:** Part 1 – Data Acquisition, Cleaning and Exploratory Data Analysis

---

# Dataset Description

This project uses the **Medical Cost Personal Dataset (Insurance Dataset)**.

The dataset contains demographic and health-related information of individuals along with their medical insurance charges.

### Dataset Features

| Feature | Description | Data Type |
|----------|-------------|-----------|
| age | Age of the insured person | Numeric |
| sex | Gender | Categorical |
| bmi | Body Mass Index | Numeric |
| children | Number of dependent children | Numeric |
| smoker | Smoking status | Categorical |
| region | Residential region | Categorical |
| charges | Medical insurance charges (Target Variable) | Numeric |

**Total Records:** 1338

**Total Columns:** 7

---

# Objective

The objective of this project is to clean, preprocess and explore the insurance dataset before applying machine learning models in later parts of the capstone.

The work performed includes:

- Data loading
- Missing value analysis
- Duplicate detection
- Data type correction
- Descriptive statistics
- Skewness analysis
- Outlier detection
- Exploratory visualizations
- Correlation analysis
- Grouped aggregation
- Saving the cleaned dataset

---

# Libraries Used

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

# Data Loading

The dataset was loaded using

```python
pd.read_csv("insurance.csv")
```

The first five rows, shape, column names, data types and dataset information were displayed successfully.

---

# Missing Value Analysis

Missing values were computed using

```python
df.isnull().sum()
```

and their percentages were calculated.

### Observation

No missing values were present in the dataset.

Therefore,

- No columns exceeded the 20% missing value threshold.
- No imputation was required.

### Why Median Instead of Mean?

If missing values had existed, the median would have been preferred because it is robust to extreme values and skewed distributions. Unlike the mean, the median is not significantly affected by outliers.

---

# Duplicate Detection

Duplicate rows were identified using

```python
df.duplicated().sum()
```

Duplicate rows were removed using

```python
df.drop_duplicates()
```

### Observation

One duplicate record was removed from the dataset.

The percentage of missing values remained unchanged after duplicate removal.

---

# Data Type Correction

The dataset was inspected for incorrect data types.

No numeric columns were incorrectly stored as object.

The following columns were converted from object to category datatype:

- sex
- smoker
- region

This reduced the overall memory usage while preserving all information.

---

# Descriptive Statistics

Descriptive statistics including

- Mean
- Median
- Standard Deviation
- Minimum
- Maximum
- Quartiles

were generated for all numeric columns using

```python
df.describe()
```

---

# Skewness Analysis

Skewness was calculated for every numeric column.

### Most Skewed Column

**charges**

### Skewness

Positive (approximately 1.52)

### Interpretation

The insurance charges distribution is positively skewed.

Most customers have relatively lower insurance charges while a small number of customers have very high medical expenses.

Because of this positive skew, the median provides a better measure of central tendency than the mean.

---

# Outlier Detection (IQR Method)

Outliers were detected using the Interquartile Range (IQR) method.

### BMI

Outliers detected: **9**

### Charges

Outliers detected: **139**

### Interpretation

The detected outliers were retained because they likely represent genuine high-cost insurance cases rather than erroneous data.

Removing them could negatively affect the performance of predictive models in later stages.

These outliers will be retained during Part 2 since several machine learning algorithms are naturally robust to such observations.

---

# Visualizations

The following visualizations were generated.

## 1. Line Plot

Displays insurance charges across dataset records.

Observation:

Insurance charges vary significantly, with a few very high-cost observations.

---

## 2. Bar Chart

Shows average insurance charges grouped by region.

Observation:

Average insurance charges differ slightly among regions.

---

## 3. Histogram

Displays the distribution of insurance charges.

Observation:

The histogram shows a positively skewed distribution with a long right tail.

---

## 4. Scatter Plot

Shows BMI versus insurance charges.

Observation:

A weak to moderate positive relationship exists between BMI and insurance charges.

Higher BMI values generally correspond to higher insurance costs.

---

## 5. Box Plot

Shows insurance charges grouped by smoking status.

Observation:

Smokers have significantly higher median insurance charges and much greater variability compared to non-smokers.

Smoking status appears to be a strong predictor of insurance cost.

---

## 6. Correlation Heatmap

A Pearson correlation heatmap was generated for all numeric variables.

### Highest Correlated Pair

Age and Charges

Correlation ≈ 0.30

### Interpretation

Although age and charges exhibit the strongest correlation among numeric variables, correlation does not necessarily imply causation.

Other variables such as smoking status, BMI, and overall health conditions may also influence insurance charges.

---

# Mean vs Median Comparison

The two most skewed variables were:

| Column | Mean | Median |
|---------|------|--------|
| Charges | 13279.12 | 9386.16 |
| Children | 1.10 | 1.00 |

Since charges is positively skewed, the median better represents the central tendency than the mean.

---

# Spearman Rank Correlation

Spearman correlation was computed and compared with Pearson correlation.

### Largest Differences

| Feature Pair | Difference |
|--------------|-----------|
| Age – Charges | 0.235 |
| BMI – Charges | 0.079 |
| Charges – Children | 0.065 |

### Interpretation

These differences suggest that some relationships are monotonic but not perfectly linear.

Spearman correlation is therefore useful for feature selection because it is more robust to skewed distributions and outliers.

---

# Grouped Aggregation

Grouped aggregation was performed using

```python
df.groupby("smoker")["charges"].agg(["mean","std","count"])
```

### Observation

- Smokers have the highest average insurance charges.
- Smokers also have the highest standard deviation.
- The ratio between the highest and lowest group mean indicates that smoking status carries strong predictive information.

High within-group standard deviation indicates that smoking status alone cannot perfectly predict insurance charges, although it remains an important feature.

---

# Cleaned Dataset

The cleaned dataset was saved as

```
cleaned_data.csv
```

This dataset will be used in Parts 2 and 3 for machine learning model development.

---

# Files Included

- Part1.ipynb
- insurance.csv
- cleaned_data.csv
- requirements.txt
- README.md
- figures/
  - line_plot.png
  - bar_chart.png
  - histogram.png
  - scatter_plot.png
  - box_plot.png
  - heatmap.png

---

# Conclusion

The dataset was successfully cleaned and analyzed.

Key findings include:

- No missing values were present.
- One duplicate record was removed.
- Categorical columns were optimized using category datatype.
- Insurance charges exhibit strong positive skewness.
- Significant outliers exist but were retained.
- Smoking status and age appear to be influential variables.
- The cleaned dataset is now ready for machine learning modelling in Part 2.
