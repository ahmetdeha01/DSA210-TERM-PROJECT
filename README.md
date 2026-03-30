# deharepolink
# A Data-Driven Analysis of Umrah Participation Patterns in Saudi Arabia

**DSA 210 – Introduction to Data Science (Spring 2025–2026)**  
**Student:** [Your Name]  
**Project Proposal**

---

## Motivation

This project aims to analyze Umrah participation patterns in Saudi Arabia using publicly available official statistics. I am interested in understanding how Umrah participation differs across demographic and behavioral categories such as gender, nationality, age group, region, entry point, and visitor type. Since Umrah is a major religious activity that attracts both domestic and international participants, I believe it is meaningful to explore how these groups differ and what factors may be associated with changes in participation patterns over time.

---

## Data Sources

The main dataset for this project will be collected from the **Saudi General Authority for Statistics (GASTAT)** Umrah Statistics page. I plan to use official quarterly and, where available, monthly statistical tables related to **internal and external Umrah performers**. These reports provide variables such as:

- number of Umrah performers
- gender distribution
- nationality
- age groups
- entry points
- accommodation type
- accompanying group type
- administrative regions
- monthly counts
- Medina visitor statistics

The data will be collected from official reports and tables and then converted into a structured tabular format such as CSV for analysis in Python.

---

## Planned Enrichment

Since this project uses a **publicly available dataset**, I will enrich it with additional public data sources.

### 1. Regional Population Data
I will use population data for Saudi administrative regions to normalize internal Umrah participation by region. Instead of only comparing raw counts, I will also calculate region-based participation rates, which will make regional comparisons more meaningful.

### 2. Weather Data
I will include weather-related data, such as temperature and possibly humidity, for **Makkah** and **Madinah**. This will help me investigate whether weather conditions are associated with monthly or quarterly variation in Umrah participation and Medina visits.

These enrichments will allow the project to go beyond descriptive summaries and support more informative statistical analysis.

---

## Data Characteristics

The dataset is expected to include multiple observations across different **time periods** and **categories**, rather than a single summary table. Depending on data availability, observations may be organized by:

- quarter
- month
- region
- gender
- nationality
- age group
- visitor type
- entry point
- accommodation type

This means the final dataset will contain grouped observations across several dimensions, which will be suitable for exploratory data analysis, visualization, and hypothesis testing.

---

## Planned Analysis

The project will follow the main steps of the data science pipeline:

1. **Data collection and cleaning**  
   Collect the official Umrah statistics and enrichment datasets, then convert them into structured and analyzable tables.

2. **Exploratory Data Analysis (EDA)**  
   Visualize distributions and patterns across gender, nationality, age groups, regions, and time.

3. **Hypothesis Testing**  
   Test whether internal and external Umrah participation differs significantly across selected categories such as gender, age, and region.

4. **Machine Learning (if data structure is sufficient)**  
   If the collected data is detailed enough, I plan to apply machine learning methods to classify or model participation patterns.

---

## Expected Contribution

This project will provide a data-driven view of Umrah participation patterns in Saudi Arabia and show how official statistics can be enriched with contextual data such as weather and population. The final analysis will aim to identify meaningful demographic, regional, and temporal differences between internal and external Umrah performers.
