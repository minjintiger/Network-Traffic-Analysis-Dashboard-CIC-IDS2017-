# Network Traffic Analysis Dashboard (CIC-IDS2017)

Interactive dashboard and statistical analysis pipeline for exploring network traffic behavior using the CIC-IDS2017 Wednesday dataset.  
This project focuses on data preprocessing, PCA, statistical validation, and visualization to understand how benign traffic differs from DoS attacks. The goal is analysis and understanding of the data structure, not building a classifier.

--------------------------------------------------
Overview
--------------------------------------------------

This project walks through a full exploratory analysis workflow:

- Dataset inspection and feature understanding
- Cleaning and preprocessing
- Outlier detection
- Normality testing
- PCA-based dimensionality reduction
- Correlation analysis
- Interactive visualization using Plotly Dash

Instead of static plots, I built an interactive dashboard so preprocessing decisions (cleaning, transformation, filtering) immediately change the visual outputs and PCA space.

The CIC-IDS2017 Wednesday dataset is heavily imbalanced and strongly non-Gaussian, so visualization and preprocessing are required before any meaningful modeling or IDS logic can be built.

--------------------------------------------------
Dataset
--------------------------------------------------

Source: CIC-IDS2017 benchmark  
File used: Wednesday-workingHours.pcap_ISCX.csv

Each row represents a bidirectional network flow with numerical flow statistics and a Label field:

BENIGN, DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest, Heartbleed

The dataset is highly imbalanced:
- ~69% BENIGN traffic
- ~29% DoS Hulk
- Remaining attacks are minority classes

This directly affects how distributions and PCA behave.

--------------------------------------------------
What I Implemented
--------------------------------------------------

1) Data Cleaning and Preprocessing

Multiple cleaning modes are available inside the dashboard:

- Keep Original
- Basic cleaning (drop invalid or non-positive values)
- Strict cleaning (remove rows with missing numeric fields)
- Mean imputation

The raw dataset contains invalid values, missing entries, and extreme spikes, so cleaning is necessary before running PCA or statistical tests.

2) Outlier Detection

Two filtering options:

- IQR filtering
- Z-score filtering

IQR filtering removes extreme flows that dominate scale without destroying the main traffic structure.

3) Statistical Analysis

I analyzed distribution behavior using:

- Histogram + KDE
- Regression plots
- Multivariate density views

Key observation: flow features are heavily skewed and not normally distributed.

4) Normality Testing

Implemented tests:

- Shapiro-Wilk
- Kolmogorov-Smirnov
- D’Agostino K²

All major features reject normality, confirming that the dataset is fundamentally non-Gaussian.

5) PCA (Dimensionality Reduction)

Pipeline:

- Standardize numeric features
- Run PCA
- Visualize PC1 vs PC2 colored by traffic label

PCA is used for visualization and sanity checking, not for building a final model.

6) Interactive Dashboard

Built entirely in Python using Plotly Dash.

Features:

- Cleaning selection
- Outlier removal
- Transformations (log1p, MinMax, Standard scaling)
- PCA visualization
- Numeric plots
- Categorical plots
- Storytelling dashboard

The dashboard updates in real time based on preprocessing choices.

--------------------------------------------------
Project Structure
--------------------------------------------------

.
├── Final_Term_Project.py
├── Wednesday-workingHours.pcap_ISCX.csv
├── CS_5764_Final_Term_Project_Report.pdf
└── README.md

--------------------------------------------------
How to Run
--------------------------------------------------

Place the following files in the same folder:

    Final_Term_Project.py
    Wednesday-workingHours.pcap_ISCX.csv

Then run:

    python Final_Term_Project.py

If the dataset location changes, update DATA_PATH inside the script.

--------------------------------------------------
Tech Stack
--------------------------------------------------

Python  
Plotly Dash  
Pandas  
NumPy  
SciPy  
Scikit-learn  
Plotly

--------------------------------------------------
Key Observations
--------------------------------------------------

- Network traffic features are heavy-tailed and strongly skewed.
- Cleaning and transformations significantly change PCA structure.
- Different DoS attacks occupy different regions in PCA space after scaling.
- Visualization is necessary before any modeling step on CIC-IDS2017.

--------------------------------------------------
Author
--------------------------------------------------

Minjin Kim  
Virginia Tech — CS 5764 Final Term Project
