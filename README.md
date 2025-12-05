<div align="center">

### Hi, welcome to my AOS C204 final project!

# Predicting Diabetes Disease Progression Using Machine Learning

Liliana Hanna • AOS C204 Final Project  

[Home](./) • [Project Report](project.md)

</div>

---

<img align="right" width="320" src="/assets/IMG/bmi_progression-2.png">

This project investigates how machine-learning models can be used to **predict diabetes disease progression one year after baseline clinical measurements**. I use the scikit-learn Diabetes dataset, which includes 442 patients and 10 standardized clinical predictors such as body mass index (BMI), blood pressure, and serum biomarkers.

The main goals of this project are to:

- Compare multiple supervised learning models  
  – Linear Regression  
  – Ridge Regression  
  – Lasso Regression  
  – Random Forest Regression  
- Evaluate how accurately each method predicts one-year disease progression (using Mean Squared Error, MSE).  
- Identify which clinical predictors are most important for prediction (using Random Forest feature importance).  

This work connects ideas from AOS C204 — including regularization, bias–variance trade-off, train–test splitting, and nonlinear models — to a real biomedical application.

---

## Report

- 📄 **Full Project Report:** [Project Report](project.md)  

The report contains:

- Detailed data description and exploratory BMI vs. progression plot  
- Modeling pipeline and assumptions for each method  
- Comparison table of test MSE for all models  
- Random Forest feature-importance visualization  
- Discussion, limitations, and conclusions

---

## Preprocessed Data

The data used in this project is the **Diabetes dataset** from `sklearn.datasets.load_diabetes`, which provides:

- 442 patients  
- 10 standardized predictors (age, sex, BMI, blood pressure, and six serum markers)  
- 1 continuous target: **disease progression after one year**

All predictors are standardized (mean 0, variance 1), making them directly suitable for regularized regression methods such as Lasso and Ridge.

---

## Code

All analysis is implemented in Python using:

- `scikit-learn` for models and metrics  
- `pandas` / `numpy` for data handling  
- `matplotlib` for plots  

> *(Optional)* If you have a Colab notebook link, replace `YOUR_COLAB_LINK_HERE` below and remove this note.

- ▶️ **Colab notebook:** [View the code in Colab](YOUR_COLAB_LINK_HERE)

---

## About This Website

This site is built using **GitHub Pages** and written entirely in Markdown, using the AOS C204 project template.  
It hosts my final report and can later be extended with more machine-learning projects.

