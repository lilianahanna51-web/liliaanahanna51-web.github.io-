<div align="center">

### Hi, welcome to my AOS C204 final project!

# Predicting Diabetes Disease Progression Using Machine Learning

Liliana Hanna • AOS C204 Final Project  

[Home](./) • [Project Report](project.md)

</div>

---

<img align="right" width="320" src="/assets/IMG/bmi_progression-2.png">

This project examines how machine-learning models can predict one-year diabetes disease progression using baseline clinical measurements from the scikit-learn Diabetes dataset. By analyzing relationships between variables such as BMI, blood pressure, and serum biomarkers, the study compares the performance of Linear Regression, Ridge Regression, Lasso Regression, and Random Forest Regression to determine which approach provides the most accurate predictions. Through exploratory visualization, model evaluation using Mean Squared Error, and interpretation of feature importance, the project highlights how different modeling techniques capture the underlying patterns in metabolic health and disease progression.

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

