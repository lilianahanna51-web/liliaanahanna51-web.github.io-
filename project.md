<div align="center">

# Predicting Diabetes Disease Progression Using Machine Learning  

### Liliana Hanna • AOS C204 Final Project  

[Home](./) • [Project Report](project.md)

</div>

---


## 1. Introduction

Type 2 diabetes is a complex metabolic disease marked by impaired glucose regulation, progressive insulin resistance, and a wide range of long-term complications that affect the cardiovascular, renal, ocular, and nervous systems. Identifying individuals who are likely to experience accelerated disease progression is a central clinical goal, as early detection enables physicians to tailor treatment plans, initiate preventive strategies, and reduce the burden of future complications.

Accurately predicting the course of diabetes remains challenging because the disease is influenced by many interdependent factors. Physiological indicators such as body mass index (BMI), blood pressure, and serum biomarkers capture different aspects of a patient’s metabolic state, while demographic variables provide additional context. No single measurement is sufficient on its own; rather, it is the combination of these predictors that offers the most meaningful insight into disease trajectory. Machine-learning methods are well suited for this task, as they can integrate diverse clinical variables, uncover underlying patterns, and generate data-driven estimates of future outcomes.

In this project, I analyze the scikit-learn Diabetes dataset, which includes 442 patients and ten standardized baseline clinical predictors, to build models that forecast one-year diabetes disease progression. This dataset is particularly well suited for methodological comparison because it contains real clinical measurements, offers a continuous target variable appropriate for regression analysis, and provides predictors that have already been standardized, simplifying model training and interpretation.

The central aim of this study is to evaluate how effectively different supervised learning approaches—Linear Regression, Ridge Regression, Lasso Regression, and Random Forest Regression—can predict future disease progression from these baseline measurements. Through this comparison, I examine both linear and nonlinear modeling behavior and assess how regularization, feature selection, and interaction effects influence predictive performance.

This analysis draws directly on core concepts from AOS C204, including the bias–variance tradeoff, the role of regularization in stabilizing models, the importance of proper train–test splitting, and the use of Mean Squared Error (MSE) as a consistent evaluation metric. By applying these techniques to a real biomedical context, this project highlights both the strengths and limitations of classical and ensemble-based machine-learning models in forecasting clinical outcomes.

---

## 2. Data Description

This study draws on the widely used diabetes dataset provided through the scikit-learn machine-learning library. The dataset contains baseline clinical measurements from 442 patients, represented through ten standardized predictors that capture essential aspects of metabolic and physiological health. These variables include patient age, a sex indicator, body mass index (BMI), blood pressure, and six serum biomarkers that together reflect insulin sensitivity, lipid metabolism, and broader metabolic function. All predictors have been normalized to have a mean of zero and unit variance, a preprocessing step that ensures each feature contributes on a comparable scale and enables regularized models such as Ridge and Lasso Regression to operate effectively. The structured, clinically meaningful nature of this dataset makes it well suited for evaluating the predictive performance of different regression-based machine-learning approaches.

### 2.1 Target Variable
The outcome of interest in this study is a continuous measure of diabetes disease progression assessed one year after the baseline clinical evaluation. This quantitative score reflects the degree to which a patient’s condition has advanced over that period, with higher values indicating more severe progression. Because it is a continuous variable, it is well suited for regression-based modeling and allows for direct comparison of model accuracy using metrics such as Mean Squared Error.

### 2.2 Exploratory Visualization
Before training predictive models, it is essential to examine how individual predictors relate to the progression outcome. To do this, I generated a scatter plot comparing body mass index (BMI) with the disease progression score. BMI is a well-established indicator of metabolic health and is closely linked to insulin resistance, systemic inflammation, and overall disease burden in individuals with diabetes, making it a meaningful variable to examine first.
The visualization reveals a clear upward relationship: individuals with higher BMI values generally exhibit higher progression scores after one year, suggesting that elevated adiposity is associated with worsening disease trajectories. Notably, the spread of progression scores becomes wider among patients with high BMI, indicating increased variability that may reflect nonlinear or interaction effects between physiological predictors. These patterns justify the need to explore both linear and nonlinear modeling approaches, as they hint at underlying complexity that simpler models may not fully capture.
### Figure 1. BMI vs Disease Progression

![BMI vs Disease Progression](/assets/IMG/bmi_progression-2.png)

*Figure 1: Scatter plot showing the relationship between BMI and one-year diabetes disease progression. Higher BMI is generally associated with higher progression scores, with wider variability at higher BMI values.*

---

## 3. Modeling Approach

All models in this study were developed using a consistent and transparent workflow to ensure fair comparison. The dataset was first divided into a training set and a held-out test set, with 70% of the observations used for model fitting and the remaining 30% reserved exclusively for evaluation. A fixed random state was applied to maintain reproducibility across runs. Each model was then trained on the standardized predictors using either default or lightly tuned hyperparameters, after which predictions were generated for the test data. Model performance was assessed using Mean Squared Error (MSE), a widely used metric for continuous outcomes that penalizes large deviations and provides a straightforward means of comparing regression approaches. This standardized pipeline allows for a clear evaluation of how different modeling assumptions influence predictive accuracy.

### 3.1 Linear Regression
Linear Regression serves as the foundational baseline for this analysis. It assumes a strictly linear relationship between each predictor and the progression outcome, with all variables contributing additively and without interaction effects. Because the predictors in the Diabetes dataset are standardized, the resulting coefficients can be interpreted on a comparable scale, providing insight into the relative influence of each clinical measurement. Although simple, this model establishes a useful benchmark and helps clarify whether more complex methods offer meaningful advantages.
### 3.2 Ridge Regression
Ridge Regression extends the linear model by adding an L2 penalty on the magnitude of the coefficients. This form of regularization discourages extreme parameter values and is particularly beneficial when predictors are correlated—a common occurrence among physiological measurements such as serum markers and metabolic indicators. By shrinking coefficients toward zero without eliminating them entirely, Ridge reduces variance and enhances model stability, especially in smaller datasets. The trade-off is a slight increase in bias, but Ridge often improves generalization when multicollinearity is present.
### 3.3 Lasso Regression
Lasso Regression applies an L1 regularization penalty, which has the unique property of forcing some coefficients to exactly zero. This results in a more parsimonious model that effectively performs embedded feature selection. In a biomedical dataset where several clinical variables may overlap in the information they provide, Lasso helps identify which predictors carry the strongest independent signal. This sparsity can reduce noise, improve interpretability, and enhance test-set performance by preventing the model from overfitting to weaker or redundant features.
### 3.4 Random Forest Regression
Random Forest Regression represents a fundamentally different modeling strategy. Instead of relying on linear relationships, it constructs an ensemble of decision trees—each trained on a bootstrap sample of the data—and aggregates their predictions. This structure allows Random Forests to capture nonlinear patterns, interaction effects, and threshold behaviors that linear or regularized models cannot express. The method is well suited for biomedical data, where physiological responses often do not follow strictly linear trajectories. Random Forests are also robust to outliers and feature scaling, and they provide a natural measure of feature importance, offering valuable insight into which clinical variables most strongly influence disease progression.
---

## 4. Results

After training all models and evaluating them on the test set, the following MSE values were obtained:

| Model                | Test MSE  |
|----------------------|-----------|
| **Linear Regression** | **2821.75** |
| **Ridge Regression**  | 3112.97 |
| **Lasso Regression**  | **2814.06** |
| **Random Forest**     | **2808.84** (Best) |
### Figure 2. Random Forest Feature Importances

![Random Forest Feature Importances](/assets/IMG/rf_feature_importance.png)

*Figure 2: Random Forest feature importances for the 10 baseline clinical predictors. Features with higher importance values contribute more strongly to predictions of diabetes progression; BMI is the most influential variable, followed by several serum markers and blood pressure.*

### 4.1 Analysis of Model Performance

- **Random Forest achieved the lowest MSE**, outperforming the linear models.  
  This suggests the presence of nonlinear patterns in the predictors.

- **Lasso Regression slightly outperformed Linear Regression**, indicating that L1 regularization helped reduce noise and emphasize key predictors.

- **Ridge Regression performed the worst**, demonstrating that L2 regularization did not suit the dataset as well as L1.

### 4.2 Interpretation

- The relatively small differences between Linear Regression, Lasso, and Random Forest imply the dataset is **mostly linear**, with mild nonlinear effects.  
- Lasso’s strong performance suggests that certain predictors (e.g., BMI, blood pressure, serum markers) play more important roles, while others contribute minimally.
- Random Forest’s slight edge indicates that interactions between features — for example, BMI × serum marker effects — may modestly improve predictive power.

---

## 5. Discussion

### 5.1 Insights from Linear and Regularized Models

Linear Regression provided a strong baseline performance, reflecting the overall linear structure of the diabetes dataset. Lasso Regression performed even better, implying that **reducing less important coefficients improved generalization**. Serum markers, which often exhibit high multicollinearity, may have benefitted from the stronger shrinkage applied by Lasso.

Ridge Regression performed worse than both Linear and Lasso, suggesting that dispersed shrinkage across all coefficients was less effective than the selective shrinkage provided by Lasso.

### 5.2 Insights from Random Forest

Random Forest was the top-performing model. This suggests:

- Some degree of nonlinearity is present  
- Predictor interactions matter  
- Decision tree ensembles capture subtle patterns

Random Forest models are often advantageous for biomedical datasets where physiological responses are rarely perfectly linear.

### 5.3 Model Limitations

- The dataset has only **442 samples**, which limits the complexity and capacity of nonlinear models.  
- Features are standardized but not normalized across medical scales, making clinical interpretation more challenging.  
- The target variable captures only **one-year progression**, which may not fully reflect long-term metabolic changes.

### 5.4 Strengths of This Approach

- Multiple models were compared systematically  
- The MSE metric provides a clear performance comparison  
- The modeling pipeline is reproducible and transparent  
- The dataset is well-suited for educational machine-learning applications  

---

## 6. Conclusion

This study demonstrates that machine learning can moderately predict diabetes disease progression using only baseline clinical measurements.

**Key findings:**

- **Random Forest Regression was the most accurate model (MSE = 2808.84).**  
- **Lasso Regression outperformed both Linear and Ridge Regression**, showing the value of variable selection.  
- **Ridge Regression performed least effectively**, suggesting L2 regularization is not ideal for this dataset.  
- The dataset appears **mostly linear**, but slight nonlinearities improve performance modestly.

This project highlights how classical and ensemble-based machine-learning models can be applied to healthcare prediction tasks and underscores the value of comparing multiple approaches before drawing conclusions.

---

## 7. References

- Scikit-learn Diabetes Dataset Documentation  
- Hastie, Tibshirani & Friedman (2009), *The Elements of Statistical Learning*  
- AOS C204 Lecture Notes, UCLA  
