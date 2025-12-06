<div align="center">

#  Predicting Diabetes Disease Progression Using Machine Learning  

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
### 4.0 Additional Model Evaluation: Linear Regression Fit and Error Metrics
To establish a baseline, I first fit a simple Linear Regression model using BMI as the sole predictor of one-year diabetes progression. This approach provides a clear view of how well a purely linear relationship captures the underlying trend in the data before introducing more complex models such as Ridge, Lasso, and Random Forest Regression. The linear model achieved an R² of **0.35**, meaning BMI alone explains approximately 35% of the variation in disease progression. To further evaluate predictive performance, I computed error metrics for the Random Forest model, which outperformed the linear approach. On the held-out test set, Random Forest achieved a Mean Absolute Error of **47.80**, a Mean Squared Error of **3532.62**, a Root Mean Squared Error of **59.36**, and an R² of **0.39**, indicating moderate accuracy and an improvement over the single-predictor linear model. To assess model stability, I conducted 5-fold cross-validation, which produced consistent RMSE values across folds, demonstrating that the model generalizes reliably. After evaluating all models, the following test MSE values were obtained:

| Model             | Test MSE           |
| ----------------- | ------------------ |
| Linear Regression | 2821.75            |
| Ridge Regression  | 3112.97            |
| Lasso Regression  | 2814.06            |
| Random Forest     | **2808.84 (Best)** |

Together, these results show that while BMI alone provides a meaningful linear signal, more flexible modeling approaches—especially Random Forest—capture additional structure in the data and yield stronger predictive performance.

**Figure 2 : Linear Regression Line Fit**
![plot](/assets/IMG/linear_regression_bmi.png)

*Figure 2: Figure X. Linear Regression of BMI vs. Disease Progression.
This plot shows a linear regression line (red) fitted to the relationship between BMI and one-year disease progression, with individual patients represented as scatter points. The positive slope indicates that higher BMI values are generally associated with greater progression severity. The wide vertical spread around the line reflects substantial variability, consistent with the model’s modest R² value (0.35).




---


### Figure 3. Random Forest Feature Importances

![Random Forest Feature Importances](/assets/IMG/rf_feature_importance.png)

*Figure 3: Random Forest feature importances for the 10 baseline clinical predictors. Features with higher importance values contribute more strongly to predictions of diabetes progression; BMI is the most influential variable, followed by several serum markers and blood pressure.*

### 4.1 Analysis of Model Performance
Among the models evaluated, Random Forest Regression achieved the lowest Mean Squared Error, indicating that it captured the underlying structure of the data more effectively than the linear methods. Its superior performance suggests that the relationship between the clinical predictors and the progression outcome includes nonlinear components or interaction effects that linear models cannot fully represent.

Lasso Regression performed slightly better than standard Linear Regression, reflecting the benefit of L1 regularization in reducing noise and emphasizing the predictors that carry the strongest independent signal. By shrinking weaker coefficients toward zero, Lasso was able to stabilize the model and improve generalization on the test set. Ridge Regression, in contrast, exhibited the highest MSE among the models. This result suggests that the type of regularization it applies—penalizing large coefficients but retaining all predictors—was less suited to this particular dataset, where eliminating weaker features appears to be more beneficial than merely shrinking them.

### 4.2 Interpretation
Although Random Forest achieved the best overall performance, the differences between the models were relatively modest, which implies that the dataset is largely governed by linear relationships with only mild nonlinearities. The strong performance of Lasso further supports this interpretation; by selectively reducing the influence of less informative predictors, it highlights the importance of core features such as BMI, blood pressure, and specific serum markers while suggesting that other variables contribute minimally to the progression outcome.
Although Random Forest achieved the best overall performance, the differences between the models were relatively modest, which implies that the dataset is largely governed by linear relationships with only mild nonlinearities. The strong performance of Lasso further supports this interpretation; by selectively reducing the influence of less informative predictors, it highlights the importance of core features such as BMI, blood pressure, and specific serum markers while suggesting that other variables contribute minimally to the progression outcome.

Random Forest’s slight advantage likely arises from its ability to model interactions and threshold effects that linear methods cannot express. For example, the combined influence of BMI with certain biochemical markers may produce subtle nonlinear patterns that the ensemble of decision trees is better equipped to detect. The improvement, however, is not dramatic, reinforcing the idea that the clinical predictors in this dataset relate to disease progression in a mostly linear but not perfectly linear manner.

### 4.3 Feature Importance
The feature-importance results from the Random Forest model provide meaningful insight into which clinical variables most strongly influence diabetes disease progression. Among all predictors, body mass index (BMI) emerged as the most influential feature, which is consistent with the well-established relationship between excess adiposity, insulin resistance, and metabolic dysregulation. High BMI is strongly associated with impaired glucose uptake and chronic low-grade inflammation, both of which contribute to worsening diabetic outcomes over time.

Several of the serum biochemical markers (s1–s6) also ranked prominently, suggesting that underlying metabolic and lipid-related processes meaningfully shape one-year progression trajectories. These markers often reflect insulin sensitivity, hepatic function, inflammatory pathways, and lipid metabolism—physiological domains closely tied to diabetes severity. The prominence of these features reinforces the multifactorial nature of the disease and highlights the value of using multiple clinical predictors rather than relying on any single measurement.

Blood pressure and additional demographic variables contributed to a lesser, yet still measurable extent. This pattern aligns with clinical expectations: while demographic factors provide important contextual information, metabolic indicators such as BMI and serum markers play a more direct role in shaping short-term progression. The distribution of importances also supports earlier findings that the dataset is only mildly nonlinear; although Random Forest captures interactions and threshold effects, the magnitude of nonlinear contributions is moderate rather than dominant.

Overall, the feature-importance analysis complements the model-performance results by revealing that diabetes progression within this dataset is influenced most strongly by metabolic stress indicators, particularly BMI and specific serum biomarkers. These findings help explain why models with embedded feature selection—such as Lasso Regression—performed well, and why Random Forest achieved the best overall accuracy by capturing subtle interactions among the most predictive clinical variables. 



---

## 5. Discussion

### 5.1 Insights from Linear and Regularized Models
The performance of the linear and regularized regression models provides valuable insight into the structure of the dataset. Standard Linear Regression served as a strong baseline, reinforcing the idea that the relationship between the predictors and the progression outcome is largely linear. Lasso Regression exceeded the performance of the baseline model, suggesting that selectively reducing the influence of weaker predictors improved its ability to generalize. Many of the serum biochemical markers are known to be interrelated, and the stronger shrinkage effect imposed by L1 regularization likely helped mitigate multicollinearity by focusing the model on the most informative physiological indicators. Ridge Regression, however, performed less effectively than both Linear Regression and Lasso, indicating that uniformly shrinking all coefficients was less advantageous than allowing certain predictors to be reduced more aggressively while preserving others.

### 5.2 Insights from Random Forest
Random Forest Regression ultimately achieved the best predictive performance, indicating that the clinical variables in the dataset contain subtle nonlinearities and interaction effects that linear models fail to capture. The ensemble of decision trees is well suited to uncovering threshold behaviors—such as sharp changes in progression risk at certain BMI or serum marker levels—as well as interactions among physiological variables that jointly influence metabolic outcomes. The modest but consistent improvement offered by Random Forest aligns with expectations for biomedical datasets, where complex biological processes rarely adhere strictly to linear patterns.

### 5.3 Model Limitations
Despite the useful insights provided by the models, several limitations should be acknowledged. The dataset contains only 442 samples, which constrains the complexity that nonlinear models can effectively learn without overfitting. Although the predictors are standardized, they are not normalized in terms of clinical meaning, which makes interpreting coefficient magnitudes or comparing variable strength across domains more difficult. Additionally, the target variable reflects only a single year of disease progression, offering a limited view of the long-term trajectory of type 2 diabetes, which often unfolds over many years or decades. These constraints do not diminish the value of the analysis but highlight the need for caution when extrapolating beyond the scope of the dataset.

### 5.4 Strengths of This Approach
The overall approach adopted in this study offers several strengths. By comparing multiple modeling frameworks—ranging from simple linear methods to regularized regression and ensemble-based models—the analysis provides a comprehensive view of how different assumptions influence predictive accuracy. The consistent use of Mean Squared Error enables a clear and objective comparison across models, while the standardized modeling pipeline ensures that the results are reproducible and transparent. Furthermore, the dataset’s structure and scale make it an effective educational tool, illustrating key concepts in regression, regularization, and nonlinear modeling within a real clinical context.

---

## 6. Conclusion
The overall approach adopted in this study offers several strengths. By comparing multiple modeling frameworks—ranging from simple linear methods to regularized regression and ensemble-based models—the analysis provides a comprehensive view of how different assumptions influence predictive accuracy. The consistent use of Mean Squared Error enables a clear and objective comparison across models, while the standardized modeling pipeline ensures that the results are reproducible and transparent. Furthermore, the dataset’s structure and scale make it an effective educational tool, illustrating key concepts in regression, regularization, and nonlinear modeling within a real clinical context.

---

## 7. References
American Diabetes Association. Standards of Medical Care in Diabetes—2022. Diabetes Care, vol. 45, suppl. 1, 2022 

Breiman, Leo. “Random Forests.” Machine Learning, vol. 45, 2001, pp. 5–32.
Efron, Bradley, et al. “Least Angle Regression.” The Annals of Statistics, vol. 32, no. 2, 2004, pp. 407–499 

Hastie, Trevor, et al. The Elements of Statistical Learning: Data Mining, Inference, and Prediction. 2nd ed., Springer, 2009 

Knowler, William C., et al. “Reduction in the Incidence of Type 2 Diabetes with Lifestyle Intervention or Metformin.” The New England Journal of Medicine, vol. 346, no. 6, 2002, pp. 393–403 

“Diabetes.” World Health Organization, 10 Nov. 2021, https://www.who.int/news-room/fact-sheets/detail/diabetes 

“Diabetes Dataset Description.” Originally Published as Part of the LARS Paper by Efron et al., 2004. Dataset accessed via scikit-learn 
