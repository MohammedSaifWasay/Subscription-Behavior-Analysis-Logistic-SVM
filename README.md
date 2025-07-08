# 📊 Subscription Behavior Analysis Using Logistic Regression and SVM

## 📌 Abstract
This study investigates declining magazine subscription rates by applying **Logistic Regression** and **Support Vector Machine (SVM)** models on customer data. Key predictors such as digital engagement, customer tenure, and income levels were identified. The models demonstrated strong predictive capabilities, providing actionable insights for targeted marketing and customer retention strategies.

---

## 📖 Introduction

### Background
The magazine industry faces challenges with declining subscriptions despite increased digital consumption. Data-driven approaches are essential to understand customer behavior and improve subscription rates.

### Objectives
- Identify factors influencing subscription decisions.
- Compare logistic regression and SVM model performances.
- Provide insights for strategic business improvements.

---

## 🧪 Methods

### Data Overview
- Dataset: 2,240 customer records from a marketing campaign.
- Features: Demographics (income, education, marital status), engagement metrics (web visits, purchases), and campaign responses.
- Target: Subscription response (subscribed or not).

### Data Preprocessing
- Imputed missing income values with median.
- One-hot encoded categorical variables.
- Engineered temporal features from customer dates to capture tenure and recency.
- Standardized numeric variables for model compatibility.

### Modeling
- **Logistic Regression** for interpretability and coefficient analysis.
- **Linear Kernel SVM** for capturing complex patterns.
- Train-test split and performance evaluation using accuracy, precision, recall, F1-score, and ROC-AUC.

---

## 📊 Results

### Logistic Regression
- Accuracy: **85%**
- Top Predictors: Number of web visits per month, catalog purchases, presence of teenagers at home.
- AUC: 0.87

### SVM
- Comparable accuracy.
- Top Predictors: Customer tenure, recency of interaction, campaign acceptance.
- AUC: 0.78

---

## 💬 Discussion

### Key Insights
- Higher income and active digital engagement correlate with subscription likelihood.
- Recency of interaction is critical for customer conversion.
- Personalized marketing based on customer behavior improves outcomes.

### Business Implications
- Enhance digital touchpoints (website, catalogs).
- Design targeted, data-driven campaigns.
- Re-engage lapsed customers with personalized offers.

### Limitations
- Assumed linearity; complex dynamics might be oversimplified.
- Dataset limited to a single campaign snapshot.

---

## ✅ Conclusion
The models effectively identified key drivers of subscription behavior. Logistic regression’s interpretability and SVM’s flexibility offer complementary strengths. Future work should explore non-linear models and incorporate broader datasets for improved predictions.

---

## 📚 References
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825-2830.
- Hunter, J. D. (2007). *Matplotlib: A 2D Graphics Environment*. Computing in Science & Engineering, 9(3), 90-95.
- Waskom, M. (2021). *Seaborn: Statistical Data Visualization*. JOSS, 6(60), 3021.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.

---

## 🧠 Author  
**Mohammed Saif Wasay**  
*Data Analytics Graduate — Northeastern University*  
*Machine Learning Enthusiast | Passionate about turning data into insights*  

🔗 [Connect with me on LinkedIn](https://www.linkedin.com/in/mohammed-saif-wasay-4b3b64199/)

---
