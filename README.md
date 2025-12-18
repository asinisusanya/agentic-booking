# 🏨 Agentic Guest Experience Optimizer  
### Proactive Hotel Satisfaction Prediction and Personalized Intervention

---

## 📌 Project Overview

The **Agentic Guest Experience Optimizer** is an end-to-end **agentic AI system** designed to **predict hotel guest satisfaction at booking time** and **proactively improve it before the stay begins** through personalized email interventions.

Unlike traditional hotel systems that rely on **post-stay feedback**, this project introduces a **pre-stay, autonomous decision-making pipeline** that integrates:

- Customer segmentation  
- Satisfaction prediction  
- Explainable AI (SHAP)  
- Intent inference  
- Intervention selection  
- Personalized email generation  

The system demonstrates how **Machine Learning + Agentic AI** can be applied to hospitality to reduce dissatisfaction risk and enhance guest experience proactively.

---

## 🎯 Objectives

- Segment hotel guests based on booking-time behavior  
- Predict expected satisfaction before the stay  
- Explain predictions transparently using SHAP  
- Infer guest intents from model explanations  
- Generate personalized, context-aware pre-stay emails  
- Demonstrate an end-to-end agentic AI pipeline  

---

## 📊 Dataset

This project uses the **Cinnamon Hotels Dataset** from Kaggle:

🔗 **Dataset Link**  
https://www.kaggle.com/datasets/asinisusanya/cinnamon-hotels-data

### Dataset Components

- **Booking Dataset**
  - Booking date, check-in/check-out dates
  - Number of guests, room pricing
  - Booking channel and stay characteristics

- **Feedback Dataset**
  - Overall satisfaction rating
  - Sub-ratings (service, cleanliness, value, etc.)
  - Used **only for training and evaluation**

- **Customer Profile Dataset**
  - Demographic attributes
  - Inferred behavioral traits:
    - Price sensitivity
    - Quality expectations
    - Travel frequency
    - Loyalty propensity

⚠️ **Data Leakage Prevention**  
Post-stay feedback is never used as an input feature. It is used only as the target variable during supervised learning.

---

## 🧠 End-to-End System Pipeline

```text
Booking Data
│
▼
Feature Engineering
│
▼
Data Preparation & Integration
│
▼
Customer Segmentation (K-Means)
│
▼
Satisfaction Prediction (LightGBM)
│
▼
SHAP Explainability
│
▼
Intent Inference Engine
│
▼
Intervention Selection Engine
│
▼
Personalized Email Generation

```
---

## 🛠 Feature Engineering

Raw booking inputs are not directly suitable for machine learning models or decision-making.  
Feature engineering transforms booking-time and profile-level information into structured signals that capture guest behavior, expectations, and risk patterns relevant to satisfaction prediction and personalization.

### Engineered Features

| Feature | Derivation / Formula | ML Relevance | Business Relevance |
|------|----------------------|-------------|------------------|
| `stay_nights` | Check-out date − Check-in date | Captures trip duration patterns | Short vs long stay experience planning |
| `lead_time_days` | Check-in date − Booking date | Indicates booking urgency | Late bookings need reassurance |
| `price_per_night` | Total booking amount ÷ stay nights | Normalizes pricing information | Distinguishes value vs premium guests |
| `is_weekend_stay` | Binary flag from check-in weekday | Behavioral segmentation | Weekend demand management |
| `number_of_guests` | Provided during booking | Group size effect on experience | Room and service planning |
| `checkin_month` | Extracted from check-in date | Seasonality modeling | Peak vs off-peak handling |
| `price_sensitivity` | Inferred from historical booking behavior | Explains dissatisfaction risk | Determines discount-based interventions |
| `quality_expectations` | Inferred from past ratings and spending | Predicts service expectations | Guides premium service allocation |
| `travel_frequency` | Historical stay count | Loyalty and familiarity signal | Repeat guest handling |
| `loyalty_propensity` | Aggregated CRM behavior | Long-term value estimation | Loyalty-based perks |

### Handling Missing and Inferred Features

- New customers may not have historical behavioral data
- Missing inferred features are approximated using:
  - Population-level medians
  - Booking-context heuristics
- This ensures deployability in real-world booking systems without blocking predictions

---

## 👥 Customer Segmentation

Guest expectations are heterogeneous.  
Customer segmentation enables **cluster-aware personalization** and tone adaptation in communication.

### Algorithm
- **K-Means clustering**
- Uses only **pre-stay features**
- Satisfaction ratings are explicitly excluded to prevent leakage

### Determining the Number of Clusters

The optimal number of clusters was selected as **K = 2**, based on:
- Elbow method (inertia reduction)
- Silhouette score (cluster separation quality)
- PCA-based visualization

Although PCA projections may visually suggest additional sub-structures, quantitative metrics indicate that two clusters provide the most stable and interpretable segmentation.

### Cluster Interpretation

| Cluster | Characteristics | Interpretation |
|------|----------------|---------------|
| Cluster 0 | Low price sensitivity, high expectations, strong loyalty | Premium-oriented guests |
| Cluster 1 | High price sensitivity, short stays, low loyalty | Value-oriented guests |

---

## 📈 Satisfaction Prediction (Supervised Learning)

### Problem Formulation
- Supervised regression task
- Predict expected satisfaction **before the stay**

### Inputs (X)
- Engineered booking-time features
- Customer cluster assignment

### Target (y)
- Overall satisfaction rating collected post-stay
- Used **only for training and evaluation**

### Model Selection Strategy

Two pipelines were evaluated:

- **Pipeline B – Traditional Models**
  - Linear Regression, Ridge, Lasso
  - Random Forest, Gradient Boosting
  - MLP Regressor

- **Pipeline A – Boosting Models**
  - XGBoost
  - LightGBM
  - CatBoost

### Evaluation Metrics
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

### Best Model Selection

🏆 **LightGBM** achieved the lowest RMSE and MAE and was selected as the final prediction model.

---

## 🔍 Explainable AI (SHAP Analysis)

Prediction accuracy alone is insufficient for proactive intervention.  
Explainability is required to understand **why dissatisfaction is predicted**.

### SHAP Methodology
Predicted Rating = Base + Feature Contributions



- **Base value:** Average predicted rating
- **SHAP values:** Feature-level contributions

### Global Explainability
- SHAP summary plots rank features by importance
- Identifies dominant drivers of satisfaction

### Local Explainability
- Per-booking SHAP values explain individual predictions
- Used directly for intent inference

---

## 🤖 Agentic Email Generation System

This project implements an **agentic AI system**, moving beyond static templates to autonomous decision-making.

### End-to-End Agentic Pipeline

1. Predict expected satisfaction
2. Explain prediction using SHAP
3. Infer guest intents
4. Select appropriate interventions
5. Render personalized email

### Intent Inference
SHAP signals are mapped to guest intents such as:
- High expectations
- Price sensitivity
- Late booking behavior
- Short stay urgency

### Intervention Selection
Each inferred intent is mapped to actionable service responses:
- Reassurance
- Discounts or incentives
- Priority services
- Experience recommendations

### Personalized Email Rendering
- Email tone adapts using customer cluster
- Content grouped into:
  - Arrival support
  - Experience suggestions
  - Courtesy and perks
- Output is a human-like, professional pre-stay email

---

## 📊 Evaluation and Discussion

### Technical Evaluation
- LightGBM demonstrated superior predictive performance
- SHAP explanations aligned with domain intuition

### Business Impact
- Enables proactive satisfaction improvement
- Reduces negative post-stay feedback risk
- Enhances personalization at scale

### Practical Feasibility
- Integrates with existing booking systems
- Scales across hotels and regions

---

## ⚠️ Limitations and Future Work

### Current Limitations
- Cold-start inference relies on approximations
- Email delivery is simulated
- No real-time feedback loop

### Future Enhancements
- Live email delivery integration
- Weather and local events APIs
- Reinforcement learning for adaptive interventions

---

## ✅ Conclusion

This project demonstrates how **agentic AI systems** can transform hospitality workflows by shifting from reactive feedback handling to **proactive experience optimization**.

By integrating prediction, explainability, and autonomous action, the system provides a practical blueprint for intelligent guest experience management.

---

## 👤 Author

**Asini Susanya**   
GitHub: https://github.com/asinisusanya

