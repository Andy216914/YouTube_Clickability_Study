---
layout: default
permalink: /midterm/
---
{% include nav.html %}

## Introduction and Literature Review {#Introduction}

The success of a YouTube video depends heavily on its ability to capture audience attention quickly, as early engagement plays a decisive role in how the platform's recommendation system promotes content. Previous research has highlighted that early viewership dynamics, discovery algorithms, and creator-level factors interact in complex ways to determine a video's reach. Chatzopoulou, Sheng, and Faloutsos [1] found that early viewers are the most likely to engage, as the probability of likes, comments, or interactions declines with increasing view count, highlighting the temporal dynamics of audience behavior. Zhou et al. [2] examined YouTube’s discovery mechanisms, including keyword search, homepage recommendations, and event-driven highlights, and identified a “rich-get-richer” effect, in which initial traction compounds visibility across multiple recommendation pathways. Bärtl [3] analyzed channels and uploads over ten years, demonstrating that audience size, channel age, and genre strongly influence long-term video success, emphasizing the role of structural, channel-level factors.


Together, these studies indicate that YouTube popularity arises from a combination of audience behavior, platform algorithms, and creator characteristics. Early engagement patterns [1] drive disproportionate interaction, recommendation systems [2] amplify visibility, and channel-level features [3] condition overall success. These complementary insights provide a foundation for predictive modeling, highlighting the need to consider both presentation-level features and contextual metadata when estimating engagement.


Building on these findings, our project investigates whether *presentation-level* features, especially video title and thumbnail, combined with *contextual* factors like channel size can predict engagement outcomes. We focus on modeling _clickability_, operationalize as *views per subscriber*, as a normalized and interpretable measure of how effectively a title and metadata convert audience size into views. The ultimate aim is to identify which textual and structural patterns most strongly correlate with success, providing actionable insights for content optimization. In the next sage of our project, we will extend this framework beyond textual and metadata features to include visual analysis of thumbnails, allowing us to evluate how presentation style and imagery contribute to clickability.

## Problem Definition {#Problem}
Our study address the question:
*Can we predict the relative clickability of a YouTube video based on its title, thumbnail, and channel metadata?*

To approach thins, we formulated two tasks:
1. **Regression:** Predict a continuous *views per subscriber* metric to quantify engagement potential
2. **Classification:** Categorize videos into *high* or *low* clickability classes based on whether they exceed the dataset's median performance

This dual approach allows us to measure both the absolute predictive accuracy of our models (via R^2 and error metrics) and their discriminative power in identifying standout titles (via precision, recall, and ROC-AUC).

## Methods {#Methods}
### Data and Preprocessing
We used the YouTube Trending Videos dataset from Kaggle, which includes metadata for thousands of trending uploads. After cleaning and filtering, our processed dataset contained 5,742 videos.

Structured numeric and categorical features included:

* subscribers (channel size)
* title_length, avg_word_len, caps_ratio
* sentiment_vader (title sentiment polarity)
* punctuation identifiers (presence of "?", "!", or digits)
  
In addition, textual features were extracted using TF-IDF vectorization, capturing the 50 most informative terms from video titles. The target variable *views_per_subscriber* was lightly clipped and scaled to mitigate the influence of extreme viral outliers while preserving meaningful variation. 

### Feature Engineering
To standardize inputs across feature types:

* Numeric features were scaled using StandardScaler
* Text features were processed through TF-IDF with a maximum of 50 components
* Both sets were concatenated into a single feature matrix

This combination enabled our models to learn relationships between structured signals (e.g., sentiment, length) and unstructured linguistic cues (specific keywords and phrasing). In the next phase of this project, we will integrate computer vision-based thumbnail features, such as brightness, contrast, color balance, and object presence, extracted via pre-trained convolutional neural networks (CNNs) to capture visual patterns that may contribute to engagement.

### Modeling Approach

We implemented and compared five supervised models:

| **Task**         | **Model**                 | **Purpose**                                             |
|------------------|---------------------------|---------------------------------------------------------|
| Regression       | Linear Regression         | Baseline for interpretability and linear patterns       |
| Regression       | Random Forest Regressor   | Captures nonlinear and interaction effects              |
| Regression       | XGBoost Regressor         | Gradient-boosted refinement for tabular data            |
| Classification   | Logistic Regression       | Linear baseline                                         |
| Classification   | Random Forest Classifier  | Balances precision, recall, and interpretability        |

Each model was trained with an 80/20 train-test split (random_state = 42) to ensure consistent comparisons.

## Results and Discussion {#Results}
### Regression Performance
| **Model**                | **R²** | **MAE** | **RMSE** |
| ------------------------ | ------ | ------- | -------- |
| Linear (Structured)      | 0.024  | 14.73   | 43.49    |
| Random Forest (Combined) | 0.257  | 10.04   | 37.96    |
| XGBoost (Combined)       | 0.258  | 10.16   | 37.93    |

Both the Random Forest and XGBoost Regressors achieved an R^2 of approximately 0.26, explaining over one-quarter of the variance in video clickability - a strong result given the inherent noise and external factors influencing YouTube viewership. 

The MAE of ~10 views per subscriber indicates reasonably tight predictions around actual performance.

Linear Regression, by constrast, performed poorly, reinforcing that title-success relationships are nonlinear and feature interactions matter.

These findings motivate our upcoming project phase, in which we will test whether incorporating thumbnail visual features improves model performance beyond the current text-and-metadata baseline. 

![image](https://github.gatech.edu/user-attachments/assets/fd2c8b2f-0590-4318-a0ac-deb4f7804288)

*Figure 1. Distribution of actual vs. predicted values (log-scaled). The model captures the overall shape but underestimates extreme viral cases.**

### Feature Importance

![image](https://github.gatech.edu/user-attachments/assets/eb789a04-e923-4e17-a213-c9f21b792457)

*Figure 2. Top 10 feature importances from the Random Forest Regressor.*

Feature importance analysis revealed that:
* Subscriber count overwhelmingly dominates predictive power, consistent with Bärtl [3]'s findings about structural importance
* Sentiment and capitalization ratio emerged as secondary predictors: positive, clearly written titles slightly improve performance
* Textual TF-IDF features contributed smalelr but meaningful signals, suggesting that specific words may increase visibility or curiosity

### Classification Performance
| **Model**           | **Accuracy** | **Precision** | **Recall** | **F1** | **ROC-AUC** |
| ------------------- | ------------ | ------------- | ---------- | ------ | ----------- |
| Logistic Regression | 0.75         | 0.00          | 0.00       | 0.00   | 0.80        |
| Random Forest       | 0.81         | 0.71          | 0.42       | 0.53   | 0.86        |

The Random Forest Classifier achieved the best overall balance, with 81% accuracy and an AUC of 0.86, demonstrating strong ability to distinguish high- vs low-performing videos. 

High precision (0.71) suggests that when the model predicts a video will perform well, it is usually correct. This is an important property if this system were used as a recommender or content optimization assistant.
Lower recall (0.42) indicates some high-performing titles are missed, likely due to natural class imbalance.

![image](https://github.gatech.edu/user-attachments/assets/4099b6f5-815a-43f2-a8bb-da8a34e78927)

*Figure 3. ROC curve of Random Forest classifier, showing strong discriminative ability (AUC = 0.86).*

### Interpretation
Across models, the findings support our central hypothesis that title characteristics combined with channel metadata can predict relative engagement potential. However, the dominace of the subscribers feature highlights that success is heavily conditioned by existing audience reach, with title wording contributing a secondary, but stil measurable, effect. These insights align with prior research on algorithmic amplification and audience dynamics, suggesting that creators with established followings benefit more from metadata optimization than new creators.

## Next Steps
1. Integrate Thumbnail Analysis
  * Extract visual embeddings and metadata (brightness, saturation, text density, and facial presence) from thumbnails
  * Use CNNs or transfer learning models (e.g., ResNet, EfficientNet) to generate thumbnail feature vectors
  * Combine these with existing textual and structured features to train multimodal models
3. Hyperparameter Optimization
  * Conduct grid search for Random Forest and XGBoost parameters to improve recall and generalization
4. Model Explainability
  * Use SHAP values to interpret how both textual and visual features contribute to predicted clickability
5. Cross-Validation and Robustness
  * Perform k-fold validation to evluate consistency across data splits and reduce overfitting risk

## Gantt Chart & Contribution Table: {#Contributions}
[Click here -> Gantt Chart](https://docs.google.com/spreadsheets/d/1y-I8I2sqUG6IM8JgbNB582FVOKyg1I9ms89l9MGjOCo/edit?usp=sharing\)
<img src="/sjin308/YouTube_Clickability_Study/blob/main/_images/ContributionChart.jpg?raw=true" alt="Proposal Contributions">

## References: {#References}
[1] G. Chatzopoulou, C. Sheng, and M. Faloutsos, “A first step towards understanding popularity in YouTube,” 2010 INFOCOM IEEE Conference on Computer Communications Workshops, Mar. 2010. doi:10.1109/infcomw.2010.5466701 


[2] R. Zhou, S. Khemmarat, L. Gao, J. Wan, and J. Zhang, “How youtube videos are discovered and its impact on video views,” Multimedia Tools and Applications, vol. 75, no. 10, pp. 6035–6058, Jan. 2016. doi:10.1007/s11042-015-3206-0 


[3] M. Bärtl, “YouTube channels, uploads and views: A statistical analysis of the past 10 years,” Convergence: The International Journal of Research into New Media Technologies, vol. 24, no. 1, pp. 16–32, Jan. 2018. doi:10.1177/1354856517736979 


[4] A. Testas, “Logistic regression with pandas, scikit-learn, and pyspark,” Distributed Machine Learning with PySpark, pp. 173–212, 2023. doi:10.1007/978-1-4842-9751-3_7


## GitHub Repository: {#GitHub}
[Click here -> GitHub Repository for this project](https://github.gatech.edu/sjin308/YouTube_Clickability_Study)

## Project Award Eligibility: {#Award}
We would like to be considered for the “Outstanding Project” award.
