---
layout: default
permalink: /final/
---
{% include nav.html %}

## Introduction and Literature Review {#Introduction}

The success of a YouTube video depends heavily on its ability to capture audience attention quickly, as early engagement plays a decisive role in how the platform's recommendation system promotes content. Previous research has highlighted that early viewership dynamics, discovery algorithms, and creator-level factors are combined in complex ways to determine a video's reach. Chatzopoulou, Sheng, and Faloutsos [1] found that early viewers are the most likely to engage, as the probability of likes, comments, or interactions declines with increasing view count, highlighting the temporal dynamics of audience behavior. Zhou et al. [2] examined YouTube’s discovery mechanisms, including keyword search, homepage recommendations, and event-driven highlights, and identified a “rich-get-richer” effect, in which initial traction compounds visibility across multiple recommendation pathways. Bärtl [3] analyzed channels and uploads over ten years, demonstrating that audience size, channel age, and genre strongly influence long-term video success, emphasizing the role of structural, channel-level factors.

Together, these studies indicate that YouTube popularity arises from a combination of audience behavior, platform algorithms, and creator characteristics. Early engagement patterns [1] drive disproportionate interaction, recommendation systems [2] amplify visibility, and channel-level features [3] condition overall success. These complementary insights provide a foundation for predictive modeling, highlighting the need to consider both presentation-level features and contextual metadata when estimating engagement.

Building on these findings, our project investigates whether *presentation-level* features, especially video title and thumbnail, combined with *contextual* factors like channel size can predict engagement outcomes. We focus on modeling clickability, in the form of *views per subscriber*, as a normalized and interpretable measure of how effectively a title and metadata convert audience size into views. The ultimate aim is to identify which textual and structural patterns most strongly correlate with success, providing insights for content optimization. In the next stage of our project, we will extend this framework beyond textual and metadata features to include visual analysis of thumbnails, allowing us to evluate how presentation style and imagery contribute to clickability.

## Problem Definition {#Problem}
Our study address the question:

*Can we predict the relative clickability of a YouTube video based on its title, thumbnail, and channel metadata?*

We approached this in two ways:
1. **Regression Task**: Predict the continuous metric views_per_subscriber for a given video, representing the average engagement level. This metric is computed as total views divided by channel subscriber count, normalized to represent engagement potential independent of channel size.
2. **Classification Task**: Predict whether a video achieves "high clickability"—defined as falling in the top 25% of the views_per_subscriber distribution. This binary classification task directly addresses the question of whether a video will be a "hit" or not.

This dual approach allows us to measure both the absolute predictive accuracy of our models (via R<sup>2</sup> and error metrics) and their discriminative power in identifying standout titles (via precision, recall, and ROC-AUC).

The overarching goal is to develop predictive models that can forecast these engagement metrics based solely on intrinsic video characteristics (thumbnail, title, description, tags, channel popularity), enabling content creators to optimize their uploads and platforms to predict which content will resonate with audiences.
## Methods {#Methods}
### Data and Preprocessing

Our project began with raw YouTube data sourced from two closely related datasets: the main YouTube US Trending Videos dataset containing detailed video metadata, and a supplementary dataset providing channel subscriber counts. Because the original trending dataset did not include subscriber information, we merged the two sources using a left join on video_id, allowing us to retain all videos while bringing in subscriber data wherever available. Missing subscriber values were then imputed using channel-level averages, leveraging the assumption that a channel’s subscriber count remains relatively stable over short time periods.

Before merging, both datasets underwent systematic preprocessing. We inspected the data structure and selected core analytical columns such as video_id, title, channel_title, views, likes, dislikes, comment_count, publish_time, tags, description, and thumbnail_link. Numeric fields were cleaned using pandas.to_numeric() with coercion to gracefully handle malformed entries, while timestamps were parsed into proper datetime formats to support future temporal analysis. Rows missing critical values were dropped to maintain dataset reliability.

To address duplicate video entries appearing at multiple timestamps, we deduplicated the dataset by keeping only the row with the highest view count for each video_id. After all cleaning, merging, and filtering steps, we obtained a final dataset of **5,905 videos**.

We engineered our primary regression target, **views_per_subscriber**, defined as:

<p align="center"><code>views_per_subscriber = views / (subscribers + 1)</code></p>

to avoid division by zero. These values were clipped to the range **[0, 500]** to reduce the impact of extreme outliers. For classification, we labeled videos in the top **25%** of `views_per_subscriber` as **high-clickability (1)** and all others as **low-clickability (0)**.

Beyond cleaning, we engineered additional presentation- and language-based features derived from video titles, including:

- **Channel size:** subscriber count  
- **Title structure:** title_length, word_count, avg_word_len, caps_ratio  
- **Sentiment:** VADER polarity score  
- **Punctuation indicators:** presence of `"?"`, `"!"`, or digits  

Finally, to capture deeper linguistic patterns, we applied **TF-IDF vectorization** to the video titles and extracted the **50 most informative components**, which were incorporated as additional features for modeling.



### Feature Engineering

Our feature engineering pipeline evolved substantially from the earlier version to support multimodal learning across structured text, full-text metadata, and thumbnail images. The final system generates three independent feature matrices:

---

#### **1. Structured Features (Traditional + Deep Features)**  
We compute two sets of structured features:

**Traditional Structured Features (8 dims)**  
These capture linguistic style and channel-level properties:  
- `title_length` (character count)  
- `word_count` (number of words)  
- `caps_ratio` (uppercase proportion)  
- `avg_word_len`  
- `has_question`, `has_exclamation`, `has_number`  
- `sentiment_vader` (compound sentiment score)  

**Deep Structured Features (13 dims)**  
To enrich linguistic nuance, we add:  
- `sentiment_tb` (TextBlob polarity)  
- `readability` (Flesch Reading Ease)  
- `emoji_count`  
- `punctuation_intensity`

These features capture emotional tone, urgency, writing complexity, and stylistic elements that may influence engagement.

---

#### **2. Text Features**  
We use two different text representations depending on the model family:

**Traditional Models (TF-IDF + SVD → 50 dims)**  
- Concatenate title + tags + description  
- Vectorize via TF-IDF (1,000 max features)  
- Reduce to 50 dimensions using Truncated SVD  
This preserves 85–90% of variance while remaining computationally efficient.

**Deep Models (Sentence-BERT + PCA → 128 dims)**  
- Encode text using the `all-mpnet-base-v2` Sentence-BERT model  
- Produce 768-dimensional semantic embeddings  
- Reduce to 128 dimensions using PCA (>95% variance retained)  
S-BERT embeddings capture deep semantic structure, synonym similarity, sentiment flow, and topic coherence—far beyond TF-IDF capabilities.

---

#### **3. Image Features**  
Thumbnail features are extracted using two approaches:

**Traditional Models (ResNet50 + PCA → 50 dims)**  
- Download thumbnails (hqdefault.jpg, 224×224)  
- Extract 2,048-dim features using pretrained ResNet50  
- Reduce to 50 dims using PCA (~80% variance retained)

**Deep Models (CLIP + Visual Metadata + PCA → 128 dims)**  
- Encode thumbnails with CLIP ViT-B/32 (768-dim features)  
- Add interpretable visual metadata:  
  - Brightness  
  - Saturation  
  - Face count  
  - Text density  
- Reduce the combined 772 dims to 128 using PCA  
CLIP produces semantically rich features aligned with natural language, improving multimodal fusion.

---

### Modeling Approach

We evaluate both traditional ML models and a full multimodal deep learning model.

---

#### **Traditional Machine Learning Models**

All structured, text, and image features are concatenated and split using an 80/20 train-test split. A `StandardScaler` fit on the training data is applied uniformly across all subsets.

We train the following models:

| **Task**         | **Model**                   | **Purpose**                                                           |
|------------------|-----------------------------|-----------------------------------------------------------------------|
| Regression       | Linear Regression           | Baseline interpretability; detects linear relationships               |
| Regression       | Random Forest Regressor     | Captures nonlinear interactions; robust to noise                      |
| Regression       | XGBoost Regressor           | Gradient-boosting for tabular optimization and high predictive power |
| Classification   | Logistic Regression         | Baseline classifier with interpretable weights                        |
| Classification   | Random Forest Classifier    | Ensemble model balancing precision, recall, and interpretability      |

These serve as benchmarks against which we compare our deep multimodal architecture.

---

### Deep Learning Pipeline

#### **Deep Feature Engineering **  
We generate Sentence-BERT text embeddings, CLIP image embeddings, and expanded structured features, each compressed with PCA to ensure stability and efficiency.

---

#### **Multimodal Deep Neural Network**  
We use a branch-and-fusion architecture:

- **Structured branch:** 13 → 32  
- **Text branch:** 128 → 64 → 32  
- **Image branch:** 128 → 64 → 32  
- **Fusion:** Concatenate (96) → 96 → 48 → 1  

Each layer uses BatchNorm, LeakyReLU, and Dropout for stability and regularization.

**Classification**: trained with BCEWithLogitsLoss  
**Regression**: trained with MSELoss  

Training uses a 70/15/15 stratified split, Adam optimizer, and early stopping based on validation AUC (classification) or RMSE (regression).






## Results and Discussion {#Results}
### Regression Performance
{% capture m %}
| **Model** | **R<sup>2</sup>** | **MAE** | **RMSE** |
|---|---|---|---|
| Linear (Structured) | 0.024 | 14.73 | 43.49 |
| Random Forest (Combined) | 0.257 | 10.04 | 37.96 |
| XGBoost (Combined) | 0.258 | 10.16 | 37.93 |
{% endcapture %}
{{ m | markdownify }}

Both the Random Forest and XGBoost Regressors achieved an R<sup>2</sup> of approximately 0.26, explaining over one-quarter of the variance in video clickability - a strong result given the inherent noise and many external factors influencing YouTube video viewership. 

The MAE of ~10 views per subscriber indicates reasonably tight predictions around actual performance.

Linear Regression, by constrast, performed poorly, reinforcing that title-success relationships are nonlinear and feature interactions matter.

In our upcoming project phase we will test whether incorporating thumbnail visual features improves model performance beyond the current text-and-metadata baseline. 

![image](https://github.gatech.edu/user-attachments/assets/fd2c8b2f-0590-4318-a0ac-deb4f7804288)

*Figure 1. Distribution of actual vs. predicted values (log-scaled). The model captures the overall shape but underestimates extreme viral cases.**

### Feature Importance

![image](https://github.gatech.edu/user-attachments/assets/eb789a04-e923-4e17-a213-c9f21b792457)

*Figure 2. Top 10 feature importances from the Random Forest Regressor.*

Feature importance analysis revealed that:
* Subscriber count overwhelmingly dominates predictive power, consistent with Bärtl [3]'s findings
* Sentiment and capitalization ratio emerged as secondary predictors, suggesting positive, clearly written titles slightly improve performance
* Textual TF-IDF features contributed smaller but meaningful signals, suggesting that specific words may increase visibility or curiosity

Dimensionality reduction on the TF-IDF title features revealed several coherent latent patterns. For example, one component emphasized words such as "trailer," "season," "video," and "HD," corresponding to streaming and media-related titles. Another component featured "official video," "live," "Christmas," and "Valentine’s Day," representing music or event-themed uploads. Other components highlighted clusters such as "react," "review," "people," and "things," characteristic of reaction or commentary videos, and "Super Bowl," "commercial," "Trump," and "Black Panther," capturing event-driven or trending topics. These groupings suggest that the TF-IDF components learned meaningful linguistic structures from YouTube titles, revealing distinct genres and content strategies that align with intuitive video categories.

### Classification Performance
{% capture m %}
| Model              | Accuracy | Precision | Recall | F1   | ROC-AUC |
|---                 |---       |---        |---     |---   |---      |
| Logistic Regression| 0.75     | 0.00      | 0.00   | 0.00 | 0.80    |
| Random Forest      | 0.81     | 0.71      | 0.42   | 0.53 | 0.86    |
{% endcapture %}
{{ m | markdownify }}

The Random Forest Classifier achieved the best overall balance, with 81% accuracy and an AUC of 0.86, demonstrating strong ability to distinguish high- vs low-performing videos. 

High precision (0.71) suggests that when the model predicts a video will perform well, it is usually correct. This is an important property if this system were used as a recommender or content optimization assistant.
Lower recall (0.42) indicates some high-performing titles are missed, likely due to natural class imbalance.

![image](https://github.gatech.edu/user-attachments/assets/f78dc157-3a3a-4414-80cb-b65d7aaf3ef9)

*Figure 3. ROC curve of Random Forest classifier, showing strong discriminative ability (AUC = 0.86).*

# Summary of Model Results

Across traditional machine learning baselines and the multimodal deep learning pipeline, the results show that **nonlinear models and semantically rich features provide the strongest predictive performance**, while linear methods struggle to capture the complexity of YouTube engagement.

---

## 1. Regression Models (Predicting Views per Subscriber)

### **Top Performers: Random Forest & XGBoost**
- **R² ≈ 0.26** — models explain ~26% of engagement variance  
- **MAE ≈ 10** — predictions typically within ±10 views/subscriber  
- **RMSE ≈ 38** — viral outliers remain difficult to capture  

### **Interpretation**
- Video performance is **nonlinear**, so Linear Regression performs poorly (**R² = 0.02**).
- Ensemble models capture meaningful structure and outperform all baselines.

### **Feature Insights**
- **Subscriber count** is the most influential predictor.
- **Sentiment**, **capitalization ratio**, and **linguistic style** meaningfully contribute.
- **TF-IDF latent components** reveal coherent genre clusters (e.g., trailers, holiday content, reaction videos, event-driven uploads).

---

## 2. Classification Models (High vs. Low Performance)

### **Best Model: Random Forest Classifier**
- **Accuracy:** 0.81  
- **Precision:** 0.71  
- **Recall:** 0.42  
- **F1:** 0.53  
- **ROC-AUC:** 0.86  

### **Interpretation**
- High **precision** → when the model predicts success, it’s usually correct.
- Lower **recall** → some high-performing videos are missed (class imbalance).
- **AUC = 0.86** → model reliably distinguishes successful vs. unsuccessful titles.

### **Logistic Regression**
- Good AUC (0.80) but **precision/recall = 0**, reinforcing that linear decision boundaries do not work for this task.

---

## 3. Overall Performance Trends

### **Traditional Models**
- Strongest when combining **semantic text features + structured metadata**.
- Nonlinear methods outperform linear ones by a large margin.
- Performance plateaus at ~0.26 R² due to inherent unpredictability of viral content.

### **Multimodal Deep Learning Model**
- Designed to fuse:
  - **S-BERT text embeddings**  
  - **CLIP image embeddings**  
  - **Deep structured features**  
- Expected to surpass traditional approaches due to:
  - richer semantic representations  
  - cross-modal learning  
  - better handling of nonlinear patterns  

(*Numerical results were not provided but the architecture is optimized for superior multimodal performance.*)

---

# **Key Takeaways**

- **Nonlinear models are essential** for engagement prediction.  
- **Text features are the single strongest signal**, especially semantic embeddings.  
- **Structured metadata** (subscriber count, sentiment, formatting style) significantly boosts accuracy.  
- **Classification is easier than exact regression**, with strong ROC-AUC performance.  
- **Viral content remains unpredictable**, as all models underestimate extreme outliers.

---

### Interpretation
Across models, the findings support our central hypothesis that title characteristics combined with channel metadata can predict relative engagement potential. However, the dominace of the subscribers feature highlights that success is heavily conditioned by existing audience reach, with title wording contributing a secondary, but stil measurable, effect. These insights align with prior research on algorithmic amplification and audience dynamics, suggesting that creators with established followings benefit more from metadata optimization than new creators. It's important to remember video titles are not the only factor contributing to views, so we should not expect perfect predictive ability based solely on that input, but rather we can evaluate to what extent titles have an impact and how that impact can be maximized.

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
[Click here -> Gantt Chart](https://docs.google.com/spreadsheets/d/1ks7QSZOliQQ410aY5oCGphUX07czt0Q8Edpbw5tmB1A/edit?usp=sharing)
<img src="/sjin308/YouTube_Clickability_Study/blob/main/_images/miterm_report_contributions.jpg?raw=true" alt="Proposal Contributions">

## References: {#References}
[1] G. Chatzopoulou, C. Sheng, and M. Faloutsos, “A first step towards understanding popularity in YouTube,” 2010 INFOCOM IEEE Conference on Computer Communications Workshops, Mar. 2010. doi:10.1109/infcomw.2010.5466701 


[2] R. Zhou, S. Khemmarat, L. Gao, J. Wan, and J. Zhang, “How youtube videos are discovered and its impact on video views,” Multimedia Tools and Applications, vol. 75, no. 10, pp. 6035–6058, Jan. 2016. doi:10.1007/s11042-015-3206-0 


[3] M. Bärtl, “YouTube channels, uploads and views: A statistical analysis of the past 10 years,” Convergence: The International Journal of Research into New Media Technologies, vol. 24, no. 1, pp. 16–32, Jan. 2018. doi:10.1177/1354856517736979 


[4] A. Testas, “Logistic regression with pandas, scikit-learn, and pyspark,” Distributed Machine Learning with PySpark, pp. 173–212, 2023. doi:10.1007/978-1-4842-9751-3_7


## GitHub Repository: {#GitHub}
[Click here -> GitHub Repository for this project](https://github.gatech.edu/sjin308/YouTube_Clickability_Study)

## Project Award Eligibility: {#Award}
We would like to be considered for the “Outstanding Project” award.
