---
layout: default
permalink: /final/
---
{% include nav.html %}

## Introduction and Literature Review {#Introduction}

The success of a YouTube video depends heavily on its ability to capture audience attention quickly, as early engagement plays a decisive role in how the platform's recommendation system promotes content. Previous research has highlighted that early viewership dynamics, discovery algorithms, and creator-level factors are combined in complex ways to determine a video's reach. Chatzopoulou, Sheng, and Faloutsos [1] found that early viewers are the most likely to engage, as the probability of likes, comments, or interactions declines with increasing view count, highlighting the temporal dynamics of audience behavior. Zhou et al. [2] examined YouTube’s discovery mechanisms, including keyword search, homepage recommendations, and event-driven highlights, and identified a “rich-get-richer” effect, in which initial traction compounds visibility across multiple recommendation pathways. Bärtl [3] analyzed channels and uploads over ten years, demonstrating that audience size, channel age, and genre strongly influence long-term video success, emphasizing the role of structural, channel-level factors.

Together, these studies indicate that YouTube popularity arises from a combination of audience behavior, platform algorithms, and creator characteristics. Early engagement patterns [1] drive disproportionate interaction, recommendation systems [2] amplify visibility, and channel-level features [3] condition overall success. These complementary insights provide a foundation for predictive modeling, highlighting the need to consider both presentation-level features and contextual metadata when estimating engagement.

Building on these findings, our project investigates whether *presentation-level* features, especially video title and thumbnail, combined with *contextual* factors like channel size can predict engagement outcomes. We focus on modeling clickability, in the form of *views per subscriber*, as a normalized and interpretable measure of how effectively a title and metadata convert audience size into views. The ultimate aim is to identify which textual, visual, and structural patterns most strongly correlate with success, providing insights for content optimization. In this project, we extend our framework beyond textual and metadata features to include visual analysis of thumbnails, allowing us to evaluate how presentation style and imagery contribute to clickability.

## Problem Definition {#Problem}
Our study addresses the question:

*Can we predict the relative clickability of a YouTube video based on its title, thumbnail, and channel metadata?*

We approached this in two ways:
1. **Regression Task**: Predict the continuous metric `views_per_subscriber` for a given video, representing the average engagement level. This metric is computed as total views divided by channel subscriber count, normalized to represent engagement potential independent of channel size.
2. **Classification Task**: Predict whether a video achieves "high clickability," defined as falling in the top 25% of the `views_per_subscriber` distribution. This binary classification task directly addresses the question of whether a video will be a "hit" or not.

This dual approach allows us to measure both the absolute predictive accuracy of our models (via R<sup>2</sup> and error metrics) and their discriminative power in identifying standout titles (via precision, recall, and ROC-AUC).

The overarching goal is to develop predictive models that can forecast these engagement metrics based solely on intrinsic video characteristics (thumbnail, title, description, tags, channel popularity), enabling content creators to optimize their uploads and platforms to predict which content will resonate with audiences.

## Methods {#Methods}
### Data and Preprocessing

Our project began with raw YouTube data sourced from two closely related datasets: the main YouTube US Trending Videos dataset containing detailed video metadata, and a supplementary dataset providing channel subscriber counts. Because the original trending dataset did not include subscriber information, we merged the two sources using a left join on `video_id`, allowing us to retain all videos while bringing in subscriber data wherever available. Missing subscriber values were then imputed using channel-level averages, leveraging the assumption that a channel’s subscriber count remains relatively stable over short time periods.

Before merging, both datasets underwent systematic preprocessing. We inspected the data structure and selected core analytical columns such as `video_id`, `title`, `channel_title`, `views`, `likes`, `dislikes`, `comment_count`, `publish_time`, `tags`, `description`, and `thumbnail_link`. Numeric fields were cleaned using `pandas.to_numeric()` with coercion to gracefully handle malformed entries, while timestamps were parsed into proper datetime formats to support future temporal analysis. Rows missing critical values were dropped to maintain dataset reliability.

To address duplicate video entries appearing at multiple timestamps, we deduplicated the dataset by keeping only the row with the highest view count for each `video_id`. After all cleaning, merging, and filtering steps, we obtained a final dataset of **5,905 videos**.

We engineered our primary regression target, **views_per_subscriber**, defined as:

<p align="center"><code>views_per_subscriber = views / (subscribers + 1)</code></p>

to avoid division by zero. These values were clipped to the range **[0, 500]** to reduce the impact of extreme outliers. For classification, we labeled videos in the top **25%** of `views_per_subscriber` as **high-clickability (1)** and all others as **low-clickability (0)**.

Beyond cleaning, we engineered additional presentation- and language-based features derived from video titles, including:

- **Channel size:** subscriber count  
- **Title structure:** `title_length`, `word_count`, `avg_word_len`, `caps_ratio`  
- **Sentiment:** VADER polarity score  
- **Punctuation indicators:** presence of `"?"`, `"!"`, or digits  

Finally, to capture deeper linguistic patterns, we applied **TF-IDF vectorization** to the video titles and extracted the **50 most informative components**, which were incorporated as additional features for modeling.

### Feature Engineering

Our feature engineering pipeline evolved substantially from the earlier version to support multimodal learning across structured text, full-text metadata, and thumbnail images. The final system generates three independent feature matrices:

---

#### **1. Structured Features (Traditional + Deep Features)**  
We compute two sets of structured features:

### **Traditional Structured Features (8 dimensions)**  
These handcrafted features capture text structure, punctuation, and basic emotional tone:

- `title_length`: total number of characters in the title  
- `word_count`: number of tokens  
- `caps_ratio`: proportion of characters that are uppercase  
- `avg_word_len`: average number of characters per word  
- `has_question`: binary indicator for presence of `?`  
- `has_exclamation`: binary indicator for presence of `!`  
- `has_number`: binary indicator for presence of digits  
- `sentiment_vader`: compound polarity score ∈ [-1, 1] computed via VADER sentiment analyzer  

These features reflect findings from linguistics and marketing literature: punctuation, capitalization, and emotional cues influence user attention and engagement.

---

### **Deep Structured Features (13 dimensions)**  
To capture deeper nuances in writing style and emotional intensity, we extend the feature set:

- `sentiment_tb`: sentiment polarity computed via TextBlob  
- `readability`: Flesch Reading Ease score (lower = harder to read)  
- `emoji_count`: count of emojis representing emotional charge  
- `punctuation_intensity`: ratio of punctuation marks to total characters  
- plus all 8 traditional features above  

These features quantify readability difficulty, affective expressiveness, style markers, and non-verbal emotional signals.

These features capture emotional tone, urgency, writing complexity, and stylistic elements that may influence engagement.

---

#### **2. Text Features**  
We use two different text representations depending on the model family:

**Traditional Models (TF-IDF + SVD → 50 dims)**  
- Concatenate `title + tags + description`  
- Vectorize via TF-IDF (1,000 max features)  
- Reduce to 50 dimensions using Truncated SVD  
This preserves 85–90% of variance while remaining computationally efficient.

**Deep Models (Sentence-BERT + PCA → 128 dims)**  
- Encode text using the `all-mpnet-base-v2` Sentence-BERT model  
- Produce 768-dimensional semantic embeddings  
- Reduce to 128 dimensions using PCA (>95% variance retained)  
Sentence-BERT embeddings capture deep semantic structure, synonym similarity, sentiment flow, and topic coherence, far beyond TF-IDF capabilities.

---

#### **3. Image Features**  
Thumbnail features are extracted using two approaches:

**Traditional Models (ResNet50 + PCA → 50 dims)**  
- Download thumbnails (`hqdefault.jpg`, 224×224)  
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

All structured, text, and image features are concatenated and split using an 80/20 train–test split. A `StandardScaler` fit on the training data is applied uniformly across all subsets.

We train the following models:

| **Task**         | **Model**                         | **Purpose**                                                           |
|------------------|-----------------------------------|-----------------------------------------------------------------------|
| Regression       | Linear Regression                 | Baseline interpretability; detects linear relationships               |
| Regression       | Random Forest Regressor           | Captures nonlinear interactions; robust to noise                      |
| Regression       | XGBoost Regressor                 | Gradient-boosting for tabular optimization and high predictive power |
| Classification   | Logistic Regression               | Baseline classifier with interpretable weights                        |
| Classification   | Random Forest Classifier          | Ensemble model balancing precision, recall, and interpretability      |
| Classification   | HistGradientBoosting Classifier   | Histogram-based gradient boosting; handles class imbalance and complex interactions |

These serve as benchmarks against which we compare our deep multimodal architecture.

---

### Deep Learning Pipeline

#### **Deep Feature Engineering**  
We generate Sentence-BERT text embeddings, CLIP image embeddings, and expanded structured features, each compressed with PCA to ensure stability and efficiency.

---

#### **Multimodal Deep Neural Network**  
We use a branch-and-fusion architecture:

- **Structured branch:** 13 → 32  
- **Text branch:** 128 → 64 → 32  
- **Image branch:** 128 → 64 → 32  
- **Fusion:** Concatenate (96) → 96 → 48 → 1  

Each layer uses BatchNorm, LeakyReLU, and Dropout for stability and regularization.

**Classification**: trained with `BCEWithLogitsLoss`  
**Regression**: trained with `MSELoss`  

Training uses a 70/15/15 stratified split, Adam optimizer, and early stopping based on validation AUC (classification) or RMSE (regression).

---

#### Model Selection Rationale

We selected these models to balance interpretability, nonlinearity, and performance.

- **Linear Regression / Logistic Regression** provide an interpretable baseline and help determine whether simple linear relations exist between title features and engagement.  
- **Random Forest** captures nonlinear interactions and provides feature importance for interpretability.  
- **XGBoost** offers strong tabular modeling performance through gradient boosting, making it ideal for noisy, high-dimensional metadata.  
- **HistGradientBoosting** was added because histogram-based boosting handles class imbalance well and is optimized for high-cardinality tabular datasets.  
- **The Multimodal Deep Neural Network** integrates text, image, and structured data, enabling richer nonlinear modeling across modalities and testing whether semantic + visual representations outperform traditional ML.

---

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

Both the Random Forest and XGBoost Regressors achieved an R<sup>2</sup> of approximately 0.26, explaining over one-quarter of the variance in video clickability, which is a strong result given the inherent noise and many external factors influencing YouTube video viewership. 

The MAE of ~10 views per subscriber indicates reasonably tight predictions around actual performance, though the RMSE reveals that extremely viral outliers are still hard to capture.

Linear Regression, by contrast, performed poorly, reinforcing that title–success relationships are nonlinear and that feature interactions matter. This pattern is clear when we examine the distribution of predictions for each model.

![image](https://github.gatech.edu/user-attachments/assets/c33a0ec5-c49e-4784-9ce3-13998f0eee15)

*Figure 1. Distribution of actual vs. predicted `views_per_subscriber` (log scale) for the Linear Regression model. Predictions are heavily shrunk toward the mean, and the model fails to capture the long right tail of high-performing videos.*

![image](https://github.gatech.edu/user-attachments/assets/9bc86ddd-4307-4246-8649-170f2a182269)

*Figure 2. Distribution of actual vs. predicted `views_per_subscriber` (log scale) for the Random Forest Regressor. The model better matches the empirical distribution but still compresses the extreme right tail of viral videos.*

![image](https://github.gatech.edu/user-attachments/assets/277f2116-0a86-44cf-a26b-053ce25ff644)

*Figure 3. Distribution of actual vs. predicted `views_per_subscriber` (log scale) for the XGBoost Regressor. XGBoost tracks the heavy tail slightly better than Random Forest, but the most viral videos remain underestimated.*

### Feature Importance

![image](https://github.gatech.edu/user-attachments/assets/eb789a04-e923-4e17-a213-c9f21b792457)

*Figure 4. Top 10 feature importances from the Random Forest Regressor.*

Feature importance analysis revealed that:
* Subscriber count overwhelmingly dominates predictive power, consistent with Bärtl [3]'s findings.
* Sentiment and capitalization ratio emerged as secondary predictors, suggesting positive, clearly written titles slightly improve performance.
* Textual TF-IDF features contributed smaller but meaningful signals, suggesting that specific words may increase visibility or curiosity.

Dimensionality reduction on the TF-IDF title features revealed several coherent latent patterns. For example, one component emphasized words such as "trailer," "season," "video," and "HD," corresponding to streaming and media-related titles. Another component featured "official video," "live," "Christmas," and "Valentine’s Day," representing music or event-themed uploads. Other components highlighted clusters such as "react," "review," "people," and "things," characteristic of reaction or commentary videos, and "Super Bowl," "commercial," "Trump," and "Black Panther," capturing event-driven or trending topics. These groupings suggest that the TF-IDF components learned meaningful linguistic structures from YouTube titles, revealing distinct genres and content strategies that align with intuitive video categories.

### Classification Performance
{% capture m %}
| Model                | Accuracy | Precision | Recall | F1    | ROC-AUC |
|----------------------|----------|-----------|--------|-------|---------|
| Logistic Regression  | 0.763    | 0.581     | 0.188  | 0.284 | 0.777   |
| Random Forest        | 0.791    | 0.712     | 0.275  | 0.397 | 0.859   |
| HistGradientBoosting | 0.822    | 0.663     | 0.589  | 0.624 | 0.853   |
{% endcapture %}
{{ m | markdownify }}

The HistGradientBoosting model achieved the strongest overall performance, with 82.2% accuracy and an F1 score of 0.62, indicating the most balanced relationship between precision and recall among the three models. Its recall of 0.59 suggests it is much more effective at identifying positive cases than the other models, which is valuable when missing true positives is costly. The ROC-AUC of 0.85 shows it maintains strong ranking ability even when threshold adjustments are taken into account.

The Random Forest classifier also performed well, with 79% accuracy and the highest precision (0.71) across all models. This means that when it predicts a positive outcome, it is correct most of the time. However, its lower recall (0.28) indicates that it misses many true positive cases, likely due to class imbalance or conservative decision boundaries. Its ROC-AUC of 0.86 is the highest of all, meaning it is particularly strong at separating classes overall, even if its chosen decision threshold produces lower recall.

Logistic Regression performed the weakest overall, with 76% accuracy and notably low recall (0.19). While its precision (0.58) is decent, the model struggles to capture positive cases effectively. The low F1 score (0.28) reflects this imbalance. However, its ROC-AUC (0.78) shows that the underlying linear decision boundary still provides some discriminatory power, just not enough to match the non-linear models.

Overall, HistGradientBoosting provides the best balance for practical use, especially in scenarios where detecting as many true positives as possible is important. Random Forest is ideal when precision is more critical than recall, and Logistic Regression serves as a reasonable but less powerful baseline.

![image](https://github.gatech.edu/user-attachments/assets/e8eb14da-5391-43e8-a97b-d0436481648c)

*Figure 5. ROC curve of the Random Forest classifier, showing strong discriminative ability (AUC ≈ 0.86).*

![image](https://github.gatech.edu/user-attachments/assets/529b9a59-61de-4369-b1b9-a5f95d6f305a)

*Figure 6. ROC curve of the Linear Regression*

![image](https://github.gatech.edu/user-attachments/assets/93f3c7c9-395a-4c7f-af63-4b1f5806e442)

*Figure 7. ROC curve of the Histogram based gradient boosting classifier*

Across both tasks, all nonlinear models (Random Forest, XGBoost, and HistGradientBoosting) substantially outperform linear baselines. Random Forest provides the best interpretability through feature importance, XGBoost provides slightly better regression accuracy, and HistGradientBoosting provides the strongest and most balanced classification performance. Linear methods fail to capture nonlinear linguistic patterns, while tree-based ensembles consistently model these interactions. The deep multimodal network is expected to outperform all traditional models once fully trained due to richer semantic and visual representation learning.

---
### Deep neural networks
**Purpose:**  
Deep neural networks (DNNs) were used to model the complex relationships between structured features, text features, and image features, enabling the system to learn how multiple modalities jointly influence video clickability.

**Architecture:**  
A branch-and-fusion multimodal DNN was designed:
- Each modality is processed by its own branch consisting of fully connected layers with BatchNorm, LeakyReLU activations, and Dropout.
- The structured branch handles handcrafted and semantic features.
- The text branch processes Sentence-BERT embeddings (compressed with PCA).
- The image branch processes CLIP or ResNet50 embeddings (also PCA-compressed) plus visual metadata.
- Outputs from the three branches are concatenated and passed through additional dense layers to produce the final prediction.

**Feature Inputs:**  
- **Structured:** Handcrafted linguistic features, VADER sentiment, TextBlob sentiment, readability, emoji count, punctuation intensity.  
- **Text:** Sentence-BERT semantic embeddings reduced to 128 dimensions with PCA.  
- **Image:** CLIP or ResNet50 embeddings reduced with PCA, plus brightness, saturation, face count, and text density.

**Training Setup**
- **Framework:** PyTorch
- **Dataset split:** 70 / 15 / 15 (train / validation / test), stratified for classification
- **Regularization:** BatchNorm, LeakyReLU, Dropout (0.1), weight decay
- **Optimizer:** Adam
- **Early stopping:** triggered by validation AUC (classification) or validation RMSE (regression)
- **Inputs:** structured features + text embeddings + image embeddings
Two supervised models were trained independently:

**Classification Model**
- **Objective:** predict `high_clickability` (0/1)
- **Metrics:** AUC, Accuracy, F1, Precision, Recall
![image](https://github.gatech.edu/user-attachments/assets/0ae0f1d4-f9a8-4d36-af9f-c075b233e2f6)
![image](https://github.gatech.edu/user-attachments/assets/62153adc-6887-4b36-bbe2-e902a500dfdd)

**Regression Model**
- **Objective:** predict `views_per_subscriber` (continuous)
- **Metrics:** RMSE, MAE, R²
![image](https://github.gatech.edu/user-attachments/assets/8f1e6f35-33fb-4a94-80bd-374b363f7157)



# Summary of Model Results

Across traditional machine learning baselines and the multimodal deep learning pipeline, the results show that **nonlinear models and semantically rich features provide the strongest predictive performance**, while linear methods struggle to capture the complexity of YouTube engagement.

---

## 1. Regression Models (Predicting Views per Subscriber)

### **Top Performers: Random Forest & XGBoost**
- **R² ≈ 0.26** — models explain ~26% of engagement variance  
- **MAE ≈ 10** — predictions typically within ±10 views/subscriber  
- **RMSE ≈ 38** — viral outliers remain difficult to capture  

### **Interpretation**
- Video performance is **nonlinear**, so Linear Regression performs poorly (**R² ≈ 0.02**).
- Ensemble models capture meaningful structure and outperform all baselines.

### **Feature Insights**
- **Subscriber count** is the most influential predictor.
- **Sentiment**, **capitalization ratio**, and **linguistic style** meaningfully contribute.
- **TF-IDF latent components** reveal coherent genre clusters (e.g., trailers, holiday content, reaction videos, event-driven uploads).

---

## 2. Classification Models (High vs. Low Performance)

### **Best Overall Model: HistGradientBoosting Classifier**
- **Accuracy:** 0.82  
- **Precision:** 0.66  
- **Recall:** 0.59  
- **F1:** 0.62  
- **ROC-AUC:** 0.85  

This model provides the best balance between precision and recall, making it the strongest choice when we care about catching as many high-performing videos as possible without producing too many false alarms.

### **Random Forest Classifier**
- **Accuracy:** 0.79  
- **Precision:** 0.71 (highest of all models)  
- **Recall:** 0.28  
- **F1:** 0.40  
- **ROC-AUC:** 0.86 (highest AUC)  

Random Forest is extremely precise and has the best ranking ability (highest AUC), but its conservative decision boundary means it misses many truly high-performing videos.

### **Logistic Regression Baseline**
- **Accuracy:** 0.76  
- **Precision:** 0.58  
- **Recall:** 0.19  
- **F1:** 0.28  
- **ROC-AUC:** 0.78  

Logistic Regression serves as a linear baseline. Its reasonable AUC indicates some separation power, but the very low recall and F1 show that linear decision boundaries are not sufficient for this task.

In summary, nonlinear tree-based methods dramatically outperform the linear baseline. HistGradientBoosting offers the most balanced practical performance, while Random Forest remains the best if we care primarily about precision and ranking quality.

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

(*Numerical results for the full multimodal model were limited by training time, but the architecture is optimized for superior multimodal performance.*)

---

### Key Takeaways
- **Nonlinear models are essential** for engagement prediction.  
- **Text features are the single strongest signal**, especially semantic embeddings.  
- **Structured metadata** (subscriber count, sentiment, formatting style) significantly boosts accuracy.  
- **Classification is easier than exact regression**, with strong ROC-AUC performance.  
- **Viral content remains unpredictable**, as all models underestimate extreme outliers.

---

### Limitations

There are several limitations in our current modeling pipeline. First, even the best regression models struggle with highly viral videos, producing systematic underestimation in the extreme right tail of the distribution. Second, metadata alone cannot fully determine engagement; external factors such as trends, timing, recommendation dynamics, and community preferences introduce irreducible noise. Third, thumbnail image quality varies widely across videos, and some missing or low-resolution thumbnails reduce signal strength. Finally, all models were trained on a trending dataset, which may not generalize perfectly to non-trending uploads.

---

### Interpretation
Across models, the findings support our central hypothesis that title characteristics combined with channel metadata can predict relative engagement potential. However, the dominance of the subscribers feature highlights that success is heavily conditioned by existing audience reach, with title wording contributing a secondary, but still measurable, effect. These insights align with prior research on algorithmic amplification and audience dynamics, suggesting that creators with established followings benefit more from metadata optimization than new creators. It is important to remember video titles and thumbnails are not the only factors contributing to views, so we should not expect perfect predictive ability based solely on those inputs. Instead, our goal is to evaluate to what extent presentation features have an impact and how that impact can be maximized.

## Next Steps

Looking ahead, several possible directions could further strengthen the system’s predictive capabilities. One natural extension would be to incorporate more detailed thumbnail information, such as higher-resolution crops, face localization, and text region extraction, to create a more complete multimodal model that mirrors how viewers process both text and imagery. Architectural adjustments, including expanded fusion layers or attention-based interactions, could also be explored to enhance the model’s ability to learn richer cross-modal representations.

Additional improvements could come from experimenting with hyperparameter tuning for ensemble models or adopting more robust evaluation strategies like stratified k-fold cross-validation. Finally, incorporating interpretability tools, such as SHAP for textual and structured features or CLIP-based attribution for thumbnail analysis, may offer valuable insights into why specific titles or thumbnails drive higher engagement. These potential steps would help guide future development and refinement of the system.

In addition, we identify three concrete next steps:

- **Hyperparameter tuning** (e.g., grid search or Bayesian optimization) for Random Forest, XGBoost, HistGradientBoosting, and the deep network to squeeze out additional performance.  
- **Collecting more diverse data**, including non-trending and long-tail videos, to improve generalization and reduce dataset-induced bias.  
- **Applying class rebalancing techniques** such as focal loss or synthetic minority oversampling (SMOTE) to further improve recall on high-performing but underrepresented videos.

---

## Gantt Chart & Contribution Table: {#Contributions}
[Click here -> Gantt Chart](https://docs.google.com/spreadsheets/d/1ks7QSZOliQQ410aY5oCGphUX07czt0Q8Edpbw5tmB1A/edit?usp=sharing)

![image](https://github.gatech.edu/user-attachments/assets/d4a9adbb-d843-47dd-8285-574a8459027d)

---

## References: {#References}
[1] G. Chatzopoulou, C. Sheng, and M. Faloutsos, “A first step towards understanding popularity in YouTube,” 2010 INFOCOM IEEE Conference on Computer Communications Workshops, Mar. 2010. doi:10.1109/infcomw.2010.5466701 

[2] R. Zhou, S. Khemmarat, L. Gao, J. Wan, and J. Zhang, “How youtube videos are discovered and its impact on video views,” Multimedia Tools and Applications, vol. 75, no. 10, pp. 6035–6058, Jan. 2016. doi:10.1007/s11042-015-3206-0 

[3] M. Bärtl, “YouTube channels, uploads and views: A statistical analysis of the past 10 years,” Convergence: The International Journal of Research into New Media Technologies, vol. 24, no. 1, pp. 16–32, Jan. 2018. doi:10.1177/1354856517736979 

[4] A. Testas, “Logistic regression with pandas, scikit-learn, and pyspark,” Distributed Machine Learning with PySpark, pp. 173–212, 2023. doi:10.1007/978-1-4842-9751-3_7

---

## GitHub Repository: {#GitHub}
[Click here -> GitHub Repository for this project](https://github.gatech.edu/sjin308/YouTube_Clickability_Study)

---

## Project Award Eligibility: {#Award}
We would like to be considered for the “Outstanding Project” award.
