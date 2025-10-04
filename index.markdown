---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
---

{% include nav.html %}
{% include sidebar.html %}

## Proposal Video: {#Video}


## Introduction and Literature Review: {#Introduction}

The ability of YouTube videos to achieve success and sustain engagement is a central concern for creators and companies seeking to maximize reach. Presentation choices such as titles, thumbnails, and surrounding metadata influence visibility and interaction, yet the extent to which these features drive success relative to broader contextual factors like channel size or recommendation mechanisms remains unclear.


Prior studies provide some key insights into this challenge. Chatzopoulou, Sheng, and Faloutsos [1] found that early viewers are the most likely to engage, as the probability of likes, comments, or interactions declines with increasing view count, highlighting the temporal dynamics of audience behavior. Zhou et al. [2] examined YouTube’s discovery mechanisms, including keyword search, homepage recommendations, and event-driven highlights, and identified a “rich-get-richer” effect, in which initial traction compounds visibility across multiple recommendation pathways. This suggests metadata optimization of titles, tags, and thumbnails is critical for gaining early attention. Bärtl [3] analyzed channels and uploads over ten years, demonstrating that audience size, channel age, and genre strongly influence long-term video success, emphasizing the role of structural, channel-level factors.


Together, these studies indicate that YouTube popularity arises from a combination of audience behavior, platform algorithms, and creator characteristics. Early engagement patterns [1] drive disproportionate interaction, recommendation systems [2] amplify visibility, and channel-level features [3] condition overall success. These complementary insights provide a foundation for predictive modeling, highlighting the need to consider both presentation-level features and contextual metadata when estimating engagement.


Building on these findings, our project integrates early viewer behavior [1], discovery dynamics [2], and channel-level influences [3] to predict video success. We will extract textual and visual features from titles and thumbnails and combine them with contextual metadata, such as likes, dislikes or subscriber count, to develop predictive models. Engagement will be measured using normalized metrics such as the like–dislike ratio and view-to-comment or view-to-like ratios. This framework allows us to extend prior work to develop a practical approach for informing content strategy and optimizing presentation choices.

## Problem Definition: {#Problem}
Our goal for this study is to develop predictive models that estimate the likelihood of YouTube video success based on measurable features of titles, thumbnails, and contextual metadata. This approach aims to enhance our ability to identify which presentation choices most strongly influence engagement outcomes such as likes, views, and comments, thereby providing a data-driven framework for optimizing video design.

## Dataset: {#Dataset}
We will use the YouTube Trending Video Dataset from Kaggle, which compiles daily trending videos across multiple regions. The dataset includes features such as titles, thumbnails, genres, views, channels, likes, and dislikes, offering a rich foundation for building predictive models and testing how presentation and contextual factors influence engagement.

## Proposed Methods: {#Methods}
The YouTube dataset will be preprocessed to remove duplicates, handle missing values, apply log scaling, and scale numeric features to 0–1 using MinMaxScaler (X_std = (X - X_min) / (X_max - X_min); X_scaled = X_std * (max - min) + min).


Text features from titles and descriptions will be tokenized, stopwords removed, converted to TF-IDF vectors, and encoded with Word2Vec embeddings. Additional features include title length, word length, capitalization, numbers, buzzwords, and sentiment (positive, neutral, negative).


Image features from thumbnails include color, brightness, saturation histograms, face and text detection, text boldness, and image embeddings capturing style, objects, and layout. PCA will reduce dimensionality, and dense TF-IDF and embedding vectors will be concatenated into a unified dataset for modeling.


Supervised learning will explore feedforward neural networks for multimodal and nonlinear relationships, ensemble tree-based models (random forests, histogram-based gradient boosting) for mixed-feature interactions, and baseline models (logistic regression, decision trees) for simpler relationships and feature importance.

## (Potential) Results and Discussion: {#Results}
Model performance will be evaluated using classification accuracy, precision, recall, F1-score, and ROC-AUC to address class imbalance and ranking capability. We expect baseline models to perform moderately, while gradient-boosted trees and neural networks should achieve higher accuracy. Feature importance analyses will identify which title and thumbnail characteristics most strongly predict engagement. The results aim to be interpretable and actionable for content creators, highlighting ethical considerations such as avoiding incentivizing clickbait while providing insights into engagement patterns.

## Gantt Chart & Contribution Table: {#Contributions}


## References: {#References}
[1] G. Chatzopoulou, C. Sheng, and M. Faloutsos, “A first step towards understanding popularity in YouTube,” 2010 INFOCOM IEEE Conference on Computer Communications Workshops, Mar. 2010. doi:10.1109/infcomw.2010.5466701 


[2] R. Zhou, S. Khemmarat, L. Gao, J. Wan, and J. Zhang, “How youtube videos are discovered and its impact on video views,” Multimedia Tools and Applications, vol. 75, no. 10, pp. 6035–6058, Jan. 2016. doi:10.1007/s11042-015-3206-0 


[3] M. Bärtl, “YouTube channels, uploads and views: A statistical analysis of the past 10 years,” Convergence: The International Journal of Research into New Media Technologies, vol. 24, no. 1, pp. 16–32, Jan. 2018. doi:10.1177/1354856517736979 


[4] A. Testas, “Logistic regression with pandas, scikit-learn, and pyspark,” Distributed Machine Learning with PySpark, pp. 173–212, 2023. doi:10.1007/978-1-4842-9751-3_7


## GitHub Repository: {#GitHub}
[Click here -> GitHub Repository for this project](https://github.gatech.edu/sjin308/YouTube_Clickability_Study)

## Project Award Eligibility: {#Award}
We would like to be considered for the “Outstanding Project” award.


