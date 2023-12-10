# **Introduction**

For this project I decided to make a recommendation system for a social network. The dataset I used only contains a list of friendships (pairs of IDs) and no other information. I used this simple dataset to explore learning from graphical data. The dataset that I used is from a Stanford paper  The data was collected from Facebook. It has 4039 users and 88234 friendships. 

Friend recommendation systems are integral to the user experience on social networks, influencing how people connect and interact online. In this project, I explored the development of such a system using a dataset that uniquely challenges us with its simplicity: it consists solely of friendship pairs from Facebook, devoid of additional user information typically leveraged in these systems. This dataset https://snap.stanford.edu/data/ego-Facebook.html, sourced from Stanford's SNAP (Stanford Network Analysis Project) repository, comprises 4,039 users and 88,234 friendships.

The objective was to build a recommender system capable of learning from graphical data, navigating the constraints posed by the dataset’s limited scope. This approach underscores the potential of utilising basic network structures to infer complex social connections, an endeavor that not only tests the boundaries of data-driven recommendations but also sheds light on the intricacies of social network dynamics.

# **Dataset and Preprocessing**

### Data Description

(Description of the dataset, its source, and basic statistics)

Facebook data was collected from survey participants. The facebook user ids were replaced by new ids ranging from 0 to 4038.

Dataset is represented in a graph in the following way: each user is a node and each friendship is an edge between two nodes.

### Evaluation

Given that recommender systems used by websites such as social networks have a lot more information such as likes, interactions and details of each user, their recommender systems are a lot more complex. If I had more data for example, when each connection was created, I could remove the most recent connections. This method would resemble the real-world scenario more closely. Also social media websites have data such as which recommendations were used (a friend request was sent based on the recommendation) and which were ignored or removed.

Since my data is limited and only contains the existing connections, to evaluate the model, my only option is to randomly select a sample of connections from the graph and treat them as positive samples. Combining these samples with randomly selected non-edges gives us the test set. 

### Class imbalance

Class imbalance is a significant challenge in machine learning, particularly in classification tasks. This issue is pronounced in my project due to the nature of social networks, where potential connections vastly outnumber actual connections.

In a fully connected network of 4,039 users, there would be 8,155,581 possible edges. However, the dataset contains only 88,234 actual friendships, which is just about 1.08% of the total possible connections. This stark disparity results in a dataset where 98.92% of the samples are negative (no connection) and only 1.08% are positive (existing connection).

In such a scenario, a classification model might learn to predict the majority class (negative samples) for all inputs, achieving high accuracy but poor predictive power. This is why relying solely on accuracy as a performance metric can be misleading in the context of class imbalance. Instead, I focused on recall (the ratio of true positives to the sum of true positives and false negatives) to gauge our model’s ability to identify actual connections accurately.

A common approach to handling class imbalance is to balance the dataset by undersampling the majority class or oversampling the minority class, often aiming for a 50/50 split. However, such an artificially balanced dataset may not reflect real-world conditions, where actual connections are indeed sparse.

I addressed the class imbalance challenge by configuring the model to identify and select the top n most likely positive connections, where n is the actual number of positive samples present in each specific test or validation set. This targeted prediction method, which aligns with real-world recommender system scenarios, is elaborated upon in the model development section.

### Train-Validation-Test Split

The unique characteristics of the dataset necessitated a custom approach to splitting the data for training, validation, and testing. This process was meticulously designed to prevent data leakage and address class imbalance.

#### Step 1: Test Set Creation

- **Selection of Test Edges**: Randomly selected 10% of edges from the original graph as positive samples for the test set.
- **Graph for Test Features**: Removed these edges from the original graph to form **`G_train`**, which was used to compute features for the test set.

#### Step 2: Cross-Validation Sets Formation

- **Partitioning for Validation/Training**: Divided the edges in **`G_train`** into 10 sets, each serving as positive samples for cross-validation sets.
- **Feature Computation**: Calculated features using graphs formed by the union of edges from the other nine sets, ensuring no data leakage.

#### Step 3: Distribution of Negative Samples

- **Negative Sample Allocation**: Partitioned all negative samples (non-edges) from the original graph into 11 equal sets.
- **Combining and Shuffling**: Assigned one set of negative samples to each cross-validation set and the test set, then combined them with the corresponding positive samples into dataframes and shuffled to prevent data leakage.

#### Step 4: Filtering Based on Common Neighbours

- **Removal of Zero Common Neighbours**: Eliminated rows (both edges and non-edges) with zero common neighbours across all dataframes.
- **Rationale**: Aimed to align recommendations with realistic social scenarios where mutual friends often facilitate connections.
- **Impact on Class Imbalance**: This filtering reduced the class imbalance from 1.08% to 8.00% positive samples.

#### Step 5: Model Training and Evaluation

- **Cross-Validation Procedure**: For each of the 10 cross-validation sets, trained the model on the aggregated data from the remaining 9 sets.
- **Performance Assessment**: Averaged the model's performance scores across all folds to determine the overall cross-validation score.

# **Feature Engineering**

I created various features for the train validation and test sets based each set’s corresponding graph. The features which I computed are as follows: Common Neighbours, Jaccard Coefficient, Adamic-Adar Index, Preferential Attachment, Resource Allocation Index, Community Commonality. Here is a detailed description of each of the features:

#### Common Neighbours

This feature represents the number of common neighbours between two nodes. In the context of a social network, if two users (nodes) have mutual friends, these mutual friends are their common neighbours. This feature is quantified as the count of such mutual connections. The underlying theory is that a higher number of common neighbours increases the likelihood of there being a direct link (edge) between the two nodes. It reflects the social network principle that people who share mutual friends are more likely to know each other or might be introduced in the future.

#### Jaccard Coefficient

The Jaccard Coefficient is a normalized measure of similarity between two nodes' neighborhoods. It is calculated as the size of the intersection of the two nodes' neighbor sets (i.e., their common neighbours) divided by the size of their union. This coefficient ranges from 0 to 1, where 0 indicates no commonality and 1 indicates identical sets of connections. Unlike the simple count of common neighbours, the Jaccard Coefficient accounts for the overall size of the two nodes' networks, thereby providing a proportionate measure of overlap. This is useful because it normalizes the common neighbours by the total number of unique neighbours both nodes have, giving a more balanced view that is less skewed by nodes with very high degrees of connectivity.

#### Adamic-Adar Index

This metric is based on the common neighbours between two nodes, but it applies more weight to less-connected neighbours. The rationale is that connections with rare mutual friends are more significant than those with mutual friends who are highly connected. The index is calculated as the sum of the inverse logarithm of the degree of common neighbours. In essence, it refines the common neighbours approach by considering the connectivity of those neighbours.

#### Preferential Attachment

This feature captures the idea that the likelihood of forming a new link is higher for nodes with a higher degree of connections. In other words, popular nodes tend to accumulate more connections over time. The preferential attachment score is calculated as the product of the number of edges of the two nodes. It's based on the assumption that nodes with a high number of connections are more likely to create additional connections.

#### Resource Allocation Index

This index is similar to the Adamic-Adar Index but with a different approach to weighting the shared connections. It is calculated by summing the inverse of the degree of common neighbours. This metric is based on the concept of resource transfer in networks, assuming that each node has a fixed amount of resource and can only distribute it equally among its neighbours.

#### Community Commonality

This feature determines whether two nodes belong to the same or different communities within the network. It operates on the premise that nodes within the same community or closely-knit groups are more likely to form connections. In our implementation, we use the Louvain method of community detection to partition the network, and then we assign a binary feature to each node pair. This binary feature is set to 1 if both nodes belong to the same community, indicating a higher likelihood of a connection, and 0 if they are in different communities.

### **Addressing Non-Gaussian Distributions**

In the dataset, most features showed a pronounced positive skew due to the large class imbalance. To address this:

- **Initial Step**: Applied log transformation (**`log(1 + x)`**) to all features to reduce skewness. This transformation was effective for the 'Preferential Attachment' feature, which approached a Gaussian distribution.
- **Evaluation of Transformation**: Despite the transformation, the remaining features continued to exhibit a significant positive skew.
- **Adapting Model Selection**: Given the limited impact of the transformation on overall feature distribution, I shifted the strategy towards selecting models that are inherently robust to non-Gaussian data. This decision aligns better with the dataset's intrinsic characteristics and the requirements of recommender systems.
- **Conclusion**: The process highlighted the need to balance between feature preprocessing and model selection, particularly in datasets common in real-world scenarios like recommender systems.

# **Model Development**

The development of this project involved a series of iterative approaches, each designed to address specific challenges and insights encountered along the way. Here's an overview of the different strategies I employed and the rationale behind them:

### **First Approach: Initial Data Splitting**

- **Method**: Initially, I split the graph's edges into three equal parts: 10% for training, 10% for validation, and 10% for testing. This left only 70% of the edges for constructing the training graph.
- **Challenge**: The primary issue with this approach was the significant reduction in the training graph's size, which lacked 30% of the edges. Moreover, the method did not provide a reliable set of negative samples for training, as edges in the validation and test sets were considered as negative samples in the training set.

### **Second Approach: Addressing Class Imbalance**

- **Adjustment**: To counter the imbalance, I sampled an equal number of non-edges (negative samples) to edges (positive samples), aiming for a balanced dataset.
- **Insight**: I realized that significant class imbalance is a characteristic and a necessity for recommender systems, as opposed to typical binary classification tasks.
- **Modification**: The non-edges were split into three parts and distributed equally among the training, validation, and test sets. This resulted in a class imbalance that was more reflective of the real-world data distribution, albeit still more unbalanced than the actual graph data.

### **Third Approach: Custom Cross-Validation and True Negatives**

- **Strategy**: I introduced a custom cross-validation procedure, allowing for the use of a larger percentage of the graph for training. This increased the amount of data available for validation.
- **Improvement**: By sampling negative samples from the complete graph for each set, I ensured the inclusion of true negatives. The non-edges were divided into 11 equal parts, achieving a class imbalance closer to the actual graph.

### **Targeted Prediction Approach: Aligning with Recommender System Dynamics**

- **Approach**: For each validation or test set, my model ranks potential connections by their predicted likelihood of being positive and selects the top 'n' predictions, where 'n' is the number of actual positive samples in the set.
- **Rationale**:
    - This approach is designed to emulate real-world recommender systems, where the utility lies in prioritizing a limited number of the most likely connections from a larger pool.
    - It shifts the focus from traditional binary classification metrics to the quality of the top predictions, which is more aligned with the objectives of practical recommender systems.
    - Recognising that relevance in recommendations is a gradient rather than a binary, this strategy aims to identify the most relevant connections at the top of the list. The goal is not just to predict connections accurately but to ensure that the highest-ranked predictions have the greatest likelihood of being meaningful and successful.

# **Model Selection and Evaluation Metrics**

Criteria for Model Selection: I needed classification models which can work with highly unbalanced classes and non-gaussian data. The models also need to have a probability feature.

Since I used the probability to select the top n samples, n being the number of positive samples in the validation/test set, The only metric I am concerned with is the recall. 

### **Individual Feature Performance**

I started from getting baseline measures from using single features.

| Feature | Recall |
| --- | --- |
| resource_allocation_index | 0.7325 |
| adamic_adar_index | 0.6584 |
| jaccard_coefficient | 0.6583 |
| common_neighbours | 0.6309 |
| preferential_attachment | 0.3924 |
| community_commonality | 0.1492 |

Resource Allocation Index was by far the most accurate. Preferential Attachment and Community Commonality don’t seem to be very usefull on their own.

### **Baseline Model Performance**

I have tried the following models:
Insert table with initial model results

| Model | Recall |
| --- | --- |
| LightGBM   | 0.744001 |
| CatBoost   | 0.742986 |
| XGBoost   | 0.742605 |
| Logistic Regression | 0.737504 |
| Random Forest | 0.737352 |
| AdaBoost | 0.736159 |
| KNN | 0.706757 |
| SDG Classifier | 0.684068 |

Most models performed slightly better than the best individual feature. KNN and SDGClassifier performed a lot worse.

## **Hyperparameter Tuning**

For hyperparameter tuning, I utilized my custom cross-validation process alongside comprehensive parameter grids for each model. However, the Random Forest Classifier was excluded from detailed tuning. This was due to its extensive run time in baseline tests (over 12 minutes) and results that, while not underwhelming, did not significantly outperform other models. This strategic choice was made to optimize computational efficiency and focus on models showing more promise. In preparing these grids, I also meticulously filtered out invalid parameter combinations to ensure each model was configured with feasible and effective settings. The specifics of these parameter grids are outlined in the following sections.

**Parameter Grids**

**XGBoost**

param_grid_xgb = {
    'learning_rate': [0.01, 0.1, 0.5, 1],
    'n_estimators': [100, 200, 500],  
    'max_depth': [3, 4, 5, 6], 
    'subsample': [0.5, 0.7, 1.0],
    'colsample_bytree': [0.5, 0.7, 1.0]
}

**KNN**

param_grid_knn = {
    'n_neighbors': [5, 7, 9, 11, 13, 15, 17, 21, 23],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'p': [1, 2]
}

**Logistic Regression**

param_grid_logistic = {
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'C': [0.01, 0.1, 1, 10],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'max_iter': [100, 200, 500],
        'l1_ratio': [0.2, 0.5, 0.8] 
    }

**LightGBM**

param_grid_lgbm = {
    'learning_rate': [0.01, 0.1, 0.5],
    'n_estimators': [50, 100, 200],
    'num_leaves': [31, 50, 100],
    'boosting_type': ['gbdt', 'dart', 'goss'],
    'subsample': [0.5, 0.7, 1.0]
}

**CatBoost**

param_grid_catboost = {
    'iterations': [100, 200, 500],
    'learning_rate': [0.01, 0.1, 0.5],
    'depth': [4, 6, 10],
    'l2_leaf_reg': [1, 3, 5, 7, 9]
}

**AdaBoost**

param_grid_adaboost = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0],
    'algorithm': ['SAMME', 'SAMME.R']
}

**SGD Classifier**

param_grid_sgd = {
    'loss': ['log_loss', 'modified_huber'],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
    'eta0': [0.01, 0.1, 1]  
}

**************Results**************

| Model | Parameters | Recall |
| --- | --- | --- |
| LightGBM   | 'learning_rate': 0.1, 'n_estimators': 200, 'num_leaves': 100, 'boosting_type': 'dart', 'subsample': 1 | 0.747618 |
| CatBoost   | 'iterations': 200, 'learning_rate': 0.1, 'depth': 10, 'l2_leaf_reg': 7 | 0.747541 |
| XGBoost   | 'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 6, 'subsample': 0.5, 'colsample_bytree': 0.7 | 0.746488 |
| SDG Classifier | 'loss': 'modified_huber', 'penalty': 'elasticnet', 'alpha': 0.0001, 'learning_rate': 'adaptive', 'eta0': 0.01 | 0.738773 |
| AdaBoost | 'n_estimators': 200, 'learning_rate': 1, 'algorithm': 'SAMME.R’ | 0.738735 |
| KNN | 'n_neighbors': 23, 'weights': 'uniform', 'metric': 'minkowski', 'p': 1 | 0.737973 |
| Logistic Regression | 'penalty': 'l2', 'C': 0.01, 'solver': 'liblinear', 'max_iter': 100 | 0.737923 |

The hyperparameter tuning process yielded varying degrees of improvement across different models. Notably, the **SDG Classifier** and **KNN** showed the most substantial improvements in recall, though they still did not emerge as the top-performing models.

- **SDG Classifier:** Improved by approximately 5.47% in recall score, from an initial 68.41% to 73.88%. Despite this significant enhancement, it didn't surpass the leading models in terms of overall performance.
- **KNN:** Exhibited an improvement of around 3.12%, increasing from 70.68% to 73.80%. This considerable improvement highlights the impact of choosing the right parameters for distance-based models.

For the other models, improvements were more modest:

- **CatBoost, XGBoost, and LightGBM:** These three models, which were already performing well, showed only slight improvements in recall after hyperparameter tuning. Their initial configurations were quite robust, leaving limited room for further enhancement.
- **AdaBoost and Logistic Regression:** These models also saw minimal improvements, indicating that their performance was relatively stable across different parameter settings.

It's worth noting that **LightGBM, CatBoost, and XGBoost** consistently stood out as the top three models. They had closely clustered recall scores, all hovering around the 74.7% mark, which was noticeably higher than the other models. This indicates a higher level of effectiveness and reliability in these models for the given dataset and problem.

In conclusion, while SDG and KNN demonstrated significant improvements, they did not reach the performance level of the top three models. LightGBM, CatBoost, and XGBoost maintained their positions as the most effective models in this analysis, showcasing their robustness and suitability for handling this specific classification task.

## **Ensemble Models**

In the next phase of my project, I explored the creation of ensemble models. These models combine the predictive power of multiple individual models, leveraging their strengths to achieve potentially better performance. The ensemble approach I adopted was based on probabilistic predictions, a suitable method considering the large and imbalanced nature of the dataset.

### **Combining Probabilities: Additive vs. Multiplicative Approach**

I had two strategies for integrating the probabilities from each model in the ensemble:

1. **Additive Approach:** This method involves summing the probabilities from each model to obtain an overall probability score for each sample. The higher the cumulative score, the more likely a sample is considered positive.
2. **Multiplicative Approach:** Here, probabilities from each model are multiplied together. This technique amplifies the agreement between models, as samples predicted as positive by all models will have higher combined probabilities.

### **Model Selection for the Ensemble**

The composition of the ensemble models was a critical decision. I experimented with two different sets:

1. **Top Performers Ensemble:** This ensemble included the three best-performing models - LightGBM, CatBoost, and XGBoost. The rationale was to combine the strengths of the models with the highest individual recall scores.
2. **Diverse Ensemble:** In this configuration, I added the SDG and Logistic Regression models to the top performers. The goal was to introduce diversity into the ensemble, potentially capturing unique patterns identified by these models, despite their slightly lower individual performance.

|  | Top 3 | Diverse |
| --- | --- | --- |
| Summed | 0.748036 | 0.746146 |
| Multiplied  | 0.748100 | 0.745371 |

## Test Results

These are the recall scores running the models on the test set. The models were trained on all the data except for the test set.

### Single features

| Feature | Recall |
| --- | --- |
| resource_allocation_index | 0.7318 |
| jaccard_coefficient | 0.6574 |
| adamic_adar_index | 0.6523 |
| common_neighbours | 0.6267 |
| preferential_attachment | 0.3782 |
| community_commonality | 0.1319 |

### Models (best parameters)

Here I used the models with the best parameters chosen during the cross validation

| Model | Recall |
| --- | --- |
| XGBoost | 0.741925 |
| LightGBM | 0.741358 |
| CatBoost | 0.740904 |
| SGD Classifer | 0.740904 |
| Logistic Regression | 0.740564 |
| AdaBoost | 0.738864 |
| KNN | 0.730024 |

### Ensemble models

Here I used the same ensemble models covered in the previous section.

|  | Top 3 | Diverse |
| --- | --- | --- |
| Summed | 0.741509 | 0.742106 |
| Multiplied  | 0.741472 | 0.742083 |

It appears that now the more diverse ensemble model performs better. This could be due to the fact that the hyperparemter tuning was performed on the cross validation set and is better fit to that data. On the other hand the diverse ensemble model which consists of more models generalisise better and avoids overfitting  

### Observations and Insights

- **Improved Performance with Diversity**: The 'Diverse' ensemble, which includes a wider range of models, exhibited marginally better performance compared to the 'Top 3' ensemble. This suggests a potential benefit in incorporating varied models into the ensemble.
- **Generalization Advantage**: The superior performance of the 'Diverse' ensemble could be attributed to its better generalization capabilities. Since the hyperparameter tuning was specifically tailored to the cross-validation data, models optimized in this manner might overfit to the nuances of that dataset. In contrast, a more varied ensemble brings together different perspectives and learning patterns, potentially reducing overfitting and enhancing performance on unseen data.
- **Summed vs. Multiplied Probabilities**: Both methods of combining probabilities (summation and multiplication) showed comparable results, indicating that the method of probability aggregation may have a minimal impact on the overall performance of these ensemble models.
