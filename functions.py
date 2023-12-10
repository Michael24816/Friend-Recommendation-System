# clean up imports

# Standard libraries
import math
import random
import itertools

# Data manipulation and analysis
import pandas as pd
import numpy as np

# Graph handling
import networkx as nx
import community as community_louvain

# Visualization
import matplotlib.pyplot as plt

# Machine learning tools
from sklearn.metrics import recall_score
from sklearn.preprocessing import RobustScaler


def compute_feature(df, graph, feature_generator, new_column_name):
    """
    Computes and adds a new feature column to the dataframe based on the given feature generator.

    :param df: DataFrame containing node pairs.
    :param graph: NetworkX graph object.
    :param feature_generator: Function to generate feature values.
    :param new_column_name: Name for the new feature column.
    """
    pairs = list(zip(df['node1'], df['node2']))    
    df[new_column_name] = [value for value in feature_generator(graph, pairs)]


def common_neighbours_generator(graph, pairs):
    """
    Generator for the number of common neighbours between pairs of nodes.

    :param graph: NetworkX graph object.
    :param pairs: Iterable of node pairs.
    """
    for e in pairs:
        yield len(list(nx.common_neighbors(graph, e[0], e[1])))


def jaccard_coefficient_generator(graph, pairs):
    """
    Generator for the Jaccard coefficient between pairs of nodes.

    :param graph: NetworkX graph object.
    :param pairs: Iterable of node pairs.
    """
    for u, v in pairs:
        neighbors_u = set(graph.neighbors(u))
        neighbors_v = set(graph.neighbors(v))
        yield len(neighbors_u & neighbors_v) / len(neighbors_u | neighbors_v) if neighbors_u | neighbors_v else 0


def adamic_adar_index_generator(graph, pairs):
    """
    Generator for the Adamic-Adar index between pairs of nodes.

    :param graph: NetworkX graph object.
    :param pairs: Iterable of node pairs.
    """
    for u, v in pairs:
        common_neighbors = set(graph.neighbors(u)) & set(graph.neighbors(v))
        yield sum(1 / math.log(graph.degree(w)) for w in common_neighbors if graph.degree(w) > 1)


def preferential_attachment_generator(graph, pairs):
    """
    Generator for the preferential attachment score between pairs of nodes.

    :param graph: NetworkX graph object.
    :param pairs: Iterable of node pairs.
    """
    for u, v in pairs:
        yield graph.degree(u) * graph.degree(v)


def resource_allocation_index_generator(graph, pairs):
    """
    Generator for the resource allocation index between pairs of nodes.

    :param graph: NetworkX graph object.
    :param pairs: Iterable of node pairs.
    """
    for u, v in pairs:
        common_neighbors = set(graph.neighbors(u)) & set(graph.neighbors(v))
        yield sum(1 / graph.degree(w) for w in common_neighbors if graph.degree(w))


def community_commonality_generator(graph, pairs):
    """
    Generator to determine if node pairs belong to the same community.

    :param graph: NetworkX graph object.
    :param pairs: Iterable of node pairs.
    """
    partition = community_louvain.best_partition(graph)
    for e in pairs:
        yield int(partition[e[0]] == partition[e[1]])


def test_split(G, percentage=0.1):
    """
    Splits the graph into test and training graphs based on a percentage of edges.

    :param G: Original graph.
    :param percentage: Percentage of edges to be used for testing.
    :return: Dictionary with test graph and test edges.
    """
    all_edges = list(G.edges())
    random.shuffle(all_edges)
    num_test_edges = int(len(all_edges) * percentage)
    x_edges_test = all_edges[:num_test_edges]
    g_edges_test = all_edges[num_test_edges:]

    G_test = nx.Graph()
    G_test.add_nodes_from(G.nodes())
    G_test.add_edges_from(g_edges_test)
    return {"graph": G_test, "x_edges": x_edges_test}


def validation_train_split(G, folds=9):
    """
    Splits graph into validation and training sets for cross-validation.

    :param G: Graph used for training or validation.
    :param folds: Number of folds for cross-validation.
    :return: List of dictionaries containing graphs and x_edges for each fold.
    """
    edges = list(G.edges())
    random.shuffle(edges)
    partition_size = len(edges) // folds
    partition = [edges[i:i + partition_size] for i in range(0, len(edges), partition_size)]
    partition = partition[:folds]

    cross_val_list = []
    for part in partition:
        complement = set(edges) - set(part)
        graph = nx.Graph()
        graph.add_nodes_from(G.nodes())
        graph.add_edges_from(complement)
        x_edges = [edge for edge in part if list(nx.common_neighbors(graph, edge[0], edge[1]))]
        cross_val_list.append({"graph": graph, "x_edges": x_edges})
    return cross_val_list


def df_from_edges(edges, non_edges, graph):
    """
    Creates a DataFrame from given edges and non-edges.

    :param edges: List of edges.
    :param non_edges: List of non-edges.
    :param graph: Graph for computing features.
    :return: DataFrame with edges, non-edges, and labels.
    """
    non_edges = [ne for ne in non_edges if list(nx.common_neighbors(graph, ne[0], ne[1]))]
    edges_dict_list = [{'node1': e[0], 'node2': e[1], 'label': 1} for e in edges]
    non_edges_dict_list = [{'node1': ne[0], 'node2': ne[1], 'label': 0} for ne in non_edges]
    train_dict_list = edges_dict_list + non_edges_dict_list
    df = pd.DataFrame(train_dict_list)
    return df.sample(frac=1).reset_index(drop=True)


def distribute_non_edges(G, G_test, test_edges, cross_val_list):
    """
    Distributes non-edges among cross-validation and test sets.

    :param G: Original graph.
    :param G_test: Test graph.
    :param test_edges: List of edges for the test set.
    :param cross_val_list: List of dictionaries for cross-validation sets.
    :return: Dictionary with dataframes for training/validation sets and the test set.
    """
    folds = len(cross_val_list)
    non_edges = list(nx.non_edges(G))
    random.shuffle(non_edges)

    num_partitions = 1 + folds
    partition_size = len(non_edges) // num_partitions
    partition = [non_edges[i:i + partition_size] for i in range(num_partitions)][0:folds]

    df_test = df_from_edges(test_edges, partition[0], G_test)
    df_test = df_test.sample(frac=1).reset_index(drop=True)

    df_list = []
    for i in range(folds):
        edges = cross_val_list[i]["x_edges"]
        graph = cross_val_list[i]["graph"]
        non_edges = partition[i]
        df = df_from_edges(edges, non_edges, graph)
        df_list.append(df.sample(frac=1).reset_index(drop=True))

    return {"train_val_df_list": df_list, "df_test": df_test}


def add_features_to_df(df, graph):
    """
    Adds network-based features to a dataframe.

    :param df: DataFrame to which features are to be added.
    :param graph: NetworkX graph object used for generating features.
    :return: DataFrame with added features.
    """
    # Compute and add various network-based features to the dataframe
    compute_feature(df, graph, common_neighbours_generator, 'common_neighbours', 'Computing Common Neighbours')
    compute_feature(df, graph, jaccard_coefficient_generator, 'jaccard_coefficient', 'Computing Jaccard Coefficient')
    compute_feature(df, graph, adamic_adar_index_generator, 'adamic_adar_index', 'Computing Adamic Adar Index')
    compute_feature(df, graph, preferential_attachment_generator, 'preferential_attachment', 'Computing Preferential Attachment')
    compute_feature(df, graph, resource_allocation_index_generator, 'resource_allocation_index', 'Computing Resource Allocation Index')
    compute_feature(df, graph, community_commonality_generator, 'community_commonality', 'Computing Community Commonality')

    return df


def scale_features(df):
    """
    Scales numerical features in the DataFrame using RobustScaler.

    :param df: DataFrame with features to scale.
    :return: DataFrame with scaled features.
    """
    # Selecting features to scale, excluding specific columns
    features_to_scale = df.drop(columns=['node1', 'node2', 'label', 'community_commonality'])

    # Applying RobustScaler to feature columns
    scaler = RobustScaler()
    scaled_features_array = scaler.fit_transform(features_to_scale)

    # Converting scaled features back to DataFrame
    scaled_features_df = pd.DataFrame(scaled_features_array, columns=features_to_scale.columns, index=features_to_scale.index)

    # Merging scaled features with original DataFrame
    return pd.concat([df[['node1', 'node2', 'label', 'community_commonality']], scaled_features_df], axis=1)


def cross_validate(model, df_list, feature_cols, verbosity=0):
    """
    Performs cross-validation for a given model and dataset.

    :param model: Model to be used for cross-validation.
    :param df_list: List of dataframes for cross-validation.
    :param feature_cols: Feature columns used for training.
    :param verbosity: Verbosity level for output messages.
    :return: Dictionary with predictions, true values, and recall score.
    """
    all_predictions, all_true_values = [], []

    for i, val_df in enumerate(df_list):
        # Preparing training and validation data
        train_df = pd.concat([df for j, df in enumerate(df_list) if j != i], ignore_index=True)
        X_train, y_train = train_df[feature_cols], train_df['label']
        X_val, y_val = val_df[feature_cols], val_df['label']

        # Training the model and predicting probabilities
        model.fit(X_train, y_train)
        probas = model.predict_proba(X_val)[:, 1]

        # Selecting top predictions based on probabilities
        top_indices = np.argsort(probas)[::-1][:y_val.sum()]
        predictions = np.zeros_like(probas, dtype=int)
        predictions[top_indices] = 1

        # Storing predictions and true values
        all_predictions.extend(predictions)
        all_true_values.extend(y_val)

        if verbosity > 0:
            print(f"Fold {i+1} / {len(df_list)} done")

    # Calculating recall score
    recall = recall_score(all_true_values, all_predictions)
    return {"all_predictions": all_predictions, "all_true_values": all_true_values, "recall": recall}


def expand_grid(dictionary):
    """
    Generates a DataFrame from all combinations of parameters in a dictionary.

    :param dictionary: Dictionary of parameters.
    :return: DataFrame with all combinations of parameters.
    """
    return pd.DataFrame([row for row in itertools.product(*dictionary.values())], 
                        columns=dictionary.keys())


def GridSearchCV(combinations, model_class, df_list, feature_cols):
    """
    Custom Grid Search Cross-Validation function.

    :param combinations: DataFrame of parameter combinations.
    :param model_class: Class of the model for grid search.
    :param df_list: List of dataframes for cross-validation.
    :param feature_cols: List of feature columns for training.
    :return: DataFrame with grid search results.
    """
    results_list = []

    # Identify integer columns
    int_cols = [col for col, dtype in combinations.dtypes.items() if dtype == 'int']

    for index, row in combinations.iterrows():
        params = {key: val for key, val in row.to_dict().items() if not pd.isna(val)}

        # Convert integer parameters to integers
        for col in int_cols:
            if col in params and not pd.isna(params[col]):
                params[col] = int(params[col])

        model = model_class(**params)
        results = cross_validate(model, df_list, feature_cols, verbosity=0)
        results_list.append({**params, 'recall': results['recall']})

    return pd.DataFrame(results_list)


def remove_recall_and_nan(params):
    """
    Removes 'recall' key and NaN values from a parameters dictionary.

    :param params: Dictionary of parameters.
    :return: Cleaned dictionary of parameters.
    """
    return {key: value for key, value in params.items() if key != 'recall' and pd.notna(value)}


def convert_to_int(params):
    """
    Converts float values to integers in a parameters dictionary.

    :param params: Dictionary of parameters.
    :return: Dictionary with converted integer values.
    """
    for key, value in params.items():
        if isinstance(value, float) and value.is_integer():
            params[key] = int(value)
    return params


def cross_validate_ensemble(models, df_list, feature_cols, multiply, verbosity=0):
    """
    Perform cross-validation for an ensemble of models and calculate the recall score.

    :param models: A list of machine learning models for the ensemble.
    :param df_list: A list of dataframes for cross-validation.
    :param feature_cols: List of feature columns for training.
    :param multiply: Boolean indicating whether to multiply or add probabilities.
    :param verbosity: Level of verbosity for output messages.
    :return: Dictionary with predictions, true values, and recall score.
    """
    all_predictions = []
    all_true_values = []
    
    for i, val_df in enumerate(df_list):
        train_df = pd.concat([df for j, df in enumerate(df_list) if j != i], ignore_index=True)
        X_train = train_df[feature_cols]
        y_train = train_df['label']
        X_val = val_df[feature_cols]
        y_val = val_df['label']

        ensemble_probas = np.ones(len(X_val)) if multiply else np.zeros(len(X_val))

        for model in models:
            model.fit(X_train, y_train)
            probas = model.predict_proba(X_val)[:, 1]
            ensemble_probas = ensemble_probas * probas if multiply else ensemble_probas + probas

        num_positives = y_val.sum()
        top_indices = np.argsort(ensemble_probas)[::-1][:num_positives]
        predictions = np.zeros_like(ensemble_probas, dtype=int)
        predictions[top_indices] = 1

        all_predictions.extend(predictions)
        all_true_values.extend(y_val)
        
        if verbosity > 0:
            print(f"Fold {i+1} / {len(df_list)} done")

    recall = recall_score(all_true_values, all_predictions)
    return {"all_predictions": all_predictions, "all_true_values": all_true_values, "recall": recall}


def test(df_test, df_list, model, feature_cols):
    """
    Test a model on a given test set and calculate the recall score.

    :param df_test: DataFrame for the test set.
    :param df_list: List of DataFrames for training.
    :param model: The machine learning model to be tested.
    :param feature_cols: List of feature columns for training.
    :return: Dictionary with predictions, true values, and recall score.
    """
    all_predictions = []
    all_true_values = []

    # Prepare the training data
    train_df = pd.concat(df_list, ignore_index=True)
    X_train = train_df[feature_cols]
    y_train = train_df['label']

    # Prepare the test data
    X_test = df_test[feature_cols]
    y_test = df_test['label']

    # Train and predict
    model.fit(X_train, y_train)
    probas = model.predict_proba(X_test)[:, 1]
    num_positives = y_test.sum()
    top_indices = np.argsort(probas)[::-1][:num_positives]
    predictions = np.zeros_like(probas, dtype=int)
    predictions[top_indices] = 1

    # Store results and calculate recall
    all_predictions.extend(predictions)
    all_true_values.extend(y_test)
    recall = recall_score(all_true_values, all_predictions)
    
    return {"all_predictions": all_predictions, "all_true_values": all_true_values, "recall": recall}


def test_ensemble(df_test, df_list, models, feature_cols, multiply):
    """
    Test an ensemble of models on a given test set and calculate the recall score.

    :param df_test: DataFrame for the test set.
    :param df_list: List of DataFrames for training.
    :param models: List of models in the ensemble.
    :param feature_cols: List of feature columns for training.
    :param multiply: Boolean indicating whether to multiply or add probabilities.
    :return: Dictionary with predictions, true values, and recall score.
    """
    all_predictions = []
    all_true_values = []

    # Prepare the training data
    train_df = pd.concat(df_list, ignore_index=True)
    X_train = train_df[feature_cols]
    y_train = train_df['label']

    # Prepare the test data
    X_test = df_test[feature_cols]
    y_test = df_test['label']

    # Ensemble prediction probabilities
    ensemble_probas = np.ones(len(X_test)) if multiply else np.zeros(len(X_test))

    # Train each model and predict probabilities
    for model in models:
        model.fit(X_train, y_train)
        probas = model.predict_proba(X_test)[:, 1]
        ensemble_probas = ensemble_probas * probas if multiply else ensemble_probas + probas

    # Select top n predictions and store results
    num_positives = y_test.sum()
    top_indices = np.argsort(ensemble_probas)[::-1][:num_positives]
    predictions = np.zeros_like(ensemble_probas, dtype=int)
    predictions[top_indices] = 1
    all_predictions.extend(predictions)
    all_true_values.extend(y_test)

    # Calculate recall
    recall = recall_score(all_true_values, all_predictions)
    
    return {"all_predictions": all_predictions, "all_true_values": all_true_values, "recall": recall}
