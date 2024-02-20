import psycopg2

def AverageCrossFoldScores(scores):
    """scores : type dict"""
    import numpy as np
    avgScores = {}

    avgScores["fit_time"] = np.nanmean(scores["fit_time"])
    avgScores["test_accuracy"] = np.nanmean(scores["test_accuracy"])
    avgScores["test_precision"] = np.nanmean(scores["test_precision"])
    avgScores["test_recall"] = np.nanmean(scores["test_recall"])
    avgScores["test_f1"] = np.nanmean(scores["test_f1"])
    avgScores["test_roc"] = np.nanmean(scores["test_roc"])
    return avgScores

def GetIntraClusterDistance(cluster_labels, cluster_num, features_df):
    import numpy as np
    cluster_indices = np.where(cluster_labels == cluster_num)[0]


    from scipy.spatial.distance import pdist, squareform
    distances = pdist(features_df.iloc[cluster_indices])
    distance_matrix = squareform(distances)

    # calculate average intra-cluster distance
    intra_cluster_distance = np.mean(distance_matrix)

    return intra_cluster_distance

def ClusterClassDistribution(y_train_labels, cluster_labels, cluster_num):
    import numpy as np
    cluster_indices = np.where(cluster_labels == cluster_num)[0]
     # get the number of images in each class for the first cluster
    n_treatment = np.count_nonzero(y_train_labels[cluster_indices] == 1)
    n_control = np.count_nonzero(y_train_labels[cluster_indices] == 0)

    return n_control, n_treatment

def GetMetricsFromCrossFold(keras_model, x, y, scoring):
    import hyperparams
    from sklearn.model_selection import cross_validate
    import helpers
    import numpy as np
    if len(x) > 4:
        scores = cross_validate(keras_model, x, y, cv=hyperparams.num_folds,
                                scoring=scoring)
        cluster_metrics = helpers.AverageCrossFoldScores(scores)
        # unpack this dict for performance metrics
        c_cross_fold_time = cluster_metrics["fit_time"]
        acc_c = cluster_metrics["test_accuracy"]
        prec_c = cluster_metrics["test_precision"]
        f1_c = cluster_metrics["test_f1"]
        recall_c = cluster_metrics["test_recall"]
        auc_c = cluster_metrics["test_roc"]

        return c_cross_fold_time, acc_c, prec_c, f1_c, recall_c, auc_c

    else:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan



