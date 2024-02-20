import imaging_data
import pandas as pd
import numpy as np
import txtlogs
from tensorflow.keras.models import load_model
import tensorflow
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import db_helpers
from datetime import date
import hyperparams
import helpers



num_folds = hyperparams.num_folds
current_instance_id = db_helpers.getNextInstanceID()
create_date = date.today()


""" DATA SET CONTROLS - change this section when changing datasets """
# 1 = fuil, 2 = half, 3 = mid, 4 = small
# sample_size_id = 3
# data definition
dataset = "Cataracts"

dataset_id = db_helpers.datasets[dataset]

#vary the sample sizes according to the dataset
#set_sizes = db_helpers.sample_sizes[dataset]
set_sizes = [200]

for i, set_size in enumerate(set_sizes):
    sample_size_id = 4
    """ Data definition """
    X_all = imaging_data.x_train[:set_size]
    y_all = imaging_data.y_train[:set_size]
    y_train_labels = imaging_data.y_train_labels[:set_size]

    txtlogs.logText(f"There are {len(X_all)} images. Control: {np.sum(y_train_labels)}, treatment: {len(X_all)-np.sum(y_train_labels)}")


    # X_all = skin_cancer_data_new.x_train
    # y_all = skin_cancer_data_new.y_train
    # y_train_labels = skin_cancer_data_new.y_train_labels


    import tensorflow as tf
    with tf.device('/cpu:0'):
        X_all = tf.convert_to_tensor(X_all, np.float32)
        y_all = tf.convert_to_tensor(y_all, np.int64)
    # model definition


    feature_extractor = tensorflow.keras.models.load_model("./c0_efficientnet.h5", compile=False)

    """
    Define the FEATURE EXTRACTOR by removing the last (dense) layer, and extract the features.
    """
    # Feature extractor
    from keras.models import Model

    # Remove the last layer (Dense layer)
    feature_extractor = Model(inputs=feature_extractor.input, outputs=feature_extractor.layers[-2].output)

    # Extract features from the training data
    features = feature_extractor.predict(X_all, batch_size=1)

    features_df = pd.DataFrame(features.reshape(features.shape[0], -1))

    """ Use K-means to define clusters. We take 4 clusters as in the original paper's Chest X-ray dataset."""
    import hyperparams

    N_CLUSTERS = hyperparams.n_clusters

    train_features = feature_extractor.predict(X_all, batch_size=1)
    train_features_df = pd.DataFrame(train_features.reshape(train_features.shape[0], -1))

    # clusters defined
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=4)
    cluster_labels = kmeans.fit_predict(features_df)
    train_cluster_labels = kmeans.predict(train_features_df)

    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = train_features_df.index.values
    cluster_map['cluster'] = train_cluster_labels


    """ ################# evaluate the base model within each cluster using cross-val #################"""
    # c0_train = cluster_map[cluster_map.cluster == 0]
    # y0_train = np.array([y_all[i] for i in c0_train.data_index])
    # x0_train = np.array([X_all[i] for i in c0_train.data_index])
    # n_c0 = len(x0_train)



    with tf.device('/cpu:0'):
        c0_train = cluster_map[cluster_map.cluster == 0]
        y0_train = [y_all[i] for i in c0_train.data_index]
        x0_train = [X_all[i] for i in c0_train.data_index]
        intra_d_c0 = helpers.GetIntraClusterDistance(cluster_labels=cluster_labels, cluster_num = 0, features_df = features_df)
        n_c0_control, n_c0_treatment = helpers.ClusterClassDistribution(y_train_labels=y_train_labels, cluster_labels=cluster_labels, cluster_num=0)
        n_c0 = n_c0_control + n_c0_treatment

        c1_train = cluster_map[cluster_map.cluster == 1]
        y1_train = [y_all[i] for i in c1_train.data_index]
        x1_train = [X_all[i] for i in c1_train.data_index]
        intra_d_c1 = helpers.GetIntraClusterDistance(cluster_labels=cluster_labels, cluster_num = 1, features_df = features_df)
        n_c1_control, n_c1_treatment = helpers.ClusterClassDistribution(y_train_labels=y_train_labels, cluster_labels=cluster_labels, cluster_num=1)
        n_c1 = n_c1_control + n_c1_treatment

        c2_train = cluster_map[cluster_map.cluster == 2]
        y2_train = [y_all[i] for i in c2_train.data_index]
        x2_train = [X_all[i] for i in c2_train.data_index]
        intra_d_c2 = helpers.GetIntraClusterDistance(cluster_labels=cluster_labels, cluster_num = 2, features_df = features_df)
        n_c2_control, n_c2_treatment = helpers.ClusterClassDistribution(y_train_labels=y_train_labels, cluster_labels=cluster_labels, cluster_num=2)
        n_c2 = n_c2_control + n_c2_treatment

        c3_train = cluster_map[cluster_map.cluster == 3]
        y3_train = [y_all[i] for i in c3_train.data_index]
        x3_train = [X_all[i] for i in c3_train.data_index]
        intra_d_c3 = helpers.GetIntraClusterDistance(cluster_labels=cluster_labels, cluster_num = 3, features_df = features_df)
        n_c3_control, n_c3_treatment = helpers.ClusterClassDistribution(y_train_labels=y_train_labels, cluster_labels=cluster_labels, cluster_num=3)
        n_c3 = n_c3_control + n_c3_treatment


    # start with a blank model, which will be trained during CV
    from keras.models import Sequential
    from keras.layers import Flatten, Dense
    from keras.applications.efficientnet import EfficientNetB0
    import preprocess_data
    def build_model():
        img_width = preprocess_data.img_width
        img_height = preprocess_data.img_height

        activation = hyperparams.activation
        optimizer = hyperparams.optimizer
        loss = hyperparams.loss
        metrics = hyperparams.metrics

        efficientnet = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

        basemodel = Sequential()
        basemodel.add(efficientnet)
        basemodel.add(Flatten())
        # first argument of Dense is 2 because of the dimensionality of the data
        basemodel.add(Dense(2, activation=activation))
        basemodel.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return basemodel


    # within each cluster
    from sklearn.model_selection import cross_validate
    scoring = [ 'test_precision_macro', 'test_recall_macro', 'f1_macro']

    from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, roc_auc_score
    # Define custom scoring metrics
    scoring  = {'accuracy': make_scorer(accuracy_score),
                'precision': make_scorer(precision_score, average='macro'),
                'recall': make_scorer(recall_score, average='macro'),
                'f1': make_scorer(f1_score, average='macro'),
                'roc': make_scorer(roc_auc_score,  average='macro')
                }

    from scikeras.wrappers import KerasClassifier
    keras_model = KerasClassifier(model = build_model, optimizer=hyperparams.optimizer, batch_size = 4, epochs=hyperparams.epochs, verbose=0)

    """ Cross fold on each cluster and save results to send to database """

    c0_cross_fold_time, acc_c0, prec_c0, f1_c0, recall_c0, auc_c0 = helpers.GetMetricsFromCrossFold(keras_model, x = x0_train, y=y0_train, scoring=scoring)
    c1_cross_fold_time, acc_c1, prec_c1, f1_c1, recall_c1, auc_c1 = helpers.GetMetricsFromCrossFold(keras_model, x = x1_train, y=y1_train, scoring=scoring)
    c2_cross_fold_time, acc_c2, prec_c2, f1_c2, recall_c2, auc_c2 = helpers.GetMetricsFromCrossFold(keras_model, x = x2_train, y=y2_train, scoring=scoring)
    c3_cross_fold_time, acc_c3, prec_c3, f1_c3, recall_c3, auc_c3 = helpers.GetMetricsFromCrossFold(keras_model, x = x3_train, y=y3_train, scoring=scoring)



    print(" ------------------------- RESULTS --------------------------")
    print(id, current_instance_id, create_date)
    print("cluster 0: ")
    print(c0_cross_fold_time, acc_c0, prec_c0, recall_c0, auc_c0, n_c0, n_c0_control, n_c0_treatment, intra_d_c0)

    print("cluster 1: ")
    print(c1_cross_fold_time, acc_c1, prec_c1, recall_c1, auc_c1, n_c1, n_c1_control, n_c1_treatment, intra_d_c1)

    print("cluster 2: ")
    print(c2_cross_fold_time, acc_c2, prec_c2, recall_c2, auc_c2, n_c2, n_c2_control, n_c2_treatment, intra_d_c2)

    print("cluster 3: ")
    print(c3_cross_fold_time, acc_c3, prec_c3, recall_c3, auc_c3, n_c3, n_c3_control, n_c3_treatment, intra_d_c3)

    """ Send all results from 4 clusters to database; insert as a single row"""
    # n_c0
    # n_c1
    # n_c2
    # n_c3


    centroids = kmeans.cluster_centers_

    # Calculate the distances between each pair of centroids
    distances = []
    for i in range(4):
        for j in range(i+1, 4):
            centroid1 = centroids[i]
            centroid2 = centroids[j]
            distance = np.linalg.norm(centroid1-centroid2)
            distances.append(distance)

    dist_c0_c1 = distances[0]
    dist_c0_c2 = distances[1]
    dist_c0_c3 = distances[2]
    dist_c1_c2 = distances[3]
    dist_c1_c3 = distances[4]
    dist_c2_c3 = distances[5]



    args = [current_instance_id, dataset_id, create_date, n_c0, n_c1, n_c2, n_c3, n_c0_control, n_c1_control, n_c2_control, n_c3_control, n_c0_treatment, n_c1_treatment, n_c2_treatment, n_c3_treatment, intra_d_c0, intra_d_c1, intra_d_c2, intra_d_c3, acc_c0, auc_c0, f1_c0, prec_c0, recall_c0, acc_c1, auc_c1, f1_c1, prec_c1, recall_c1, acc_c2, auc_c2, f1_c2, prec_c2, recall_c2, acc_c3, auc_c3, f1_c3, prec_c3, recall_c3, dist_c0_c1, dist_c1_c2, dist_c2_c3, dist_c0_c2, dist_c0_c3, dist_c1_c3, c0_cross_fold_time, c1_cross_fold_time, c2_cross_fold_time, c3_cross_fold_time, sample_size_id]
    # cleaned_args = []
    # for arg in args:
    #     cleaned_args.append(helpers.nan_to_null(arg))


    sql_query = """INSERT INTO public.results(
        instance_id, dataset_id, create_date, n_c0, n_c1, n_c2, n_c3, n_c0_control, n_c1_control, n_c2_control, n_c3_control, n_c0_treatment, n_c1_treatment, n_c2_treatment, n_c3_treatment, intra_d_c0, intra_d_c1, intra_d_c2, intra_d_c3, acc_c0, auc_c0, f1_c0, prec_c0, recall_c0, acc_c1, auc_c1, f1_c1, prec_c1, recall_c1, acc_c2, auc_c2, f1_c2, prec_c2, recall_c2, acc_c3, auc_c3, f1_c3, prec_c3, recall_c3, dist_c0_c1, dist_c1_c2, dist_c2_c3, dist_c0_c2, dist_c0_c3, dist_c1_c3, c0_cross_fold_time, c1_cross_fold_time, c2_cross_fold_time, c3_cross_fold_time, sample_size_id)
        VALUES ({}, {}, '{}', {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {});""".format(*args)
    # sql_query1 = """INSERT INTO public.results(
    # 	instance_id, dataset_id, create_date, n_c0, n_c1, n_c2, n_c3, n_c0_control, n_c1_control, n_c2_control, n_c3_control, n_c0_treatment, n_c1_treatment, n_c2_treatment, n_c3_treatment, intra_d_c0, intra_d_c1, intra_d_c2, intra_d_c3, acc_c0, auc_c0, f1_c0, prec_c0, recall_c0, acc_c1, auc_c1, f1_c1, prec_c1, recall_c1, acc_c2, auc_c2, f1_c2, prec_c2, recall_c2, acc_c3, auc_c3, f1_c3, prec_c3, recall_c3, dist_c0_c1, dist_c1_c2, dist_c2_c3, dist_c0_c2, dist_c0_c3, dist_c1_c3, c0_cross_fold_time, c1_cross_fold_time, c2_cross_fold_time, c3_cross_fold_time)
    # 	VALUES ({}, {}, '{}', {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             }, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {});""".format(*cleaned_args)

    print(sql_query)
    # print(sql_query1)
    txtlogs.logText(sql_query)

    try:
        db_helpers.populateResults(sql_query)
    except:
        print("Datatypes led to error. Data was not inserted automatically.")