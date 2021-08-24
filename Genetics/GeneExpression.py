import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,"
import pandas as pd
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow import keras
from sklearn.manifold import TSNE
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.preprocessing import LabelEncoder
from tableone import TableOne
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE, SVMSMOTE, BorderlineSMOTE, SMOTEN, SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from numpy import mean
import collections

def ReadCSV(path):  ## Read Csv files

    df = pd.read_csv(path, dtype='object', engine='c', na_filter=False)
    return df

def ReadCSV2(path):  ## Read Csv files

    df = pd.read_csv(path, engine='c', na_filter=False)
    return df

def make_model(X_soc, X_gen):

    input_soc = keras.Input(X_soc.shape[-1], )
    input_gen = keras.Input(X_gen.shape[-1], )
    x_1 = keras.layers.Dense(16, activation="relu")(input_soc)
    x_3 = keras.layers.Dropout(0.1)(x_1)

    y_1 = keras.layers.Dense(80, activation="relu")(input_gen)
    y_2 = keras.layers.Dropout(0.1)(y_1)
    y_2 = keras.layers.Dense(128, activation='relu')(y_2)
    y_3 = keras.layers.Dropout(0.2)(y_2)

    c = keras.layers.concatenate([x_3, y_3])
    output = keras.layers.Dense(1, activation='sigmoid', name='outputs')(c)

    model = keras.Model(inputs=[input_soc, input_gen], outputs=output)
    print(model.summary())

    return model

def sklearn_model(X_scale_soc, y_resampled_soc, X_pca_train_gen, y_resampled_gen, X_scale_val_soc, X_pca_val_gen):

    # Supervised transformation based on random forest:
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=3)
    rf_model.fit(X_scale_soc, y_resampled_soc)
    rf_prod_train = rf_model.predict_proba(X_scale_soc)
    rf_prod = rf_model.predict_proba(X_scale_val_soc)
    #print(accuracy_score(test_y, rf_pred))

    svm_model = SVC(C=0.1, kernel='rbf', probability=True)
    svm_model.fit(X_pca_train_gen, y_resampled_gen)
    svm_prod_train = svm_model.predict_proba(X_pca_train_gen)
    svm_prod = svm_model.predict_proba(X_pca_val_gen)
    #print(accuracy_score(test_y, svm_pred))
    final_prod_train = (rf_prod_train + svm_prod_train)/2
    final_prod = (rf_prod + svm_prod)/2
    preds_train = final_prod_train[:,1]
    preds_test = final_prod[:,1]
    train_pred = np.argmax(final_prod_train, axis=1)
    test_pred = np.argmax(final_prod, axis=1)


    return preds_train, preds_test, train_pred, test_pred


def tsne_plot(x1, y1, name="graph.png"):
    tsne = TSNE(n_components=2, random_state=0)
    X_t = tsne.fit_transform(x1)

    pyplot.figure(figsize=(12, 8))
    pyplot.scatter(X_t[np.where(y1 == 0), 0], X_t[np.where(y1 == 0), 1], marker='o', color='g', linewidth='1', alpha=0.8, label='Non Fraud')
    pyplot.scatter(X_t[np.where(y1 == 1), 0], X_t[np.where(y1 == 1), 1], marker='o', color='r', linewidth='1', alpha=0.8, label='Fraud')

    pyplot.legend(loc='best')
    pyplot.savefig(name)

def plot_roc_curve(fpr_train, tpr_train, fpr_test, tpr_test, auc_train, auc_test, name):

    plt.figure(2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_train, tpr_train, label='Keras (area = {:.3f})'.format(auc_train))
    plt.plot(fpr_test, tpr_test, label='Sklearn (area = {:.3f})'.format(auc_test))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(name)
    plt.close()


if __name__ == "__main__":
    loc_soc = 'socioeconomicdata.csv'
    loc_exp = 'expression_data.csv'

    data_soc = ReadCSV2(loc_soc)

    # Creating new age bins
    # data_soc['Age'] = data_soc['Age'].astype(int)
    # data_soc[['Age', 'YrsEducation', 'bmi', 'sbp', 'dbp', 'pp', 'Urasgmcr', 'dead', 'PatientID', 'cigsmoke', 'Gender']] = \
    # data_soc[['Age', 'YrsEducation', 'bmi', 'sbp', 'dbp', 'pp', 'Urasgmcr', 'dead', 'PatientID', 'cigsmoke', 'Gender']].apply(pd.to_numeric)

    # print(data_soc['bp_med'].head())
    # print(data_soc['Urasgmcr'].head())

    names = ['25-34', '35-44', '45-54', '55-70']
    data_soc["Age_binned"] = pd.cut(data_soc["Age"], bins=[24, 35, 45, 55, 70], labels=names)

    # Creating new bin for years education
    years = ['0-4', '5-9', '10-20']
    data_soc['YrsEducation_binned'] = pd.cut(data_soc['YrsEducation'], bins=[-np.inf, 5, 10, 20], labels=years)
    data_soc['Urasgmcr'] = data_soc['Urasgmcr'].apply(pd.to_numeric)
    print(data_soc.dtypes)

    data_soc = data_soc.drop(['BaselineBlood', 'Age', 'YrsEducation', 'bp_med', 'DateDeath'],
                             axis=1)
    #print(list(data_soc.columns))
    #print(data_soc['YrsEducation_binned'].value_counts())

    # Tableone statistics:

    columns = ['Gender', 'cigsmoke', 'bmi', 'sbp', 'dbp', 'pp', 'Urasgmcr', 'Age_binned', 'YrsEducation_binned','dead']
    cat_cols = ['Age_binned', 'YrsEducation_binned', 'Gender', 'cigsmoke', 'dead']
    mytable = TableOne(data_soc, columns=columns, categorical= cat_cols)
    #mytable.to_csv('summary.csv')

    data_exp = ReadCSV2(loc_exp)
    data_exp = data_exp.drop(columns=['Unnamed: 0', 'BaselineBlood', 'dead'])
    #print(data_exp.shape)
    numeric_ = list(data_exp.columns)
    unwanted = ['PatientID']
    numeric_ = [e for e in numeric_ if e not in unwanted]
    #
    soc_cols = ['Gender', 'cigsmoke', 'bmi', 'Urasgmcr']
    numeric_cols = numeric_ + soc_cols

    final_data = pd.merge(data_exp, data_soc, on=['PatientID'], how='inner')
    # print(list(final_data.columns))
    # print(len(final_data))
    # print(final_data['PatientID'].nunique())

    final_data = final_data.drop(columns=['PatientID'])

    print(final_data['dead'].value_counts())

    X = final_data.drop(columns=['dead'], axis=1)


    X['Age_binned'] = X['Age_binned'].astype('category')
    X['Age_cat'] = X['Age_binned'].cat.codes

    X['YrsEducation_binned'] = X['YrsEducation_binned'].astype('category')
    X['YrsEducation_cat'] = X['YrsEducation_binned'].cat.codes

    X = X.drop(columns=['Age_binned', 'YrsEducation_binned'])
    X = X.fillna(0)
    Y = final_data[['dead']]

    train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size=0.1, shuffle=True, random_state=88, stratify=Y)


    X_soc = train_X[['Gender', 'cigsmoke', 'bmi', 'sbp', 'dbp', 'pp', 'Urasgmcr','Age_cat', 'YrsEducation_cat']].values
    X_gen = train_X.drop(columns=['Gender', 'cigsmoke', 'bmi', 'sbp', 'dbp', 'pp', 'Urasgmcr', 'Age_cat', 'YrsEducation_cat'], axis=1).values
    Y1 = train_y.values


    # test data
    test_soc = test_X[['Gender', 'cigsmoke', 'bmi', 'sbp', 'dbp', 'pp', 'Urasgmcr','Age_cat', 'YrsEducation_cat']].values
    test_gen = test_X.drop(columns=['Gender', 'cigsmoke', 'bmi', 'sbp', 'dbp', 'pp', 'Urasgmcr', 'Age_cat', 'YrsEducation_cat'], axis=1).values
    test_y = test_y.values
    print(test_y.shape)


    over = SMOTE(sampling_strategy=0.4)
    under = RandomUnderSampler(sampling_strategy=1)
    steps = [('over', over), ('under', under)]
    pipeline = imbPipeline(steps=steps)

    # fit and apply the pipeline

    X_resampled_soc, y_resampled_soc = under.fit_resample(X_soc, Y1)
    X_resampled_gen , y_resampled_gen = under.fit_resample(X_gen, Y1)
    print('Resampled dataset shape %s' % Counter(y_resampled_soc))
    print(y_resampled_gen.shape)
    print(y_resampled_soc.shape)

    sc = StandardScaler()
    X_scale_soc = sc.fit_transform(X_resampled_soc)
    X_scale_val_soc = sc.transform(test_soc)

    X_scale_gen = sc.fit_transform(X_resampled_gen)
    X_scale_val_gen = sc.transform(test_gen)

    # PCA
    pca = PCA(n_components=10,svd_solver='full')
    X_pca_train_gen = pca.fit_transform(X_scale_gen)
    X_pca_val_gen = pca.transform(X_scale_val_gen)

    model = make_model(X_scale_soc, X_pca_train_gen)
    file_name = '/home/sshah23/Genetics/genetic_model_design.png'
    tf.keras.utils.plot_model(model, file_name, show_shapes=True)
    metrics = [
        # keras.metrics.FalseNegatives(name="fn"),
        # keras.metrics.FalsePositives(name="fp"),
        # keras.metrics.TrueNegatives(name="tn"),
        # keras.metrics.TruePositives(name="tp"),
        # keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        # keras.metrics.BinaryAccuracy(name='accuracy'),
        #keras.metrics.AUC(name='auc'),
    ]

    model.compile(
        optimizer=keras.optimizers.Nadam(), loss="binary_crossentropy", metrics=metrics
    )

    callbacks = [keras.callbacks.ModelCheckpoint("fraud_model_at_epoch_{epoch}.h5")]

    history = model.fit([X_scale_soc, X_pca_train_gen],
                        y_resampled_gen,
                        batch_size=50,
                        epochs=100,
                        verbose=1,
                        shuffle=True,
                        validation_split=0.1
                        )
    plt.figure(1)
    plt.plot(history.history['loss'], label='train loss')
    #pyplot.plot(history.history['val_loss'], label='val loss')
    plt.legend()
    plt.savefig('training_loss.png')
    filename = 'Genetics_model.h5'
    model.save(filename)

    # evaluate the model
    scores1 = model.evaluate([X_scale_soc, X_pca_train_gen], y_resampled_gen, verbose=0)
    print(scores1)
    scores2 = model.evaluate([X_scale_val_soc, X_pca_val_gen], test_y, verbose=0)
    print(scores2)

    # Predict train classes Keras:
    pred_prod_train = model.predict([X_scale_soc, X_pca_train_gen])
    pred_prod_train_1 = pred_prod_train.ravel()
    pred_class_train = (pred_prod_train > 0.5).astype(int)
    # Predict test classes keras:
    pred_prob_test = model.predict([X_scale_val_soc, X_pca_val_gen])
    pred_prob_test_1 = pred_prob_test.ravel()
    pred_class_test = (pred_prob_test > 0.5).astype(int)

    ## Sklearn model:

    prod_sklearn_train, prod_sklearn, sklearn_pred_train, sklearn_pred = sklearn_model(X_scale_soc, y_resampled_soc, X_pca_train_gen, y_resampled_gen, X_scale_val_soc, X_pca_val_gen)
    # print(sklearn_pred_train)
    # print(sklearn_pred)

    # Training classification report:
    class_labels = [0, 1]
    report_train = classification_report(y_resampled_gen, pred_class_train)
    print('Training cr: ')
    print(report_train)
    print('Keras Training cf: ')
    print(confusion_matrix(y_resampled_gen, pred_class_train))

    # Test classification report:
    report = classification_report(test_y, pred_class_test)
    print('Keras Testing cr: ')
    print(report)
    print('Keras Testing cf: ')
    print(confusion_matrix(test_y, pred_class_test))

    print('Sklearn training cf: ')
    print(classification_report(y_resampled_soc, sklearn_pred_train))
    print('Sklearn training: ')
    print(confusion_matrix(y_resampled_soc, sklearn_pred_train))
    print('Sklearn test cf: ')
    print(classification_report(test_y, sklearn_pred))
    print('Sklearn test: ')
    print(confusion_matrix(test_y, sklearn_pred))

    ## plotting the roc curve
    from sklearn.metrics import roc_curve, roc_auc_score, auc

    ## Training comparision auc roc:
    fpr_train, tpr_train, thresholds_train = roc_curve(y_resampled_gen, pred_prod_train_1)
    auc_train = auc(fpr_train, tpr_train)

    fpr_train_sklearn, tpr_train_sklearn, thresholds_sklearn = roc_curve(y_resampled_soc, prod_sklearn_train)
    auc_sklearn = auc(fpr_train_sklearn, tpr_train_sklearn)

    training = 'roc_auc_training2.png'
    plot_roc_curve(fpr_train, tpr_train, fpr_train_sklearn, tpr_train_sklearn, auc_train, auc_sklearn, training)

    ## Testing comparision auc roc:
    fpr_test, tpr_test, thresholds_test = roc_curve(test_y, pred_prob_test_1)
    auc_test = auc(fpr_test, tpr_test)
    print('test auc: ', auc_test)

    fpr_test_sklearn, tpr_test_sklearn, thresholds_sklearn_test = roc_curve(test_y, prod_sklearn)
    auc_sklearn_test = auc(fpr_test_sklearn, tpr_test_sklearn)

    testing = 'roc_auc_testing2.png'
    plot_roc_curve(fpr_test, tpr_test, fpr_test_sklearn, tpr_test_sklearn, auc_test, auc_sklearn_test, testing)





