import csv
import os
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from script.utils import compute_embeddings

if __name__ == '__main__':
    '''
    RQ1: configuration 1/3 BEYOND APPS
    Doc2Vec trained on DS + commoncrawl
    Classifiers trained on Labeled(DS)
    Classifiers tested on SS
    '''

    # load Labeled(DS)
    df_labeled_ds = pd.read_csv('D:\\doc2vec\\dataset\\training_sets\\DS_threshold_set.csv')
    df_labeled_ds['answer'] = [int(row != 2) for row in df_labeled_ds['HUMAN_CLASSIFICATION']]
    df_labeled_ds = df_labeled_ds.dropna(subset=['state1_content', 'state2_content',
                                                 'state1_tags', 'state2_tags',
                                                 'state1_content_tags', 'state2_content_tags'])

    # load SS
    df_ss = pd.read_csv('D:\\doc2vec\\dataset\\training_sets\\SS_threshold_set.csv')
    df_ss['answer'] = [int(row != 2) for row in df_ss['HUMAN_CLASSIFICATION']]
    df_ss = df_ss.dropna(subset=['state1_content', 'state2_content',
                                 'state1_tags', 'state2_tags',
                                 'state1_content_tags', 'state2_content_tags'])

    trained_models_path = 'D:\\doc2vec\\trained_model\\'
    embedding_type = ['content', 'tags', 'content_tags']
    # embedding_type = ['content', 'tags']
    vector_size = ['modelsize100']
    epochs = 31  # this is the best model in terms of accuracy from 07.classifier_scores_DS

    if not os.path.exists(r'..\\csv_results_table\\rq1-beyond-apps.csv'):
        header = ['Model', 'Embedding', 'Classifier', 'Accuracy', 'Precision', 'Recall', 'F1']
        with open('..\\csv_results_table\\rq1-beyond-apps.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(header)

    for ep in epochs:
        for emb in embedding_type:
            print("epoch: %s\tembedding: %s" % (str(ep), emb))
            name = trained_models_path + 'DS_' + emb + '_' + vector_size[0] + 'epoch' + str(ep) + '.doc2vec.model'
            model = Doc2Vec.load(name)

            comparison_df = pd.read_csv('..\\csv_results_table\\rq1-beyond-apps.csv')

            # load Labeled(DS) as training set
            X_train = []
            y_train = []
            pbar = tqdm(total=df_labeled_ds.shape[0])
            for inp in df_labeled_ds.iterrows():
                # TODO: too expensive. Replace with reading precomputed distances/embeddings
                ret_val = compute_embeddings(inp, model, emb, compute_similarity=True)
                if ret_val is None:
                    continue
                X_train.append(ret_val[0])
                y_train.append(ret_val[1])
                pbar.update()

            # load SS as training set
            X_test = []
            y_test = []
            pbar = tqdm(total=df_ss.shape[0])
            for inp in df_labeled_ds.iterrows():
                # TODO: too expensive. Replace with reading precomputed distances/embeddings
                ret_val = compute_embeddings(inp, model, emb, compute_similarity=True)
                if ret_val is None:
                    continue
                X_test.append(ret_val[0])
                y_test.append(ret_val[1])
                pbar.update()

            names = [
                "Nearest Neighbors",
                "SVM RBF",
                "Decision Tree",
                "Gaussian Naive Bayes",
                "Random Forest",
                "Ensemble",
                "Neural Network"
            ]

            classifiers = [
                KNeighborsClassifier(),
                SVC(),
                DecisionTreeClassifier(),
                GaussianNB(),
                RandomForestClassifier(),
                VotingClassifier(estimators=[('knn', KNeighborsClassifier()),
                                             ('svm', SVC()),
                                             ('dt', DecisionTreeClassifier()),
                                             ('gnb', GaussianNB()),
                                             ('rf', RandomForestClassifier())]),
                MLPClassifier()
            ]

            for name, model in zip(names, classifiers):

                # fit the classifier
                model = model.fit(X_train, y_train)

                # predict the scores
                y_pred = model.predict(X_test)

                # compute metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                print(f'{name}, accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}')

                a = ''
                if emb == 'content':
                    a = 'Content only'
                elif emb == 'tags':
                    a = 'Tags only'
                elif emb == 'content_tags':
                    a = 'Content and tags'
                elif emb == 'all':
                    a = "Ensemble"
                else:
                    print('nope')

                d1 = pd.DataFrame(
                    {'Model': ['DS_' + emb + '_' + vector_size[0] + 'epoch' + str(ep)], 'Embedding': [a],
                     'Classifier': [name], 'Accuracy': [accuracy], 'Precision': [precision], 'Recall': [recall],
                     'F1': [f1]})

                comparison_df = pd.concat([comparison_df, d1])

            comparison_df.to_csv('..\\csv_results_table\\rq1-beyond-apps.csv', index=False)
