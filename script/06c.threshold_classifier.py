import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

if __name__ == '__main__':
    '''
    Compute accuracy for threshold based classifier
    '''

    embedding_type = ['content', 'tags', 'content_tags']
    # embedding_type = ['content']
    # dataset = ["DS", "SS"]
    dataset = ["DS"]

    df = None
    if "DS" in dataset:
        # load Labeled(DS)
        df = pd.read_csv('D:\\doc2vec\\dataset\\training_sets\\DS_threshold_set.csv')
    elif "SS" in dataset:
        print("SS distances not computed yet.")
        exit()
        # df = pd.read_csv('D:\\doc2vec\\dataset\\training_sets\\SS_threshold_set.csv')

    for emb in embedding_type:
        X = df['doc2vec_distance_' + emb]
        y = df['HUMAN_CLASSIFICATION']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)

        df_train = pd.DataFrame(list(zip(X_train, y_train)),
                                columns=['doc2vec_distance_' + emb, 'HUMAN_CLASSIFICATION'])

        # 0, 1 = clones; 2 = distinct
        df_clones = df_train.query("HUMAN_CLASSIFICATION != 2")
        df_clones = df_clones['doc2vec_distance_' + emb].to_list()

        df_distinct = df_train.query("HUMAN_CLASSIFICATION == 2")
        df_distinct = df_distinct['doc2vec_distance_' + emb].to_list()

        # plt.hist(df_clones, bins=5, label="similarities clones")
        # plt.hist(df_distinct, bins=5, label="similarities distinct")
        # plt.legend()
        # plt.show()
        #
        # data = [df_clones, df_distinct]
        # plt.boxplot(x=data, showmeans=True)
        # plt.legend()
        # plt.show()
        #
        # plt.clf()

        threshold = 0.8

        # TODO: test set here should be SS
        df_test = pd.DataFrame(list(zip(X_test, y_test)),
                               columns=['doc2vec_distance_' + emb, 'HUMAN_CLASSIFICATION'])

        # 0, 1 = clones; 2 = distinct
        df_clones = df_test.query("HUMAN_CLASSIFICATION != 2")
        df_clones_test = df_clones['doc2vec_distance_' + emb]
        tp = df_clones_test[df_clones_test > threshold].count()
        fn = len(df_clones_test) - tp

        df_distinct = df_test.query("HUMAN_CLASSIFICATION == 2")
        df_distinct_test = df_distinct['doc2vec_distance_' + emb]
        fp = df_distinct_test[df_distinct_test > threshold].count()
        tn = len(df_distinct_test) - fp

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        print(accuracy)
