import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from tqdm import tqdm

from script.utils import compute_embeddings

if __name__ == '__main__':
    '''
    Compute embeddings for the sets DS and/or SS
    '''

    trained_models_path = 'D:\\doc2vec\\trained_model\\'
    vector_size = ['modelsize100']
    epochs = 31  # this is the best model in terms of accuracy from 07.classifier_scores_DS
    # embedding_type = ['content', 'tags', 'content_tags']
    embedding_type = ['content_tags']
    # dataset = ["DS", "SS"]
    dataset = ["SS"]
    compute_similarity = True

    apps = ['addressbook', 'mantisbt', 'mrbs', 'pagekit', 'petclinic', 'phoenix', 'ppma']

    df = None
    if "DS" in dataset:
        # load Labeled(DS)
        df = pd.read_csv('D:\\doc2vec\\dataset\\training_sets\\DS_threshold_set.csv')
    elif "SS" in dataset:
        df = pd.read_csv('D:\\doc2vec\\dataset\\training_sets\\SS_threshold_set.csv')

    df['answer'] = [int(row != 2) for row in df['HUMAN_CLASSIFICATION']]
    df = df.dropna(subset=['state1_content', 'state2_content',
                           'state1_tags', 'state2_tags',
                           'state1_content_tags', 'state2_content_tags'])

    # for app in apps:
    for emb in embedding_type:
        print("computing embedding: %s\tsimilarity: %s" % (emb, str(compute_similarity)))
        name = trained_models_path + 'DS_' + emb + '_' + vector_size[0] + 'epoch' + str(epochs) + '.doc2vec.model'
        model = Doc2Vec.load(name)

        embeddings = []
        pbar = tqdm(total=df.shape[0])
        for inp in df.iterrows():
            ret_val = compute_embeddings(inp, model, emb, compute_similarity=compute_similarity)
            if ret_val is None:
                continue
            embeddings.append(ret_val[0][0])
            pbar.update()

        df['doc2vec_distance_' + emb] = embeddings
        if "DS" in dataset:
            df.to_csv('D:\\doc2vec\\dataset\\training_sets\\DS_threshold_set.csv')
        elif "SS" in dataset:
            df.to_csv('D:\\doc2vec\\dataset\\training_sets\\SS_threshold_set.csv')
