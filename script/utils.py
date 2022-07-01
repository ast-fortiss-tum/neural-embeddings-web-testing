import json
from os.path import join

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

all_html_path = 'D:\\doc2vec\\dataset\\all_html'


def compute_embeddings(inp, trained_model, s='all', compute_similarity=False):
    '''

    '''
    try:
        (_, row) = inp
        if s == 'all':
            # create content embeddings
            metadata_content1 = json.loads(row['state1_content'])
            metadata_content2 = json.loads(row['state2_content'])
            emb1_content = load_and_create_embedding(metadata_content1, trained_model)
            emb2_content = load_and_create_embedding(metadata_content2, trained_model)

            # create tags embeddings
            metadata_tags1 = json.loads(row['state1_tags'])
            metadata_tags2 = json.loads(row['state2_tags'])
            emb1_tags = load_and_create_embedding(metadata_tags1, trained_model)
            emb2_tags = load_and_create_embedding(metadata_tags2, trained_model)

            # create content+tags embeddings
            metadata_content_tags1 = json.loads(row['state1_content_tags'])
            metadata_content_tags2 = json.loads(row['state2_content_tags'])
            emb1_content_tags = load_and_create_embedding(metadata_content_tags1, trained_model)
            emb2_content_tags = load_and_create_embedding(metadata_content_tags2, trained_model)

            if compute_similarity:
                # calculate the similarity between embeddings
                cos_sim_content = cosine_similarity(emb1_content, emb2_content)
                cos_sim_tags = cosine_similarity(emb1_tags, emb2_tags)
                cos_sim_content_tags = cosine_similarity(emb1_content_tags, emb2_content_tags)
                final_sim = np.array([cos_sim_content[0, 0], cos_sim_tags[0, 0], cos_sim_content_tags[0, 0]])
            else:
                emb1 = np.array([np.concatenate((emb1_content[0], emb2_content[0]))])
                emb2 = np.array([np.concatenate((emb1_tags[0], emb2_tags[0]))])
                emb3 = np.array([np.concatenate((emb1_content_tags[0], emb2_content_tags[0]))])

                final_sim = np.array([np.concatenate((emb1[0], emb2[0], emb3[0]))])[0]

            return final_sim, row['answer']

        if s == 'content':
            metadata_content1 = json.loads(row['state1_content'])
            metadata_content2 = json.loads(row['state2_content'])

            # create content embeddings
            emb1_content = load_and_create_embedding(metadata_content1, trained_model)
            emb2_content = load_and_create_embedding(metadata_content2, trained_model)

            if compute_similarity:
                # calculate the similarity between embeddings
                cos_sim_content = cosine_similarity(emb1_content, emb2_content)
                final_sim = np.array([cos_sim_content[0, 0]])
            else:
                final_sim = np.array([np.concatenate((emb1_content[0], emb2_content[0]))])[0]

            return final_sim, row['answer']

        if s == 'tags':
            metadata_tags1 = json.loads(row['state1_tags'])
            metadata_tags2 = json.loads(row['state2_tags'])

            # create tags embeddings
            emb1_tags = load_and_create_embedding(metadata_tags1, trained_model)
            emb2_tags = load_and_create_embedding(metadata_tags2, trained_model)

            if compute_similarity:
                # calculate the similarity between embeddings
                cos_sim_tags = cosine_similarity(emb1_tags, emb2_tags)
                final_sim = np.array([cos_sim_tags[0, 0]])
            else:
                final_sim = np.array([np.concatenate((emb1_tags[0], emb2_tags[0]))])[0]

            return final_sim, row['answer']

        if s == 'content_tags':
            metadata_content_tags1 = json.loads(row['state1_content_tags'])
            metadata_content_tags2 = json.loads(row['state2_content_tags'])

            # create content+tags embeddings
            emb1_content_tags = load_and_create_embedding(metadata_content_tags1, trained_model)
            emb2_content_tags = load_and_create_embedding(metadata_content_tags2, trained_model)

            if compute_similarity:
                # calculate the similarity between embeddings
                cos_sim_content_tags = cosine_similarity(emb1_content_tags, emb2_content_tags)
                final_sim = np.array([cos_sim_content_tags[0, 0]])
            else:
                final_sim = np.array([np.concatenate((emb1_content_tags[0], emb2_content_tags[0]))])[0]

            return final_sim, row['answer']

    except Exception as e:
        print(e, row)
        return None


def load_and_create_embedding(metadata, model):
    with open(join(all_html_path, metadata['path'])) as fp:
        data = json.load(fp)
    return model.infer_vector(data).reshape(1, -1)
