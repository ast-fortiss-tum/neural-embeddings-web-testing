from os.path import join
from os import mkdir
import gensim
from tqdm import tqdm
from gensim.models.callbacks import CallbackAny2Vec
from multiprocessing import cpu_count


class MonitorCallback(CallbackAny2Vec):
    def __init__(self, epochs, vector_size, output_name):
        self.pbar = tqdm(total=epochs)
        self.epoch_number = 0
        self.vector_size = vector_size
        self.output_name = output_name

    def on_epoch_end(self, model):
        self.epoch_number += 1
        if self.epoch_number % 10 == 0:
            model.save(self.output_name + f'size{self.vector_size}epoch{self.epoch_number}.doc2vec.model')
        self.pbar.update()


def train_model(train_model_set_path, output_trained_model_path, vector_size, epochs):
    model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, min_count=4, epochs=epochs, workers=cpu_count())
    model.build_vocab(corpus_file=train_model_set_path)

    monitor = MonitorCallback(model.epochs, vector_size, output_trained_model_path)

    model.train(corpus_file=train_model_set_path, total_examples=model.corpus_count,
                total_words=model.corpus_total_words, epochs=model.epochs, callbacks=[monitor])

    model.save(output_trained_model_path + f'size{vector_size}epoch{epochs}.doc2vec.model')


try:
    mkdir('D:\\doc2vec\\trained_model')
except:
    pass

try:
    mkdir('D:\\doc2vec\\VERY_SMALL')
except:
    pass

print('Training Doc2Vec models on corpus VERY_SMALL')
train_model('D:\\doc2vec\\dataset\\train_model_corpus\content_model_train_set_VERY_SMALL.line_sentence',
            'D:\\doc2vec\VERY_SMALL\\content_model_train_set_VERY_SMALL', vector_size=30, epochs=300)
train_model('D:\\doc2vec\\dataset\\train_model_corpus\tags_model_train_set_VERY_SMALL.line_sentence',
            'D:\\doc2vec\VERY_SMALL\\tags_model_train_set_VERY_SMALL', vector_size=30, epochs=300)
train_model('D:\\doc2vec\\dataset\\train_model_corpus\content_tags_model_train_set_VERY_SMALL.line_sentence',
            'D:\\doc2vec\VERY_SMALL\\content_tags_model_train_set_VERY_SMALL', vector_size=30, epochs=300)

train_model('D:\\doc2vec\\dataset\\train_model_corpus\content_model_train_set_VERY_SMALL.line_sentence',
            'D:\\doc2vec\VERY_SMALL\\content_model_train_set_VERY_SMALL', vector_size=100, epochs=300)
train_model('D:\\doc2vec\\dataset\\train_model_corpus\tags_model_train_set_VERY_SMALL.line_sentence',
            'D:\\doc2vec\VERY_SMALL\\tags_model_train_set_VERY_SMALL', vector_size=100, epochs=300)
train_model('D:\\doc2vec\\dataset\\train_model_corpus\content_tags_model_train_set_VERY_SMALL.line_sentence',
            'D:\\doc2vec\VERY_SMALL\\content_tags_model_train_set_VERY_SMALL', vector_size=100, epochs=300)

try:
    mkdir('D:\\doc2vec\\SMALL')
except:
    pass
print('Training Doc2Vec models on corpus SMALL')
train_model('D:\\dataset\\train_model_corpus\content_model_train_set_SMALL.line_sentence',
            'D:\\doc2vec\SMALL\\content_model_train_set_SMALL', vector_size=100, epochs=150)
train_model('D:\\doc2vec\\dataset\\train_model_corpus\tags_model_train_set_SMALL.line_sentence',
            'D:\\doc2vec\SMALL\\tags_model_train_set_SMALL', vector_size=100, epochs=150)
train_model('D:\\doc2vec\\dataset\\train_model_corpus\content_tags_model_train_set_SMALL.line_sentence',
            'D:\\doc2vec\SMALL\\content_tags_model_train_set_SMALL', vector_size=100, epochs=150)

try:
    mkdir('D:\\doc2vec\\FULL')
except:
    pass
print('Training Doc2Vec models on corpus FULL')
train_model('D:\\dataset\\train_model_corpus\\content_model_train_set.line_sentence',
            'D:\\doc2vec\FULL\\content_model_train_set', vector_size=300, epochs=50)
train_model('D:\\doc2vec\\dataset\\train_model_corpus\tags_model_train_set.line_sentence',
            'D:\\doc2vec\FULL\\tags_model_train_set', vector_size=300, epochs=50)
train_model('D:\\doc2vec\\dataset\\train_model_corpus\content_tags_model_train_set.line_sentence',
            'D:\\doc2vec\FULL\\content_tags_model_train_set', vector_size=300, epochs=50)
