from flair.data import Corpus
from flair.data import Sentence
from flair.datasets import SENTEVAL_SST_GRANULAR
from flair.embeddings import WordEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer


def train():
    # 1. Get the SST-5 corpus
    corpus: Corpus = SENTEVAL_SST_GRANULAR()

    # 2. create the label dictionary
    label_dict = corpus.make_label_dictionary()

    # 3. make a list of word embeddings ( Using Glove for testing )
    word_embeddings = [WordEmbeddings('glove')]

    # 4. initialize document embedding by passing list of word embeddings
    document_embeddings = DocumentRNNEmbeddings(word_embeddings, hidden_size=256)

    # 5. create the text classifier
    classifier = TextClassifier(document_embeddings, label_dictionary=label_dict)

    # 6. initialize the text classifier trainer
    trainer = ModelTrainer(classifier, corpus)

    # 7. start the training
    trainer.train('resources/taggers/trec',
                  learning_rate=0.1,
                  mini_batch_size=32,
                  anneal_factor=0.5,
                  patience=5,
                  max_epochs=15)


def predict():
    classifier = TextClassifier.load('resources/taggers/trec/best-model.pt')

    sent_list = [
        'We urge the US to immediately withdraw its wrong decision, and stop any words and actions that interfere in China internal affairs and harm China interests spokeswoman Hua Chunying said.',
        'Bad dog eats a gross treat and he dislikes it',
        'good people refrain from eating dog food',
        'bad dog eats a healthy treat before winning the race',
        'beautifully ugly cars have windows in Romania']

    for sent in sent_list:
        # create example sentence
        sentence = Sentence(sent)

        # predict class and print
        classifier.predict(sentence)

        print(sentence.labels)


if __name__ == '__main__':
    train()
    predict()

    ''' 
    Prediction results after a quick 10 epochs
    [1 (0.4764)]
    [1 (0.4626)]
    [1 (0.3892)]
    [1 (0.2842)]
    [1 (0.4014)]
    '''
    # df = pd.read_csv('data.csv')
