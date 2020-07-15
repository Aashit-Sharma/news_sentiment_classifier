from flair.data import Corpus
from flair.data import Sentence
from flair.datasets import SENTEVAL_SST_GRANULAR
from flair.embeddings import WordEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer


def train():
    # Get the SST-5 corpus
    corpus: Corpus = SENTEVAL_SST_GRANULAR()

    # create the label dictionary
    label_dict = corpus.make_label_dictionary()

    # make a list of word embeddings ( Using Glove for testing )
    word_embeddings = [WordEmbeddings('glove')]

    # initialize document embedding by passing list of word embeddings
    document_embeddings = DocumentRNNEmbeddings(word_embeddings, hidden_size=256)

    # create the text classifier
    classifier = TextClassifier(document_embeddings, label_dictionary=label_dict)

    # initialize the text classifier trainer
    trainer = ModelTrainer(classifier, corpus)

    # start the training
    trainer.train('resources/taggers/trec',
                  learning_rate=0.1,
                  mini_batch_size=32,
                  anneal_factor=0.5,
                  patience=5,
                  embeddings_storage_mode='gpu',
                  max_epochs=15)


def predict(doc):
    classifier = TextClassifier.load('resources/taggers/trec/final-model.pt')
    # create example sentence
    sentence = Sentence(doc)

    # predict class and print
    classifier.predict(sentence)

    return sentence.labels


def predict_article_list(articles: list):
    classifier = TextClassifier.load('resources/taggers/trec/final-model.pt')
    predictions = list()
    for article in articles:
        sentence = Sentence(article)

        # predict class and print
        classifier.predict(sentence)
        predictions.append(sentence.labels)

    return predictions

def test_pred():

    sent_list = [
        'We urge the US to immediately withdraw its wrong decision, and stop any words and actions that interfere in China internal affairs and harm China interests spokeswoman Hua Chunying said.',
        'Bad dog eats a gross treat and he dislikes it',
        'good people refrain from eating dog food',
        'bad dog eats a healthy treat before winning the race',
        'beautifully ugly cars have windows in Romania'
        ]
    classifier = TextClassifier.load('resources/taggers/trec/final-model.pt')

    for sent in sent_list:
        # create example sentence
        sentence = Sentence(sent)
        classifier.predict(sentence)

        # predict class and print
        classifier.predict(sentence)

        print(type(sentence.labels))
        print(sentence.embedding)
        print(list(sentence.labels))


if __name__ == '__main__':
    # train()
    # predict()
    test_pred()
