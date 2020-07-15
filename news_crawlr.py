import pandas as pd
import texthero as hero
from goose3 import Goose
from googlesearch import search
import BERT_Classfier


def get_url(topic, site, num_urls=10):
    urls = list()
    for url in search(f'"{topic}" site:{site}', stop=num_urls, tpe='nws'):
        urls.append(url)

    return urls


def score_article(articles):
    sentiment = BERT_Classfier.predict_article_list(articles)

    return sentiment


def get_webdata(url_list):
    """

    :param url_list: List of URLs
    :return: Dataframe with cleaned text extracted from the URLS and their Sentiment Classifier predictions

    """

    webdata_df = pd.DataFrame()

    for url in url_list:
        # Getting html and Extracting the Metadata

        g = Goose()
        article = g.extract(url=url)
        webdata_df = webdata_df.append({'url': url, 'title': article.title, 'article_desc': article.meta_description,
                                        'date_pub': article.publish_date, 'text': article.cleaned_text},
                                       ignore_index=True)

        g.close()

    webdata_df['clean_text'] = (
        webdata_df['text'].pipe(hero.clean)
    )

    sentiment = score_article(webdata_df['clean_text'].to_list())

    webdata_df['Predicted_sentiment'] = sentiment

    webdata_df.drop(['text'], axis=1)

    return webdata_df
