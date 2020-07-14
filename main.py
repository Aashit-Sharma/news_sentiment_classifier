import pandas as pd
import texthero as hero
from googlesearch import search
from goose3 import Goose
import BERT_Classfier
import swifter


def get_url(topic, site, num_urls=10):
    urls = list()
    for url in search(f'"{topic}" site:{site}', stop=num_urls, tpe='nws'):
        urls.append(url)

    return urls


def score_article(articles):
    sentiment = BERT_Classfier.predict_article_list(articles)

    return sentiment


def get_webdata(url_list):
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

    # TODO: String together topic and desc to use it for prediction.
    # After that build a state of the art summarizer and
    # compare the results of predictions on it with the one above


    sentiment = score_article(webdata_df['clean_text'].to_list())

    # webdata_df['Predicted_sentiment'] = webdata_df['clean_text']\
    #     .swifter.apply(BERT_Classfier.predict_article_list)

    webdata_df['Predicted_sentiment'] = sentiment

    # webdata_df['tfidf'] = (
    #     webdata_df['text'].pipe(hero.clean).pipe(hero.tfidf)
    # )
    #
    # webdata_df['kmeans_labels'] = (
    #     webdata_df['tfidf'].pipe(hero.kmeans, n_clusters=5).astype(str)
    # )

    return webdata_df


if __name__ == '__main__':
    topic = "Trump"
    site = 'bbc.com'
    urls = get_url(topic=topic, site=site, num_urls=10)

    get_webdata(urls).to_csv(f'results/{topic}_{site}.csv')
