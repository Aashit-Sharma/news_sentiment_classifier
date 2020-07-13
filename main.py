import pandas as pd
import texthero as hero
from googlesearch import search
from goose3 import Goose


def get_url(topic, site, num_urls=10):
    urls = list()
    for url in search(f'"{topic}" site:{site}', stop=num_urls, tpe='nws'):
        urls.append(url)

    return urls


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

    webdata_df['clean'] = (
        webdata_df['text'].pipe(hero.clean)
    )

    webdata_df['tfidf'] = (
        webdata_df['text'].pipe(hero.clean).pipe(hero.tfidf)
    )

    webdata_df['kmeans_labels'] = (
        webdata_df['tfidf'].pipe(hero.kmeans, n_clusters=5).astype(str)
    )

    # print(hero.top_words(webdata_df['clean']))
    return webdata_df


if __name__ == '__main__':
    urls = get_url("india", 'bbc.com', num_urls=10)

    get_webdata(urls).to_csv('data.csv')
