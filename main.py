import news_crawlr as crawl

if __name__ == '__main__':
    topic = "China"
    site = 'bbc.com'
    urls = crawl.get_url(topic=topic, site=site, num_urls=10)

    crawl.get_webdata(urls).to_csv(f'results/{topic}_{site[:-4]}.csv')
