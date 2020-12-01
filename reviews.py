from selectorlib import Extractor
import requests 
import json 
from time import sleep
import csv
from dateutil import parser as dateparser
import pandas as pd
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from nltk.sentiment import vader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
nltk.download("stopwords")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Create an Extractor by reading from the YAML file
e = Extractor.from_yaml_file('selectors.yml')

def scrape(url):    
    headers = {
        'authority': 'www.amazon.com',
        'pragma': 'no-cache',
        'cache-control': 'no-cache',
        'dnt': '1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'sec-fetch-site': 'none',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-dest': 'document',
        'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
    }

    # Download the page using requests
    print("Downloading %s"%url)
    r = requests.get(url, headers=headers)
    # Simple check to check if page was blocked (Usually 503)
    if r.status_code > 500:
        if "To discuss automated access to Amazon data please contact" in r.text:
            print("Page %s was blocked by Amazon. Please try using better proxies\n"%url)
        else:
            print("Page %s must have been blocked by Amazon as the status code was %d"%(url,r.status_code))
        return None
    # Pass the HTML of the page and create 
    return e.extract(r.text)

def generate_input_data():
    with open("urls.txt",'r') as urllist, open('data.csv','w', newline='', encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["title","content","date","variant","images","verified","author","rating","product","url"],quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for url in urllist.readlines():
            data = scrape(url) 
            if data:
                for r in data['reviews']:
                    # print(r)
                    r["product"] = data["product_title"]
                    r['url'] = url
                    if 'verified' in r:
                        if not r['verified'] or 'Verified Purchase' in r['verified']:
                            r['verified'] = 'Yes'
                        else:
                            r['verified'] = 'Yes'
                    if 'rating' in r and r['rating']:
                        r['rating'] = r['rating'].split(' out of')[0]
                    date_posted = r['date'].split('on ')[-1]
                    if r['images']:
                        r['images'] = "\n".join(r['images'])
                    r['date'] = dateparser.parse(date_posted).strftime('%d %b %Y')
                    writer.writerow(r)
                # sleep(5)


def get_continuous_chunks(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    continuous_chunk = []
    current_chunk = []
    for i in chunked:
            if type(i) == Tree:
                    current_chunk.append(" ".join([token for token, pos in i.leaves()]))
            if current_chunk:
                    named_entity = " ".join(current_chunk)
                    if named_entity not in continuous_chunk:
                            continuous_chunk.append(named_entity)
                            current_chunk = []
            else:
                    continue
    return continuous_chunk

def processing_data():
    input_dataframe = pd.read_csv("data.csv")
    input_dataframe["entities"] = input_dataframe["content"].apply(get_continuous_chunks)
    entity_sentiment_list = []
    for row in input_dataframe.iterrows():
        for entity in row[1]["entities"]:
            entity_sentiment_dict = {}
            entity_sentiment_dict["Text Content"] = row[1]["content"]
            entity_sentiment_dict["Entity Name"] = entity
            analyzer = SentimentIntensityAnalyzer()
            sentiment = analyzer.polarity_scores(row[1]["content"])
            if sentiment["neg"] > 0.05:
                entity_sentiment_dict["sentiment"] = "Negative"
            elif sentiment["pos"] > 0.05:
                entity_sentiment_dict["sentiment"] = "Positive"
            else:
                entity_sentiment_dict["sentiment"] = "Neutral"
            entity_sentiment_list.append(entity_sentiment_dict)
            
    output_dataframe = pd.DataFrame(entity_sentiment_list)
    output_dataframe.to_csv("output.csv")
            