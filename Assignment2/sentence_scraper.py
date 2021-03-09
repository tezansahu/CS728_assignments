import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import concurrent.futures
import time
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--words_data", type=str, default="./train.csv", help="path to list of noun compounds data")
parser.add_argument("--k", type=int, default=15, help="number of sentences to retrieve per noun compound")

def scrape_sentences(word, k):
    url = f"https://www.wordhippo.com/what-is/sentences-with-the-word/{word}.html"
    try:
        sentences = []
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for i, tr in enumerate(soup.find_all("tr", class_="exv2row1")):
            if i >= k:
                break
            print(word)
            sentences.append(tr.find_all("td")[0].text)
        return sentences
    except Exception:
        print("No sentences found for", word)
        return None



def get_sentences_for_noun_compounds(words_data, k=15):
    df = pd.read_csv(words_data, header=None, names=["word1", "word2", "interpretation"])
    # df = df.iloc[:20]
    noun_compounds = df.apply(lambda s: s["word1"] + "_" + s["word2"], axis=1)

    scraped_sentences = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_res = {executor.submit(scrape_sentences, word, k): word for word in noun_compounds}
        for future in concurrent.futures.as_completed(future_to_res):
            word = future_to_res[future]
            scraped_sentences[word] = future.result()
    
    with open("sentences.json", "w") as fout:
        fout.write(json.dumps(scraped_sentences, indent=4))



def main():
    args = parser.parse_args()
    print("Starting to scrape sentences...")
    s = time.time()
    get_sentences_for_noun_compounds(args.words_data, args.k)
    print("Completed scraping! Time taken: %.2f" % (time.time() - s))

if __name__ == "__main__":
    main()