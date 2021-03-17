import os
import json
import pandas as pd
import numpy as np
import gensim.downloader as api
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--data", type=str, default="./train.csv", help="path to training data")
parser.add_argument("--approach", type=int, default=1, choices=[1, 2, 3], help="approach to be followed for the task (based on the problem statement)")
parser.add_argument("--train", action='store_true', help="flag to indicate that dataset is to be used for training (needed for approaches 2 & 3)")
parser.add_argument("--save_data_as", type=str, default="nc_emb.pkl", help="save the created dataset as")

class DatasetPreparationApproach1:
    def __init__(self, data_path):
        print("Loading glove embeddings...")
        self.df = pd.read_csv(data_path, header=None, names=["word1", "word2", "interpretation"])
        self.model_gigaword = api.load("glove-wiki-gigaword-300")
    
    def __getNounCompoundEmbeddings(self, word1, word2):
        try:
            emb1 = self.model_gigaword[word1]
        except Exception:
            emb1 = np.random.rand(300)
        
        try:
            emb2 = self.model_gigaword[word2]
        except Exception:
            emb2 = np.random.rand(300)

        return (emb1 + emb2) / 2

    def prepareCompoundWordEmbeddings(self, save_as):
        print("Obtaining embeddings for noun compounds...")
        self.df["embedding"] = self.df.apply(lambda s: self.__getNounCompoundEmbeddings(s["word1"], s["word2"]), axis=1)
        if not os.path.exists(os.path.join(os.path.dirname(__file__), "data")):
            os.makedirs(os.path.join(os.path.dirname(__file__), "data"))
        
        self.df.to_pickle(os.path.join(os.path.dirname(__file__), "data", save_as))


class DatasetPreparationApproach2_3:
    def __init__(self, json_path, is_train=True):
        print("Loading context sentences for noun compounds...")
        with open(json_path, "r") as fin:
            self.data = json.loads(fin.read())
        self.is_train = is_train
        if is_train:
            self.df = pd.DataFrame({"nc": [], "context":[], "label": []})
        else:
            self.df = pd.DataFrame({"nc": [], "context":[]})
    
    def prepareDataframe(self, save_as):
        print("Preparing dataframe...")
        for i in range(len(self.data)):
            if len(self.data[i]["context"]) == 0:
                if self.is_train:
                    self.df= self.df.append({"nc": self.data[i]["nc"], "context": self.data[i]["nc"], "label": self.data[i]["label"]}, ignore_index = True)
                else:
                    self.df = self.df.append({"nc": self.data[i]["nc"], "context": self.data[i]["nc"]}, ignore_index = True)
            
            for j in range(len(self.data[i]["context"])):
                if self.is_train:
                    self.df = self.df.append({"nc": self.data[i]["nc"], "context": self.data[i]["context"][j], "label": self.data[i]["label"]}, ignore_index = True)
                else:
                    self.df = self.df.append({"nc": self.data[i]["nc"], "context": self.data[i]["context"][j]}, ignore_index = True)
    
        self.df.to_pickle(os.path.join(os.path.dirname(__file__), "data", save_as))


def main():
    args = parser.parse_args()
    if args.approach == 1:
        dp = DatasetPreparationApproach1(args.data)
        dp.prepareCompoundWordEmbeddings(args.save_data_as)
    else:
        dp = DatasetPreparationApproach2_3(args.data, args.train)
        dp.prepareDataframe(args.save_data_as)

if __name__ == "__main__":
    main()