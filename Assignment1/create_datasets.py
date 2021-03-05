"""
@author: Tezan Sahu
"""

from gensim.models import KeyedVectors
import xml.etree.ElementTree as ET
import os
import re
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

parser = ArgumentParser()
parser.add_argument("--dataDir", type=str, default="./data/Train/Source", help="directory containing raw data")
parser.add_argument("--outDir", type=str, default="./datasets", help="directory to store dataset with features")
parser.add_argument('--test', action='store_true', help="indicator for test data")
parser.add_argument('--kl', type=int, default=2, help="number of words forming left context of focus word")
parser.add_argument('--kr', type=int, default=2, help="number of words forming right context of focus word")

class PSDDatasetGenerator:
    """Class to generate datasets from raw data for Preposition Sense Disambiguation"""

    def __init__(self, data_dir):
        print("Loading word2vec model...")
        self.model = KeyedVectors.load_word2vec_format("./GoogleNews-vectors-negative300.bin", binary=True)

        print("Loading TPP mappings...")
        tpp_mappings_df = pd.read_csv("./data/Tratz Variations/mappings_for_tpp_data", sep="\t", header=None, names=["id", "prep_sense"])
        self.tpp_mappings = tpp_mappings_df.set_index('id').T.to_dict('records')[0]
        self.data_dir = data_dir

    
    def __context_feature_vector(self, left_context, right_context):
        """Obtain the context feature vectors using word embeddings, given the left & right context words

        Parameters
        ----------
        left_context: Array of words forming the context to the left of the focus word
        right_context: Array of words forming the context to the right of the focus word
        """
        vl = vr = vi = 0
        n_vl = n_vr = 0
        V = []
        for word in left_context:
            try:
                vl_i = self.model[word]
            except Exception:
                vl_i = np.random.rand(300)
            finally:
                vl += vl_i
                n_vl += 1
                V.append(vl_i)
        
        for word in right_context:
            try:
                vr_i = self.model[word]
            except Exception:
                vr_i = np.random.rand(300)
            finally:
                vr += vr_i
                n_vr += 1
                V.append(vr_i)

        try:
            vl /= n_vl
            vr /= n_vr

            V = np.array(V)
            pca = PCA(1)
            pca.fit(V)
            vi = pca.components_[0]


            return (vl, vr, vi)
        except Exception:
            return (np.random.rand(300), np.random.rand(300), np.random.rand(300))

    def createTestDataset(self, output_dir, k_l=2, k_r=2):
        """Create the test datasets containing feature vectors from samples

        Parameters
        ----------
        output_dir: Directory to save the test datasets created
        k_l: Number of context words for left context of focus word
        k_r: Number of context words for right context of focus word
        """
        for prep_file in os.listdir(self.data_dir):
            preposition = re.findall(r"([a-z]*)\.out", prep_file)[0]
            print("Preposition:\t", preposition)

            ctxts_l = []
            ctxts_r = []

            with open(os.path.join(self.data_dir, prep_file), "r") as f:
                lines = f.readlines()

                for sent in lines:
                    words = sent.lower().split()
                    focus_word_index = words.index(preposition)

                    ctxts_l.append(words[:focus_word_index][-k_l:])
                    ctxts_r.append(words[focus_word_index+1:][: k_r])

            print("Preparing context feature vectors...")
            df = pd.DataFrame({"left_context": ctxts_l, "right_context": ctxts_r})
            df["vl"], df["vr"], df["vi"] = zip(*df.apply(lambda s: self.__context_feature_vector(s["left_context"], s["right_context"]), axis=1))

            print("%d Testing Instances" % (len(df)))

            print("Saving dataset...")
            test_dir = os.path.join(output_dir, "test")
            if not os.path.exists(test_dir):
                os.makedirs(test_dir)
            df.to_pickle(os.path.join(test_dir, preposition + ".pkl"))
            print("====================================================================")


    def createDataset(self, output_dir, k_l=2, k_r=2):
        """Create the required datasets containing feature vectors from samples

        Parameters
        ----------
        output_dir: Directory to save the training & validation datasets created
        k_l: Number of context words for left context of focus word
        k_r: Number of context words for right context of focus word
        """
        print("Parsing XML files...")
        for prep_file in os.listdir(self.data_dir):
            preposition = re.findall(r"pp-([a-z]*)\.", prep_file)[0]
            print("Preposition:\t", preposition)

            instances = []
            prep_senses = []
            ctxts_l = []
            ctxts_r = []

            tree = ET.parse(os.path.join(self.data_dir, prep_file)) 
            root = tree.getroot()

            for item in root.findall('./instance'):
                try:
                    instance_id = item.attrib["id"]
                    sense = self.tpp_mappings[instance_id]
                    prep_senses.append(sense)
                    instances.append(instance_id)

                    sent = item.find("./context")
                    ctxts_l.append(list(sent.text.split())[-k_l:])
                    ctxts_r.append(list(sent[0].tail.split())[: k_r])
                except Exception:
                    pass
        
            print("Preparing context feature vectors...")
            df = pd.DataFrame({"id": instances, "left_context": ctxts_l, "right_context":ctxts_r, "preposition_sense": prep_senses})
        
            df["vl"], df["vr"], df["vi"] = zip(*df.apply(lambda s: self.__context_feature_vector(s["left_context"], s["right_context"]), axis=1))
            df = df.dropna().reset_index(drop=True)

            print("Saving dataset...")
            train_dir = os.path.join(output_dir, "train")
            valid_dir = os.path.join(output_dir, "valid")
            if not os.path.exists(train_dir):
                os.makedirs(train_dir)
            if not os.path.exists(valid_dir):
                os.makedirs(valid_dir)
            train_df, valid_df = train_test_split(df, test_size=0.2)
            print("%d Training Instances\t %d Validation Instances" % (len(train_df), len(valid_df)))
            
            train_df.to_pickle(os.path.join(train_dir, preposition + ".pkl"))
            valid_df.to_pickle(os.path.join(valid_dir, preposition + ".pkl"))
            
            print("====================================================================")

def main():
    args = parser.parse_args()
    data_generator = PSDDatasetGenerator(args.dataDir)
    if args.test:
        data_generator.createTestDataset(args.outDir, k_l=args.kl, k_r=args.kr)
    else:
        data_generator.createDataset(args.outDir, k_l=args.kl, k_r=args.kr)

if __name__ == "__main__":
    main()