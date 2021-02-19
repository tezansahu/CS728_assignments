"""
@author: Tezan Sahu
"""

from gensim.models import KeyedVectors
import xml.etree.ElementTree as ET
import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

parser = ArgumentParser()
parser.add_argument("--dataDir", type=str, help="directory containing raw data")
parser.add_argument("--outDir", type=str, help="directory to store dataset with features")
parser.add_argument('--test', action='store_true', help="indicator for test data")


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
                vl += vl_i
                n_vl += 1
                V.append(vl_i)
            except Exception:
                pass
        
        for word in right_context:
            try:
                vr_i = self.model[word]
                vr += vr_i
                n_vr += 1
                V.append(vr_i)
            except Exception:
                pass

        try:
            vl /= n_vl
            vr /= n_vr

            V = np.array(V)
            pca = PCA(1)
            pca.fit(V)
            vi = pca.components_[0]


            return (vl, vr, vi)
        except Exception:
            return (None, None, None)

    
    def createDataset(self, output_dir, k_l=2, k_r=2, is_test=False, save_train_as="train.pkl", save_valid_as="valid.pkl", save_test_as="test.pkl"):
        """Create the required datasets containing feature vectors from samples

        Parameters
        ----------
        output_dir: Directory to save the training & validation datasets created
        k_l: Number of context words for left context of focus word
        k_r: Number of context words for right context of focus word
        is_test: Boolean to indicate if the data is for creating a test dataset
        save_train_as: Filename to save training dataset
        save_valid_as: Filename to save validation dataset
        save_test_as: Filename to save test dataset
        """

        instances = []
        prep_senses = []
        ctxts_l = []
        ctxts_r = []

        print("Parsing XML files...")
        for prep_file in os.listdir(self.data_dir):
            tree = ET.parse(os.path.join(self.data_dir, prep_file)) 
            root = tree.getroot()

            for item in root.findall('./instance'):
                try:
                    instance_id = item.attrib["id"]
                    if not is_test:
                        sense = self.tpp_mappings[instance_id]
                        prep_senses.append(sense)
                    instances.append(instance_id)

                    sent = item.find("./context")
                    ctxts_l.append(list(sent.text.split())[-k_l:])
                    ctxts_r.append(list(sent[0].tail.split())[: k_r])
                except Exception:
                    pass
        
        print("Preparing context feature vectors...")
        if not is_test:
            df = pd.DataFrame({"id": instances, "left_context": ctxts_l, "right_context":ctxts_r, "preposition_sense": prep_senses})
        else:
            df = pd.DataFrame({"id": instances, "left_context": ctxts_l, "right_context":ctxts_r})
        
        df["vl"], df["vr"], df["vi"] = zip(*df.apply(lambda s: self.__context_feature_vector(s["left_context"], s["right_context"]), axis=1))
        df = df.dropna().reset_index(drop=True)

        print("Saving dataset...")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not is_test:
            train_df, valid_df = train_test_split(df, test_size=0.2)
            train_df.to_pickle(os.path.join(output_dir, save_train_as))
            valid_df.to_pickle(os.path.join(output_dir, save_valid_as))
        else:
            df.to_pickle(os.path.join(output_dir, save_test_as))

def main():
    args = parser.parse_args()
    data_generator = PSDDatasetGenerator(args.dataDir)
    data_generator.createDataset(args.outDir, is_test=args.test)

if __name__ == "__main__":
    main()