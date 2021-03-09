from argparse import ArgumentParser
import pandas as pd
import numpy as np
import gensim.downloader as api
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import os
import pickle
import matplotlib.pyplot as plt

parser = ArgumentParser()

parser.add_argument("--data", type=str, default="./train.csv", help="path to training data")
parser.add_argument("--approach", type=int, default=1, choices=[1, 2, 3], help="approach to be followed for the task (based on the problem statement)")
parser.add_argument("--sentences_data", default=None, help="path to sentences to obtain contextual embeddings (appoaches 2 & 3)")

# Creation of embeddings dataset
parser.add_argument("--create_emb_data", action='store_true', help="flag to prepare appropriate embeddings dataset based on approach")
parser.add_argument("--save_data_as", type=str, default="noun_compound_embeddings.pkl", help="save the created embeddings as")

# Training
parser.add_argument("--train", action='store_true', help="flag to train a model")
parser.add_argument("--save_model_as", type=str, default="model_approach1.h5", help="save the trained model as")


class DatasetPreparation:
    def __init__(self, approach, data_path, sent_path = None):
        print("Loading glove embeddings...")
        self.approach = approach
        self.df = pd.read_csv(data_path, header=None, names=["word1", "word2", "interpretation"])
        self.model_gigaword = api.load("glove-wiki-gigaword-300")
        if sent_path:
            # Read & store the sentences
            pass
    
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


class Approach1:
    def __init__(self, data, test_split=0.1):
        self.data = pd.read_pickle(data)
        self.num_classes = len(self.data["interpretation"].unique())
        self.interpretations = {k: v for v, k in enumerate(self.data["interpretation"].unique())}
        # self.data["class"] = self.data["interpretation"].apply(lambda x: self.interpretations[x])

        self.train_data, self.valid_data = train_test_split(self.data, test_size=test_split)

        self.model = self.__initializeModel()
        self.history = None

    def __initializeModel(self):
        model = keras.Sequential(
            [
                layers.Dense(1024, activation="relu", input_shape=(300,)),
                layers.Dense(512, activation="relu"),
                layers.Dense(self.num_classes, activation="softmax"),
            ]
        )
        model.compile(
            optimizer=keras.optimizers.Adagrad(lr=0.05),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        print(model.summary())
        return model
    
    def train(self, batch_size=32, epochs=20, save_as="model_approach1.h5"):
        X_train = np.stack(self.train_data["embedding"].values, axis=0)
        y_train = pd.get_dummies(self.train_data["interpretation"]).values

        X_valid = np.stack(self.valid_data["embedding"].values, axis=0)
        y_valid = pd.get_dummies(self.valid_data["interpretation"]).values 

        print("Fitting model on training data...")
        history = self.model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_valid, y_valid),
        )
        self.history = history

        if not os.path.exists(os.path.join(os.path.dirname(__file__), "models")):
            os.makedirs(os.path.join(os.path.dirname(__file__), "models"))
        self.model.save(os.path.join(os.path.dirname(__file__), "models", save_as))

    def plotTraining(self):
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs = range(1, len(acc)+1)

        plt.figure(figsize=(16, 8))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'b*--', label="Training Accuracy")
        plt.plot(epochs, val_acc, 'rD:', label="Validation Accuracy")
        plt.legend()
        plt.title('Training and Validation Accuracy')
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'b*--', label="Training Loss")
        plt.plot(epochs, val_loss, 'rD:', label="Validation Loss")
        plt.legend()
        plt.title('Training and Validation Loss')
        
        if not os.path.exists(os.path.join(os.path.dirname(__file__), "plots")):
            os.makedirs(os.path.join(os.path.dirname(__file__), "plots"))
        
        plt.savefig(os.path.join(os.path.dirname(__file__), "plots", "plots_approach1.png"), bbox_inches="tight")
        plt.show()

def main():
    args = parser.parse_args()
    if args.create_emb_data:
        dp = DatasetPreparation(args.approach, args.data, args.sentences_data)
        dp.prepareCompoundWordEmbeddings(args.save_data_as)
    elif args.train:
        if args.approach == 1:
            classifier = Approach1(args.data)
        else:
            pass
        classifier.train(save_as=args.save_model_as)
        classifier.plotTraining()

if __name__ == "__main__":
    main()