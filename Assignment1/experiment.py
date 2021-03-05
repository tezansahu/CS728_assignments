"""
@author: Tezan Sahu
"""

import os
import re
import numpy as np
import pandas as pd
import json
import pickle
from argparse import ArgumentParser
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import sys
import warnings

parser = ArgumentParser()
# For training & validation
parser.add_argument("--trainDir", type=str, default="./datasets/train", help="path to training data")
parser.add_argument("--validDir", type=str, default="./datasets/valid",help="path to validation data")
parser.add_argument("--outDir", type=str, default="experiments", help="directory to store trained model & classification report")
parser.add_argument("--features", type=str, default="lri", choices=["lri", "lr", "li", "ri"], help="combination of context feature vectors for the experiment")
parser.add_argument("--modelType", type=str, default="svm", choices=["knn", "svm", "mlp"], help="type of model to experiment")

# For testing
parser.add_argument('--test', action='store_true', help="indicator for test data")
parser.add_argument("--testDir", type=str, default="./datasets/test", help="path to test data")
parser.add_argument("--testOutDir", type=str, default="./test_out", help="path to output files for storing test predictions")
parser.add_argument("--modelsDir", type=str, help="path to trained models")

# Tunable params for weighted k-NN
parser.add_argument("--k", type=int, default=5, help="[k-NN] no. of neighbors")
parser.add_argument("--beta", type=float, default=1, help="[k-NN] relative weight of right context vector")
parser.add_argument("--gamma", type=float, default=1, help="[k-NN] relative weight of interplay context vector")

# Tunable params for MLP
parser.add_argument("--numNeurons", type=int, default=100, help="[MLP] no. of neurons in hidden layer")

# Tunable params for SVM
parser.add_argument("--C", type=float, default=1, help="[SVM] regularization parameter")


class PSDExperiment:
    def __init__(self, model_type, params, features, train_path, valid_path, output_dir, test=False):
        self.features = features
        self.model_type = model_type
        self.params = params
        if not test:
            print("Initializing experiment...")
            self.train_dir = train_path
            self.valid_dir = valid_path

            self.model_identifier = f"{features}_{model_type}"
            for key, value in params.items():
                self.model_identifier += f"_{key}={value}"

            self.output_dir = output_dir
            self.models_dir = os.path.join(self.output_dir, "models", self.model_identifier)
            self.reports_dir = os.path.join(self.output_dir, "reports")
            for folder in [self.models_dir, self.reports_dir ]:
                if not os.path.exists(folder):
                    os.makedirs(folder)

            self.prepositions_with_one_sense = {}
        else:
            with open("sense_mapping.json", "r") as f:
                self.sense_mappings = json.load(f)

    def __initializeModel(self):
        if self.model_type == "knn":
            return KNeighborsClassifier(n_neighbors=self.params["k"])
        
        elif self.model_type == "mlp":
            return MLPClassifier(hidden_layer_sizes=(self.params["num_neurons"],), random_state=1, max_iter=200)

        elif self.model_type == "svm":
            return SVC(kernel="linear", C=self.params["C"])
        
        else:
            print("Invalid model! Exitting...")
            exit(1)

    def __getFeatures(self, df):
        X_vl = np.stack(df["vl"].values, axis=0)
        X_vr = np.stack(df["vr"].values, axis=0)
        X_vi = np.stack(df["vi"].values, axis=0)

        if self.features == "lri":
            if self.model_type == "knn":
                return X_vl + self.params["beta"]*X_vr + self.params["gamma"]*X_vi
            else:
                return np.concatenate((X_vl, X_vr, X_vi), axis=1)
        elif self.features == "lr":
            if self.model_type == "knn":
                return X_vl + self.params["beta"]*X_vr
            else:
                return np.concatenate((X_vl, X_vr), axis=1)
        elif self.features == "li":
            if self.model_type == "knn":
                return X_vl + self.params["gamma"]*X_vi
            else:
                return np.concatenate((X_vl, X_vi), axis=1)
        elif self.features == "ri":
            if self.model_type == "knn":
                return self.params["beta"]*X_vr + self.params["gamma"]*X_vi
            else:
                return np.concatenate((X_vr, X_vi), axis=1)


    def trainModels(self):
        print(f"Training {self.model_type} models...")
        for prep_train_data in os.listdir(self.train_dir):
            preposition = re.findall(r"([a-z]*)\.pkl", prep_train_data)[0]
            
            train_df = pd.read_pickle(os.path.join(self.train_dir, prep_train_data))
            X = self.__getFeatures(train_df)
            y = train_df["preposition_sense"]
            num_senses = len(y.unique())

            print("Preposition: %s \tNumber of senses: %d" % (preposition, num_senses))

            if num_senses > 1:
                # Train a model to disambiguate each preposition
                model = self.__initializeModel()
                model.fit(X, y)
                pickle.dump(model, open(os.path.join(self.models_dir, preposition + ".sav"), 'wb'))
            else:
                self.prepositions_with_one_sense[preposition] = y[0]
        print("Training completed!")
        print("==================================================================")

    def validateModels(self):
        print("Validating models...")
        y_actual_all = pd.Series([], dtype=str)
        y_pred_all = np.array([])
        for prep_valid_data in os.listdir(self.valid_dir):
            preposition = re.findall(r"([a-z]*)\.pkl", prep_valid_data)[0]

            valid_df = pd.read_pickle(os.path.join(self.valid_dir, prep_valid_data))
            y_actual = valid_df["preposition_sense"]
            y_actual_all = y_actual_all.append(y_actual)
            
            if preposition in self.prepositions_with_one_sense.keys():
                y_pred = pd.Series([self.prepositions_with_one_sense[preposition]]*len(valid_df))
            else:
                X = self.__getFeatures(valid_df)
                model = pickle.load(open(os.path.join(self.models_dir, preposition + ".sav"), 'rb'))
                y_pred = model.predict(X)
            
            y_pred_all = np.append(y_pred_all, y_pred)
            print("Preposition: %s \tValidation Accuracy: %.4f" % (preposition, accuracy_score(y_actual, y_pred)))
        
        report = classification_report(y_actual_all, y_pred_all, output_dict=True)
        valid_report = pd.DataFrame(report).transpose()
        valid_report.to_csv(os.path.join(self.reports_dir, self.model_identifier + ".csv"))

        print("==================================================================")
        print("Overall Validation Accuracy: %.4f" % accuracy_score(y_actual_all, y_pred_all))
        print("==================================================================")

    def testModels(self, test_dir, test_out_dir, models_dir):
        print("Testing models...")
        
        for prep_test_data in os.listdir(test_dir):
            preposition = re.findall(r"([a-z]*)\.pkl", prep_test_data)[0]
            print("Predicting for: %s ..." % preposition)

            test_df = pd.read_pickle(os.path.join(test_dir, prep_test_data))
            if not os.path.exists(os.path.join(models_dir, preposition + ".sav")):
                target_vals = set(self.sense_mappings[preposition].values())
                assert len(target_vals) == 1
                y_pred = pd.Series([list(target_vals)[0]]*len(test_df))
            else:
                X = self.__getFeatures(test_df)
                model = pickle.load(open(os.path.join(models_dir, preposition + ".sav"), 'rb'))
                y_pred = model.predict(X)

            with open(os.path.join(test_out_dir, f"{preposition}.out"), "r") as out_file:
                lines = out_file.readlines()
                assert len(lines) == len(y_pred)
                new_lines = [x.replace("\n", "") + " | " + y_pred[i] + "\n" for i,x in enumerate(lines)]

            with open(os.path.join(test_out_dir, f"{preposition}.out"), "w") as out_file:
                out_file.writelines(new_lines)
        print("==================================================================")

def main():
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    args = parser.parse_args()
    if args.modelType == "knn":
        params = {
            "k": args.k,
            "beta": args.beta,
            "gamma": args.gamma
        }
    elif args.modelType == "mlp":
        params = {
            "num_neurons": args.numNeurons
        }
    elif args.modelType == "svm":
        params = {
            "C": args.C
        }
    else:
        print("Invalid model! Exitting...")
        exit(1)
    
    expt = PSDExperiment(args.modelType, params, args.features, args.trainDir, args.validDir, args.outDir, args.test)
    if args.test:
        expt.testModels(args.testDir, args.testOutDir, args.modelsDir)
    else:
        expt.trainModels()
        expt.validateModels()
    print("Completed")

if __name__ == "__main__":
    main()