# Assignment 1: Preposition Sense Disambiguation

## Problem Statement
In this first assignment we'll tackle the problem of preposition sense disambiguation. You have to follow the approach as described in [this paper](https://www.aclweb.org/anthology/D18-1180.pdf)

While you are encouraged to read the whole paper, section 4 is what you have to follow most closely. Note that you don't have to tackle the preposition sense representation task which is discussed in section 5 of the paper. There are four approaches discussed in the paper - 

- Supervised
    - SVM
    - MLP
    - kNN
- Unsupervised
    - Clustering


You have to implement a minimum of two approaches to be eligible for full credit. Credit will further be allotted based on the number of approaches you implement, and the accuracy you are able to achieve on the test data. Data for this task has been uploaded as a zip file below. The test data is already available but we are not providing the key to the test data. You'll have to submit the following things - 

- Code written by you
- A report describing your approach and other implementation details
- The outputs to the test data (format to be told shortly)


### Data format description: 

Each preposition has its own XML file (e.g., underneath.xml), included in the 'Source' directory under the train/test folders. Each file contains a number of instances (only one is shown in the example below). The sentences in each corpus follow the lexical sample format, as given in the following example:

```xml
<lexelt item="underneath" pos="prep">

  <instance id="underneath.p.fn.635810" docsrc="FN">

    <answer instance="underneath.p.fn.635810" senseid="1(1)"/>

    <context>

      He always used to tuck it <head>underneath</head> the water butt . 

    </context>

  </instance>

</lexelt>
```

The first line identifies the lexical item and its part of speech (always "prep" in these corpora). Each instance is given an identifying number and a document source. The next line gives the answer for the instance, identifying the instance number and the sense identifier. The next line gives the sentence (the context), with the target preposition surrounded by a "head" tag. Each sentence has been tokenized using TreeTagger, that is, separated into space-separated strings, so that, for example, an apostrophe and the letter s forms a possessive token ('s) and the terminal period is separated from the preceding word.


> _**NOTE:** We'd be running plagiarism checkers on your submitted codes, so avoid any form of inter-group interaction at all costs. Also, the problem description might have some tweaks over time._

## [Detailed Report on Implementation](https://github.com/tezansahu/CS728_assignments/blob/main/Assignment1/170100035_A1_Report.pdf)

## Code & Usage

### Creating the Datasets from Raw Data

Use the `create_datasets.py` script to create datasets containing the context feature vectors from the raw XML files as follows:

```bash
# Create the training & validation datasets
$ python create_datasets.py --dataDir ./data/Train/Source --outDir ./datasets --kl 2 --kr 2

# Create the test datasets
$ python create_datasets.py --dataDir ./test_out --outDir ./datasets --test --kl 2 --kr 2
```

> **Note:** Although the number of left & right context words can be tuned using the `--kl` & `--kr` arguments, for this assignment we use `kl = kr = 2` to create the datasets (as suggested in the paper)

### Experimenting with Models

The following approaches have been implemented for the PSD task:

- Weighted k-NN *(tunable hyperparameters: k, beta, gamma)*
- MLP *(tunable hyperparameters: # neurons in hidden layer)*
- SVM *(tunable hyperparameters: C)*

Use the `experiment.py` script to tune the hyperparameters & conduct an ablation analysis for these approaches. Some ways to use the script are shown below:

```bash
# For k-NN
$ python experiment.py --trainDir ./datasets/train --validDir ./datasets/valid --outDir experiments--modelType knn --beta 0.1 --gamma 0.1 --k 5

# For MLP
$ python experiment.py --trainDir ./datasets/train --validDir ./datasets/valid --outDir experiments--modelType mlp --numNeurons 50

# For SVM
$ python experiment.py --trainDir ./datasets/train --validDir ./datasets/valid --outDir experiments--modelType svm --C 0.5

# Testing any of the models on the test dataset
$ python experiment.py --test --testDir ./datasets/test --testOutDir ./test_out --modelsDir ./experiments/models/lri_svm_C=0.5 --modelType svm --C 0.5
```

For ablation analysis, we can use the `--features` argument, which can take values `lri` (default), `lr`, `li` & `ri`, indicating the context feature vectors to be used in the model training.

> **Note:** The classification reports of all the models trained during our experiments & analysis can be found in `./experiments/reports/`