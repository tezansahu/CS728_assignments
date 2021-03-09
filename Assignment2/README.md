# Noun Compound Classification

In this assignment, you'd be tackling the problem of noun compound classification. Noun compounds, inspite of being a very common part of the language, lack uniformity, and simplicity in their formulation and usage. For example, consider the following two noun compounds

- Olive Oil
- Baby Oil

Here olive oil signifies oil made from olives. However, baby oil obviously does not mean oil made from babies but oil for use on babies. Your task will be to distinguish among such interpretations of noun compounds.

You are given a dataset of about 16000 noun compounds along with their intended interpretations. There are a total of 37 different interpretations used in the dataset. Your task is to treat this as a multiclass classification problem where given a noun compound you have to decide upon its intended interpretations.

You'd be trying the following approaches for this task: 

1. Use a simple feed-forward network (tailed by a softmax layer) on the word embeddings for the two words in the noun compound. You can either take mean or concatenation of the two-word embeddings as input to the feed-forward network.

2. For each noun compound, gather a set of k (10-20) sentences where that noun compound has been used. Those sentences can be fed to an LSTM network and the contextual embeddings of the noun compound thus obtained can be then used as input in the feed-forward network,

3. Augment approach 2 by incorporating self-attention in the LSTM you implement


Gathering sentences for task 2 - You can either scrape for these such sentences from the web or get a Wikipedia dump followed by its indexing (lucene or elasticsearch) and query. We'll provide the sentences down the line in case many groups are facing difficulty doing so.


The dataset has been provided to you as a CSV file. You can use this to get train and validation data for yourself. A test data file will be provided later on.