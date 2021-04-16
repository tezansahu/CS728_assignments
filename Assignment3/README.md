# Assignment 3: Query Answer Type Classification using BERT & Entity Linking

## Problem Statement

This assignment is based on the answer type classification of queries. You are given in the [dataset](https://drive.google.com/file/d/1oRMV-wX6iGqj9gnbNYcEVNUZ6-hE_iBD/view?usp=sharing) queries from the **WebQSP dataset** and the corresponding target answer types. You have to train a multiclass BERT based classifier for the task. You can input the question to a pretrained BERT base model and use the CLS embeddings for multiclass classification. This constitutes the basic approach to this task. The questions often have mentions of named entities which often contribute little in terms of signal to the classification model. This issue can be tackled in the following manner -

- Use an entity linker to link the named entity to a Wikipedia/DBpedia entity
- Use DBpedia dataset to get a type (finest granularity) for the named entity
- Replace the named entity with the corresponding type name and retrain the classification model

Implement this as your second approach. Compare the results from the two approaches and support your observations with reason in your report.


[Tagme](https://sobigdata.d4science.org/web/tagme/tagme-help) can be used for entity linking. You can also search for its python API

You can use [this dataset](https://downloads.dbpedia.org/repo/dbpedia/mappings/instance-types/2020.06.01/instance-types_lang%3den_specific.ttl.bz2) for getting types of DBPedia entities


Huggingface is the standard goto library for BERT implementations. For computation resources, you are expected to use Google Colab. Logging in with your IITB Ldap provides you with more GPU memory than otherwise on Google Colab


## Solution Code & Model

The trained models for the 2 approaches could be download using the following links:
- [Model (Approach 1)](https://drive.google.com/file/d/1_vaSyqR9t3ZQbQkSwVgf9OBzL9mhvKxd/view?usp=sharing)
- [Model (Approach 2)](https://drive.google.com/file/d/1-6U2aHthJt4TYwQWj1c8ZQFUs4Mjcx5J/view?usp=sharing)

The IPython notebook for training the models using both the approaches can be found [here](./Assignment3.ipynb)

### Observations

- **No. of Queries:** 1763
- **No. of Unique Answer Types:** 137
- **Validation Split:** 10%

| Metric | Approach 1 | Approach 2 |
| :---: | :---: | :---: |
| Micro F1 | 0.2846 | 0.2984 |
| Accuracy * | 91.49 % | 96.54 % |
| Type Fineness Score ** | 0.4317 | 0.5543 |


> **Note:**
>
>\* _Since answer type of each query is represented by a list of possible values, accuracy is calculated such that model prediction is considered 'correct' if it belongs to the list of answer types for that query_
>
>\** _We observe that the answer type lists are such that the most specific type appears first, followed by more generic types. Since we want our model to be able predict more specific types, we define this custom metric, which gives a score 1 for a query if the predicted answer type is the most specific, & penalizes the score linearly as the specificity decreases._

**We observe that implementing Entity Linking using TagMe & DBPedia has improved the overall performance of our model**

***