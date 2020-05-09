# Named Entity Recognition with Spacy

## Introduction

This is an implementation of multi-lingual Named Entity Recognition (NER) using spaCy (<https://spacy.io/>). This codebase loads Germand and English models to provide NER predictions on various user input. It will run with Google Colab, and [MLflow](https://mlflow.org/) integration is currently in progress. A proof of concept for fine-tuning these models to custom tags is also shown. 

## Models

spaCy allows the user to interface with several core language models. Currently, there are models available in the following [languages](https://spacy.io/usage/models):

- German	
- Greek	
- English	
- Spanish	
- French	
- Italian	
- Lithuanian	
- Norwegian 
- Dutch	
- Portuguese	

Additionally, spaCy interfaces with [Hugging Face's](https://explosion.ai/blog/spacy-transformers) implementation of several transformers, including BERT, GPT-2, and XLNet. 

### Custom Tags

One of the major limitations with NER is the limited range of tags these models are trained on. Using embeddings, such as BERT, allow us to start with a good representation of language in general and fine-tune to the NLP problem at hand. However, fine-tuning must be done carefully, as models can be brittle and exhibit issues such as [catastrophic forgetting](https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting). We therefore not only need a way to generate new tags and train the model, but also ensure that we mix in a sufficient amount of data from the original distribution of tags such that we don't overfit to the new tags. 

Here a function `train_model` is developed which does exactly this, by allowing the user to input new training data, as well as revision data. 

```
model = train_model(
    nlp,                             # spaCy base model
    training_data,                   # training data
    revision_data = revision_data,   # revision data 
    labels = new_labels              # new labels
)
```

Additionally, an example of training a new label "Model" is shown by creating a custom pattern. Two methods, manual and parsing, are used to add this pattern. It is also shown how the `GoldParse` functionality is used to parse documents and add revision data. 

### Usage

By training on this new named entity, predictions are possible with the saved NER model, which can be loaded using `trained_model = spacy.load("./model_name")`. Serialization and deserialization of these models can be challenging and is also shown in detail. 

Predictions can be made easily by creating a document:
```
doc = trained_model("How do I lease a fortwo coupe?")
```

As an example of using this to create custom entities for a company, text is extracted from [Smart](https://www.smartusa.com/) and used to train a tag "MODEL", on different car models. Here we train specifically on the **fortwo** models using 

```
texts = [
    "Lease a 2019 smart EQ fortwo coupe.",
    "You can lease a 2019 smart EQ fortwo cabrio for as little as $199/month.",
    "For navigating your city, or escaping it altogether, the 2019 smart EQ fortwo features a high-tech interior.",
    "I Love the smart EQ fortwo car, it's amazing.",
    "Reviews of the smart EQ fortwo have been phenomenal! Smart EQ fortwo cars are a hit."
    "Fortwo?",
    "Fortwo is for everyone.",
    "I am thinking of buying a fortwo coupe."
    "How much does a fortwo coupe cost?",
    "At signing, how much will be due for a new fortwo?",
    "Is fortwo the best car from Smart?",
    "Fortwo is a type of Smart car."
    "The fortwo model is sick!"   
]
```

Then we parse out instances of this tag, train the model, and run predictions. 

**Make a prediction**:
```
sentence = "How do I lease a fortwo coupe?" 
doc = trained_model(sentence)                       # make prediction
```

**Output** (positions, entity, label):
```
for ent in doc.ents:
    print(doc, ent.start, ent.end, ent.text, ent.label_)

# 5 6 fortwo MODEL
```


