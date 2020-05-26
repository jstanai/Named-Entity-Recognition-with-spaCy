# Named Entity Recognition with spaCy

## Introduction

This is an implementation of multi-lingual Named Entity Recognition (NER) using spaCy (<https://spacy.io/>). This codebase loads German and English models to provide NER predictions on various user input. It will run with Google Colab, and [MLflow](https://mlflow.org/) integration is currently in progress. A proof of concept for fine-tuning these models to custom tags is also shown. 

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

Additionally, spaCy interfaces with [Hugging Face's](https://explosion.ai/blog/spacy-transformers) implementation of several transformers, including BERT, GPT-2, and XLNet. Shown here, however, is only the statistical model. This statistical model does a reasonable job and is very fast to train. 

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

Additionally, an example of training a new label "COMPANY" is shown by creating a custom pattern. Two methods, manual and parsing, are used to add this pattern. It is also shown how the `GoldParse` functionality is used to parse documents and add revision data. 

### Usage

By training on this new named entity, predictions are possible with the saved NER model, which can be loaded using `trained_model = spacy.load("./model_name")`. Serialization and deserialization of these models can be challenging and is also shown in detail. 

Predictions can be made easily by creating a document:
```
doc = trained_model("Why do people like Tesla?")
```

As an example of using this to create custom entities, for example a car model. Here text is created to train a custom tag "COMPANY". Here we train specifically on the **tesla** company name using these training texts. Thus our entity value is "tesla" and our entity type is "COMPANY".

```
texts = [
    "Tesla does not currently have full self-driving, but is level 2 ready.",
    "Investors are mixed on Tesla stock.",
    "If you want to buy a Tesla, it will likely cost you more than $50,000.",
    "People like Tesla because it is a environmentally friendly company.",
    "Tesla really moved the needle on EV adoption."
    "Tesla?",
    "The new Gigafactory in Shanghai was built by Tesla in about a year.",
    "Elon Musk was very close to failing with Tesla.",
    "Tesla is also heavily focused on battery and solar products.",
    "The supercharger network created by Tesla is helping support EV adoption."
    "A new battery is in development from Tesla for over 1 million miles."   
]
```

Then we parse out instances of this tag, train the model, and run predictions. 

**Make a prediction**:
```
sentence = "Can you tell that I love Tesla?" 
doc = trained_model(sentence)                       # make prediction
```

**Output** (positions, entity, label):
```
for ent in doc.ents:
    print(ent.start, ent.end, ent.text, ent.label_)

# Can you tell that I love Tesla? 6 7 Tesla COMPANY
```


