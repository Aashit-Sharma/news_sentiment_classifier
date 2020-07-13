# Fine-grained Sentiment(FGS) Classification of News Articles using BERT

Started: 12th July 2020

To analyze news articles and classify their sentiment using SOTA methods to uncover latent biases/agendas of different news companies against or for specific entities, individuals and topics.

We will be using the Stanford Sentiment Treebank (SST-5) dataset for building a model that classifies document sentiment in the range 0 to 4 (4 being the most positive)

## Architecture
(Subject to change)

[text]-->[pre process]-->[BERT Embedding]-->[Dropout]-->[Softmax Classifier]-->[Sentiment Label]

## Plan

### Phase 1

- [x] API independent web scraper for building the dataset
- [x] Text Pre-processing
- [ ] Text Summarization
- [ ] Topic Modelling
- [ ] Implementing the FGS-Classifier architecture using PyTorch
- [ ] Fine Tuning the model on the SST-5 dataset (Stanford Sentiment Treebank)
- [ ] Analysis of the results and conclusions

### Phase 2
- [ ] Deeper analysis of propagandist patterns ( https://propaganda.qcri.org/annotations/definitions.html)