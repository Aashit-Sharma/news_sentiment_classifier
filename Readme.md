# Fine-grained Sentiment(FGS) Classification of News Articles using BERT

Started: 12th July 2020

To analyze news articles and classify their sentiment using SOTA methods to uncover latent biases of certain news companies against or for specific entities, individuals and topics.

## Architecture
(Subject to change)

[text]-->[pre process]-->[BERT Embedding]-->[Dropout]-->[Softmax Classifier]-->[Sentiment Label]

## Plan
- [x] Build a API independent web scraper for building the dataset
- [x] Text Pre-processing
- [ ] Text Summarization
- [ ] Topic Modelling
- [ ] Implementing the FGS-Classifier architecture using PyTorch
- [ ] Fine Tuning the model on the SST-5 dataset (Stanford Sentiment Treebank)
- [ ] Analysis of the results and conclusions
