# Extractive Summarization with Discourse Tree Attention
This is the official code for the paper 'Do We Really Need That Many Parameters In Transformer For Extractive Summarization? Discourse Can Help !' (CODI at EMNLP 2020)

## Data and Trained Models
The CNNDM dataset with generated attention maps (C-tree w/ Nuc) can be found [here](https://drive.google.com/drive/folders/17UNXb7ls-t18OKN54HHM7YMksqawBtGa?usp=sharing). It is based on the dataset from [DiscoBERT](https://github.com/jiacheng-xu/DiscoBERT) with segmented EDUs.

The trained model with discourse tree attention can be found [here](https://drive.google.com/drive/folders/1Hg0IkE42YPyUZpufZXm6OOMsEdhoQSy3?usp=sharing)

## Discourse Parser
We use the state-of-the-art [Discourse Parser](https://github.com/yizhongw/StageDP)

## How to train the model
Run 
```
python main.py 
```
with following arguments:

1. -bert_dir indicates where to store the pretrained BERT model
2. -d_v, -d_k, -d_inner, -d_mlp, -n_layers, -n_head, -dropout are the parameters of the Transformer-based Document Encoder model
3. -lr, -warmup_steps are the parameter for the adam optimizer
4. -inputs_dir, -val_inputs_dir are the address of the data
5. -unit, -unit_length_limit, -word_length_limit indicates whether you want to use sentence or edu as the basic unit, and the length limits of generated summaries
6. -batch_size indicates the number of instances per batch
7. -attention_type choose from 'tree', 'dense', 'fixed_rand', 'learned_rand', 'none' and 'self-attention'
8. -device indicates which gpu device you want to use

## How to evaluate the model
Run 
```
python test.py
```
with following arguments:

1. -model_path, -model_name indicates the folder and name of the saved model, the model to evalueate is 'model_path/model_name'
2. -test_inputs_dir indicates the address of test data
3. -device, indicates which gpu device you want to use
4. -d_v, -d_k, -d_inner, -d_mlp, -n_layers, -n_head, -dropout are the parameters of the Transformer-based Document Encoder model
5. -unit, -unit_length_limit, -word_length_limit indicates whether you want to use sentence or edu as the basic unit, and the length limits of generated summaries
6. -attention_type choose from 'tree', 'dense', 'fixed_rand', 'learned_rand', 'none' and 'self-attention'
