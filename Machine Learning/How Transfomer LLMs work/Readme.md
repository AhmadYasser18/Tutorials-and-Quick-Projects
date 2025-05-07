Contains notes from [How Transformer LLMs work - DeepLearning.AI](https://learn.deeplearning.ai/courses/how-transformer-llms-work)
-----------
## Topics:

1. [Introduction](#intro)
2. [Understanding Language Models: Language as a Bag-of-Words](#bagOFwords)
3. [Understanding Language Models: Word Embeddings](#embeddings)
4. [Understanding Language Models: Encoding and Decoding Context with Attention](#EncodeDecode)
5. [Understanding Language Models: Transformers](#transformers)
6. [Tokenizers](#tokenizers)
7. [Architectural Overview](#arch_over)
8. [Transformer Block](#transformer_block)

-----------
# Introduction
<a id="intro"></a>
The **transformer** architecture was first introduced in the 2017 paper - Attention Is All You Need by Ashish Vaswani and others for machine translation tasks. The same architecture turned out to be great at inputting a "*prompt*" and outputting a "*response*".  

The original transformer architecture consisted of two main parts: an *Encoder* and a *Decoder*.  
The **Encoder:**
- provides rich, context-sensitive representation of the input text
- Basis of:
    - Bert model
    - most Embedding models using RAG applications

The **Decoder:**
- performs text generation tasks such as: summarizing text, writing code, answering questions ....
- Basis for most popular LLMs 

# Understanding Language Models: Language as a Bag=of-Words
<a id="bagOFwords"></a>

**Bag of words:** an algorithm that represents words as large sparse vectors or arrays of numbers.  

![alt text](image.png)  
- Earlier techniques such as **Bag-of-Words** and  **Word2Vec** have been the foundation of modern tools. Although they lack contextualized representations, they are a good baseline to start with. These are called non-transformer models since today's models are typically powered by transformer models.

- **Encoder-Only models:** great at representing language in numerical representations.

- **Decoder-Only models:** are generative in their nature. Their main usage is to generate text. 

- **Encoder-Decoder models:** attempt to get the best of both. 

**Tokenization** is the process of converting input text into pieces (tokens). A taken may even be smaller than a word.  
Tokenization of different sentences results in sets of tokens used to build a **Vocabulary**. The Vocabulary will contain fewer words than generated tokens as it has only unique words. Number of tokens in a Vocabulary is referred to as a Vocabulary Size. A numerical representation of how often token of an input appear in the vocabulary is called a Bag-of-Words.

# Understanding Language Models: Word Embeddings
<a id="embeddings"></a>

Although Bag-of-words is a useful approach, it is flawed in how it considers language to be nothing more than a literal bag of words; meaning it does not consider the semantic nature of text.  

Following it was the release of **Word2Vec** which was one of the first successful attempts at capturing the meaning of text in embeddings. This was done by training it on large amounts of textual data and leveraging neural networks making it learn the semantic representation of words.  
Using neural networks **Word2Vec** generates word embeddings by looking at which words tend to appear next to each other. 
- In the beginning each word is assigned a vector embedding (initialized with random values). 
- In every training step, words pairs are taken from training data and the model attempts to predict whether they are likely to be neighbors in a sentence.
- During the training process, **Word2Vec** learns the relationship between words and distills that information into the embedding.  

Note that there are many types of embeddings that can be used. Models similar to **Word2Vec** which aim to convert textual input to embeddings are referred to as **Representation Models** as they attempt to represent text as values.

# Understanding Language Models: Encoding and Decoding Context with Attention
<a id="EncodeDecode"></a>

Capturing text context is important to perform some language tasks, such as translation. 
**Word2Vec** creates static embeddings meaning that the same embedding is generated for a given word regardless of context.  
**RNN**s were used due to their ability to model entire sequences. RNN in the encoding step takes the static embeddings generated using Word2Vec and processes it, taking into account the context of the embeddings, generating the context in the form of embedding.  
The decoder step aimed to generate language (response) by leveraging the previously generated context embeddings. Output tokens are generated one at a time, which is called **autoregressive**.  
This method makes it harder to deal with longer sentences as the embedding used to represent the text may fail to  capture the entire context as it gets longer and more complex. 

**Attention** was later introduced which allows a model to focus on parts of the input sequence that are relevant to each other and amplify their signal. 

# Understanding Language Models: Transformers
<a id="transformers"></a>

**Transformers** architecture is based solely on **Attention** without the **RNN**. This architecture allows the model to be trained in parallel which speeds up calculations significantly.

The transformer consists with stacked encoder and decoder blocks. By stacking the blocks, the encoder and decoder strength is amplified.   
- Encoder: 
    - input is converted to embeddings starting with *random values* instead of *Word2Vec*.
    - Self-Attention processes these embeddings and updates them.
    - The updated embeddings contain more contextualized information as a result of the intention mechanism.
    - The embeddings are then passed to a feedforward neural network which finally output **contextualized token word embedding**.
After the encoder is done processing, 
- Decoder:
    - takes any previously generated words and pass it to the masked Self-Attention.
    - intermediate embeddings are generated and passed to another Attention network together with Encoder's embeddings, processing both what was just generated and what was already in possession.
    - This output is passed to a neural network generating the next word in sequence.

![alt text](image-1.png)

The original architecture (encoder-decoder) serves well in translation tasks but can not be used easily for other tasks like tet classification 

## Representation models - BERT

A new a architecture called **Bidirectional Encoder Representations from Transformers (BERT)** was introduced which could be used for a wide variety of tasks. BERT is an encoder only architecture that focuses on representing language and generating contextual word embeddings.  
The input includes an additional token, the CLS. The CLS, or classification token, is used as a representation for the entire input. It is often used a the input embedding for fine tuning the model on specific tasks.  

![alt text](image-2.png)

To train a BERT-like model *mask language modeling technique* could be used.  
This is done by randomly masking a number of words from the input sequence then having the model predict the masked words. By doing so, the model learns to represent language as it attempts to deconstruct the masked words. This step is called pre-training.  
The pre-trained model is then fine-tuned on a number of downstream tasks [classification, named entity recognition, paraphrase identification...].

## Generative Models

Unlike BERT models, generative models tend to only stack decoders. Generative Pre-trained Transformer (**GPT**) is one of its famous implementations. 

# Tokenizers
<a id="tokenizers"></a>

Given an input sentence, it's tokenized into smaller pieces. Tokens can be entire words or pieces of a word. When these pieces are combinded they form the original word. This porcess is necessary since tokenizers have a limited number of tokens. Thus, whenever an unkown word is encountered, it can be still represented using those sub tokens.  
Each token has a fixed ID to easily encode and decode the tokens.  
These are fed to the language model which internally creates the token embeddings. The output of a generative model would be another token ID which is decoded to represent an actual token.

# Architectural Overview
<a id="arch_over"></a>

The transformer is made up of three major components:  
    - **Tokenizer**: breaks down text into multiple chunks. Token embeddings are associated for each token.
    - **Stack of Transformer Blocks**: contains majority of computing, neural networks..   
    - **Language Modeling Head**: based on all of the processing in the stack of transformer blocks results  in a token probability scoring indicating percentage of each token.  
    One method of choosing the output token depends on the highest probability [Greedy decoding]. Other strategies use different methods in choosing the output token.

# Transformer Block
<a id="transformer_block"></a>

