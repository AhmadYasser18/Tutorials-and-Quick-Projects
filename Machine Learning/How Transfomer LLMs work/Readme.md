Contains notes from [How Transformer LLMs work - DeepLearning.AI](https://learn.deeplearning.ai/courses/how-transformer-llms-work)
-----------
## Topics:

1. [Introduction](#intro)
2. [Understanding Language Models: Language as a Bag-of-Words](#bagOFwords)
3. [Understanding Language Models: Word Embeddings](#embeddings)
4. [Understanding Language Models: Encoding and Decoding Context with Attention](#EncodeDecode)

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