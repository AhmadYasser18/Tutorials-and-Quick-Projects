Contains notes from [How Transformer LLMs work - DeepLearning.AI](https://learn.deeplearning.ai/courses/how-transformer-llms-work)
-----------
## Topics:

1. [Introduction](#intro)
2. [Understanding Language Models: Language as a Bag=of-Words](#bagOFwords)
3. 

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
