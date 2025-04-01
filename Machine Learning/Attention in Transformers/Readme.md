Contains notes from [Attention in Transformers - OpenAI](https://learn.deeplearning.ai/courses/attention-in-transformers-concepts-and-code-in-pytorch/)
-----------
# Transfromers 
A transformer requires 3 main parts:
1. **Word embedding:** 
    - converts words, bits of words and symbols (called *Tokens*) into numbers.
    - needed since transformers are neural networks (only take numbers as input). 
2. **Positional encoding:**
    - helps keep track of word order.
    - multiple methods exist to implement it.
3. **Attention:**
    - mechanism used by the transformer to establish relationships among words.
    - there are a few different types of *Attention*:
        - *Self-Attenion:* works by seeing how similar each word is to all of the words in the sentence, including itself. Once the similarities are calculated, they are used to determine how the transformer encodes each word.


# The Matrix Math for Self-Attention
$$
\text{Attention}(Q, K, V) = \text{SoftMax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V
$$
Q: Query  
K: Key  
V: Value  
$d_k$ : Dimension of the Key matrix 

These variablea are part of databse terminology.   
The Query [ Q ] is used to search the Keys [ K ] and returns the Value [ V ].  

In order to create the queries for each word, the encodings are stacked in a matrix which is then multiplied by a **nxn** (n: number of encoded values/Word embeddings, usally 512) of query wieghts.

The same is repeated but with a Keys weights and Value weights to create the Keys and Values.

# Self-Attention vs Masked Self-Attention

### Transforming words into numbers with **Word Embedding**  
An easy way is to just assign each word to a random number. In theory this works well, however, it overlooks words having similar meanings or used similarly as they will be given different numbers.  
As result, the Neural Network will have to be more complex and require more training. 

A better method would assign similar words similar numbers. In addition, since a words can be used in different contexts having different meanings, they may be assigned more than one number so that the Neural Network adjusts more easily.  

Stand alone Neural Networks previously built that create Word Embeddings.  
A simple Neural Network is trained to predict the word that comes after the input which helps in assigning similar number to similar words as it takes the context into consideration. Each input is connected to at least one activation function (number of functions represent number of values assigned to each word). Instead of just using one word, the preceding 3-4 words could be used. Increasing context helps in improving the Word Embedding value.  
However, this ignores word order.

The **Positional Encoding** layer plays the part of taking word order into consideration when creating Embeddings.

The **Attention** layer helps establish relationships among words.  
Using *Self-Attention* factors all words including those that come after the word of interest resulting in the formation of a new embedding refered to as **Context Aware Embedding** or **Contextualized Embedding**. Compared to **Word Embedding** which only clusters individual words, **Context Aware Embedding** can help cluster sentences and can even cluster similar documents.

![alt text](image.png)

Transformers that only use Self-Attention are called **Encoder-Only Transformers**. **Context Aware Embeddings** can also be used as input to a normal neural network that classifies the sentiment of the input.   

## Decoder-Only Transformer

![alt text](image-1.png)

Similar to an Encoder-Only Transformer, a **Decoder-Only Transformer** starts with Word Embedding and Positional Encoding. However, instead of Self-Attention, it uses **Masked Self-Attention**. Unlike **Self-Attention**, **Masked Self-Attention** ignores words that appear after the word of interest.  

Due to their nature, **Decoder-Only Transformers** can be trained to generate responses to prompts 
Decoder-Only Transformers generate responses to prompts by predicting the next token based on previous ones.
During training, only the first part of sentence is given, then model weights are modified until it completes the sentence.
ChatGPT is an example of a Decoder-Only Transformer. It was trained to generate text following a prompt. 

To sum it up, in contrast to an **Encoder-Only Transformer** which creates ***Context Aware Embeddings***, a **Decoder-Only Transformer** creates ***Generative Inputs*** that can be plugged into a simple Neural Network generating new tokens. 

# The Matrix Math for Masked Self-Attention
$$
\text{MaskedAttention}(Q, K, V, M) = \text{SoftMax} \left( \frac{QK^T}{\sqrt{d_k}} + M \right) V
$$
Q: Query  
K: Key  
V: Value  
M: Mask
$d_k$ : Dimension of the Key matrix 

Similarly as in Self-Attention, matrices for Q, K, and V are calculated. Then similarity between the Query and Key are is calculated which is then scaled.
However, before proeceding to applying the Softmax function, matrix **M** is added. The purpose of the **Mask** matrix is to prevent tokens from including anything coming after them when calculating attention. The Mask matrix consists of **0** for values to be included [0 + value = value];  
and **-∞** for values to be masked [-inf + value = -inf] which will retrun 0 when the function is applied.

# Encoder-Decoder Attenion

The first Transformer had an **Encoder** which used a Self-Attention and **Decoder** whch used Masked Self-Attentions. The encoder and the decoder were connected to each other so they could calculate **Encoder-Decoder Attention** [also called **Cross-Attention**], which uses the output from the **Encoder** to calculate the Keys and Values while the Queries are calculated from the output of the **Decoder**'s Masked Self-Attention.  
Usind the Keys, Values, and Queries, the **Encoder-Decoder Attention** is calculated using every similarity just like in Self-Attention.  
This transformer was based on something called a **Seq2Seq** or **Encoder-Decoder** model. **Seq2Seq** models were designed to translate text in one language into another.  
While Seq2Seq models are as widely used for language modeling, it is still used for **Multi-Modal Models**.  An example for a **Multi-Modal Model**, it may have an Encoder trained on images or sound, and the *Context Aware Embeddings* could be fed into a text based Decoder via Encoder-Decoder Attention in order to generate captions or reponds to audio prompts.

# Multi-Head Attention

In order to correctly establish how words are related in longer, more complicated sentences and paragraphs, Attention is applied to Encoded Values multiple times simultaneously. Each Attention unit is called a **Head** and has its own sets of **Weights** for calculating Queries, Keys, and Values. When having multiple **Heads** calculating Attention it is refered to as **Multi-Head Attention**.

![alt text](image-2.png)
In order to get back to the original number of Encoded values [new number is: **number of Heads** X **Attention values per Head**], the Attention values are connected to a fully connected layer having same number of outputs as original number of Encoded values. 


Another method to reduce the number of outputs is to modify the shape of the Value Weight matrix and choose a matching number of Heads.