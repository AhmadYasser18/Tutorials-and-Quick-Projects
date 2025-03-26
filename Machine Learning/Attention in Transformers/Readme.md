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



