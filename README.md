## What is this project?
This project contains the implementation in Python of the Contextual Forest, an unsupervised model for language disambiguation that I developed during [my bachelor's thesis](http://t.ly/bBNd). It includes a demo notebook with
code snippets showing the main functionality and informally exposes the ideas behind the model as well as 
the key points discussed in my bachelor's thesis regarding the task of Word Sense Disambiguation (WSD).

## Introduction

Like many other NLP (Natural Language Processing) models, Contextual Forests operate under the assumption that the context of a word (i.e., the other words in the sentence) can fully determine its meaning. This hypothesis seems reasonable to assume, considering that essentially this is what we humans do when we communicate with each other.

Unfortunately, the context has proven to be rather challenging to figure out using even the most advanced techniques in many situations. This difficulty lies within the knowledge-based component that context has; let's illustrate this with an example. Consider the following sentence:

_"The best Queen songs redefined rock"_


Now let's focus our attention on the words Queen and rock. For us humans, if familiar with 70's rock, it's a straightforward and almost automatic process to recognize that we are talking about the British rock band Queen and that rock is a music genre. For a language model instead, this could present a far more complicated situation since both Queen and rock, as separate words, can be referring to a considerable number of different things (for example, a female monarch and a solid aggregate of minerals).

So how do language models deal with this problem? One of the key points is that these words appear together in the same sentence. Probabilistically speaking, in the example sentence, I could be talking about the Queen Elizabeth II of the United Kingdom, but that seems highly unlikely since I'm also talking about some concept represented by the word rock which is not statistically associated with the context of Queen Elizabeth II.

From this point, it's only natural to wonder how language models assign both these probabilities and association statistics. Depending on the answer to this question, we can roughly classify language models into two types:

* **Context-free models** such as [Word2Vec](https://arxiv.org/pdf/1301.3781.pdf) or [GloVe](https://nlp.stanford.edu/pubs/glove.pdf), are based on creating a 1:1 mapping between words and vectors (usually referred to as word embeddings). Although specifics in the implementation may be different, this models' general idea consists in training a neural network over a large corpus of text to get a representation that captures semantic properties. For example, if we consider the vectors k, w, and m corresponding to the words "king", "woman" and "man" respectively.  Then the vector q = (k - m) + w is very close to the one assigned to the word "queen". These algebraic properties are desirable in many NLP problems, but it's complicated to solve the WSD (Word Sense Disambiguation) problem with these models since the embedding for every word is unique and independent from the context.

    <figure>
    <img src="./imgs/glove.png" alt="drawing" width="500"/>
    </figure>

* **Dynamic-embedding models** like [ELMo](https://arxiv.org/pdf/1802.05365.pdf), [BERT](https://arxiv.org/pdf/1810.04805.pdf) and [RoBERTa](https://arxiv.org/pdf/1907.11692.pdf) are based on an architecture called "transformers", and since their first appearance around 2018, they have been at the top of the NLP world. The difference between these models and the context-free ones lies in the embedding association process. Rather than a fixed embedding, dynamic models will assign embeddings depending on the other words in the context, which means that the same word can have different embeddings depending on the sentence. They also need a massive corpus for training and considerably more computational power than the context-free models, but on the other hand, the WSD problem results are significantly better.

    <figure>
    <center><img src="./imgs/green.png" alt="drawing" width="500"/></center>
    </figure>

Although dynamic-embedding models have a significantly better performance in the WSD problem, they share a crucial aspect in the training process with the context-free models; they both need an extensive text corpus. Having this enormous corpus for training means that with enough computational power and some fancy architectures, one can build a pretty decent model entirely based on statistics inferred from the training data with no understanding of the concepts that language represents.

So what are we looking at when we are face to face with the state-of-the-art models? Language understanding or statistical inference? The answer is somewhere in between; It is clear that language models have mastered the syntactic rules of language, but they have a long way to go to understand the subtleties within a complex semantic context (see this article from Niven and Kao for an example).

While thinking about how people solve the WSD problem daily in conversations, I concluded that disambiguation could not be a memory-based process. In the previous example, we don't know that Queen is a British rock band because one night, while we were discussing music in a bar, we heard a friend of a friend using the words Queen and rock in the same sentence. We know that Queen is a British rock band, and rock is a music genre because the meaning of both words is, from all the possibilities, the most consistent option considering the rest of the words in the sentence. Specifically, when we read the first three words (The best Queen), and until the next word, in our head, Queen can mean many things, but the moment we read the word songs we know we are talking about music, so this Queen must be the rock band. This kind of association is precisely the idea behind the Contextual Forest model.


## Contextual Forest

So I wanted to build a disambiguation system through semantical connections using the different possible contexts of the words in a sentence. I thought that using context for making connections to find common ground between word meanings sounded a lot like spanning nodes in searching algorithms over graphs, so I decided to model this disambiguation process as different Trees (one for each possible meaning) trying to make connections with each other.  Given an initial word, I needed context about possible meanings organized in a structure similar to a graph to make the model work, so I decided to use Wikipedia. Unfortunately, Wikipedia only provides articles about nouns (objects, people, events, etc.), so due to this impasse and time restrictions, the implemented version of Contextual Forests only works for disambiguating nouns. Nevertheless, the process is easily scalable if one finds another resource for covering more words.

<figure>
<center><img src="./imgs/model_idea.png" alt="drawing" width="500"/></center>
</figure>

### Step by step
 
 Firstly, after identifying nouns in a sentence, it was necessary to find all possible meanings for every one of them. This requirement could have been a problem since many words have possible meanings associated with historical events or songs that would not appear in a standard dictionary. Fortunately, Wikipedia has pages specifically designed for this task:


<figure>
<img src="./imgs/disambiguation.png" alt="drawing" width="500"/>
</figure>



It's worth mentioning that an average Wikipedia page has a relatively large number of links that recursively expanded can lead to a computationally infeasible search problem. For this reason, the next thing the algorithm needed was a "relevance function" that could evaluate which links to expand to quickly find a connection between Trees (In computer science, this is called a heuristic function). This heuristic function needed to represent how close two Wikipedia articles are. At first, I thought in just finding shared links between pages, but as it turns out, it's not uncommon for two Wikipedia pages that are not related at all to share a few links:

```python
    >>> from ContextualForest import wiki
    >>> A = wiki.page('potato').links.keys()
    >>> B = wiki.page('Microsoft').links.keys()
    >>> len(A & B)
    12
```

The basis of the idea was correct, but it needed some refinement. Instead of considering all links, one can better capture the similarity of two Wikipedia pages by only considering the relevant links for computations. 

So how do we define a relevant link? For figuring that out, we need to determine what words are semantically relevant on a Wikipedia page. I identified relevant words with those uniformly distributed over the text (Note that a relevant word doesn't necessarily mean a very frequent word).

<figure>
<center><img src="./imgs/metrics.png" alt="drawing" width="500"/></center>
</figure>

After that, I defined link relevance as the average relevance of the words composing the link's title. This approach proved ineffective as a relevance metric because using a non-weighted average can be pretty sensitive to outliers, resulting in a bias towards links with the most relevant word as part of the title. To solve this issue, I studied the distribution of the relevance score, which I realized could be approximated by Zipf's distribution.

<figure>
<center><img src="./imgs/zipf.png" alt="drawing" width="500"/></center>
</figure>

Instead of computing link relevance as a simple average, I defined it as the inverse image of an average over ranking positions, capturing the scoring values' decreasing factor to correct the bias.

<figure>
<center><img src="./imgs/link_relevance.png" alt="drawing" width="500"/></center>
</figure>

Finally, by using the Trees, we have all the necessary tools to disambiguate context with this non-supervised technique

```python
    >>> from ContextualForest import contextual_forest
    >>> fr = fr = contextual_forest("Queen redefined rock with their songs")
    >>> for word, node in fr.dic.items():
            possible_meanings = len(disambiguation(word))
            if not possible_meanings:
                #no disambiguation page
                possible_meanings = 1
            print(f"Word: {word}\t possible meanings: {possible_meanings}\n Choosen: {node.page.text[:100]} ...")
    
    Word: rock	 possible meanings: 49
    Chosen: Rock music is a broad genre of popular music that originated as "rock and roll" in the United States ...
    Word: songs	 possible meanings: 1
    Chosen: A song is a musical composition intended to be performed by the human voice. This is often done at d ...
    Word: queen	 possible meanings: 40
    Chosen: Queen are a British rock band formed in London in 1970. Their classic line-up was Freddie Mercury (l ...
    
```

Note that the algorithm is far from perfect and sometimes can fail to disambiguate some words. Still, these results prove a potential alternative approach to successfully disambiguate context by mining specific information in graph-based structures (Knowledge Graphs) and mimicking a reasoning process instead of training with millions of examples and learning the statistical information needed for disambiguation from them. This idea to put effort into how models learn instead of how large model capabilities are, is, in my opinion, something that deserves consideration if one day we want to build models that could ultimately reason like we humans do.