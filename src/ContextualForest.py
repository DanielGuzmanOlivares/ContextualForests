#!/usr/bin/env python

import numpy as np
import re
import string
import wikipediaapi
import en_core_web_lg
import es_core_news_lg
import itertools
from nltk.stem.snowball import SnowballStemmer
from scipy.optimize import curve_fit
from nltk.stem import WordNetLemmatizer
from queue import PriorityQueue as pq
import spacy


# Global variables

stemmer = SnowballStemmer(language='english')
wiki = wikipediaapi.Wikipedia('en') # Wikipedia API object configured to English
nlp = en_core_web_lg.load()
stopwords_path = "stopwords.txt"

lemmatizer = WordNetLemmatizer() 
with open(stopwords_path) as file:
        stop_words = [x.strip() for x in file.readlines()]

####################### Interface with Wikipedia ##########################

def _internal_page(title):
    """Checks if a Wikipedia page is categorized as an internal page.

    Parameters
    ----------
    title : str
        Wikipedia page title as shown in the url (www.wikipedia.com/en/<title>).

    Returns
    -------
    bool
        True if the page is an internal page, False otherwise.
    """
    return ':' in title or "List of" in title or "(disambiguation)" in title

def disambiguation(noun):
    """Returns the possible links from a Wikipedia disambiguation page.
     
    Parameters
    ----------
        noun : str 
            noun to disambiguate.
    Returns
    -------
        list
            The Wikipedia pages possibly associated with the `noun`.
    
    Notes
    -----
    If the `noun` to disambiguate corresponds with only one page
    (and consequently no disambiguation page exists) the list returned
    will be empty.

    Examples
    --------
    >>> disambiguation('Bohemian Rhapsody')
    ['Bohemian Rhapsody',
    "Bohemian Rhapsody (That '70s Show)",
    'Bohemian Rhapsody (The Muppets)',
    'Bohemian Rhapsody (film)',
    'The Story of Bohemian Rhapsody']

    """
    #dis_title = f"{noun} (disambiguation)"
    dis_title = f"{noun} (desambiguacion)"
    return [x for x in wiki.page(dis_title).links if not _internal_page(x) and noun.lower() in x.lower()]


def clean_text(text):
    """Replaces return (\n) and commas (,) and points (. ) by blank spaces.
       Any other punctuation will be removed along with posessive forms.

    Parameters
    ----------
        text : str
            Text to be cleaned.
    Returns
    -------
        str
            Cleaned text.
    
    Examples
    --------
    >>> text = "Earth's atmosphere consists mostly of nitrogen and oxygen. "
    >>> clean_text(text)
    'Earth atmosphere consists mostly of nitrogen and oxygen  '
    """
    text = re.sub(r"\n|,|[.]+\s", " ", text)
    text = re.sub(r"'es|'s", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def stem_text(text):
    """Uses snowball stemmer to produce a statistics dictionary
    Where words -> 
          occur -> indexes of (key) occurrences in the text
          relevance -> set to 0 (see set_relevance())
    
    Parameters
    ----------
        text : str
            Text from which the dictionary will be created.
    Returns
    -------
        tuple : (dict of dict, int)
        Where first element is a dictionary where:
                * a key is a tuple with format (stemmed, part_of_speech):
                    1. stemmed : str
                        the lexical root (stem) of a word.
                    2. part_of_speech : str
                        part of speech, can be "NOUN","PROPN","VERB" or "ADJ".
                * and values have format:
                    {
                        words : set
                            different derivations of the stemmed word in the text.
                        occurr : list
                            indexes of every occurrence of derivations of 
                            the stemmed word in the text
                        relevance : float
                            relevance of the stemmed word in the text
                            set to 0 initially.
                        pos : float
                            relevance-ranking position of the stemmed word
                            set to 0 intially.
                    }
        and second element is the number of different words in text after cleaning.
    
    Examples
    --------
    >>> text = "Factorization algorithms can become computationally infeasable. \
                Factoring a number efficently is an algorithmically complicated process that requires\
                not only mathematical knowledge in order to take advantage of the factor properties that\
                some numbers have but also good programming skills"
    >>> stem_text(text)

    ({('factor', 'NOUN'): {'words': {'factor', 'factorization'},
        'occurr': [0, 30],
        'relevance': 0.0,
        'pos': 0},

        ('algorithm', 'NOUN'): {'words': {'algorithms'},
        'occurr': [1],
        'relevance': 0.0,
        'pos': 0},
        ...
    } , 39)
    
    """
    dic = {}
    content = clean_text(text).split() #split spaces
    for i,x in enumerate(nlp(clean_text(text))):
        target_pos = ["NOUN","PROPN","VERB","ADJ"]
        if x.pos_ not in target_pos:
            continue
        else:
            word = x.text.lower()
            #wikipedia in word -> internal redirection
            if word not in stop_words and "wikipedia" not in word:
                stemmed = stemmer.stem(word)
                key = (stemmed,x.pos_)
                if key not in dic:
                    dic[key] = {"words":set([word]), 
                                "occurr":[i], 
                                "relevance":.0, #initially 0, updated later
                                "pos":0 #initially 0, updated later
                               }
                else:
                    dic[key]["occurr"].append(i)
                    dic[key]["words"].add(word)
    return dic, len(content)

def set_relevance(stem_dict, text_len, n_chunks=100):
    """Fixes relevance and position for every word in the statistical dictionary.

    Parameters
    ----------
        stem_dict : dict
            statistical dictionary created from a text (see stem_text()).
        text_len : int
            the number of different words in text after cleaning.
        n_chunks : int
            size of the sliding window used to compute relevance value.


    Returns
    -------
        dict
            statistical dictionary with updated relevance and position.

    Examples
    --------
    >>> text = wiki.page('bohemian rhapsody').text
    >>> d = set_relevance(*stem_text(text))
    >>> max(d.keys(), key=lambda x: d[x]['relevance'])
    ('song', 'NOUN')

    See Also
    --------
    set_relevance : get statistical dictionary.
    """
    if text_len < 300: #articles with less than 300 words are not accepted
        return None
    else:
        for word in stem_dict:
            occ,_ = np.histogram(stem_dict[word]["occurr"],text_len//n_chunks)
            stem_dict[word]["relevance"] = (occ > 0).mean()
    return stem_dict

####################### Node ##########################

class Node():
    """ Implementation of a semantical node associated with a Wikipedia page

        Attributes
        ----------
        page : wikipediaapi.WikipediaPage
            Wikipedia page associated with this node.
        depth : float
            Depth in the disambiguation tree or distance to the inital word.
        parent : ContextualForest.Node
            Parent node, default: None.
        root : ContextualForest.Node
            The root Node from which this node was expanded, default: None.
        d : int
            Expansion factor, the most relevant d links will be considered for expansion.
        title : str
            Title of the prime ancestor's associated Wikipedia page, default: None.
        dic : dict
            Statistical dictionary associated with the node's Wikipedia page.

        Methods
        -------
        expand()
            Expands the node using the links associated to it's Wikipedia page.
        link_relevance(l)
            Computes the importance of a link in the associated Wikipedia page.
        similarity(other)
            Calculates the similarity score with respect to other node.
        """
    def __init__(self, page, depth, parent=None, root=None, d=20, title=None):
        self.page = page
        self.parent = [parent]
        self.children = {}
        self.d = d
        self.title = title
        if depth == 0:
            self.depth = 0 #root node, from disambiguation page
            self.root = None
        else:
            self.depth = depth
            self.root = self if self.depth == 1 else root
            self.dic = set_relevance(*stem_text(self.page.text))
            if self.dic == None: #article too short
                return None
            keys = sorted(self.dic.keys(),key=lambda k:self.dic[k]["relevance"], reverse=True) #word-relevance sort
            y_data = np.zeros(len(keys)) 
            # set relevance index and prepare zipf y-data
            for i,key in enumerate(keys): 
                self.dic[key]["pos"] = i
                y_data[i] = self.dic[key]["relevance"]
            
            y_data = y_data[y_data != 0] # only consider words with positive relevance 
            x_data = np.linspace(1,len(y_data),len(y_data))
            
            # zipf's estimation for word with index x
            def zipf(x,alpha): 
                return self.dic[keys[0]]["relevance"] / (x ** alpha)

            popt, _ = curve_fit(zipf,x_data,y_data)
            self.model = lambda x: zipf(x,popt)
    
    def __clean_links(self,links):
        """ Obtains a unique mapping between trimmed links and associated Wikipedia pages.

            Parameters
            ----------
            links : list of tuple
                Where every tuple has format link : str, page : wikipediaapi.WikipediaPage)
                and the list represents all links in a Wikipedia page.

            Returns
            -------
                list of tuple
                    where every tuple has format (link : str, page : wikipediaapi.WikipediaPage).
        """
        maps = {}
        for link,page in links:
            ref = link.lower().split(" (")[0].strip()
            if ref not in maps:
                maps[link] = page
        return maps
            
    def expand(self):
        """ Expands a node using the links in it's associated Wikipedia page.

            Returns
            -------
            list of ContextualForest.Node
                Children nodes discovered in expansion
            
            Notes
            -----
            The number of children selected for expansion from all possible links
            will be determined by the expansion factor.
        """
        # disambiguation page case
        if self.depth == 0: 
            links = self.page.links.items()
            title = self.title
            rel_set = lambda _ : 1/len(self.page.links)
        # regular expansion
        else:
            links = sorted(self.page.links.items(), key = lambda l: self.link_relevance(l[0]), reverse=True)
            # d most relevant links (with no loops)
            links = [x for x in self.__clean_links(links).items() if self.__no_loop(self.page.title, x[0])][:self.d]
            title = ""
            rel_set = lambda l : self.link_relevance(l)
        for link,page in links:
            if not _internal_page(link) and (self.title.lower() in link.lower() or self.depth != 0):
                n = Node(page, self.depth+1, parent=self, root=self.root, title=title, d=self.d)
                self.children[n] = rel_set(link)

        return self.children
    
    def link_relevance(self,link):
        """ Computes the relevance of a link in the associated Wikipedia page
            by using the image of the Zipf distribution associated with this node.

            Parameters
            ----------
            link : str
                The link for computing the relevance, if the words forming the link
                are not in the text from thge Wikipedia page, relevance value will be 0.

            Returns
            -------
                float
                    Link relevance value.

        """
        words = nlp(link) #tokenization
        rel = []
        # contingency measure for not expanding the same link or an internal link
        if (self.title.lower() in link.lower() and self.depth == 1) or ":" in link:
                return 0
        for word in words:
            try:
                #token in statistical dictionary
                pos = self.dic[(stemmer.stem(word.text.lower()),word.pos_)]["pos"]
                rel.append(pos)
            except KeyError:
                #token not in statistical dictionary -> max value possible
                rel.append(len(self.dic))
        if len(rel) == 0:
            return 0
        elif sum(rel) == 0:
            return 0
        else:
            m = np.mean(rel)
            return self.model(m)[0]

    def similarity(self,other):
        """ Computes the similarity score between two nodes.
            
            Parameters
            ----------
                other : ContextualForest.Node
                    The other node

            Returns
            -------
                float
                    similarity between two nodes.
            Notes
            -----
            More information in http://t.ly/bBNd
              
        """
        try:
            # Common links only important for both 
            both = [link for link in self.page.links if link in other.page.links 
                                                        and (other.link_relevance(link) > 0.01 
                                                        and self.link_relevance(link) > 0.01)
                    ]
            if len(both) == 0:
                return 0
            diff = np.sum([min(self.link_relevance(link), other.link_relevance(link)) for link in both]) * len(both)
            return diff
        except:
            return 0
    
    def __hash__(self):
        """Gives the node class the hashable property."""
        return hash(self.page.title)
    
    def __eq__(self, other):
        """Gives the node class the equality comparison property."""
        if other == None:
            return False
        return self.page.title == other.page.title
    
    def __gt__(self, other):
        """Gives the node class the greater-equality comparison property."""
        return self.page.title > other.page.title
    
    def __no_loop(self,title,link):
        """Checks wether if keywords of the title appear in the link (loop) or viceversa."""
        A = title.lower().split()
        B = link.lower().split()
        for a in A:
            for b in B:
                if a in b or b in a:
                    return False
        return True
    
    
####################### Tree ##########################

class Tree():
    """ Implementation of a semantical Tree, a set of connected nodes that have been
        expanded from one initial node.

        Attributes
        ----------
        root_word : str
            Root word to disambiguate.
        exp : int
            Expansion factor, the most relevant `exp` links will be considered
            in the expansion of each node

        Methods
        -------
        expand_node(node)
            Expands the given node from the tree.
    
    """
    def __init__(self, root_word, exp=10):
        word = root_word.lower()
        self.word = word
        # disambiguation page exists
        if len(disambiguation(root_word)) != 0:
            #self.root = Node(wiki.page("{} (disambiguation)".format(root_word)), 0, title = word, d=exp)
            self.root = Node(wiki.page("{} (desambiguacion)".format(root_word)), 0, title = word, d=exp)
            #expands first level to every possibility
            self.root.expand()
            self.expanded, self.to_expand = [self.root], list(self.root.children.keys())
        # only one page exists
        elif wiki.page(word).exists():
            self.root = Node(wiki.page(word), 1,title = word, d=exp)
            self.expanded, self.to_expand = [], [self.root]
        # no page exists associated with root word
        else:
            return
        
    def expand_node(self,node):
        """ Expand a given node an adds the new descendants to the tree.
            
            Parameters
            ----------
                node : ContextualForest.Node
                    The node to expand

            Returns
            -------
                list of ContextualForest.Node
                    Descendants of the node expanded.
        """
        news = node.expand()
        self.expanded.append(node)
        try:
            self.to_expand.remove(node)
        except:
            pass
        self.to_expand += news
        return news
    
    def __hash__(self):
        """ Gives the Tree object the hashable property."""
        return hash(self.root.page.title)
    
    def __eq__(self,other):
        """ Gives the Tree object the equality comparison property."""
        return self.root.page.title == other.root.page.title
    
class Forest():
    """ Implementation of the contextual forest main data structure for 
        context-based semantic disambiguation.

        Attributes
        ----------
        words : list of str
            The keywords to disambiguate
        trees : dict
            Mapping between ContextualForest.Tree objects and the root nodes
            associated with that tree objects.
        dic : dict
            Mapping between the words provided to disambiguate and root nodes
            associated with tree objects formed in the disambiguation process.
        connections : dict
            Where keys are combinations of two possible trees and values are 
            boolean indicating wether or not the pair of trees is connected.
        Q : queue.PriorityQueue
            The priority queue structure to manage connections and expansion order
        Methods
        -------
        disambiguate()
            Performs the disambiguation process (forward).
        recover_words()
            Recovers words synsets once the disambiguation process has completed.
    """
    def __init__(self,words):
        self.words = None
        self.trees = {}
        self.dic = {}
        for word in words:
            self.dic[word] = None
            self.trees[Tree(word)]= None
        self.tree_combs = itertools.combinations(self.trees,2) #possible pairs of trees
        self.connections = {pair:False for pair in self.tree_combs}
        self.Q = pq()
        for tree1,tree2 in itertools.combinations(self.trees,2):
            for u in tree1.to_expand:
                for v in tree2.to_expand:
                    sim = -u.similarity(v) #negative because pq orders naturally
                    if sim == 0:
                        continue
                    self.Q.put((sim,u,v,tree1,tree2)) # (similarity, node_1, node_2, tree_1, tree_2)

    def disambiguate(self):
        """ Performs the forward disambiguation process expanding the nodes till
            all trees are connected by a path.
        """
        #while the are connections to check or some tree is not connected
        while not all(self.connections.values()) and not self.Q.empty(): 
            _,u,v,t1,t2 = self.Q.get()
            key = (t1,t2) if (t1,t2) in self.connections else (t2,t1) #depends on itertools
            while self.connections[key]:
                #while key belongs to an already connectyed tree, pop from pq
                _,u,v,t1,t2 = self.Q.get()
                key  =  (t1,t2) if (t1,t2) in self.connections else (t2,t1)
            #expand both nodes
            news_t1 = t1.expand_node(u)
            news_t2 = t2.expand_node(v)
            #check for intersection
            if any([True if n in t2.to_expand else False for n in t1.to_expand]):
                self.connections[key] = True
                if self.trees[t1] == None:
                    self.trees[t1] = u.root
                if self.trees[t2] == None:
                    self.trees[t2] = v.root
            else: #no connection, add new nodes
                #expansion:
                for u in news_t1:
                    for v in news_t2:
                        sim = -u.similarity(v)
                        if sim == 0:
                            continue
                        self.Q.put((sim,u,v,t1,t2))
                           
    def recover_words(self):
        """ Recovers the Nodes associated with the disambiguation of every word provided
            in the instanziation of the class and stores it in the `dic` attribute.
        """
        found = False
        for tree,node in self.trees.items():
            for link,page in node.page.links.items():
                if tree.word.lower() == link.lower():
                    self.dic[tree.word] =  Node(page,1)
                    found = True
            if not found:
                connection = node
                while connection.depth != 1:
                    connection = connection.parent
                self.dic[tree.word] = connection
            found = False


def contextual_forest(text):
    """ Performs a contextual forest disambiguation.
        
        Parameters
        ----------
            text : str
                Sentence with nouns to disambiguate.

        Returns
        -------
            ContextualForest.Forest
                Disambiguation forest object with information about word synsets
                stored in the `dic` attribute
        
        Examples
        --------
        >>> fr = contextual_forest("the FDA has approved the first shot for COVID-19")
        >>> print(fr.dic["fda"].page.text[:100])
        The United States Food and Drug Administration (FDA or USFDA) is a federal agency of the Department 
        >>> print(fr.dic["shot"].page.text[:100])
        An injection (often and usually referred to as a "shot" in US English, a "jab" in UK English, or a "
        >>> print(fr.dic["covid-19"].page.text[:100])
        Coronavirus disease 2019 (COVID-19) is a contagious disease caused by severe acute respiratory syndr

        
        Notes
        -----
            Note that in the expansion, internally the program calls the Wikipedia API
            on every node to get the page so depending on internet connection and expansion
            factor it might take a while to disambiguate a sentence (arround 3-5 mins on average).
            An option to use locally stored pages will be implemented soon.
    """
    tokens = nlp(text)
    # get a dictionary token : lemma if token is noun and not a stopword
    key_words = {token : token.lemma_.lower() for token in tokens if token.pos_ in ["PROPN","NOUN"] 
                                                                    and token.text.lower() not in stop_words}
    l = []
    # obtain words to disambiguate (including NER)
    for chunk in tokens.noun_chunks:
        words = str(chunk).split()
        i = 0
        new_words = words.copy()
        # while adding words from noun chunk results in a valid wiki page
        while len(new_words) > 1 and not wiki.page(" ".join(new_words)).exists():
            new_words=words[i:]
            i += 1
        try: 
            #if only one word
            l.append(key_words[" ".join(new_words)])
        except KeyError:
            # multiple words -> not in key_words
            l.append(" ".join(new_words))
    
    # l has the lemmatization of single words + NER (if not stopwords)
    sentence = [" ".join([x for x in ele.split() if x.lower() not in stop_words]) 
                        for ele in l if ele.lower() not in stop_words]
    fr = Forest(sentence) 
    fr.disambiguate()
    fr.recover_words()
    return fr
        

