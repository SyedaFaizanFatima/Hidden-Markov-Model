#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import itertools
import operator, functools
from nltk.corpus import brown


# In[2]:


tags_train = brown.tagged_sents(categories= 'news', tagset = 'universal')[:-500]
tags_test  = brown.tagged_sents(categories = 'news', tagset = 'universal')[-500:]


# # Training Set

# In[3]:


tags_train


# # Test Set

# In[4]:


tags_test


# # Final tags to be predicted 

# In[5]:


# Tag        Meaning                  English Examples

# ADJ        adjective                new, good, high, special, big, local
# ADP        adposition               on, of, at, with, by, into, under
# ADV        adverb                   really, already, still, early, now
# CONJ       conjunction              and, or, but, if, while, although
# DET        determiner, article      the, a, some, most, every, no, which
# NOUN       noun                     year, home, costs, time, Africa
# NUM        numeral                  twenty-four, fourth, 1991, 14:24
# PRT        particle                 at, on, out, over per, that, up, with
# PRON       pronoun                  he, their, her, its, my, I, us
# VERB       verb                     is, say, told, given, playing, would
# .          punctuation marks        . , ; !
# X          other                    ersatz, esprit, dunno, gr8, univeristy

tags = ["ADJ", "ADP", "ADV", "CONJ", "DET", "NOUN", "NUM", "PRT", "PRON", "VERB", ".", "X"]


# In[6]:


# Reduce the list of list to a single list of words
word_tags_train = list(itertools.chain.from_iterable(tags_train))

# coversion to lower case 
word_tags_train = [(str.lower(a),b) for (a,b) in word_tags_train]
word_tags_train


# In[7]:


word_tag_pairs_train = list(nltk.bigrams(word_tags_train))


# # Transmission Model

# In[8]:


def tagTransition(tag):
    """
    Count of each tag that follows a given tag
    """
    return [b[1] for (a, b) in word_tag_pairs_train if a[1] == tag]


# In[9]:


adjective_transitions    = tagTransition("ADJ")
adposition_transitions   = tagTransition("ADP")
adverb_transitions       = tagTransition("ADV")
conjunction_transitions  = tagTransition("CONJ")
determiner_transitions   = tagTransition("DET")
noun_transitions         = tagTransition("NOUN")
numeral_transitions      = tagTransition("NUM")
particle_transitions     = tagTransition("PRT")
pronoun_transitions      = tagTransition("PRON")
verb_transitions         = tagTransition("VERB")
punctuation_transitions  = tagTransition(".")
other_transitions        = tagTransition("X")


adjective_fdist     = nltk.FreqDist(adjective_transitions)
adposition_fdist    = nltk.FreqDist(adposition_transitions)
adverb_fdist        = nltk.FreqDist(adverb_transitions)
conjunction_fdist   = nltk.FreqDist(conjunction_transitions)
determiner_fdist    = nltk.FreqDist(determiner_transitions)
noun_fdist          = nltk.FreqDist(noun_transitions)
numeral_fdist       = nltk.FreqDist(numeral_transitions)
particle_fdist      = nltk.FreqDist(particle_transitions)
pronoun_fdist       = nltk.FreqDist(pronoun_transitions)
verb_fdist          = nltk.FreqDist(verb_transitions)
punctuation_fdist   = nltk.FreqDist(punctuation_transitions)
other_fdist         = nltk.FreqDist(other_transitions)


# In[10]:


tag_fdist = [adjective_fdist, adposition_fdist, adverb_fdist, conjunction_fdist, determiner_fdist, noun_fdist, numeral_fdist,
             particle_fdist, pronoun_fdist, verb_fdist, punctuation_fdist, other_fdist]


# In[11]:


def tagFreq(tag_fdist):
    """
    Sum of all the transitions in a given tag
    """
    return sum([a for (_,a) in tag_fdist.most_common()])


# In[12]:


total_adjective    = tagFreq(adjective_fdist) 
total_adposition   = tagFreq(adposition_fdist) 
total_adverb       = tagFreq(adverb_fdist) 
total_conjunction  = tagFreq(conjunction_fdist) 
total_determiner   = tagFreq(determiner_fdist) 
total_noun         = tagFreq(noun_fdist) 
total_numeral      = tagFreq(numeral_fdist) 
total_particle     = tagFreq(particle_fdist) 
total_pronoun      = tagFreq(pronoun_fdist) 
total_verb         = tagFreq(verb_fdist) 
total_punctuation  = tagFreq(punctuation_fdist) 
total_other        = tagFreq(other_fdist)


# In[13]:


tag_total = [total_adjective, total_adposition, total_adverb, total_conjunction, total_determiner, total_noun, total_numeral,
        total_particle, total_pronoun, total_verb, total_punctuation, total_other]


# In[14]:


tag_total_dic = dict(zip(tags, tag_total))


# In[15]:


# Make matrix a where each aij represents the probability of moving
#from state i to state j, such that the sum of all j in an i = 1
tag_transition = {}

for tagi, tag_fdisti in zip(tags, tag_fdist):
    # Normalize each tag to get sum of prob = 1
    value = {k: v/tag_total_dic[tagi] for k,v in tag_fdisti.most_common()}
    
    tmp = {tagi: value}
    tag_transition.update(tmp)


# In[16]:


tag_emission_fdist = nltk.FreqDist(word_tags_train)


# # Emission Model

# In[17]:


def tagEmission(tag):
    return dict([(a[0], b / tag_total_dic[tag]) for a,b in tag_emission_fdist.items() if a[1] == tag])


# In[18]:


tag_emission = {}
for tagi in tags:
    tmp = {tagi: tagEmission(tagi)}
    tag_emission.update(tmp)


# In[19]:


def _tagTransition():
    """
    Some tags are missing naturally.
    This adds their keys with 0 prob    
    """
    tag_dict = {}
    for tag, transition in tag_transition.items():
        tmp = {}
        for _tags in tags:
            if _tags in transition.keys():
                tmp[_tags] = transition[_tags]
            else:
                tmp[_tags] = 0
        tag_dict[tag] = tmp
    return tag_dict


# In[20]:


def _wordEmission(sent):
    """
    Simplifying tag emission
    by only listing observation emissions
    """
    word_dict = {}
    for tag, emission in tag_emission.items():
        tmp = {}
        for word in sent:
            if word in emission.keys():
                tmp[word] = emission[word]
            else:
                tmp[word] = 0
        word_dict[tag] = tmp            
    return word_dict


# In[21]:


def _setPI(pi):
    """
    pi represents initial transmission probability
    Here we update tag_transition with the start
    probabilities if given and set it ourselves
    otherwise
    """
    k = len(tag_emission.keys())
    tmp = {}
    if pi == None:
        for x in tag_emission.keys():
            tmp[x] = 1/k

    elif type(pi) == list:
        # Make sure the right number of tags were sent
        l = k - len(pi)
        pi.extend([0]*l)
        for x, i in zip(tag_emission.keys(), pi):
            tmp[x] = i
    elif type(pi) == dict:
        tmp = pi
    if sum(tmp.values()) != 1:
        raise ValueError('Pi values given does not sum to 1')
    return tmp


# # Viterbi Algorithm 

# In[22]:


def _viterbi(obs, states, start_p, trans_p, emit_p):
    """
    Code taken from https://en.wikipedia.org/wiki/Viterbi_algorithm#Pseudocode
    Modified slightly for tie breakers so that unknown words do not give error
    """
    V = [{}]
    for st in states:
        V[0][st] = {"prob": start_p[st] * emit_p[st][obs[0]], "prev": None}
        
    # Run Viterbi when t > 0
    for t in range(1, len(obs)):
        V.append({})
        for st in states:
            max_tr_prob = V[t-1][states[0]]["prob"]*trans_p[states[0]][st]
            prev_st_selected = states[0]
            for prev_st in states[1:]:
                tr_prob = V[t-1][prev_st]["prob"]*trans_p[prev_st][st]
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = prev_st
                    
            max_prob = max_tr_prob * emit_p[st][obs[t]]
            V[t][st] = {"prob": max_prob, "prev": prev_st_selected}
    opt = []
    max_prob = 0.0
    previous = None
    
    # Get most probable state and its backtrack
    for st, data in V[-1].items():
        if data["prob"] >= max_prob:
            max_prob = data["prob"]
            best_st = st
    opt.append(best_st)
    previous = best_st
    
    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]
    res='[ ' + ' '.join(opt) + ' ]' 
    return res


# In[23]:


def viterbi(sent, pi = None):
    """
    Enter observations in sent either as a sentence or a list of words.
    pi refers to the initial transition probabilities. Enter as either a list or a dictionary, or leave empty.
    """
    sent = sent.split() if isinstance(sent, str) else sent
    obs  = [s.lower() for s in sent]
    start_p = _setPI(pi)
    trans_p = _tagTransition()
    emit_p  = _wordEmission(obs)
    states  = list(tag_transition.keys())
    prediction=_viterbi(obs, states, start_p, trans_p, emit_p)
    return prediction


# In[24]:


import xlwt 
from xlwt import Workbook
wb = Workbook() 
sheet1 = wb.add_sheet("Sheet 1", cell_overwrite_ok=True)


# In[31]:


test_words=[]
test_tags=[]
srno=1
sentence=''
sheet1.write(0,0, 'Sr. No') 
sheet1.write(0,1, 'Sentence') 
sheet1.write(0,2, 'Actual Tags') 
sheet1.write(0,3, 'Predicted Tags') 
r=1
for i in tags_test:
    for(x,y) in i:
        test_words.append(x)
        test_tags.append(y)
        sentence=sentence+x+' '
    tag='[ '
    for i in test_tags:
        tag=tag+i+' '
    tag=tag+']'
    print('\n')
    print('Sentence No '+str(srno))  
    predictedtags=viterbi(test_words)
    print('\n\tPredicted Tags\n\t'+predictedtags)
    print('\n\tActual Tags\n\t'+tag)
    
    sheet1.write(r,0,srno) 
    sheet1.write(r,1,sentence)
    sheet1.write(r,2,tag)
    sheet1.write(r,3,predictedtags)
    r=r+1
    test_words.clear()
    test_tags.clear()
    sentence=''
    srno=srno+1
wb.save('nlp-assign2-320476.xls') 

        

