#sfrom emoji import UNICODE_EMOJI
import csv
import re
import functools
import collections
import numpy as np
#import matplotlib.pyplot as plt
import math
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

from sklearn.metrics.pairwise import cosine_similarity

import nltk
#nltk.download()
from nltk import word_tokenize
from nltk.tokenize.casual import TweetTokenizer
#from nltk.tokenize import TweetTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import gensim
from gensim.models.word2vec import LineSentence

import string


emoji_list=[]
emoji_description=[]
data=[]
text=[]
twt_emoji=[]
emoji_map_front={}
emoji_map_back={}
clean_text=[]
hashtags=[]

sid = SentimentIntensityAnalyzer()

with open("emoji_list.tsv") as fhand:
    reader=csv.reader(fhand)
    for index, line in enumerate(reader):
        
        
        
        emoji_description.append(line[2])
        #map from emoji to a emoji string
        emoji_list.append(line[0])
        emoji_map_front[line[0]]=str(index)+"emojireplace"
        emoji_map_back[str(index)+"emojireplace"]= line[0]


with open("sample2016") as fhand:
    for line in fhand:
        tokens=line.split('\t')
        #print (tokens)
        #exit(0)
        if (len(tokens)==10) and (tokens[5]=="en"):
            #data.append(line)
            text.append(tokens[3])

            #hashtags.append(tokens[9].split())

'''

hashtags_indices=collections.defaultdict(list
for index, hts in enumerate(hashtags):
    for ht in hts:
        hashtags_indices[ht].append(index)

'''

#star(*) behaves weird, so add escape before *

emoji_reg_list=list(emoji_list)
emoji_reg_list[2051]="\\"+emoji_reg_list[2051]
emoji_reg_list=sorted(emoji_reg_list, key=lambda x:len(x), reverse=True)

emoji_pattern="|".join (emoji_reg_list)

#print (emoji_pattern)




def preprocess_tweet(tweet, pipeline):
    for pipe in pipeline:
        tweet = pipe(tweet)
    return tweet

#tknzr = TweetTokenizer()
tknzr = TweetTokenizer()

def myTokenizer(s):
    #return s.split()
    #tokens=word_tokenize(s)
    tokens=tknzr.tokenize(s)
    
    new_tokens=[ x for x in tokens if x not in string.punctuation]

    return new_tokens
    #return tknzr.tokenize(s)


HASHTAGS_REGEX = re.compile('#')
MENTIONS_REGEX = re.compile('@[^\s]+')
EMOJI_NAME_REGEX = re.compile(':[a-z_-]+:')

#EMOJI_NAME_REGEX = re.compile(':[a-z_-]+:')

EMOJI_NAME_REGEX = re.compile(emoji_pattern)

LINK_REGEX = re.compile('https?://[^\s]+')
EXTRA_SPACES_REGEX = re.compile('\s{2,}')
HAYSTACK_REGEX = re.compile('(RT)')
ASCII_REGEX = re.compile('[[:ascii:]]')




def preprocess_tweet(tweet, pipeline):
    for pipe in pipeline:
        tweet = pipe(tweet)
    return tweet

def preprocess_hashtags(tweet):
    return HASHTAGS_REGEX.sub('', tweet)

def preprocess_mentions(tweet):
    return MENTIONS_REGEX.sub('', tweet)

def remove_extra_spaces(tweet):
    return EXTRA_SPACES_REGEX.sub(' ', tweet).strip()

def remove_hyperlinks(tweet):
    return LINK_REGEX.sub('', tweet)

def remove_haystack(tweet):
    return HAYSTACK_REGEX.sub('', tweet)

def remove_unicode(tweet):
    return ASCII_REGEX.sub('', tweet)

def extract_emoji(tweet):
    emojis = EMOJI_NAME_REGEX.findall(tweet)
    
    #print (emojis)
    
    for e in emojis:
        tweet=tweet.replace(e, " "+str(emoji_map_front[e])+" ")
    #tweet = EMOJI_NAME_REGEX.sub(' ', tweet)
    return [tweet, emojis]



preprocessing_pipeline = [
    #preprocess_hashtags,
    preprocess_mentions,
    remove_hyperlinks,
    #remove_unicode,
    remove_haystack,
]


number_emojis=0

has_emojis=0

output=[]


t0 = time()

for tweet in text[:]:

    new_tweet, results = extract_emoji(preprocess_tweet(tweet, preprocessing_pipeline))

    twt_emoji.append(results)
    clean_text.append(new_tweet)

    number_emojis+=len(results)
    


    if len(results)>0:
        has_emojis+=1
    

print("preprocessing done in %0.3fs." % (time() - t0))
print ('Number of valid Tweeter data set', len(text))
print ( number_emojis, " number of emojis are used in total")
print ( has_emojis, " tweets have emojis")




t0 = time()
# this part is for tokenizing the text

tokens=(myTokenizer(x.lower()) for x in clean_text)

#tokens=[myTokenizer(x.lower()) for x in clean_text]



model = gensim.models.Word2Vec(size=100, window=10, min_count=20, workers=4)
model.build_vocab(tokens)
model.train(tokens)


#model = gensim.models.Word2Vec(tokens, size=100, window=10, min_count=20, workers=4)
fname="my_word2vec"
model.save(fname)
#model = gensim.models.Word2Vec.load(fname)


vocab = model.vocab.keys()

print (len(vocab), " the size of the vocabulary")
print("embedding is done in %0.3fs." % (time() - t0))


t0 = time()

model.init_sims();

used_emojis=vocab & emoji_map_back.keys()

#print (used_emojis)

vader_lex=sid.make_lex_dict()
#print (type(vader_lex))

top10_list_word={}
#top 20 meaningful words

for emoji in used_emojis:
    sims = model.most_similar(emoji, topn=1000)
    temp=[]
    limit=20
    for sim in sims:
        if limit<1:
            break
        if sim[0] not in used_emojis and sim[0] in vader_lex:
            temp.append(sim)
            limit-=1

    top10_list_word[emoji]=temp

output=[]
emoji_sentiment_score={}
emoji_sentiment_score2={}
emoji_sentiment_score3={}

# calculating sentiment score
for emoji in top10_list_word:
    top_words_with_emoji=[ emoji_map_back.get(w[0],w[0]) for w in top10_list_word[emoji]]
    
    scores=[sid.polarity_scores(i)["pos"]- sid.polarity_scores(i)["neg"] for i in top_words_with_emoji]
    scores2=scores[:10]
    scores3=scores[:5]
    

    if len(scores)==0:
        emoji_sentiment_score[emoji]=0
        emoji_sentiment_score2[emoji]=0
        emoji_sentiment_score3[emoji]=0
    else:
        emoji_sentiment_score[emoji]=sum(scores)/len(scores)
        emoji_sentiment_score2[emoji]=sum(scores2)/len(scores2)
        emoji_sentiment_score3[emoji]=sum(scores3)/len(scores3)
    
    output.append([emoji, emoji_map_back[emoji], emoji_sentiment_score[emoji], emoji_sentiment_score2[emoji], emoji_sentiment_score3[emoji], ' '.join(top_words_with_emoji) , ' '.join([str(s) for s in scores])])

print("similarity is done in %0.3fs." % (time() - t0))


emoji_flatmap= [ j for v in twt_emoji for j in v ]
emoji_counter=collections.defaultdict(float)
for e in emoji_map_back:
    emoji_counter[e]=0


for e in emoji_flatmap:
    try:
        emoji_counter[emoji_map_front[e[0]]]+=1
    except:
        continue

print (len(emoji_sentiment_score))
print (len(emoji_counter))

emoji_distri=[ (e[0], emoji_map_back[e[0]] , emoji_sentiment_score2.get(e[0], "None"), e[1]/len(emoji_flatmap)) for e in emoji_counter.items()]
top_emoji_distri=sorted(emoji_distri, key=lambda x: x[3], reverse=True)



with open("emoji_sentiment_tweet_g.tsv", "w") as outfile:
    writer = csv.writer(outfile, delimiter = '\t')
    writer.writerows(output)



with open("emoji_distribution_20_g.tsv", "w") as outfile:
    writer = csv.writer(outfile, delimiter = '\t')
    writer.writerows(top_emoji_distri)
