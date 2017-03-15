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

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import gensim
from gensim.models.word2vec import LineSentence

import string

from datetime import datetime


emoji_list=[]
emoji_description=[]
data=[]
text=[]
twt_emoji=[]
emoji_map_front={}
emoji_map_back={}
clean_text=[]
hashtags=[]
twt_time=[]



t0 = time()


with open("emoji_list.tsv") as fhand:
    reader=csv.reader(fhand)
    for index, line in enumerate(reader):
        emoji_description.append(line[2])
        
        #map from emoji to a emoji string
        emoji_list.append(line[0])
        emoji_map_front[line[0]]=str(index)+"emojireplace"
        emoji_map_back[str(index)+"emojireplace"]= line[0]


index=0

time_emoji=collections.defaultdict(list)

top_hash_tag="#PokemonGO"

with open("twt2emoji_test.tsv") as fhand:
    for line in fhand:
        tokens=line.split('\t')
        
        temp_time=tokens[2].strip()
        
        hashtag_list=tokens[1].split()
        
        
        if top_hash_tag in hashtag_list:
            time_emoji[temp_time]+=tokens[0].split()


for key in time_emoji:
    print (key, len(time_emoji[key]))



emoji_sentiment_score={}


with open("emoji_distribution_20_g.tsv") as fhand:
    for line in fhand:
        tokens=line.strip().split('\t')
        emoji_id=tokens[0]
        emoji_sentiment_score[emoji_id]=tokens[2]
        #base_distri[emoji_id]=float(tokens[3].strip())



print ("data collection done in %0.3fs." % (time() - t0))

t0 = time()

print (top_hash_tag)


print ("preprocessing done in %0.3fs." % (time() - t0))


emoji_flatmap= [ i for j in time_emoji.values() for i in j]


print (len(emoji_flatmap),' number of emojis for the hashtag')



#to get base distribution

base_distri={}


num_emoji=len(emoji_flatmap)

emoji_counter=collections.defaultdict(float)

for e in emoji_map_back:
    emoji_counter[e]=0
    
for e in emoji_flatmap:
    try:
        emoji_counter[emoji_map_front[e]]+=1
    except:
        print (e)
        continue

for e in emoji_counter.items():
    
    base_distri[e[0]]=(e[1]/num_emoji)



emoji_distri=[]

for t in sorted(list(time_emoji.keys())):
    
    emoji_flatmap=time_emoji[t]
    num_emoji=len(emoji_flatmap)
    emoji_counter=collections.defaultdict(float)


    for e in emoji_map_back:
        emoji_counter[e]=0
    
    for e in emoji_flatmap:
        emoji_counter[emoji_map_front[e]]+=1

    for e in emoji_counter.items():
        try:
            emoji_distri.append((e[0], emoji_map_back[e[0]], emoji_sentiment_score[e[0]], base_distri[e[0]],e[1]/num_emoji, t))
        except:
            continue


top_emoji_distri=sorted(emoji_distri, key=lambda x: x[4], reverse=True)

with open("hashtag_distribution_"+str(top_hash_tag)+"_by_month.tsv", "w") as outfile:
    writer = csv.writer(outfile, delimiter = '\t')
    writer.writerow(("Emoji_id", "Emoji", "Sentiment Score","Expectation", "Distribution", "Time"))
    writer.writerows(emoji_distri)
