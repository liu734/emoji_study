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

t0 = time()


with open("emoji_list.tsv") as fhand:
    reader=csv.reader(fhand)
    for index, line in enumerate(reader):
        emoji_description.append(line[2])
        
        #map from emoji to a emoji string
        
        emoji_list.append(line[0])
        emoji_map_front[line[0]]=str(index)+"emojireplace"
        emoji_map_back[str(index)+"emojireplace"]= line[0]


hashtags_indices=collections.defaultdict(list)

index=0

with open("hashtag-2016-en") as fhand:
    for line in fhand:
        tokens=line.split('\t')
        
        #print (len(tokens))
        if ((len(tokens)==11) or (len(tokens)==10)) and (tokens[5]=="en"):
            #data.append(line)
            text.append(tokens[3])
            
            
            
            if (len(tokens)==11):
                #print (tokens[10])
                hashtag_list=tokens[10].split()
                for hash in hashtag_list:
                    hashtags_indices[hash].append(index)
            else:
                #print (tokens[9])
                hashtag_list=tokens[9].split()
                for hash in hashtag_list:
                    hashtags_indices[hash].append(index)

            index+=1
                
emoji_sentiment_score={}
base_distri={}

with open("emoji_distribution_20_g.tsv") as fhand:
    for line in fhand:
        tokens=line.strip().split('\t')

        emoji_id=tokens[0]

        emoji_sentiment_score[emoji_id]=tokens[2]
        base_distri[emoji_id]=tokens[3].strip()




#star(*) behaves weird, so add escape before *
emoji_reg_list=list(emoji_list)
emoji_reg_list[2051]="\\"+emoji_reg_list[2051]
emoji_reg_list=sorted(emoji_reg_list, key=lambda x:len(x), reverse=True)
emoji_pattern="|".join (emoji_reg_list)



top_hash_tag="#Trump"

top_hash_tag=(sorted(hashtags_indices.items(), key=lambda x: len(x[1]), reverse=True)[0][0])




#print (emoji_pattern)

def preprocess_tweet(tweet, pipeline):
    for pipe in pipeline:
        tweet = pipe(tweet)
    return tweet

#tknzr = TweetTokenizer()
tknzr = TweetTokenizer()

def myTokenizer(s):

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




print ("data collection done in %0.3fs." % (time() - t0))

t0 = time()

print (top_hash_tag)

#print (hashtags_indices[top_hash_tag])

print (len(text))

hashtag_text= [  text[i] for i in hashtags_indices[top_hash_tag]]


for tweet in hashtag_text[:]:

    new_tweet, results = extract_emoji(preprocess_tweet(tweet, preprocessing_pipeline))
    twt_emoji.append(results)
    #clean_text.append(new_tweet)
    
    #number_emojis+=len(results)
    
    #if len(results)>0:
    #    has_emojis+=1


print ("preprocessing done in %0.3fs." % (time() - t0))
print ('Number of',top_hash_tag , 'Tweets', len(text))
#print ( number_emojis, " number of emojis are used in total")
#print ( has_emojis, " of tweets have emojis")


#print (hashtags_indices.items())




#print (top_hash_tag)
#print (hashtags_indices[top_hash_tag])


#emoji_flatmap= [ j for i in hashtags_indices[top_hash_tag] for j in twt_emoji[i]]

emoji_flatmap= [ j for j in twt_emoji]

#print ([ (i,text[i]) for i in hashtags_indices[top_hash_tag]][:20] )

#print (emoji_flatmap)
emoji_counter=collections.defaultdict(float)

for e in emoji_map_back:
    emoji_counter[e]=0


for e in emoji_flatmap:
    try:
        emoji_counter[emoji_map_front[e[0]]]+=1
    except:
        #print (e[0])
        continue


emoji_distri=[(e[0], emoji_map_back[e[0]],  base_distri[e[0]], e[1]/len(emoji_flatmap), emoji_sentiment_score[e[0]]) for e in emoji_counter.items()]



top_emoji_distri=sorted(emoji_distri, key=lambda x: x[3], reverse=True)


#print (top_emoji_distri)

with open("hashtag_distribution_"+str(top_hash_tag)+".tsv", "w") as outfile:
    writer = csv.writer(outfile, delimiter = '\t')
    writer.writerows(top_emoji_distri)
