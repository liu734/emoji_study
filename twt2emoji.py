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


t0 = time()


with open("emoji_list.tsv") as fhand:
    reader=csv.reader(fhand)
    for index, line in enumerate(reader):
        emoji_description.append(line[2])
        #map from emoji to a emoji string
        emoji_list.append(line[0])
        emoji_map_front[line[0]]=str(index)+"emojireplace"
        emoji_map_back[str(index)+"emojireplace"]= line[0]

#base_distri[emoji_id]=float(tokens[3].strip())
#star(*) behaves weird, so add escape before *

emoji_reg_list=list(emoji_list)
emoji_reg_list[2051]="\\"+emoji_reg_list[2051]
emoji_reg_list=sorted(emoji_reg_list, key=lambda x:len(x), reverse=True)
emoji_pattern="|".join (emoji_reg_list)




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



with open("hashtag-2016-en") as fhand:
    
    with open("twt2emoji_test.tsv","w") as outfile:
    
        writer = csv.writer(outfile, delimiter = '\t')
    
        
        
        
    
        for line in fhand:
            tokens=line.split('\t')
            
            #print (len(tokens))
            if ((len(tokens)==11) or (len(tokens)==10)) and (tokens[5]=="en"):
                #data.append(line)
                tweet =tokens[3]

                new_tweet, results = extract_emoji(preprocess_tweet(tweet, preprocessing_pipeline))
                
                #get time
                temp_time=tokens[4].split()
                temp_time=" ".join(temp_time[1:4]+temp_time[5:])
                temp_time=datetime.strptime(temp_time, "%b %d %H:%M:%S %Y")
                twt_time=str(temp_time.year)+"/"+str(temp_time.month)
                
                
                if (len(tokens)==11):
                    hashtag_list=tokens[10]
                    writer.writerow((" ".join(results),tokens[10].strip(),twt_time))
        
                else:
                    hashtag_list=tokens[9]
                    writer.writerow((" ".join(results),tokens[9].strip(),twt_time))


print ("preprocessing done in %0.3fs." % (time() - t0))


