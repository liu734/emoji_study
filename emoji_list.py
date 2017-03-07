from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import csv

url="http://unicode.org/emoji/charts/full-emoji-list.html"
html=urlopen(url).read()

soup = BeautifulSoup(html, 'html.parser')

td_list= (soup.find_all("td",class_="code"))


temp_unicode_list = [x.a.string for x in td_list]
ascii_list=[]
unicode_list=[]
escape_list=[]

for text in temp_unicode_list:
    tokens=text.split()
    
    wide_unicode=bytes("",'ascii')
    
    for token in tokens:
        num_zero=10-len(token)
        token=token.replace("+",'0'*num_zero)
        token="\\"+token
        token=bytes(token,'ascii')
        wide_unicode+=token
    
        i.encode('unicode_escape')
    
    
    ascii_list.append(wide_unicode)
    
    wide_unicode=wide_unicode.decode('unicode_escape')

    escape_unicode.encode('unicode_escape')

    escape_list.append(escape_unicode)

    unicode_list.append(wide_unicode)




td_list= (soup.find_all("td",class_="name"))

name_list=[]
description=""

for i, j in enumerate(td_list):
    if i%2==0:
        description=j.string
    else:
        temp=""
        
        for ahref in j.find_all(["a"]):
            temp+=(" "+ahref.string)

        description+=temp
        name_list.append(description)



print (len(unicode_list))
print (unicode_list[-10:])

print (len(name_list))
print (name_list[-10:])

#csv output code, description
output=[[unicode_list[i],ascii_list[i], escape_list[i], j] for i, j in enumerate(name_list)]

with open("emoji_list.tsv", "w") as outfile:
    writer = csv.writer(outfile)
    writer.writerows(output)




