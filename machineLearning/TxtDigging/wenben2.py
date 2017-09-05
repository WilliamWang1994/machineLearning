from gensim import corpora ,models,similarities
import jieba
from collections import defaultdict
doc1="E:/machineLearning/machineLearning/TxtDigging/txtNovel/d1.txt"
doc2="E:/machineLearning/machineLearning/TxtDigging/txtNovel/d2.txt"
d1=open(doc1,encoding='utf-8').read()
d2=open(doc2,encoding='utf-8').read()
data1=jieba.cut(d1)
data2=jieba.cut(d2)
data11=''
for item in data1:
    data11+=item+' '
data21=''
for item in data2:
    data21+=item+' '
documents=[data11,data21]
texts=[[word for word in document.split()]
       for document in documents]
# print (texts)
frequency=defaultdict(int)
for text in texts:
    for token in text:
        frequency[token]+=1
# print(frequency)

dictionary=corpora.Dictionary(texts)
dictionary.save("E:/machineLearning/machineLearning/TxtDigging/txtNovel/wenben2.txt")