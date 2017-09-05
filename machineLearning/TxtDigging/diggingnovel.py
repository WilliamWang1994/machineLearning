import codecs
import jieba.analyse
data=open("E:/machineLearning/machineLearning/TxtDigging/txtNovel/红楼梦1.txt",encoding='utf-8').read()
tag=jieba.analyse.extract_tags(data)
print(tag)