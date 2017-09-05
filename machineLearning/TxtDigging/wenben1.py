import jieba
sentence="我喜欢上海东方明珠"
#全模式
w1=jieba.cut(sentence,cut_all=True )
for item in w1:
    print(item)
print("--------------")
#精准模式
w2=jieba.cut(sentence)
for item in w2:
    print(item)
print("--------------")
#搜索引擎模式
sentence2=""
w3=jieba.cut_for_search(sentence)
for item in w3:
    print(item)
print("--------------")
#词性标注
"""
a:形容词
c:连词
d:副词
e:叹词 
f:方位词
i：成语
m:数词
n：名词
nr：人名
ns：地名
nt：机构团体
nz：其他专业名词
r:代词
p:介词
t：时间
u：助词
v:动词
vn：动名词
w：标点符号
un：未知词语
"""
import jieba.posseg
w4=jieba.posseg.cut(sentence)
for item in w4:
    print(item.word+"--"+item.flag)
#加载用户自定义词典
# jieba.load_userdict("E:/python3.5.3/Lib/site-packages/jieba/dict.txt")

#更改词频
sentence2="王成龙是做机器学习的"
w5=jieba.cut(sentence2)
for item in w5:
    print(item)
print("--------------")
#提取关键词
import jieba.analyse
tag=jieba.analyse.extract_tags(sentence2,4)
print(tag)

#返回词语位置
w6=jieba.tokenize(sentence)
for item in w6:
    print(item)