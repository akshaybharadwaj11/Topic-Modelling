from collections import defaultdict
from gensim import corpora, models, similarities
from pprint import pprint

#Ask Ubuntu dataset
txt = open('F:/python/askubuntu-master/askubuntu_body.txt',encoding='utf8')
txt_title = open('F:/python/askubuntu-master/askubuntu_title.txt',encoding='utf8')

#Reddit dataset
reddit = open('F:/softs/2008/RC_2008-01.txt',encoding='utf8')


def topic_modelling(txt):
    punctuations = '""#,.?`()!<>:'
    data = []

    stoplist = set('for a of the in and to an i how'.split())

    data = [[word for word in item.lower().split() if word not in stoplist and word not in punctuations] for item in txt]
    #print(data)
    frequency = defaultdict(int)
    for text in data:
        for token in text:
            frequency[token] += 1

    data = [[token for token in text if frequency[token] > 1]
         for text in data]
    #pprint(data)
    return data


doc1 = topic_modelling(txt)
doc2 = topic_modelling(txt_title)
doc3 = topic_modelling(reddit)

#Title data (ask ubuntu)
dictionary1 = corpora.Dictionary(doc1)
dictionary1.save('F:/python/askubuntu-master/body.dict')  
#print(dictionary1.token2id)

#Body data (ask ubuntu)
dictionary2 = corpora.Dictionary(doc2)
dictionary2.save('F:/python/askubuntu-master/title.dict')  

#Reddit data
dictionary3 = corpora.Dictionary(doc3)
dictionary3.save('F:/python/reddit.dict')  


corpus1 = [dictionary1.doc2bow(text) for text in doc1]
corpora.MmCorpus.serialize('F:/python/askubuntu-master/body.mm', corpus1)   

corpus2 = [dictionary2.doc2bow(text) for text in doc2]
corpora.MmCorpus.serialize('F:/python/askubuntu-master/title.mm', corpus2)   

corpus3 = [dictionary3.doc2bow(text) for text in doc3]
corpora.MmCorpus.serialize('F:/python/reddit.mm', corpus3)


#LSI Model
lsi = models.LsiModel(corpus3, id2word=dictionary3, num_topics=400) 
corpus_lsi = lsi[corpus3] 
#for doc in corpus_lsi:
#    print(doc)

#LDA Model
lda = models.LdaModel(corpus3, id2word=dictionary3, num_topics=100)
corpus_lda = lda[corpus3]


#similarity reddit dataset
index = similarities.MatrixSimilarity(corpus_lda)
sims = [index[item] for item in corpus_lda]
print(list(enumerate(sims)))


#HDP Model
hdp1 = models.HdpModel(corpus1, id2word=dictionary1)

hdp2 = models.HdpModel(corpus2, id2word=dictionary2)

'''
#Similarity
index = similarities.MatrixSimilarity(hdp1[corpus1])
sims = index[hdp2[corpus2]]
print(list(enumerate[sims]))
'''
