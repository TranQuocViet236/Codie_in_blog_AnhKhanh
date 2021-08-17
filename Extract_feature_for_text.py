from functools import reduce
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# Assume that a texts have three sentences
texts = [['i', 'have', 'a', 'cat'],
         ['he', 'have', 'a', 'dog'],
         ['he', 'and', 'i', 'have', 'a', 'cat', 'and', 'a', 'dog']]

#Create a dictionary from three above sentences
dictionary = list(enumerate(set(reduce(lambda  x,y: x+y, texts))))
#Dictionnary contains set of all of words of texts

def bag_of_word(sentence):
    #Initialize a vector have its length is len of dictionary
    vector = np.zeros(len(dictionary))

    #Count words of sentence in dictionary
    for i, word in dictionary:
        count = 0
        #Count number of words in a sentence
        for w in sentence:
            if w == word:
                count += 1
        vector[i] = count
    return vector

for i in texts:
    print(bag_of_word(i))
'''
The disadvantage of representing bag of words is that we cannot distinguish two sentences
where  the words in the sentence are the same but have completely different meanings
For instance: "No, you have no dog" and " you have no dog"
Therefore, N-gram method will be used to solve that problem
'''

# def CountVectorizer():
#     vect = CountVectorizer((ngram_range=(1,1)))
#     vect = vect.fit_transform(['you have no dog', 'no, you have dog']).toarray()
#     return vect
# print(CountVectorizer())

'''
The approach TF-IDF(Term Frequency- Inverse Document Frequency) help us judge the importance of words base on frequency of occurrence
'''
def Tfidf():
    corpus = [
        'tôi thích ăn bánh mì nhân thịt',
        'cô ấy thích ăn bánh mì, còn tôi thích ăn xôi',
        'thị trường chứng khoán giảm làm tôi lo lắng',
        'chứng khoán sẽ phục hồi vào thời gian tới. danh mục của tôi sẽ tăng trở lại',
        'dự báo thời tiết hà nội có mưa vào chiều và tối. tôi sẽ mang ô khi ra ngoài'
    ]
    #initilize model to compute tfidf for each word
    # Parameter max_df to remove stopwords appear at more than 90% sentences
    vectorizer = TfidfVectorizer(max_df=0.9)

    #tokenize sentences by tfidf
    X = vectorizer.fit_transform(corpus)
    print('words in dictionary:')
    print(X)
    print(vectorizer.get_feature_names())
    print(f"X shape: {X.shape}")

Tfidf()

'''
Word2Vec model
king - man + woman = queen
'''

