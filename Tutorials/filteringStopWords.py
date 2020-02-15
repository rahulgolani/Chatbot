from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

text="My name is Rahul Golani. I am a Computer Science Graduate."
filterWords=[]
stopwords=set(stopwords.words('english'))
words=word_tokenize(text)
for word in words:
    if word not in stopwords:
        filterWords.append(word)

print(filterWords)
