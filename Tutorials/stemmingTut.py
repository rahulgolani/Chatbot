from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

sentence="I am enjoying writing this tutorial; I love to write and I have written 266 words so far. I wrote more than you did; I am a writer"
words=word_tokenize(sentence)
ps=PorterStemmer()
for word in words:
    print(f"{word}:{ps.stem(word)}")
