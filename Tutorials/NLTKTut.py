from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import wordnet
text="My name is Rahul Golani. I am a Computer Science Graduate!"
a=sent_tokenize(text)
b=word_tokenize(text)
print(a)
print(b)
syn=wordnet.synsets('love')
print(syn)
