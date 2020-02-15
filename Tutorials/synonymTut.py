from nltk.corpus import wordnet

synonym=[]
for syn in wordnet.synsets('hi'):
    for lemma in syn.lemmas():
        synonym.append(lemma.name())
print(synonym)
