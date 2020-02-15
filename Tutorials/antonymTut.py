from nltk.corpus import wordnet
synonym=[]
antonym=[]
for syn in wordnet.synsets('happy'):
    for lemma in syn.lemmas():
        synonym.append(lemma.name())

for syn in wordnet.synsets('happy'):
    for lemma in syn.lemmas():
        #for ant in lemma.antonyms():
        if lemma.antonyms():
            antonym.append(lemma.antonyms()[0].name())

print(synonym)
print(antonym)
