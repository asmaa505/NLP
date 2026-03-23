

#? downloads
import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('maxent_ne_chunker_tab')
nltk.download('words')



#!  word tokenization
# import nltk
# from nltk.tokenize import word_tokenize

# text = "asmaa is a beautiful girl , she loves music , dance , philosophy and sport"

# tokens = word_tokenize(text)
# print(tokens)



#!  sentance tokenization
# import nltk
# from nltk.tokenize import sent_tokenize
# paragraph = "asmaa is a beautiful girl , she loves music , dance , philosophy and sport . asmaa is a beautiful girl , she loves music , dance , philosophy and sport .asmaa is a beautiful girl , she loves music , dance , philosophy and sport .asmaa is a beautiful girl , she loves music , dance , philosophy and sport."

# sentances = sent_tokenize(paragraph)

# print(sentances)


#! POS tagging
## POS == part of speech

# import nltk
# from nltk import word_tokenize , pos_tag

# text = "asmaa is a beautiful girl , she loves music , dance , philosophy and sport"

# tokens = word_tokenize(text)

# pos_tagging = pos_tag(tokens)

# print(pos_tagging)



#! NER ==  named entity recognition
import nltk
from nltk import word_tokenize , pos_tag , ne_chunk

text = "asmaa is a beautiful girl , she loves music , dance , philosophy and sport"

# word toknization
tokens = word_tokenize(text)

# POS tagging
pos_tagging = pos_tag(tokens)

# NER
neRecognition = ne_chunk(pos_tagging)

print(neRecognition)
