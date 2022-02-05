import string
from collections import Counter
import matplotlib.pyplot as plt
# Opening the file for the program to read
text = open('read.txt',encoding = 'utf-8').read()

# Converting the text to lower case
lower_case= text.lower()

# Cleaning the text and making it free of punctuations
cleaned_text = lower_case.translate(str.maketrans('','',string.punctuation))

# Tokenization
tokenized_words = cleaned_text.split()

stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
              "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
              "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
              "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
              "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
              "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
              "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
              "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
              "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
              "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

final_words=[]
for word in tokenized_words:
    if word not in stop_words:
        final_words.append(word)
# print(final_words)

# NLP emotion algorithm

emotion_list=[]
# Opening the dataset
with open('emotions_dataset.txt','r') as file:
    for line in file:
        clean_line = line.replace('\n','').replace(',','').replace("'","").strip()     # Removing blank lines, commas and single quotes
        word, emotion = clean_line.split(":")

        if word in final_words:
            emotion_list.append(emotion)
print(emotion_list)
w= Counter(emotion_list)  #using counter to calculate the dominant emotion
print(w)

# Plotting the emotions
fig,ax1 = plt.subplots()
ax1.bar(w.keys(), w.values())
fig.autofmt_xdate()    # to adjust the values accordingly on xaxis for large amount of emotions
plt.savefig("Emotions_plot.png")
plt.show()