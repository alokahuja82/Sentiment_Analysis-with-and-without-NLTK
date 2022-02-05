import string
from collections import Counter
import matplotlib.pyplot as plt
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Opening the file for the program to read
text = open('read.txt', encoding='utf-8').read()

# Converting the text to lower case
lower_case = text.lower()

# Cleaning the text and making it free of punctuations
cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))

# Tokenization
tokenized_words = word_tokenize(cleaned_text, "english")

final_words = []
for word in tokenized_words:
    if word not in stopwords.words("english"):
        final_words.append(word)
# print(final_words)

# NLP emotion algorithm

emotion_list = []
# Opening the dataset
with open('emotions_dataset.txt', 'r') as file:
    for line in file:
        clean_line = line.replace('\n', '').replace(',', '').replace("'","").strip()  # Removing blank lines, commas and single quotes
        word, emotion = clean_line.split(":")

        if word in final_words:
            emotion_list.append(emotion)
print(emotion_list)
w = Counter(emotion_list)  # using counter to calculate the dominant emotion
print(w)


# Sentiment Analyser
def sentiment_analyser(sentiment_text):
    emotion_score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    neg = emotion_score['neg']
    pos = emotion_score['pos']
    if neg > pos:
        print("Negative")
    if pos > neg:
        print("Positive")
    else:
        print("Neutral")


# Analyser function call
sentiment_analyser(cleaned_text)

# Plotting the emotions
fig, ax1 = plt.subplots()
ax1.bar(w.keys(), w.values())
fig.autofmt_xdate()  # to adjust the values accordingly on x axis for large amount of emotions
plt.savefig("Emotions_plot.png")
plt.show()
