import pandas as pd
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

#read exported date to pandas

df=pd.read_csv('C:\\Users\\huawei\\Documents\\whatsapp_chat.txt',sep='\t',header=None,names=['text'])

#remove metadata and keep only the msgs

df['text']=df['text'].str.replace('[\.,*\]','')

#remove empty messages

df=df[df['text'].str.strip().astype(bool)]

#initialize sentiment analysis
sia=SentimentIntensityAnalyzer()

#apply sentiment analysis to each msgs
df['sentiment_scores']=df['text'].apply (lambda text:sia.polarity_scores(text))

df['compound_score']=df['sentiment_scores'].apply(lambda score:score['compound'])
df['positive_score']=df['sentiment_scores'].apply(lambda score:score['pos'])
df['negative_score']=df['sentiment_scores'].apply(lambda score:score['neg'])
df['neutral_score']=df['sentiment_scores'].apply(lambda score:score['neu'])

import matplotlib.pyplot as plt


# Plot sentiment scores over time
plt.plot(df.index, df['compound_score'], label='Compound')
plt.plot(df.index, df['positive_score'], label='Positive')
plt.plot(df.index, df['negative_score'], label='Negative')
plt.plot(df.index, df['neutral_score'], label='Neutral')


# Set plot labels and legend
plt.xlabel('Message Index')
plt.ylabel('sentiment Scores')
plt.legend()

# Show the plot
plt.show()


