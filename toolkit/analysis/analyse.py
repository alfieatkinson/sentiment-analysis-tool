import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime, timezone

import toolkit

class Analyser(object):
    def __init__(self, dataset: pd.DataFrame) -> None:
        self.dataset = dataset
        self._set_sentiment_score(toolkit.get_score_weighting())

    def _sentiment_to_int(self, sentiment: str) -> int:
        if sentiment == 'Positive':
            return 1
        elif sentiment == 'Negative':
            return -1
        else:
            return 0
        
    def _set_sentiment_score(self, weighted: bool) -> None:
        if weighted:
            self.dataset['SentimentScore'] = self.dataset['Score'] * self.dataset['Sentiment'].apply(self._sentiment_to_int)
        else:
            self.dataset['SentimentScore'] = self.dataset['Sentiment'].apply(self._sentiment_to_int)

    def _save_graph(self, name: str, extension: str = 'png') -> None:
        plt.savefig(toolkit.get_dir() + f'src/figures/{name}.{extension}', format=extension)
        plt.clf()

    def _trim_dataset(self, start_date: float, end_date: float) -> pd.DataFrame:
        if (start_date is not None) and (end_date is not None):
            df = self.dataset[(self.dataset['Date/Time'] >= start_date) & (self.dataset['Date/Time'] <= end_date)]
        elif start_date is not None:
            df = self.dataset[self.dataset['Date/Time'] >= start_date]
        elif end_date is not None:
            df = self.dataset[self.dataset['Date/Time'] <= end_date]
        else:
            df = self.dataset.copy()
        
        return df

    def generate_line(self, title: str, labels: tuple[str, str], start_date: float = None, end_date: float = None, split_subs: bool = False) -> None:
        df = self._trim_dataset(start_date, end_date)

        df['Date/Time'] = pd.to_datetime(df['Date/Time'], unit='s') # Convert float to datetime
        df['Date'] = df['Date/Time'].dt.date # Extract date only

        if split_subs:
            df = df.groupby(['Date', 'Subreddit'])['SentimentScore'].sum().reset_index()
            subgroups = df.groupby('Subreddit')
            for name, group in subgroups:
                plt.plot(group['Date'], group['SentimentScore'], label=f"/r/{name}")
            plt.legend()
        else:
            df = df.groupby('Date')['SentimentScore'].sum().reset_index()
            plt.plot(df['Date'], df['SentimentScore'])

        plt.title(title)
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.xticks(rotation=45)
        plt.tight_layout()

        self._save_graph('line-chart')

    def generate_pie(self, title: str, start_date: float = None, end_date: float = None) -> None:
        df = self._trim_dataset(start_date, end_date)

        total_positive = abs(df[df['Sentiment'] == 'Positive']['SentimentScore'].sum())
        total_negative = abs(df[df['Sentiment'] == 'Negative']['SentimentScore'].sum())

        plt.figure(figsize=(8, 8))
        labels = ['Positive', 'Negative']
        sizes = [total_positive, total_negative]
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.title(title)
        plt.axis('equal') # Equal aspect ratio ensures that the pie is drawn as a circle

        self._save_graph('pie-chart')

    def generate_bar(self, title: str, labels: tuple[str, str], start_date: float = None, end_date: float = None) -> None:
        df = self._trim_dataset(start_date, end_date)

        sub_sentiments = df.groupby('Subreddit')['SentimentScore'].sum()
        sub_names = [f'/r/{name}' for name in sub_sentiments.index]

        plt.figure(figsize=(10, 6))
        sub_sentiments.plot(kind='bar')
        plt.title(title)
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.xticks(range(len(sub_names)), sub_names, rotation=45)
        plt.tight_layout()

        self._save_graph('bar-chart')

    def generate_summary(self) -> None:
        return self.dataset.describe()

