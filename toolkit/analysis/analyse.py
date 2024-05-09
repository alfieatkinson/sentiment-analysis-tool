import numpy as np

import pandas as pd
from pandas.tseries.offsets import Day, Hour

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from datetime import datetime, timezone

import toolkit

class Analyser(object):
    def __init__(self, dataset: pd.DataFrame) -> None:
        self.dataset = dataset
        self._set_sentiment_score(toolkit.get_config('score_weighting'))

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

    def _trim_dataset(self, start_date: float, end_date: float) -> pd.DataFrame:
        if (start_date is not None) and (end_date is not None):
            df = self.dataset[(self.dataset['Date/Time'] >= start_date) & (self.dataset['Date/Time'] <= end_date)].copy()
        elif start_date is not None:
            df = self.dataset[self.dataset['Date/Time'] >= start_date].copy()
        elif end_date is not None:
            df = self.dataset[self.dataset['Date/Time'] <= end_date].copy()
        else:
            df = self.dataset.copy()
        
        return df

    def generate_line(self, canvas: FigureCanvas, title: str, labels: tuple[str, str], start_date: float, end_date: float) -> None:
        df = self._trim_dataset(start_date, end_date).copy()

        canvas.axes.cla() # Clear the canvas

        if df.empty:
            canvas.axes.text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center')
            canvas.axes.set_title(title)
            canvas.axes.set_xlabel(labels[0])
            canvas.axes.set_ylabel(labels[1])
            return

        df['Date/Time'] = pd.to_datetime(df['Date/Time'], unit='s') # Convert float to datetime
        df['Date'] = df['Date/Time'].dt.date # Extract date only

        if toolkit.get_config('split_subs'):
            df = df.groupby(['Date', 'Subreddit'])['SentimentScore'].sum().reset_index()
            subgroups = df.groupby('Subreddit')
            for name, group in subgroups:
                canvas.axes.plot(group['Date'], group['SentimentScore'], label=f"/r/{name}")
            canvas.axes.legend()
        else:
            df = df.groupby('Date')['SentimentScore'].sum().reset_index()
            canvas.axes.plot(df['Date'], df['SentimentScore'])

        canvas.axes.set_title(title)
        canvas.axes.set_xlabel(labels[0])
        canvas.axes.set_ylabel(labels[1])
        canvas.axes.tick_params(axis='x', rotation=45)

    def generate_pie(self, canvas: FigureCanvas, title: str, start_date: float = None, end_date: float = None) -> None:
        df = self._trim_dataset(start_date, end_date)

        canvas.axes.cla() # Clear the canvas

        if df.empty:
            canvas.axes.text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center')
            canvas.axes.set_title(title)
            return

        total_positive = abs(df[df['Sentiment'] == 'Positive']['SentimentScore'].sum())
        total_negative = abs(df[df['Sentiment'] == 'Negative']['SentimentScore'].sum())

        labels = ['Positive', 'Negative']
        sizes = [total_positive, total_negative]

        canvas.axes.set_title(title)
        canvas.axes.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        canvas.axes.axis('equal') # Equal aspect ratio ensures that the pie is drawn as a circle

    def generate_bar(self, canvas: FigureCanvas, title: str, labels: tuple[str, str], start_date: float = None, end_date: float = None) -> None:
        df = self._trim_dataset(start_date, end_date)

        canvas.axes.cla() # Clear the canvas

        if df.empty:
            canvas.axes.text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center')
            canvas.axes.set_title(title)
            canvas.axes.set_xlabel(labels[0])
            canvas.axes.set_ylabel(labels[1])
            return

        sub_sentiments = df.groupby('Subreddit')['SentimentScore'].sum()
        sub_names = [f'/r/{name}' for name in sub_sentiments.index]

        canvas.axes.set_title(title)
        sub_sentiments.plot(kind='bar', ax=canvas.axes)
        canvas.axes.set_xlabel(labels[0])
        canvas.axes.set_ylabel(labels[1])
        canvas.axes.set_xticklabels(sub_names, rotation=45, ha='right')

    def generate_summary(self) -> None:
        return self.dataset.describe()

