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
    """
    Class for analyzing sentiment data.

    Attributes:
        dataset (pd.DataFrame): DataFrame containing sentiment data.
    """
    def __init__(self, dataset: pd.DataFrame) -> None:
        """
        initialises the Analyser object.

        Args:
            dataset (pd.DataFrame): DataFrame containing sentiment data.
        """
        self.dataset = dataset
        self._set_sentiment_score()

    def _sentiment_to_int(self, sentiment: str) -> int:
        """
        Converts sentiment strings to integers.

        Args:
            sentiment (str): Sentiment string.

        Returns:
            int: Integer representation of sentiment.
        """
        if sentiment == 'Positive':
            return 1
        elif sentiment == 'Negative':
            return -1
        else:
            return 0
        
    def _set_sentiment_score(self) -> None:
        """
        Sets sentiment scores in the dataset.
        """
        if toolkit.get_config('score_weighting'):
            self.dataset['SentimentScore'] = self.dataset['Score'] * self.dataset['Sentiment'].apply(self._sentiment_to_int)
        else:
            self.dataset['SentimentScore'] = self.dataset['Sentiment'].apply(self._sentiment_to_int)

    def _trim_dataset(self, start_date: float, end_date: float) -> pd.DataFrame:
        """
        Trims the dataset based on start and end dates.

        Args:
            start_date (float): Start date.
            end_date (float): End date.

        Returns:
            pd.DataFrame: Trimmed DataFrame.
        """
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
        """
        Generates a line plot.

        Args:
            canvas (FigureCanvas): Figure canvas for plotting.
            title (str): Title of the plot.
            labels (tuple[str, str]): Labels for the x and y axes.
            start_date (float): Start date for data selection.
            end_date (float): End date for data selection.
        """
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
        """
        Generates a pie chart.

        Args:
            canvas (FigureCanvas): Figure canvas for plotting.
            title (str): Title of the plot.
            start_date (float, optional): Start date for data selection.
            end_date (float, optional): End date for data selection.
        """
        df = self._trim_dataset(start_date, end_date)

        canvas.axes.cla() # Clear the canvas

        if df.empty:
            canvas.axes.text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center')
            canvas.axes.set_title(title)
            return

        sentiment_counts = df['Sentiment'].value_counts()

        labels = sentiment_counts.index.tolist()
        sizes = sentiment_counts.values.tolist()

        canvas.axes.set_title(title)
        canvas.axes.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        canvas.axes.axis('equal') # Equal aspect ratio ensures that the pie is drawn as a circle

    def generate_bar(self, canvas: FigureCanvas, title: str, labels: tuple[str, str], start_date: float = None, end_date: float = None) -> None:
        """
        Generates a bar chart.

        Args:
            canvas (FigureCanvas): Figure canvas for plotting.
            title (str): Title of the plot.
            labels (tuple[str, str]): Labels for the x and y axes.
            start_date (float, optional): Start date for data selection.
            end_date (float, optional): End date for data selection.
        """
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
        """
        Generates a summary of the dataset.

        Returns:
            None
        """
        return self.dataset.describe()
    
    def update_dataset(self, dataset: pd.DataFrame) -> None:
        """
        Updates the dataset.

        Args:
            dataset (pd.DataFrame): New DataFrame containing sentiment data.
        """
        self.dataset = dataset
        self._set_sentiment_score()