import os
import time

import praw
import tweepy
import pandas as pd

import toolkit

class XScraper(object):
    def __init__(self):
        try:
            # Create the OAuth handler
            self.auth = tweepy.OAuth1UserHandler(os.getenv('TWITTER_CONSUMER_KEY'), # Fetch the consumer key
                                                 os.getenv('TWITTER_CONSUMER_SECRET'), # Fetch the consumer secret
                                                 os.getenv('TWITTER_ACCESS_TOKEN'), # Fetch the access token
                                                 os.getenv('TWITTER_ACCESS_TOKEN_SECRET')) # Fetch the access token secret
        except:
            toolkit.error("Could not load environment variables for Twitter.") 

        self.api = tweepy.API(self.auth, wait_on_rate_limit=True) # Create API instance

    def search(self, query: str) -> pd.DataFrame:
        tweets = self.api.search_tweets(q=query)
        attributes = [[tweet.created_at, tweet.text, tweet.favorite_count] for tweet in tweets]
        columns = ['Date', 'Text', 'Likes']

        return pd.DataFrame(attributes, columns=columns)
    
    def search_user(self, username: str, n: int) -> pd.DataFrame:
        tweets = self.api.user_timeline(screen_name=username, count=n)
        attributes = [[tweet.created_at, tweet.text, tweet.favorite_count] for tweet in tweets]
        columns = ['Date', 'Text', 'Likes']

        return pd.DataFrame(attributes, columns=columns)
    

class RedditScraper(object):
    def __init__(self):
        self.api = praw.Reddit(client_id=os.getenv('REDDIT_CLIENT_ID'), # Fetch the client ID
                               client_secret=os.getenv('REDDIT_CLIENT_SECRET'), # Fetch the client secret
                               user_agent=os.getenv('REDDIT_USER_AGENT')) # Fetch the user agent
        
    def search_all(self, query: str, n: int = 100) -> pd.DataFrame:
        posts = self.api.subreddit('all').search(query, sort='comments', limit=n)
        attributes = [[post.id, post.subreddit.display_name, post.created_utc, post.title, post.selftext, post.score] for post in posts if post.is_self]
        columns = ['ID', 'Subreddit', 'Date/Time', 'Title', 'Body', 'Score']

        return pd.DataFrame(attributes, columns=columns)
    
    def search_subreddits(self, subs: list[str], n: int = 1000, scrape_comments: bool = False) -> 'pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]':
        posts = []
        comments = []
        for sub in subs:
            try:
                for post in self.api.subreddit(sub).hot(limit=n):
                    posts.append(post)

                    if scrape_comments:
                        post_comments = []
                        post.comments.replace_more(limit=0) # Remove all MoreComments
                        for comment in post.comments:
                            post_comments.append(comment)
                        comments.append(post_comments)
            except:
                toolkit.alert(f"Error scraping subreddit: {sub}.")

        n = 0
        for post_comments in comments:
            n += len(post_comments)

        print(f"POSTS: {len(posts)}\nCOMMENTS: {n}")

        post_attributes = []
        comment_attributes = []
        for post in posts:
            post_comments = []
            if post.is_self:
                if scrape_comments:
                    for comment in comments[posts.index(post)]:
                        post_comments.append(comment.id)
                        comment_attributes.append([comment.id, comment.submission.id, comment.subreddit.display_name, comment.created_utc, comment.body, comment.score])
                post_attributes.append([post.id, post.subreddit.display_name, post.created_utc, post.title, post.selftext, post.score, tuple(comment for comment in post_comments)])

        columns = ['ID', 'Subreddit', 'Date/Time', 'Title', 'Body', 'Score', 'Comments']
        posts_df = pd.DataFrame(post_attributes, columns=columns)
        #posts_df['Date/Time'] = posts_df['Date/Time'].map(lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x)))
        if scrape_comments: 
            columns = ['ID', 'PostID', 'Subreddit', 'Date/Time', 'Body', 'Score']
            comments_df = pd.DataFrame(comment_attributes, columns=columns)
            return posts_df, comments_df
        return posts_df