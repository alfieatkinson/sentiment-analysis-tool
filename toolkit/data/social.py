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
    
    def search_subs(self, subs: dict[str, list[str]], n: int = 20) -> 'pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]':
        scrape_comments = toolkit.get_config('scrape_comments')
        posts = []
        comments = []
        for sub, search_terms in subs.items():
            temp_posts, temp_comments = self.search_sub(sub, search_terms, n)
            posts += temp_posts
            comments += temp_comments
            toolkit.console(f"'{', '.join(search_terms)} in /r/{sub}\nPosts: {len(posts)}\nComments: {len(comments)}")

        print(f"POSTS: {len(posts)}\nCOMMENTS: {sum(map(len, comments))}")

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
    
    def search_sub(self, sub: str, search_terms: list[str], n: int) -> tuple[list, list]:
        toolkit.console(f"Searching posts in /r/{sub}.")
        scrape_comments = toolkit.get_config('scrape_comments')
        posts = []
        comments = []
        try:
            if search_terms:
                for search_term in search_terms:
                    for post in self.api.subreddit(sub).search(search_term):
                        posts.append(post)
                        if scrape_comments:
                            post.comment_sort = 'hot'
                            comments.append(self.search_comments(post))
            else:
                for post in self.api.subreddit(sub).hot(limit=n):
                    posts.append(post)
                    if scrape_comments:
                        post.comment_sort = 'hot'
                        comments.append(self.search_comments(post))
        except Exception as e:
            toolkit.error(f"Error scraping subreddit: {sub}. {e}.")

        return posts, comments
    
    def search_comments(self, post, limit: int = 5) -> list:
        toolkit.console(f"Searching comments in post {post.name}.")
        post_comments = []
        post.comment_limit = limit
        post.comments.replace_more(limit=0) # Remove all MoreComments
        for comment in post.comments:
            post_comments.append(comment)
        return post_comments