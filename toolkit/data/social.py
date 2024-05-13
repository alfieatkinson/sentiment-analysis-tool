import os
import time

import praw
import tweepy
import pandas as pd

import toolkit

class XScraper(object):
    """
    Class for scraping data from external sources.
    Non-functional due to API limitations.

    Attributes:
        auth: OAuth handler for Twitter.
        api: Tweepy API instance for accessing Twitter data.
    """
    def __init__(self):
        """
        initialises the XScraper class.
        """
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
        """
        Searches Twitter for tweets based on the given query.

        Args:
            query (str): The search query.

        Returns:
            pd.DataFrame: DataFrame containing the search results.
        """
        tweets = self.api.search_tweets(q=query)
        attributes = [[tweet.created_at, tweet.text, tweet.favorite_count] for tweet in tweets]
        columns = ['Date', 'Text', 'Likes']

        return pd.DataFrame(attributes, columns=columns)
    
    def search_user(self, username: str, n: int) -> pd.DataFrame:
        """
        Searches Twitter for tweets by a specific user.

        Args:
            username (str): The Twitter username.
            n (int): The number of tweets to retrieve.

        Returns:
            pd.DataFrame: DataFrame containing the search results.
        """
        tweets = self.api.user_timeline(screen_name=username, count=n)
        attributes = [[tweet.created_at, tweet.text, tweet.favorite_count] for tweet in tweets]
        columns = ['Date', 'Text', 'Likes']

        return pd.DataFrame(attributes, columns=columns)
    

class RedditScraper(object):
    """
    Class for scraping data from Reddit.

    Attributes:
        api: Reddit API instance for accessing Reddit data.
    """
    def __init__(self):
        """
        initialises the RedditScraper class.
        """
        self.api = praw.Reddit(client_id=os.getenv('REDDIT_CLIENT_ID'), # Fetch the client ID
                               client_secret=os.getenv('REDDIT_CLIENT_SECRET'), # Fetch the client secret
                               user_agent=os.getenv('REDDIT_USER_AGENT')) # Fetch the user agent
    
    def search_subs(self, subs: dict[str, list[str]], n: int = 10) -> 'pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]':
        """
        Searches Reddit for posts and comments in specified subreddits.

        Args:
            subs (dict): Dictionary containing subreddit names and search terms.
            n (int): The number of posts to retrieve per subreddit.

        Returns:
            pd.DataFrame or tuple[pd.DataFrame, pd.DataFrame]: DataFrame(s) containing the search results.
        """
        scrape_comments = toolkit.get_config('scrape_comments')
        posts = []
        comments = []
        for sub, search_terms in subs.items():
            temp_posts, temp_comments = self.search_sub(sub, search_terms, n)
            posts += temp_posts
            comments += temp_comments
            toolkit.console(f"'{', '.join(search_terms)} in /r/{sub} Posts: {len(posts)} Comments: {len(comments)}")

        toolkit.console(f"POSTS: {len(posts)} COMMENTS: {sum(map(len, comments))}")

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
    
    def search_sub(self, sub: str, search_terms: list[str], n: int = 10) -> tuple[list, list]:
        """
        Searches Reddit for posts and comments in a specific subreddit.

        Args:
            sub (str): The subreddit to search.
            search_terms (list): List of search terms.
            n (int): The number of posts to retrieve.

        Returns:
            tuple[list, list]: Lists of posts and comments.
        """
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
        """
        Searches for comments in a Reddit post.

        Args:
            post: The Reddit post object.
            limit (int): The maximum number of comments to retrieve.

        Returns:
            list: List of comments.
        """
        toolkit.console(f"Searching comments in post {post.name}.")
        post_comments = []
        post.comment_limit = limit
        post.comments.replace_more(limit=0) # Remove all MoreComments
        for comment in post.comments:
            post_comments.append(comment)
        return post_comments