import textwrap

import pandas as pd

import toolkit

class PostCollector(object):
    def __init__(self, model, scraper):
        self.model = model
        self.scraper = scraper
        self.path = f'{toolkit.get_dir()}/src/profiles/'
        self.posts = pd.DataFrame(columns=['ID', 'Subreddit', 'Date/Time', 'Title', 'Body', 'Score', 'Comments' 'Sentiment'])
        self.comments = pd.DataFrame(columns=['ID', 'PostID', 'Subreddit', 'Date/Time', 'Body', 'Score', 'Sentiment'])
        if not self.from_json():
            self.scrape_posts()

    def to_json(self) -> bool:
        try:
            self.posts.to_json(f'{self.path}/posts.json')
            self.comments.to_json(f'{self.path}/comments.json')
            return True
        except:
            return False

    def from_json(self) -> bool:
        try:
            self.posts = pd.read_json(f'{self.path}/posts.json')
            self.comments = pd.read_json(f'{self.path}/comments.json')
            self.posts['Comments'] = self.posts['Comments'].map(lambda x: tuple(x))
            return True
        except:
            return False

    def get_record(self, df: pd.DataFrame, ID: str, column: str) -> any:
        try:
            return df.loc[df['ID'] == ID, column].item()
        except:
            toolkit.error(f"Could not get {column} from {ID} in {df}")
            return None

    def scrape_posts(self, subreddits: dict[str, str]) -> None:
        scrape_comments = toolkit.get_config('scrape_comments')

        if scrape_comments:
            new_posts, new_comments = self.scraper.search_subreddits(subreddits, scrape_comments=True) # Scrape the subreddits for new posts and comments
            print(new_posts)
            print(new_comments)
        else:
            new_posts = self.scraper.search_subreddits(subreddits) # Scrape the subreddits for new posts
            print(new_posts)

        sentiment = [] # Initialise list for sentiment
        for ID in new_posts['ID']:
            text = f"{self.get_record(new_posts, ID, 'Title')} {self.get_record(new_posts, ID, 'Body')}" # Merge title and body of each post
            prediction = self.model.predict(text) # Predict using the combined text
            sentiment.append(prediction) # Add the predicted sentiment to list
        new_posts.insert(6, 'Sentiment', sentiment, True) # Insert sentiment column into new_posts

        self.posts = pd.concat([self.posts, new_posts], ignore_index=True) # Add new_posts to self.posts
        self.posts.drop_duplicates(keep='last', inplace=True) # Drop exact duplicates, keep the last occurrence 
        self.posts.drop_duplicates('ID', keep='last', inplace=True) # Drop ID duplicates, keep the last occurrence

        if scrape_comments:
            sentiment = [] # Initialise list for sentiment
            for ID in new_comments['ID']:
                text = self.get_record(new_comments, ID, 'Body') # Get the body of the comment
                prediction = self.model.predict(text) # Predict using the body
                sentiment.append(prediction) # Add the predicted sentiment to list
            new_comments.insert(6, 'Sentiment', sentiment, True) # Insert sentiment column into new_comments

            self.comments = pd.concat([self.comments, new_comments], ignore_index=True) # Add new_comments to self.comments
            self.comments.drop_duplicates(keep='last', inplace=True) # Drop exact duplicates, keep the last occurrence
            self.comments.drop_duplicates('ID', keep='last', inplace=True) # Drop ID duplicates, keep the last occurrence

        if self.to_json():
            print("Successfully scraped new posts.")
        else:
            toolkit.alert("Could not scrape new posts.")
            

    def show_post(self, ID: str, show_comments: bool = False) -> None:
        sub = self.get_record(self.posts, ID, 'Subreddit')
        title = self.get_record(self.posts, ID, 'Title')
        body = self.get_record(self.posts, ID, 'Body')
        sentiment = self.get_record(self.posts, ID, 'Sentiment')

        print()
        print("============================================================")
        print(f"\033[1m/r/{sub.upper()}\033[0m")
        print("============================================================")
        print(f"\033[1m{textwrap.fill(title.upper(), 60)}\033[0m")
        print()
        print(f"{textwrap.fill(body, 60)}")
        print()
        print(f"\033[1mPREDICTION: {sentiment.upper()}\033[0m")
        print("============================================================")

        if show_comments:
            print(f"    \033[1mCOMMENTS\033[0m")
            for CID in self.get_record(self.posts, ID, 'Comments'):
                body = self.get_record(self.comments, CID, 'Body')
                sentiment = self.get_record(self.comments, CID, 'Sentiment')

                print()
                print("    ------------------------------------------------------------")
                print(f"{textwrap.indent(textwrap.fill(body, 60), '    ')}")
                print()
                print(f"    \033[1mPREDICTION: {sentiment.upper()}\033[0m")
                print("    ------------------------------------------------------------")

    def show_posts(self, show_comments: bool = False) -> None:
        for ID in self.posts['ID']:
            self.show_post(ID, show_comments=show_comments)

    def merge_data(self) -> pd.DataFrame:
        filtered_posts = self.posts[['ID', 'Subreddit', 'Date/Time', 'Score', 'Sentiment']]
        filtered_comments = self.comments[['ID', 'Subreddit', 'Date/Time', 'Score', 'Sentiment']]

        filtered_posts.columns = ['ID', 'Subreddit', 'Date/Time', 'Score', 'Sentiment']
        filtered_comments.columns = ['ID', 'Subreddit', 'Date/Time', 'Score', 'Sentiment']

        merged_df = pd.concat([filtered_posts, filtered_comments], ignore_index=True)
        return merged_df
