import textwrap

import pandas as pd

import toolkit

class PostCollector(object):
    """
    Class for collecting and managing posts and comments data.

    Attributes:
        model: The sentiment analysis model.
        scraper: The web scraper for collecting posts and comments.
        profile (dict): Profile information containing user details.
        path (str): File path to store collected data.
        posts (DataFrame): DataFrame to store posts data.
        comments (DataFrame): DataFrame to store comments data.
    """
    def __init__(self, model, scraper, profile: dict[str, any]) -> None:
        """
        initialise the PostCollector object.

        Args:
            model: The sentiment analysis model.
            scraper: The web scraper for collecting posts and comments.
            profile (dict): Profile information containing user details.
        """
        self.model = model
        self.scraper = scraper
        self.profile = profile
        self.path = f'{toolkit.get_dir()}/src/profiles/{self.profile["id"]}'
        self.posts = pd.DataFrame(columns=['ID', 'Subreddit', 'Date/Time', 'Title', 'Body', 'Score', 'Comments', 'Sentiment'])
        self.comments = pd.DataFrame(columns=['ID', 'PostID', 'Subreddit', 'Date/Time', 'Body', 'Score', 'Sentiment'])
        if not self.from_json():
            pass

    def to_json(self) -> bool:
        """
        Save collected data to JSON files.

        Returns:
            bool: True if saving is successful, False otherwise.
        """
        try:
            self.posts.to_json(f'{self.path}/posts.json')
            self.comments.to_json(f'{self.path}/comments.json')
            return True
        except:
            return False

    def from_json(self) -> bool:
        """
        Load collected data from JSON files.

        Returns:
            bool: True if loading is successful, False otherwise.
        """
        try:
            self.posts = pd.read_json(f'{self.path}/posts.json')
            self.comments = pd.read_json(f'{self.path}/comments.json')
            self.posts['Comments'] = self.posts['Comments'].map(lambda x: tuple(x))
            return True
        except:
            return False

    def get_record(self, df: pd.DataFrame, ID: str, column: str) -> any:
        """
        Get a specific record from a DataFrame.

        Args:
            df (DataFrame): DataFrame to search for the record.
            ID (str): ID of the record to retrieve.
            column (str): Name of the column to retrieve the record from.

        Returns:
            any: The value of the specified record.
        """
        try:
            record = df.loc[df['ID'] == ID, column]
            if len(record) == 1:
                return record.item()
            elif len(record) > 1:
                # Handle case where there are multiple records with the same ID
                toolkit.error(f"Found multiple records with ID: {ID}")
                return None
            else:
                # Handle case where there are no records with the specified ID
                toolkit.error(f"No record found with ID: {ID}")
                return None
        except Exception as e:
            toolkit.error(f"Could not get {column} from {ID}: {e}")
            return None

    def scrape_posts(self, n: int) -> None:
        """
        Scrape posts and comments data from specified subreddits.

        Args:
            n (int): Number of posts to scrape.
        """
        text_processor = toolkit.TextProcessor()

        subs = self.profile['subs']

        scrape_comments = toolkit.get_config('scrape_comments')
        if scrape_comments:
            new_posts, new_comments = self.scraper.search_subs(subs, n=n) # Scrape the subreddits for new posts and comments
            print(new_posts)
            print(new_comments)
        else:
            new_posts = self.scraper.search_subs(subs, n=n) # Scrape the subreddits for new posts
            print(new_posts)

        all_texts = []

        for ID in new_posts['ID']:
            title = self.get_record(new_posts, ID, 'Title')
            body = self.get_record(new_posts, ID, 'Body')

            if title is None or body is None:
                index = new_posts.index[new_posts['ID'] == ID]
                new_posts.drop(index)
                toolkit.console(f"Dropped post {ID}")
                continue

            title = text_processor.clean(title)
            body = text_processor.clean(body)

            text = f"{title} [SEP] {body}" # Merge title and body of each post

            if not text.strip():  # Check if text is empty or contains only whitespace
                print(f"Empty text for post ID: {ID}")
            else:
                all_texts.append(text)  # Add non-empty text to list for batch prediction

        # Predict sentiment for all posts at once
        post_sentiments = self.model.predict(all_texts)

        # Insert sentiment column into new_posts
        new_posts.insert(6, 'Sentiment', post_sentiments, True)

        # Add new_posts to self.posts
        self.posts = pd.concat([self.posts, new_posts], ignore_index=True)

        # Drop duplicates
        self.posts.drop_duplicates(keep='last', inplace=True)
        self.posts.drop_duplicates('ID', keep='last', inplace=True)

        if scrape_comments:
            all_texts = []

            for ID in new_comments['ID']:
                post_id = self.get_record(new_comments, ID, 'PostID')
                
                title = self.get_record(new_posts, post_id, 'Title')
                body = self.get_record(new_posts, post_id, 'Body')
                comment = self.get_record(new_comments, ID, 'Body')

                if title is None or body is None:
                    index = new_posts.index[new_posts['ID'] == ID]
                    new_posts.drop(index)
                    toolkit.console(f"Dropped comment {ID}")
                    continue

                title = text_processor.clean(title)
                body = text_processor.clean(body)
                comment = text_processor.clean(comment)

                text = f"{title} [SEP] {body} [SEP] {comment}" # Get the body of the comment

                if not text.strip():  # Check if text is empty or contains only whitespace
                    print(f"Empty text for comment ID: {ID}")
                else:
                    all_texts.append(text)  # Add non-empty text to list for batch prediction

            # Predict sentiment for all comments at once
            comment_sentiments = self.model.predict(all_texts)

            # Insert sentiment column into new_comments
            new_comments.insert(6, 'Sentiment', comment_sentiments, True)

            # Add new_comments to self.comments
            self.comments = pd.concat([self.comments, new_comments], ignore_index=True)

            # Drop duplicates
            self.comments.drop_duplicates(keep='last', inplace=True)
            self.comments.drop_duplicates('ID', keep='last', inplace=True)

        if self.to_json():
            print("Successfully scraped new posts.")
        else:
            toolkit.alert("Could not scrape new posts.")
            

    def show_post(self, ID: str, show_comments: bool = False) -> None:
        """
        Display information about a specific post and its comments.

        Args:
            ID (str): ID of the post to display.
            show_comments (bool): Flag indicating whether to display comments.
        """
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
        """
        Display information about all collected posts.

        Args:
            show_comments (bool): Flag indicating whether to display comments.
        """
        for ID in self.posts['ID']:
            self.show_post(ID, show_comments=show_comments)

    def merge_data(self) -> pd.DataFrame:
        """
        Merge posts and comments data into a single DataFrame.

        Returns:
            DataFrame: Merged DataFrame containing posts and comments data.
        """
        filtered_posts = self.posts[['ID', 'Subreddit', 'Date/Time', 'Score', 'Sentiment']]
        filtered_comments = self.comments[['ID', 'Subreddit', 'Date/Time', 'Score', 'Sentiment']]

        filtered_posts.columns = ['ID', 'Subreddit', 'Date/Time', 'Score', 'Sentiment']
        filtered_comments.columns = ['ID', 'Subreddit', 'Date/Time', 'Score', 'Sentiment']

        merged_df = pd.concat([filtered_posts, filtered_comments], ignore_index=True)
        return merged_df
