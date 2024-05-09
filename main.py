import sys
import textwrap
from datetime import datetime

from PyQt6.QtWidgets import QApplication

import toolkit

print(f"""
         _____            _   _                      _        
        / ____|          | | (_)                    | |       
       | (___   ___ _ __ | |_ _ _ __ ___   ___ _ __ | |_      
        \___ \ / _ \ '_ \| __| | '_ ` _ \ / _ \ '_ \| __|     
        ____) |  __/ | | | |_| | | | | | |  __/ | | | |_      
       |_____/ \___|_| |_|\__|_|_| |_| |_|\___|_|_|_|\__|   _ 
     /\               | |         (_)     |__   __|        | |
    /  \   _ __   __ _| |_   _ ___ _ ___     | | ___   ___ | |
   / /\ \ | '_ \ / _` | | | | / __| / __|    | |/ _ \ / _ \| |
  / ____ \| | | | (_| | | |_| \__ \ \__ \    | | (_) | (_) | |
 /_/    \_\_| |_|\__,_|_|\__, |___/_|___/    |_|\___/ \___/|_|
                          __/ |                               
                         |___/        by Alfie Atkinson       

Running main.py {toolkit.version()}   
""")

def main() -> None:
    #M = toolkit.BertModel()
    #X = toolkit.XScraper()
    #R = toolkit.RedditScraper()
    #C = toolkit.PostCollector(M, R)

    #C.scrape_posts(scrape_comments=True)
    #C.show_posts(show_comments=True)

    #A = toolkit.Analyser(C.merge_data())

    #A.generate_line("Sentiment Over Time", ('Date', 'Sentiment'))
    #A.generate_pie("Overall Sentiment")
    #A.generate_bar("Subreddit Sentiment", ('Subreddit', 'Sentiment'))
    #A.generate_line("Sentiment Over Time", ('Date', 'Sentiment'), start_date=datetime(2024, 5, 1, 14, 53).timestamp(), split_subs=True)

    app = QApplication([])
    window = toolkit.MainWindow()
    window.show()
    app.exec()

    while False:
        text = input("\nEnter text to analyse (empty input closes the program): ")

        if text == '':
            running = False
            break
        result = M.predict([text])
        print(f"Your statement was: {result}")

if __name__ == "__main__":
    main()
