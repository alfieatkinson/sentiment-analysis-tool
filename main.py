import toolkit
import textwrap

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
    M = toolkit.BertModel()
    #X = toolkit.XScraper()
    R = toolkit.RedditScraper()
    C = toolkit.PostCollector(M, R)

    C.scrape_posts(scrape_comments=True)
    C.show_posts(show_comments=True)

    while False:
        text = input("\nEnter text to analyse (empty input closes the program): ")

        if text == '':
            running = False
            break
        result = M.predict([text])
        print(f"Your statement was: {result}")

if __name__ == "__main__":
    main()
