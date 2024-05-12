import sys
import textwrap

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
    app = QApplication([])
    window = toolkit.MainWindow()
    window.show()
    app.exec()

if __name__ == "__main__":
    main()
