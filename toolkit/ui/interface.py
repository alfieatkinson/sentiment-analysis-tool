import os

from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from superqt import QCollapsible

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import pandas as pd

from datetime import datetime, timedelta

import traceback, sys

import toolkit

class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    progress
        int indicating % progress
    """
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)

class Worker(QRunnable):
    """
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function
    """

    def __init__(self, fn: callable, *args, **kwargs) -> None:
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        #self.kwargs['progress_callback'] = self.signals.progress
    
    @pyqtSlot()
    def run(self):
        """
        Initialise the runner function with passed args, kwargs.
        """

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs) # Execute the passed function
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result) # Return the result of the processing
        finally:
            self.signals.finished.emit() # Done

class MplCanvas(FigureCanvas):
    """
    Matplotlib canvas widget.

    Inherits from FigureCanvas for embedding matplotlib figures into PyQt6 applications.
    """
    def __init__(self, parent = None, width: int = 5, height: int = 5, dpi: int = 100):
        """
        initialise the matplotlib canvas.

        Args:
            parent: Parent widget.
            width (int): Width of the canvas.
            height (int): Height of the canvas.
            dpi (int): Dots per inch for the canvas.
        """
        fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class MainWindow(QMainWindow):
    """
    Main application window.

    Inherits from QMainWindow for creating the main application window.
    """
    def __init__(self, *args, **kwargs):
        """
        initialise the main window.
        """
        super(MainWindow, self).__init__(*args, **kwargs)

        self.threadpool = QThreadPool()
        print(f"Multithreading with maximum {self.threadpool.maxThreadCount()} threads.") 

        self.model = toolkit.BertModel()
        self.scraper = toolkit.RedditScraper()

        self.profile = toolkit.get_profile(0)

        profiles = toolkit.get_profiles()
        del profiles['next_index']

        self.profile = toolkit.get_profile(0)
        self.collector = toolkit.PostCollector(self.model, self.scraper, self.profile)
        self.analyser = toolkit.Analyser(self.collector.merge_data())

        self.data_start_date = self.collector.posts['Date/Time'].min()
        self.data_end_date = datetime.now().timestamp()

        self.line_chart = MplCanvas(self)
        self.pie_chart = MplCanvas(self)
        self.bar_chart = MplCanvas(self)

        self.font = QFont('Arial', 12)
        self.font_heading = QFont('Arial', 20)
        self.font_heading.setBold(True)

        self.window_profile_editor = None
        self.window_settings = None

        self.setWindowTitle("Sentiment Analysis Tool") # Set the title of the window

        # Get the screen width and height and use it to set window geometry
        screen_resolution = QApplication.instance().primaryScreen().size()
        screen_width = screen_resolution.width()
        screen_height = screen_resolution.height()
        width = screen_width // 2
        height = screen_height // 2
        x = screen_width - width * 1.5
        y = screen_height - height * 1.5
        self.setGeometry(x, y, width, height) # Set the window size and position
        print()
        print(f"Screen resolution is {screen_width}x{screen_height}, initialising window at {width}x{height}.")

        self.widget = QWidget(self)
        self.setCentralWidget(self.widget)

        self._make_menu_bar()

        self.layout = self._make_layout()
        
        self.widget.setLayout(self.layout)   

        self.timer = QTimer()
        self.timer.setInterval(toolkit.get_config('update_interval'))
        self.timer.timeout.connect(self._collect_new_posts)
        self.timer.start()

    def _update(self):
        """
        Update the main window layout and charts.
        """
        self.widget = QWidget(self)
        self.setCentralWidget(self.widget)
        self.layout = self._make_layout()
        self.analyser.update_dataset(self.collector.merge_data())
        self._update_charts()
        self.widget.setLayout(self.layout)

    def _make_menu_bar(self):
        """
        Create the menu bar with menu buttons.
        """
        menu = self.menuBar() # Create the menu bar

        # Create menu buttons
        # Button to create new brand profile
        button_new_brand_profile = QAction("New Brand Profile", self)
        button_new_brand_profile.setStatusTip("Creates a new brand profile.")
        button_new_brand_profile.triggered.connect(self._new_brand_profile)

        # Button to create new brand profile
        button_edit_brand_profile = QAction("Edit Brand Profile", self)
        button_edit_brand_profile.setStatusTip("Edits the current brand profile.")
        button_edit_brand_profile.triggered.connect(self._edit_brand_profile)

        # Button to train new model
        button_train = QAction("Re-Train Model", self)
        button_train.setStatusTip("Trains a new model.")
        button_train.triggered.connect(self._train_model)

        # Button to open settings window
        button_settings = QAction("Settings", self)
        button_settings.setStatusTip("Opens settings window.")
        button_settings.triggered.connect(self._open_settings_window)
        # Add buttons to the menu
        file_menu = menu.addMenu("&File")
        file_menu.addAction(button_new_brand_profile)
        file_menu.addAction(button_edit_brand_profile)
        file_menu.addAction(button_train)
        menu.addAction(button_settings)

    def _make_layout(self):
        """
        Create the main layout containing top and main layouts.
        """
        layout = QVBoxLayout() # Create the main layout containing everything

        layout.addLayout(self._make_top_layout())
        layout.addLayout(self._make_main_layout())

        return layout

    def _make_top_layout(self):
        """
        Create the layout for the top bar.
        """
        layout = QHBoxLayout() # Create the top bar layout

        label = QLabel(f"Analysis of {self.profile['name']}")
        label.setFont(self.font_heading)

        layout.addStretch()
        layout.addWidget(label)
        layout.addStretch()

        return layout

    def _make_main_layout(self):
        """
        Create the main layout containing left and right layouts.
        """
        layout = QHBoxLayout() # Create the main layout

        # Add components to the layout
        layout.addLayout(self._make_left_layout())
        layout.addLayout(self._make_right_layout())

        return layout

    def _make_left_layout(self):
        """
        Create the layout for the left side.
        """
        layout = QVBoxLayout() # Create the layout for the left side
    
        # Add components to the layout
        layout.addLayout(self._make_profile_changer_layout())
        layout.addLayout(self._make_time_changer_layout())
        layout.addWidget(self._make_subs_scrollable())
        layout.addStretch()

        return layout
    
    def _make_right_layout(self):
        """
        Create the layout for the right side.
        """
        layout = QVBoxLayout()

        layout.addWidget(self._make_tabs()) # Add tabs to the layout

        return layout

    def _make_profile_changer_layout(self):
        """
        Create the layout for changing profiles.
        """
        layout = QHBoxLayout()

        combo = QComboBox()
        profiles = toolkit.get_profiles()
        del profiles['next_index']
        combo.addItems([toolkit.get_profile(str(key))['name'] for key in profiles.keys()])
        combo.activated.connect(lambda: self._switch_profile(combo.currentIndex()))
        combo.setCurrentText(self.profile['name'])

        button = QPushButton("Refresh")
        button.clicked.connect(self._collect_new_posts)

        layout.addStretch()
        layout.addWidget(QLabel("Profile:"))
        layout.addWidget(combo)
        layout.addWidget(button)

        return layout

    def _make_time_changer_layout(self):
        """
        Create the layout for changing time periods.
        """
        layout = QVBoxLayout() # Create the layout for the time changer
        time_period_layout = self._make_time_period_layout() # Create the layout for the time period selector
        custom_time_layout = self._make_custom_time_layout() # Create the layout for the custom time selector

        # Add components to the layout
        layout.addWidget(QLabel("Time Period"))
        layout.addLayout(time_period_layout)
        layout.addWidget(QLabel("Custom"))
        layout.addLayout(custom_time_layout)

        return layout

    def _make_time_period_layout(self):
        """
        Create the layout for selecting predefined time periods.
        """
        layout = QHBoxLayout()

        button_day = QPushButton("Day")
        button_week = QPushButton("Week")
        button_month = QPushButton("Month")
        button_year = QPushButton("Year")
        button_alltime = QPushButton("All Time")

        button_day.clicked.connect(self._time_period_day)
        button_week.clicked.connect(self._time_period_week)
        button_month.clicked.connect(self._time_period_month)
        button_year.clicked.connect(self._time_period_year)
        button_alltime.clicked.connect(self._time_period_alltime)

        layout.addWidget(button_day)
        layout.addWidget(button_week)
        layout.addWidget(button_month)
        layout.addWidget(button_year)
        layout.addWidget(button_alltime)
        layout.addStretch()

        return layout

    def _make_custom_time_layout(self):
        """
        Create the layout for selecting custom time period.
        """
        layout = QHBoxLayout()

        selector_from = QDateEdit()
        selector_to = QDateEdit()
        apply = QPushButton("Apply")

        apply.clicked.connect(lambda: self._set_dates(selector_from, selector_to))

        layout.addWidget(QLabel("FROM:"))
        layout.addWidget(selector_from)
        layout.addWidget(QLabel("TO:"))
        layout.addWidget(selector_to)
        layout.addWidget(apply)
        layout.addStretch()

        return layout
    
    def _make_subs_scrollable(self):
        """
        Create a scrollable area containing the collection of posts.
        """
        scroll = QScrollArea()
        widget = QWidget()
        layout = QVBoxLayout()

        label = QLabel("COLLECTION")
        layout.addWidget(label)

        posts = self.collector.posts

        if posts.empty:
            layout.addWidget(QLabel("No posts available."))
        else:
            for sub in posts['Subreddit'].value_counts().index.tolist():
                layout.addLayout(self._make_sub_layout(sub))  # Add sub layout

        widget.setLayout(layout)

        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)

        return scroll

    def _make_sub_layout(self, name: str):
        """
        Create a collapsible layout for each subreddit containing posts.
        """
        layout = QVBoxLayout()
        label_sub = QLabel(f"/r/{name}")

        layout.addWidget(self._make_full_separator_line())
        layout.addWidget(label_sub)
        layout.addWidget(self._make_full_separator_line())

        posts = self.collector.posts
        posts = posts[posts['Subreddit'] == name]

        for ID in posts['ID']:
            layout.addLayout(self._make_post_row_layout(ID))  # Add post layout
            layout.addWidget(self._make_full_separator_line())  # Add dotted separator line after each post layout

        return layout

    def _make_post_row_layout(self, ID: str):
        """
        Create the layout for a post row.
        """
        layout = QVBoxLayout()

        posts = self.collector.posts
        title = self.collector.get_record(posts, ID, 'Title')
        body = self.collector.get_record(posts, ID, 'Body')
        sentiment = self.collector.get_record(posts, ID, 'Sentiment')
        comments = self.collector.get_record(posts, ID, 'Comments')

        layout.addLayout(self._make_post_info_layout(title, body, sentiment))  # Add post info layout
        layout.addWidget(self._make_full_separator_line())  # Add full separator line after post info

        for comment in comments:
            layout.addLayout(self._make_comment_row_layout(comment))  # Add comment layout

        return layout

    def _make_post_info_layout(self, title: str, body: str, sentiment: str):
        """
        Create the layout for post information.
        """
        layout = QHBoxLayout()

        label_title = QLabel(title)
        label_body = QLabel(body)
        label_sentiment = QLabel(sentiment)

        label_title.setWordWrap(True)
        label_body.setWordWrap(True)

        layout.addLayout(self._make_indent_layout(8))  # Add indent layout for post
        layout.addWidget(label_title)
        layout.addStretch()
        layout.addWidget(label_sentiment)

        return layout

    def _make_comment_row_layout(self, ID: str):
        """
        Create the layout for a comment row.
        """
        layout = QHBoxLayout()

        comments = self.collector.comments
        body = self.collector.get_record(comments, ID, 'Body')
        sentiment = self.collector.get_record(comments, ID, 'Sentiment')

        label_body = QLabel(body)
        label_sentiment = QLabel(sentiment)

        label_body.setWordWrap(True)

        layout.addLayout(self._make_indent_layout(16))  # Add indent layout for comment
        layout.addWidget(label_body)
        layout.addStretch()
        layout.addWidget(label_sentiment)

        return layout

    def _make_indent_layout(self, margin: int):
        """
        Create an indent layout with specified margin.
        """
        layout = QHBoxLayout()
        indent = QLabel(" " * margin)
        layout.addWidget(indent)
        return layout

    def _make_full_separator_line(self):
        """
        Create a full separator line.
        """
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        return line

    def _make_tabs(self):
        """
        Create tabs for displaying different types of charts.
        """
        tabs = QTabWidget()
        tabs.addTab(self._make_line_tab(), "Sentiment Over Time")
        tabs.addTab(self._make_pie_tab(), "Overall Sentiment")
        tabs.addTab(self._make_bar_tab(), "Sentiment by Subreddit")
        return tabs

    def _make_line_tab(self):
        """
        Create tab for displaying line chart.
        """
        tab = QWidget()
        layout = QVBoxLayout()

        self.analyser.generate_line(self.line_chart, "Sentiment Over Time", ("Date", "Sentiment"), self.data_start_date, self.data_end_date)

        button_split_subs = QCheckBox("Split by Subreddit?")
        button_split_subs.setChecked(toolkit.get_config('split_subs'))
        button_split_subs.stateChanged.connect(lambda: self._split_subs(button_split_subs.isChecked()))

        layout.addWidget(self.line_chart)
        layout.addWidget(button_split_subs)
        tab.setLayout(layout)
        return tab

    def _make_pie_tab(self):
        """
        Create tab for displaying pie chart.
        """
        tab = QWidget()
        layout = QVBoxLayout()

        self.analyser.generate_pie(self.pie_chart, "Overall Sentiment", self.data_start_date, self.data_end_date)

        layout.addWidget(self.pie_chart)
        tab.setLayout(layout)
        return tab

    def _make_bar_tab(self):
        """
        Create tab for displaying bar chart.
        """
        tab = QWidget()
        layout = QVBoxLayout()

        self.analyser.generate_bar(self.bar_chart, "Subreddit Sentiment", ("Subreddit", "Sentiment"), self.data_start_date, self.data_end_date)

        layout.addWidget(self.bar_chart)
        tab.setLayout(layout)
        return tab

    def _update_charts(self):
        """
        Update all charts with new data.
        """
        self.analyser.generate_line(self.line_chart, "Sentiment Over Time", ("Date", "Sentiment"), self.data_start_date, self.data_end_date)
        self.analyser.generate_pie(self.pie_chart, "Overall Sentiment", self.data_start_date, self.data_end_date)
        self.analyser.generate_bar(self.bar_chart, "Subreddit Sentiment", ("Subreddit", "Sentiment"), self.data_start_date, self.data_end_date)

        self.line_chart.draw()
        self.pie_chart.draw()
        self.bar_chart.draw()

    def _time_period_day(self):
        """
        Set time period to one day.
        """
        self.data_start_date = (datetime.now() - timedelta(days=2)).timestamp()
        self.data_end_date = datetime.now().timestamp()

        self._update()
        print("Time period set to day.")

    def _time_period_week(self):
        """
        Set time period to one week.
        """
        self.data_start_date = (datetime.now() - timedelta(days=8)).timestamp()
        self.data_end_date = datetime.now().timestamp()

        self._update()
        print("Time period set to week.")

    def _time_period_month(self):
        """
        Set time period to one month.
        """
        self.data_start_date = (datetime.now() - timedelta(days=31)).timestamp()
        self.data_end_date = datetime.now().timestamp()

        self._update()
        print("Time period set to month.")

    def _time_period_year(self):
        """
        Set time period to one year.
        """
        self.data_start_date = (datetime.now() - timedelta(days=366)).timestamp()
        self.data_end_date = datetime.now().timestamp()

        self._update()
        print("Time period set to year.")

    def _time_period_alltime(self):
        """
        Set time period to all time.
        """
        self.data_start_date = self.collector.posts['Date/Time'].min()
        self.data_end_date = datetime.now().timestamp()

        self._update()
        print("Time period set to alltime.")

    def _set_dates(self, selector_from: QDateEdit, selector_to: QDateEdit) -> None:
        """
        Set custom time period using date selectors.

        Args:
            selector_from (QDateEdit): Date selector for start date.
            selector_to (QDateEdit): Date selector for end date.
        """
        start_date = selector_from.date().toPyDate()
        end_date = selector_to.date().toPyDate()

        self.data_start_date = datetime(start_date.year, start_date.month, start_date.day).timestamp()
        self.data_end_date = datetime(end_date.year, end_date.month, end_date.day).timestamp()

        self._update()

    def _split_subs(self, state: bool):
        """
        Enable or disable splitting data by subreddit.

        Args:
            state (bool): State of the checkbox.
        """
        toolkit.set_config('split_subs', int(state))

        self._update()

    def _switch_profile(self, index: str):
        """
        Switch to a different profile.

        Args:
            index (str): Index of the selected profile.
        """
        keys = [key for key in toolkit.get_profiles().keys()]
        print(keys)
        self.profile = toolkit.get_profile(keys[index + 1])
        self.collector = toolkit.PostCollector(self.model, self.scraper, self.profile)
        self._update()

    def _new_brand_profile(self):
        """
        Open window for creating a new brand profile.
        """
        self.window_profile_editor = ProfileEditorWindow()
        self.window_profile_editor.submit_clicked.connect(self._update)
        self.window_profile_editor.show()

    def _edit_brand_profile(self):
        """
        Open window for editing the current brand profile.
        """
        self.window_profile_editor = ProfileEditorWindow(self.profile)
        self.window_profile_editor.submit_clicked.connect(self._update)
        self.window_profile_editor.show()

    def _open_settings_window(self):
        """
        Open settings window.
        """
        self.window_settings = SettingsWindow()
        self.window_settings.submit_clicked.connect(self._update)
        self.window_settings.show()

    def _collect_new_posts(self):
        """
        Collect new posts from Reddit.
        """
        worker = Worker(lambda: self.collector.scrape_posts(toolkit.get_config('n_posts')))
        worker.signals.finished.connect(self._update)
        self.threadpool.start(worker)
        toolkit.console(f"Collecting new posts for {self.profile['name']} profile.")

    def _train_model(self):
        """
        Train the sentiment analysis model.
        """
        df = pd.read_csv(toolkit.get_dir() + '/datasets/sentiment140/processed-dataset.csv', nrows=5000, keep_default_na=False)
        worker = Worker(lambda: self.model.train(df, toolkit.get_dir() + '/models/'))
        worker.signals.finished.connect(self._update)
        self.threadpool.start(worker)

class ProfileEditorWindow(QWidget):
    """
    Window for navigating and editing profiles.

    Inherits from QWidget for creating a window for profile editing.
    """
    submit_clicked = pyqtSignal()

    def __init__(self, profile: dict[str, any] = {}):
        """
        initialise the profile editor window.

        Args:
            profile (dict): Profile information.
        """
        super().__init__()
        self.profile = profile
        if not self.profile:
            self.profile['id'] = toolkit.get_profiles()['next_index']
            self.profile['name'] = ''
            self.profile['subs'] = {}
            toolkit.increment_profiles_next_index()

        self.setWindowTitle("Profile Editor")

        self.layout = self._make_layout()

        self.setLayout(self.layout)

    def _update(self):
        """
        Update the profile editor window layout.
        """
        QWidget().setLayout(self.layout)
        self.layout = self._make_layout()
        self.setLayout(self.layout)

        self.submit_clicked.emit()

    def _make_layout(self):
        """
        Create the layout for the profile editor window.
        """
        layout = QVBoxLayout()
        
        label_heading = QLabel(f"Profile {self.profile['name']}")

        label_subs = QLabel("Edit Subreddits")

        layout.addWidget(label_heading)
        layout.addLayout(self._make_name_entry_layout())
        layout.addWidget(label_subs)
        layout.addLayout(self._make_sub_entry_layout())
        layout.addLayout(self._make_subs_layout())
        layout.addStretch()

        return layout


    def _make_name_entry_layout(self):
        """
        Create the layout for entering profile name.
        """
        layout = QHBoxLayout()

        label = QLabel("Name:")

        textbox = QLineEdit()
        textbox.setMaxLength(16)
        textbox.setPlaceholderText("Enter profile name")

        button_enter = QPushButton("Enter")
        button_enter.clicked.connect(lambda: self._add_profile(textbox.text()))
        
        button_del = QPushButton("Delete")
        button_del.clicked.connect(self._del_profile)

        layout.addWidget(label)
        layout.addWidget(textbox)
        layout.addWidget(button_enter)
        layout.addWidget(button_del)

        return layout

    def _make_sub_entry_layout(self):
        """
        Create the layout for entering subreddit information.
        """
        layout = QHBoxLayout()

        label_name = QLabel("Name:")
        
        textbox_name = QLineEdit()
        textbox_name.setMaxLength(21)
        textbox_name.setPlaceholderText("Enter subreddit name")

        label_search = QLabel("Search:")

        textbox_search = QLineEdit()
        textbox_search.setMaxLength(256)
        textbox_search.setPlaceholderText("Optional search terms (separated by commas)")

        button_add = QPushButton("Add")
        button_add.clicked.connect(lambda: self._add_sub(textbox_name.text(), textbox_search.text()))

        layout.addWidget(label_name)
        layout.addWidget(textbox_name)
        layout.addWidget(label_search)
        layout.addWidget(textbox_search)
        layout.addWidget(button_add)

        return layout
    
    def _make_subs_layout(self):
        """
        Create the layout for displaying and editing subreddit information.
        """
        layout = QVBoxLayout()

        if not self.profile['subs']:
            label = QLabel("No subs")
            layout.addWidget(label)
            return layout
        
        for name, search in self.profile['subs'].items():
            layout.addLayout(self._make_row_layout(name, search))

        return layout

    def _make_row_layout(self, name: str, search_terms: list[str]):
        """
        Create a layout for a single row in the subreddit information display.

        Args:
            name (str): Subreddit name.
            search_terms (list): List of search terms.

        Returns:
            QHBoxLayout: Layout for a single row.
        """
        layout = QHBoxLayout()

        if search_terms:
            search_terms_string = ', '.join(search_terms)
            label = QLabel(f"'{search_terms_string}' in /r/{name}")
        else:
            label = QLabel(f"/r/{name}")

        button_del = QPushButton("Delete")
        button_del.clicked.connect(lambda: self._del_sub(name))

        layout.addWidget(label)
        layout.addStretch()
        layout.addWidget(button_del)

        return layout

    def _add_profile(self, name: str):
        """
        Add a new profile.

        Args:
            name (str): Profile name.
        """
        self.profile['name'] = name
        toolkit.set_profile(self.profile['id'], self.profile)

        path = f'{toolkit.get_dir()}/src/profiles/{self.profile["id"]}/'
        if not os.path.exists(path):
            os.makedirs(path)
        
        self._update()
        self.close()

    def _del_profile(self):
        """
        Delete the current profile.
        """
        try:
            toolkit.del_profile(self.profile['id'])
        except:
            pass
        self._update()
        self.close()

    def _add_sub(self, name: str, search_terms: str):
        """
        Add a subreddit to the current profile.

        Args:
            name (str): Subreddit name.
            search_terms (str): Search terms.
        """
        search_terms_list = [search_term.strip() for search_term in search_terms.split(',')]
        try:
            self.profile['subs'][name] = search_terms_list
            self._update()
        except Exception as e:
            toolkit.messages.error(f"Could not add subreddit /r/{name} to profile {self.profile['name']}. {e}.")

    def _del_sub(self, name: str):
        """
        Delete a subreddit from the current profile.

        Args:
            name (str): Subreddit name.
        """
        del self.profile['subs'][name]
        toolkit.set_profile(self.profile['id'], self.profile)
        self._update()

class SettingsWindow(QWidget):
    """
    SettingsWindow class for displaying and modifying application settings.

    This class provides a user interface for viewing and modifying various settings of the application.
    Changes to settings are deferred until the user clicks the Apply button.
    """
    submit_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Settings")

        self.layout = self._make_layout()
        self.settings = {}  # Internal dictionary to store settings

        self.setLayout(self.layout)

    def _make_layout(self):
        layout = QVBoxLayout()

        # Create tabs on the left side
        tabs = self._make_tabs()
        layout.addWidget(tabs)

        button_apply = QPushButton("Apply")
        button_apply.clicked.connect(self._apply_settings)

        layout.addWidget(button_apply)

        return layout

    def _make_tabs(self):
        """
        Create tabs for displaying different settings.
        """
        tabs = QTabWidget()

        # Add tabs with separators and buttons
        general_tab = self._make_general_tab()
        tabs.addTab(general_tab, "General")

        analysis_tab = self._make_analysis_tab()
        tabs.addTab(analysis_tab, "Analysis")

        privacy_security_tab = self._make_privacy_security_tab()
        tabs.addTab(privacy_security_tab, "Privacy & Security")

        personalisation_tab = self._make_personalisation_tab()
        tabs.addTab(personalisation_tab, "Personalisation")

        export_sharing_tab = self._make_export_sharing_tab()
        tabs.addTab(export_sharing_tab, "Export & Sharing")

        feedback_support_tab = self._make_feedback_support_tab()
        tabs.addTab(feedback_support_tab, "Feedback & Support")

        # Set tab position to the left
        tabs.setTabPosition(QTabWidget.West)

        return tabs

    def _make_general_tab(self):
        """
        Create widgets for the general settings tab.
        """
        tab_layout = QVBoxLayout()

        # Add section headers
        tab_layout.addWidget(self._make_section_header("Post Scraping"))

        update_interval_spinbox = QSpinBox()
        update_interval_spinbox.setMinimum(1000)  # 1 second minimum interval
        update_interval_spinbox.setMaximum(86400000)  # 24 hours maximum interval
        update_interval_spinbox.setValue(toolkit.get_config("update_interval"))
        update_interval_spinbox.valueChanged.connect(lambda value, name="update_interval": self._update_setting(name, value))
        tab_layout.addWidget(QLabel("Update Interval (ms):"))
        tab_layout.addWidget(update_interval_spinbox)

        n_posts_spinbox = QSpinBox()
        n_posts_spinbox.setMinimum(1)
        n_posts_spinbox.setMaximum(1000)
        n_posts_spinbox.setValue(toolkit.get_config("n_posts"))
        n_posts_spinbox.valueChanged.connect(lambda value, name="n_posts": self._update_setting(name, value))
        tab_layout.addWidget(QLabel("Number of Posts:"))
        tab_layout.addWidget(n_posts_spinbox)

        scrape_comments_checkbox = QCheckBox("Scrape Comments")
        scrape_comments_checkbox.setChecked(bool(toolkit.get_config("scrape_comments")))
        scrape_comments_checkbox.stateChanged.connect(lambda state, name="scrape_comments": self._update_setting(name, state == 2))
        tab_layout.addWidget(scrape_comments_checkbox)

        tab_layout.addStretch()
        tab_widget = QWidget()
        tab_widget.setLayout(tab_layout)
        return tab_widget

    def _make_analysis_tab(self):
        """
        Create widgets for the analysis settings tab.
        """
        tab_layout = QVBoxLayout()

        # Add section headers
        tab_layout.addWidget(self._make_section_header("Text Processing"))

        lowercase_checkbox = QCheckBox("Lowercase")
        lowercase_checkbox.setChecked(bool(toolkit.get_config("lowercase")))
        lowercase_checkbox.stateChanged.connect(lambda state, name="lowercase": self._update_setting(name, state == 2))
        tab_layout.addWidget(lowercase_checkbox)

        soup_checkbox = QCheckBox("BeautifulSoup")
        soup_checkbox.setChecked(bool(toolkit.get_config("soup")))
        soup_checkbox.stateChanged.connect(lambda state, name="soup": self._update_setting(name, state == 2))
        tab_layout.addWidget(soup_checkbox)

        lemmatise_checkbox = QCheckBox("Lemmatise")
        lemmatise_checkbox.setChecked(bool(toolkit.get_config("lemmatise")))
        lemmatise_checkbox.stateChanged.connect(lambda state, name="lemmatise": self._update_setting(name, state == 2))
        tab_layout.addWidget(lemmatise_checkbox)

        tab_layout.addWidget(self._make_section_header("Predicting"))

        confidence_threshold_spinbox = QDoubleSpinBox()
        confidence_threshold_spinbox.setMinimum(0.0)
        confidence_threshold_spinbox.setMaximum(1.0)
        confidence_threshold_spinbox.setSingleStep(0.05)
        confidence_threshold_spinbox.setValue(toolkit.get_config("confidence_threshold"))
        confidence_threshold_spinbox.valueChanged.connect(lambda value, name="confidence_threshold": self._update_setting(name, value))
        tab_layout.addWidget(QLabel("Confidence Threshold:"))
        tab_layout.addWidget(confidence_threshold_spinbox)

        tab_layout.addWidget(self._make_section_header("Training"))

        cross_validation_checkbox = QCheckBox("Cross Validation")
        cross_validation_checkbox.setChecked(bool(toolkit.get_config("cross_validation")))
        cross_validation_checkbox.stateChanged.connect(lambda state, name="cross_validation": self._update_setting(name, state == 2))
        tab_layout.addWidget(cross_validation_checkbox)

        tab_layout.addWidget(self._make_section_header("Graphing"))

        score_weighting_checkbox = QCheckBox("Score Weighting")
        score_weighting_checkbox.setChecked(bool(toolkit.get_config("score_weighting")))
        score_weighting_checkbox.stateChanged.connect(lambda state, name="score_weighting": self._update_setting(name, state == 2))
        tab_layout.addWidget(score_weighting_checkbox)

        chart_ticks_spinbox = QSpinBox()
        chart_ticks_spinbox.setMinimum(1)
        chart_ticks_spinbox.setMaximum(20)
        chart_ticks_spinbox.setValue(toolkit.get_config("chart_ticks"))
        chart_ticks_spinbox.valueChanged.connect(lambda value, name="chart_ticks": self._update_setting(name, value))
        tab_layout.addWidget(QLabel("Chart Ticks:"))
        tab_layout.addWidget(chart_ticks_spinbox)

        tab_layout.addStretch()
        tab_widget = QWidget()
        tab_widget.setLayout(tab_layout)
        return tab_widget

    def _make_privacy_security_tab(self):
        """
        Create widgets for the privacy and security settings tab.
        """
        tab_layout = QVBoxLayout()

        # Add security settings
        tab_layout.addWidget(self._make_section_header("Security Settings"))

        encryption_checkbox = QCheckBox("Encryption")
        encryption_checkbox.setChecked(bool(toolkit.get_config("encryption")))
        encryption_checkbox.stateChanged.connect(lambda state, name="encryption": self._update_setting(name, state == 2))
        tab_layout.addWidget(encryption_checkbox)

        tab_layout.addStretch()
        tab_widget = QWidget()
        tab_widget.setLayout(tab_layout)
        return tab_widget

    def _make_personalisation_tab(self):
        """
        Create widgets for the personalisation settings tab.
        """
        tab_layout = QVBoxLayout()

        # Add section headers
        tab_layout.addWidget(self._make_section_header("Styling"))

        font_button = QPushButton("Change Font")
        font_button.clicked.connect(self._change_font)
        tab_layout.addWidget(font_button)

        main_colour_button = QPushButton("Change Main Color")
        main_colour_button.clicked.connect(self._change_main_colour)
        tab_layout.addWidget(main_colour_button)

        accent_colour_button = QPushButton("Change Accent Color")
        accent_colour_button.clicked.connect(self._change_accent_colour)
        tab_layout.addWidget(accent_colour_button)

        # Add section headers
        tab_layout.addWidget(self._make_section_header("Graph Styling"))

        # Add graph styling options
        graph_colour_button = QPushButton("Change Graph Color")
        graph_colour_button.clicked.connect(self._change_graph_colour)
        tab_layout.addWidget(graph_colour_button)

        # Add personalisation settings
        tab_layout.addWidget(self._make_section_header("Other Personalisation"))

        dummy_option_checkbox = QCheckBox("Dummy Option")
        dummy_option_checkbox.setChecked(bool(toolkit.get_config("dummy_option")))
        dummy_option_checkbox.stateChanged.connect(lambda state, name="dummy_option": self._update_setting(name, state == 2))
        tab_layout.addWidget(dummy_option_checkbox)

        tab_layout.addStretch()
        tab_widget = QWidget()
        tab_widget.setLayout(tab_layout)
        return tab_widget

    def _make_export_sharing_tab(self):
        """
        Create widgets for the export and sharing settings tab.
        """
        tab_layout = QVBoxLayout()

        # Add export and sharing settings
        tab_layout.addWidget(self._make_section_header("Export & Sharing"))

        export_button = QPushButton("Export Data")
        export_button.clicked.connect(self._export_data)
        tab_layout.addWidget(export_button)

        share_button = QPushButton("Share Analysis")
        share_button.clicked.connect(self._share_analysis)
        tab_layout.addWidget(share_button)

        tab_layout.addStretch()
        tab_widget = QWidget()
        tab_widget.setLayout(tab_layout)
        return tab_widget

    def _make_feedback_support_tab(self):
        """
        Create widgets for the feedback and support settings tab.
        """
        tab_layout = QVBoxLayout()

        # Add feedback and support information
        tab_layout.addWidget(self._make_section_header("About the Creator"))

        about_label = QLabel("I am the creator of this application.")
        tab_layout.addWidget(about_label)

        linkedin_link = QLabel("<a href='https://www.linkedin.com/in/alfieatkinson/'>LinkedIn Profile</a>")
        linkedin_link.setOpenExternalLinks(True)
        tab_layout.addWidget(linkedin_link)

        tab_layout.addWidget(self._make_section_header("Contact Me"))

        email_button = QPushButton("Compose Email")
        email_button.clicked.connect(self._compose_email)
        tab_layout.addWidget(email_button)

        tab_layout.addWidget(self._make_section_header("GitHub Repository"))

        github_link = QLabel("<a href='https://github.com/alfieatkinson/sentiment-analysis-tool'>GitHub Repository</a>")
        github_link.setOpenExternalLinks(True)
        tab_layout.addWidget(github_link)

        tab_layout.addStretch()
        tab_widget = QWidget()
        tab_widget.setLayout(tab_layout)
        return tab_widget

    def _make_section_header(self, title):
        """
        Create a section header label.
        """
        section_label = QLabel(title)
        section_label.setStyleSheet("font-weight: bold; color: #666; text-decoration: underline;")
        return section_label

    def _update_setting(self, name, value):
        """
        Update a setting in the internal dictionary.
        """
        self.settings[name] = value

    def _apply_settings(self):
        """
        Apply settings when Apply button is clicked.
        """
        # Apply settings stored in the internal dictionary
        for setting_name, value in self.settings.items():
            toolkit.set_config(setting_name, value)

        self.submit_clicked.emit()

    def _compose_email(self):
        """
        Compose an email to alf.atkinson13@gmail.com.
        """
        # Open default email client with pre-filled details
        import webbrowser
        webbrowser.open("mailto:alf.atkinson13@gmail.com")

    def _change_font(self):
        """
        Open a font selection dialog to change font.
        """
        # Dummy functionality to change font
        font, ok = QFontDialog.getFont()
        if ok:
            print("Selected font:", font.family())

    def _change_main_colour(self):
        """
        Open a color picker dialog to change main colour.
        """
        # Dummy functionality to change main colour
        color = QColorDialog.getColor()
        if color.isValid():
            print("Selected main colour:", color.name())

    def _change_accent_colour(self):
        """
        Open a color picker dialog to change accent colour.
        """
        # Dummy functionality to change accent colour
        color = QColorDialog.getColor()
        if color.isValid():
            print("Selected accent colour:", color.name())

    def _change_text_colour(self):
        """
        Open a color picker dialog to change text colour.
        """
        # Dummy functionality to change text colour
        color = QColorDialog.getColor()
        if color.isValid():
            print("Selected text colour:", color.name())

    def _change_graph_colour(self):
        """
        Open a color picker dialog to change graph colour.
        """
        # Dummy functionality to change graph colour
        color = QColorDialog.getColor()
        if color.isValid():
            print("Selected graph colour:", color.name())

    def _export_data(self):
        """
        Export data to a file.
        """
        # Dummy functionality to export data
        print("Data exported successfully.")

    def _share_analysis(self):
        """
        Share analysis results.
        """
        # Dummy functionality to share analysis
        print("Analysis shared successfully.")
