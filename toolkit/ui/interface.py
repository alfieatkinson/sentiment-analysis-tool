import os

from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

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

    def __init__(self, parent = None, width: int = 5, height: int = 5, dpi: int = 100):
        fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.threadpool = QThreadPool()
        print(f"Multithreading with maximum {self.threadpool.maxThreadCount()} threads.") 

        self.model = toolkit.BertModel()
        self.scraper = toolkit.RedditScraper()

        self.profile = toolkit.get_profile(0)

        profiles = toolkit.get_profiles()
        del profiles['next_index']
        for ID in profiles.keys():
            print(f"Profile {ID}: {toolkit.get_profile(ID)}")
            self.collector = toolkit.PostCollector(self.model, self.scraper, toolkit.get_profile(ID))
            self._collect_new_posts()

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
        self.widget = QWidget(self)
        self.setCentralWidget(self.widget)
        self.layout = self._make_layout()
        self.analyser.update_dataset(self.collector.merge_data())
        print(self.analyser.dataset)
        self._update_charts()
        self.widget.setLayout(self.layout)

    def _make_menu_bar(self):
        # MENU BAR
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

        # Button to open settings window
        button_settings = QAction("Settings", self)
        button_settings.setStatusTip("Opens settings window.")
        button_settings.triggered.connect(self._open_settings_window)
        # Add buttons to the menu
        file_menu = menu.addMenu("&File")
        file_menu.addAction(button_new_brand_profile)
        file_menu.addAction(button_edit_brand_profile)
        menu.addAction(button_settings)

    def _make_layout(self):
        layout = QVBoxLayout() # Create the main layout containing everything

        layout.addLayout(self._make_top_layout())
        layout.addLayout(self._make_main_layout())

        return layout

    def _make_top_layout(self):
        layout = QHBoxLayout() # Create the top bar layout

        label = QLabel(f"Analysis of {self.profile['name']}")
        label.setFont(self.font_heading)

        layout.addStretch()
        layout.addWidget(label)
        layout.addStretch()

        return layout

    def _make_main_layout(self):
        layout = QHBoxLayout() # Create the main layout

        # Add components to the layout
        layout.addLayout(self._make_left_layout())
        layout.addLayout(self._make_right_layout())

        return layout

    def _make_left_layout(self):
        layout = QVBoxLayout() # Create the layout for the left side
    
        # Add components to the layout
        layout.addLayout(self._make_profile_changer_layout())
        layout.addLayout(self._make_time_changer_layout())
        layout.addStretch()

        return layout
    
    def _make_right_layout(self):
        layout = QVBoxLayout()

        layout.addWidget(self._make_tabs()) # Add tabs to the layout

        return layout

    def _make_profile_changer_layout(self):
        layout = QHBoxLayout()

        combo = QComboBox()
        profiles = toolkit.get_profiles()
        del profiles['next_index']
        combo.addItems([toolkit.get_profile(str(key))['name'] for key in profiles.keys()])
        combo.activated.connect(lambda: self._switch_profile(combo.currentIndex()))
        combo.setCurrentText(self.profile['name'])

        layout.addStretch()
        layout.addWidget(QLabel("Profile:"))
        layout.addWidget(combo)

        return layout

    def _make_time_changer_layout(self):
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

    def _make_tabs(self):
        tabs = QTabWidget()
        tabs.addTab(self._make_line_tab(), "Sentiment Over Time")
        tabs.addTab(self._make_pie_tab(), "Overall Sentiment")
        tabs.addTab(self._make_bar_tab(), "Sentiment by Subreddit")
        return tabs

    def _make_line_tab(self):
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
        tab = QWidget()
        layout = QVBoxLayout()

        self.analyser.generate_pie(self.pie_chart, "Overall Sentiment", self.data_start_date, self.data_end_date)

        layout.addWidget(self.pie_chart)
        tab.setLayout(layout)
        return tab

    def _make_bar_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        self.analyser.generate_bar(self.bar_chart, "Subreddit Sentiment", ("Subreddit", "Sentiment"), self.data_start_date, self.data_end_date)

        layout.addWidget(self.bar_chart)
        tab.setLayout(layout)
        return tab

    def _update_charts(self):
        self.analyser.generate_line(self.line_chart, "Sentiment Over Time", ("Date", "Sentiment"), self.data_start_date, self.data_end_date)
        self.analyser.generate_pie(self.pie_chart, "Overall Sentiment", self.data_start_date, self.data_end_date)
        self.analyser.generate_bar(self.bar_chart, "Subreddit Sentiment", ("Subreddit", "Sentiment"), self.data_start_date, self.data_end_date)

        self.line_chart.draw()
        self.pie_chart.draw()
        self.bar_chart.draw()

    def _time_period_day(self):
        self.data_start_date = (datetime.now() - timedelta(days=2)).timestamp()
        self.data_end_date = datetime.now().timestamp()

        self._update()
        print("Time period set to day.")

    def _time_period_week(self):
        self.data_start_date = (datetime.now() - timedelta(days=8)).timestamp()
        self.data_end_date = datetime.now().timestamp()

        self._update()
        print("Time period set to week.")

    def _time_period_month(self):
        self.data_start_date = (datetime.now() - timedelta(days=31)).timestamp()
        self.data_end_date = datetime.now().timestamp()

        self._update()
        print("Time period set to month.")

    def _time_period_year(self):
        self.data_start_date = (datetime.now() - timedelta(days=366)).timestamp()
        self.data_end_date = datetime.now().timestamp()

        self._update()
        print("Time period set to year.")

    def _time_period_alltime(self):
        self.data_start_date = self.collector.posts['Date/Time'].min()
        self.data_end_date = datetime.now().timestamp()

        self._update()
        print("Time period set to alltime.")

    def _set_dates(self, selector_from: QDateEdit, selector_to: QDateEdit) -> None:
        start_date = selector_from.date().toPyDate()
        end_date = selector_to.date().toPyDate()

        self.data_start_date = datetime(start_date.year, start_date.month, start_date.day).timestamp()
        self.data_end_date = datetime(end_date.year, end_date.month, end_date.day).timestamp()

        self._update()

    def _split_subs(self, state: bool):
        toolkit.set_config('split_subs', int(state))

        self._update()

    def _switch_profile(self, index: str):
        keys = [key for key in toolkit.get_profiles().keys()]
        print(keys)
        self.profile = toolkit.get_profile(keys[index + 1])
        self.collector = toolkit.PostCollector(self.model, self.scraper, self.profile)
        self._update()

    def _new_brand_profile(self):
        self.window_profile_editor = ProfileEditorWindow()
        self.window_profile_editor.submit_clicked.connect(self._update)
        self.window_profile_editor.show()

    def _edit_brand_profile(self):
        self.window_profile_editor = ProfileEditorWindow(self.profile)
        self.window_profile_editor.submit_clicked.connect(self._update)
        self.window_profile_editor.show()

    def _open_settings_window(self):
        print("SETTINGS OPENED!")

    def _collect_new_posts(self):
        worker = Worker(self.collector.scrape_posts)
        worker.signals.finished.connect(self._update)
        self.threadpool.start(worker)
        toolkit.console(f"Collecting new posts for {self.profile['name']} profile.")

    def _train_model(self):
        worker = Worker(self.model.train)
        worker.signals.finished.connect(self._update)
        self.threadpool.start(worker)

class ProfileEditorWindow(QWidget):
    """
    Window for navigating and editing profiles
    """
    submit_clicked = pyqtSignal()

    def __init__(self, profile: dict[str, any] = {}):
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
        QWidget().setLayout(self.layout)
        self.layout = self._make_layout()
        self.setLayout(self.layout)

        self.submit_clicked.emit()

    def _make_layout(self):
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
        layout = QVBoxLayout()

        if not self.profile['subs']:
            label = QLabel("No subs")
            layout.addWidget(label)
            return layout
        
        for name, search in self.profile['subs'].items():
            layout.addLayout(self._make_row_layout(name, search))

        return layout

    def _make_row_layout(self, name: str, search_terms: list[str]):
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
        self.profile['name'] = name
        toolkit.set_profile(self.profile['id'], self.profile)

        path = f'{toolkit.get_dir()}/src/profiles/{self.profile["id"]}/'
        if not os.path.exists(path):
            os.makedirs(path)
        
        self._update()
        self.close()

    def _del_profile(self):
        try:
            toolkit.del_profile(self.profile['id'])
        except:
            pass
        self._update()
        self.close()

    def _add_sub(self, name: str, search_terms: str):
        search_terms_list = [search_term.strip() for search_term in search_terms.split(',')]
        try:
            self.profile['subs'][name] = search_terms_list
            self._update()
        except Exception as e:
            toolkit.messages.error(f"Could not add subreddit /r/{name} to profile {self.profile['name']}. {e}.")

    def _del_sub(self, name: str):
        del self.profile['subs'][name]
        toolkit.set_profile(self.profile['id'], self.profile)
        self._update()