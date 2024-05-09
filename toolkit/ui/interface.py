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
        self.kwargs['progress_callback'] = self.signals.progress
    
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

        self.model = toolkit.BertModel()
        self.scraper = toolkit.RedditScraper()
        self.collector = toolkit.PostCollector(self.model, self.scraper)
        self.analyser = toolkit.Analyser(self.collector.merge_data())

        self.data_start_date = self.collector.posts['Date/Time'].min()
        self.data_end_date = datetime.now().timestamp()

        self.line_chart = MplCanvas(self)
        self.pie_chart = MplCanvas(self)
        self.bar_chart = MplCanvas(self)

        self.setWindowTitle("Sentiment Analysis Tool") # Set the title of the window

        # Get the screen width and height and use it to set window geometry
        screen_resolution = QApplication.instance().primaryScreen().size()
        width = screen_resolution.width() / 2
        height = screen_resolution.height() / 2
        x = screen_resolution.width() - width * 1.5
        y = screen_resolution.height() - height * 1.5
        self.setGeometry(x, y, width, height) # Set the window size and position

        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)

        self._make_menu_bar()

        # LAYOUTS
        main_layout = QVBoxLayout() # Create the main layout containing everything

        # Create sub-layouts
        middle_layout = QHBoxLayout() # Create the middle layout

        left_layout = QVBoxLayout() # Create the layout for the left side
        time_changer_layout = QVBoxLayout() # Create the layout for the time changer
        time_period_layout = self._make_time_period_layout() # Create the layout for the time period selector
        custom_time_layout = self._make_custom_time_layout() # Create the layout for the custom time selector

        time_changer_layout.addWidget(QLabel("Time Period"))
        time_changer_layout.addLayout(time_period_layout)
        time_changer_layout.addWidget(QLabel("Custom"))
        time_changer_layout.addLayout(custom_time_layout)

        left_layout.addLayout(time_changer_layout)

        left_layout.addStretch()

        right_layout = QVBoxLayout()
        right_layout.addWidget(self._make_tabs())

        middle_layout.addLayout(left_layout)
        middle_layout.addLayout(right_layout)


        main_layout.addWidget(QLabel("Analysis of BRAND"))
        main_layout.addLayout(middle_layout)
        

        
        main_widget.setLayout(main_layout)

        self.threadpool = QThreadPool()
        print(f"Multithreading with maximum {self.threadpool.maxThreadCount()} threads.")

        self.timer = QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._update_charts)
        self.timer.start()

    def _make_menu_bar(self):
        # MENU BAR
        menu = self.menuBar() # Create the menu bar

        # Create menu buttons
        # Button to create new brand profile
        button_new_brand_profile = QAction("New Brand Profile", self)
        button_new_brand_profile.setStatusTip("Creates a new brand profile.")
        button_new_brand_profile.triggered.connect(self._new_brand_profile)
        # Button to delete brand profile
        button_del_brand_profile = QAction("Delete Brand Profile", self)
        button_del_brand_profile.setStatusTip("Deletes the currently selected brand profile.")
        button_del_brand_profile.triggered.connect(self._del_brand_profile)
        # Button to open settings window
        button_settings = QAction("Settings", self)
        button_settings.setStatusTip("Opens settings window.")
        button_settings.triggered.connect(self._open_settings_window)
        # Add buttons to the menu
        file_menu = menu.addMenu("&File")
        file_menu.addAction(button_new_brand_profile)
        file_menu.addAction(button_del_brand_profile)
        menu.addAction(button_settings)

    def _make_layout(self):
        pass

    def _make_left_layout(self):
        pass

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
        button_split_subs.stateChanged.connect(self._split_subs)

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
        print("Time period set to day.")

    def _time_period_week(self):
        self.data_start_date = (datetime.now() - timedelta(days=8)).timestamp()
        print("Time period set to week.")

    def _time_period_month(self):
        self.data_start_date = (datetime.now() - timedelta(days=31)).timestamp()
        print("Time period set to month.")

    def _time_period_year(self):
        self.data_start_date = (datetime.now() - timedelta(days=366)).timestamp()
        print("Time period set to year.")

    def _time_period_alltime(self):
        self.data_start_date = self.collector.posts['Date/Time'].min()
        print("Time period set to alltime.")

    def _set_dates(self, selector_from: QDateEdit, selector_to: QDateEdit) -> None:
        start_date = selector_from.date().toPyDate()
        end_date = selector_to.date().toPyDate()

        self.data_start_date = datetime(start_date.year, start_date.month, start_date.day).timestamp()
        self.data_end_date = datetime(end_date.year, end_date.month, end_date.day).timestamp()

    def _split_subs(self, state: bool):
        toolkit.set_config('split_subs', int(state))

    def _open_settings_window(self):
        print("SETTINGS OPENED!")

    def _new_brand_profile(self):
        pass

    def _del_brand_profile(self):
        pass