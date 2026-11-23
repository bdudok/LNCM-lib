'''
Queue pre-processing and motion correction of Bruker sessions selected from Baserow.
Set the filters and search; select sessions in the list (Shift and Ctrl for several); set the parameters (the preset
buttons fill in the sensor settings); Queue sends them to the job queue with MotionCorrect.queue_sessions, the same
code as the template script. Launch with GUIs/launch_MotionCorrect.py.
'''
import sys

import requests
from PySide6 import QtCore, QtGui, QtWidgets

from BaserowAPI.BaserowRequests import GetSessions
from BaserowAPI.config import config
from JobQueue.common import QueueError, dashboard_port
from JobQueue.client import QueueClient
from Proc2P.Bruker import MotionCorrect

TASKS = ('MotionCorr', 'Segment', 'Process', 'Done', 'Broken')
ANY = '(any)'
COLUMNS = ('Prefix', 'Date', 'Channels', 'LFP', 'Opto', 'Status')
STATUS_POLL_MS = 5000
# registration settings set by the form; the presets fill them in
REG_FIELDS = ('do_bidiphase', 'two_step_registration', 'nonrigid', 'block_size', 'batch_size', 'snr_thresh',
              'maxregshift')


def project_names():
    '''Project values that occur in the sessions table (read-only token)'''
    names = set()
    url, params = config['session_url'], {'size': 200, 'include': 'Project'}
    while url:
        resp = requests.get(url, headers={"Authorization": f"Token {config['api_token']}"}, params=params,
                            timeout=30)
        resp.raise_for_status()
        page = resp.json()
        names.update(r['Project']['value'] for r in page['results'] if r.get('Project'))
        url, params = page['next'], None  # the next url has the parameters
    return sorted(names)


def select_value(v):
    '''Display value of a Baserow single select or boolean field'''
    if isinstance(v, dict):
        return v.get('value', '')
    if isinstance(v, bool) or type(v).__name__ == 'bool_':
        return 'yes' if v else ''
    return '' if v is None or v != v else str(v)  # v != v: NaN


class MotionCorrectGUI(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Motion correction queue')
        self.client = QueueClient()
        self.sessions = None  # DataFrame of the search result
        self.jobs = {}  # job id: prefix, of jobs queued from this window that haven't finished
        self.status = {}  # prefix: status of the jobs queued from this window
        self.presets = {p: MotionCorrect.preset_settings(p)['registration'] for p in MotionCorrect.PRESETS}

        central = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(central)
        layout.addLayout(self.make_filters())
        body = QtWidgets.QHBoxLayout()
        body.addWidget(self.make_table(), stretch=3)
        body.addWidget(self.make_parameters(), stretch=1)
        layout.addLayout(body)
        layout.addLayout(self.make_queue_bar())
        self.setCentralWidget(central)
        self.apply_preset('Calcium')

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(STATUS_POLL_MS)
        self.timer.timeout.connect(self.update_status)
        self.timer.start()
        self.resize(1300, 750)
        self.show()

    # --- layout
    def make_filters(self):
        row = QtWidgets.QHBoxLayout()
        self.project = QtWidgets.QComboBox()
        self.project.setEditable(True)
        self.project.setMinimumWidth(180)
        try:
            self.project.addItems(project_names())
        except requests.RequestException as e:
            print('Could not read the projects from Baserow:', e)
        self.project.setCurrentText('')
        self.task = QtWidgets.QComboBox()
        self.task.addItems((ANY, *TASKS))
        self.task.setCurrentText('MotionCorr')
        self.incltag = QtWidgets.QLineEdit()
        self.user = QtWidgets.QLineEdit()
        self.prefix = QtWidgets.QLineEdit()
        for label, widget in (('Project', self.project), ('Task', self.task), ('InclTag contains', self.incltag),
                              ('User contains', self.user), ('Prefix contains', self.prefix)):
            row.addWidget(QtWidgets.QLabel(label))
            row.addWidget(widget)
            if isinstance(widget, QtWidgets.QLineEdit):
                widget.returnPressed.connect(self.search)
        button = QtWidgets.QPushButton('Search')
        button.clicked.connect(self.search)
        row.addWidget(button)
        return row

    def make_table(self):
        self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)  # Shift and Ctrl
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self.update_selection_label)
        return self.table

    def make_parameters(self):
        box = QtWidgets.QGroupBox('Parameters')
        form = QtWidgets.QFormLayout(box)
        presets = QtWidgets.QHBoxLayout()
        for p in MotionCorrect.PRESETS:
            button = QtWidgets.QPushButton(p)
            button.clicked.connect(lambda checked=False, p=p: self.apply_preset(p))
            presets.addWidget(button)
        form.addRow('Preset', presets)
        self.ref_ch = QtWidgets.QComboBox()
        self.ref_ch.addItems(('Ch2', 'Ch1'))
        self.ref_ch.setToolTip('Reference channel for 2-channel movies. Ch1 is red, Ch2 is green.')
        form.addRow('Reference channel', self.ref_ch)
        self.switches = {}
        for name, default, tip in (
                ('pre_only', False, 'Only pre-process, no motion correction'),
                ('overwrite_previous', True, 'Delete the previous motion corrected movie and traces'),
                ('overwrite_preproc', False, 'Delete the previous pre-processing output'),
                ('mc_only', True, 'Only motion correct, skip the rest of the suite2p pipeline')):
            self.switches[name] = QtWidgets.QCheckBox()
            self.switches[name].setChecked(default)
            self.switches[name].setToolTip(tip)
            form.addRow(name, self.switches[name])
        self.reg = {}
        for name in REG_FIELDS:
            if isinstance(self.presets['Calcium'][name], bool):
                self.reg[name] = QtWidgets.QCheckBox()
            else:
                self.reg[name] = QtWidgets.QLineEdit()
            form.addRow(name, self.reg[name])
        return box

    def make_queue_bar(self):
        row = QtWidgets.QHBoxLayout()
        self.selection_label = QtWidgets.QLabel('No sessions')
        row.addWidget(self.selection_label)
        row.addStretch()
        dashboard = QtWidgets.QPushButton('Dashboard')
        dashboard.clicked.connect(
            lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl(f'http://localhost:{dashboard_port()}')))
        row.addWidget(dashboard)
        queue = QtWidgets.QPushButton('Queue selected')
        queue.clicked.connect(self.queue_selected)
        row.addWidget(queue)
        return row

    # --- parameters
    def apply_preset(self, preset):
        for name, widget in self.reg.items():
            value = self.presets[preset][name]
            if isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(value, (list, tuple)):
                widget.setText(', '.join(str(x) for x in value))
            else:
                widget.setText(str(value))

    def get_settings(self):
        '''suite2p settings from the form. Raises ValueError if a field can't be parsed.'''
        settings = MotionCorrect.preset_settings(None, self.switches['mc_only'].isChecked())
        reg = settings['registration']
        for name, widget in self.reg.items():
            if isinstance(widget, QtWidgets.QCheckBox):
                reg[name] = widget.isChecked()
                continue
            text = widget.text().strip()
            default = self.presets['Calcium'][name]
            try:
                if isinstance(default, (list, tuple)):
                    reg[name] = [int(x) for x in text.split(',')]
                elif isinstance(default, int):
                    reg[name] = int(text)
                else:
                    reg[name] = float(text)
            except ValueError:
                raise ValueError(f'{name}: can not read "{text}"') from None
        return settings

    # --- sessions
    def search(self):
        task = self.task.currentText()
        filters = []
        if self.prefix.text().strip():
            filters.append((f"filter__field_{config['FieldID']['Image.ID']}__contains", self.prefix.text().strip()))
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            result = GetSessions(timeout=60).search(
                project=self.project.currentText().strip() or None, task=None if task == ANY else task,
                incltag=self.incltag.text().strip() or None, user=self.user.text().strip() or None,
                more_filters=filters)
        except requests.RequestException as e:
            result = e
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if not hasattr(result, 'iterrows'):  # search returns the response if Baserow refused the request
            QtWidgets.QMessageBox.warning(self, 'Search', f'Baserow search failed: {result}')
            return
        self.sessions = result.reset_index(drop=True)
        self.fill_table()

    def fill_table(self):
        self.table.setRowCount(len(self.sessions))
        for i, item in self.sessions.iterrows():
            prefix = item['Image.ID'][:-4]
            values = (prefix, select_value(item.get('Date')), select_value(item.get('Channels')),
                      select_value(item.get('LFP')), select_value(item.get('Opto')), self.status.get(prefix, ''))
            for j, v in enumerate(values):
                self.table.setItem(i, j, QtWidgets.QTableWidgetItem(v))
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.update_selection_label()

    def selected_rows(self):
        return sorted({index.row() for index in self.table.selectionModel().selectedRows()})

    def update_selection_label(self):
        n = 0 if self.sessions is None else len(self.sessions)
        self.selection_label.setText(f'{len(self.selected_rows())} of {n} sessions selected')

    # --- queue
    def queue_selected(self):
        rows = self.selected_rows()
        if not rows:
            QtWidgets.QMessageBox.information(self, 'Queue', 'Select sessions in the list first.')
            return
        try:
            settings = self.get_settings()
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, 'Parameters', str(e))
            return
        switches = {name: box.isChecked() for name, box in self.switches.items() if name != 'mc_only'}
        steps = 'pre-processing only' if switches['pre_only'] else 'pre-processing and motion correction'
        warn = '\nThe previous motion corrected movies and traces will be deleted.' \
            if switches['overwrite_previous'] and not switches['pre_only'] else ''
        answer = QtWidgets.QMessageBox.question(self, 'Queue', f'Queue {steps} of {len(rows)} sessions?{warn}')
        if answer != QtWidgets.QMessageBox.Yes:
            return
        try:
            jobs = MotionCorrect.queue_sessions(settings, self.sessions.iloc[rows], ref_ch=self.ref_ch.currentText(),
                                                client=self.client, **switches)
        except QueueError as e:
            QtWidgets.QMessageBox.warning(self, 'Queue', str(e))
            return
        self.jobs.update(jobs)
        for prefix in jobs.values():
            self.status[prefix] = 'queued'
        self.show_status()

    def update_status(self):
        if not self.jobs:
            return
        try:
            for job in self.client.job_status(list(self.jobs)):
                prefix = self.jobs[job['job_id']]
                status = {'pending': 'queued', 'assigned': 'starting'}.get(job['status'], job['status'])
                if job['status'] == 'failed':
                    print(f'{prefix} failed:\n{job["error"]}')
                if job['status'] in ('done', 'failed'):
                    del self.jobs[job['job_id']]
                self.status[prefix] = status
        except QueueError as e:
            print(e)
            return
        self.show_status()

    def show_status(self):
        col = COLUMNS.index('Status')
        for i in range(self.table.rowCount()):
            prefix = self.table.item(i, 0).text()
            if prefix in self.status:
                self.table.item(i, col).setText(self.status[prefix])


def launch_GUI():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    gui = MotionCorrectGUI()  # noqa: F841 - keeps the window referenced while the app runs
    sys.exit(app.exec())
