import time
import pickle
import datetime
import os
import json
import logging
from threading import Thread

IFL = '....'


def _check_types_recursively(valid_dict, compare_dict):
    """
    Utility function for checking validity of configuration.
    Scans recursively dictionaries and raise exceptions in case of
    errors.

    **Args**:
       valid_dict (dict) : reference dictionary for validation
       compare_dict (dict) : dictionary to validate
    """
    for k in valid_dict:
        elem = valid_dict[k]
        if k in compare_dict:
            if type(compare_dict[k]) != elem['type']:
                raise TypeError('Config error: key ' + k +
                                ' should be of type ' + str(elem['type']))
            if elem['type'] == dict and 'keys' in elem:
                _check_types_recursively(elem['keys'], compare_dict[k])

            if elem['type'] == unicode and 'path' in elem and elem['path']:
                if not os.access(compare_dict[k], os.F_OK):
                    raise IOError('Config error: path specified by key ' +
                                  k + ' not found. Path: ' + compare_dict[k])
        else:
            if 'optional' not in elem or not elem['optional']:
                raise KeyError('Config error: mandatory key ' +
                                   k + ' is missing')

class DSolverBase(object):
    """
    Base class for all distributed solvers.
    Implements logging and utility functions.
    """

    def __init__(self, local_solver_cls, dsolver_config_file):
        """
        Sets up basic config.

        **Args**:
           local_solver (obj) : Solver object class which incapsulates model and training algorithm.
                                Solver should be picklable and have standard interface.
           dsolver_config_file (str) : Configuration filename in json format
        """
        self._config = json.load(open(dsolver_config_file, 'r'))
        self._validate_config(self._config)
        self._web_ui = self._config.get('web_ui', False)
        self.local_solver_cls = local_solver_cls

        self.logger = logging.getLogger('distributed_training_logger')
        self.logger.propagate = False
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler('training_' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S') + '.log')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def _validate_config(self, dsolver_config):
        """
        Checks if the configuration is valid. Raises exception in case it is not

        **Args**:
           dsolver_config (dict) : dictionary containing keys of parsed config file
        """
        valid_keys = {'web_ui'  : {'type': bool}, # Use or not use web interface to monitor training process
                      'cluster' : {'type': dict, 'keys': {}},
                      'training': {'type': dict, 'keys': {}}}
        valid_keys['cluster']['keys'] = \
            {'working_dir'           : {'type': unicode}, # Working directory, where all intermediate results including logs and snapshots will be saved
             'train_db_name'         : {'type': unicode, 'path': True, 'optional': True}, # Location of train database
             'test_db_name'          : {'type': unicode, 'path': True, 'optional': True}, # Location of train database
             'dependencies_path'     : {'type': unicode, 'path': True}, # Path to the build facade
             'facade_name'           : {'type': unicode}, # Name of the facade to use
             'solver_def_file'       : {'type': unicode, 'path': True}, # Solver definition file
             'model_def_file'        : {'type': unicode, 'path': True, 'optional': True}, # Model definition file
             'metainfo_file'         : {'type': unicode, 'path': True, 'optional': True}, # Meta information to use during training
             'log_prefix'            : {'type': unicode}, # Prefix for log files
             'warm_start_resume_file': {'type': unicode, 'path': True, 'optional': True}, # Snapshot file
             'net_snapshot_file'     : {'type': unicode, 'path': True, 'optional': True}} # Snapshot file
        valid_keys['training']['keys'] = \
            {'gradient_sync_interval': {'type': int}, # nsync parameter value (number of iterations between synchronizations of master and worker)
             'warm_start'            : {'type': bool}} # Use or not to use warm start
        _check_types_recursively(valid_keys, dsolver_config)

    def __enter__(self):
        if self._web_ui:
            flask_thread = Thread(target=self.start_ui)
            flask_thread.daemon = True
            flask_thread.start()
        return self

    def __exit__(self, type, value, traceback):
        pass

    def start_ui(self):
        from flask import Flask
        from flask import render_template

        app = Flask(__name__)

        @app.route('/')
        def index():
            return render_template('web_ui.html')

        @app.route('/plots.html')
        def plots():
            return render_template('nvd3_line.html')

        @app.route('/plots2.html')
        def nvd3_line():
            return render_template('plots.html')

        @app.route('/summary', methods=['GET'])
        def summary():
            # Return basic information about training session
            return self.summary()

        @app.route('/training_logs', methods=['GET'])
        def traininge_logs_flask():
            if self.log:
                return json.dumps(self.log)
            else:
                return 'No logs yet'

        print 'Listening to 0.0.0.0:8080...'
        app.run(host='0.0.0.0', port = 8080, debug=True, threaded=True, use_reloader=False)

    def _log_training(self):
        """
        Log training process, and save snapshots.
        Method should be called at every iteration for correct work.
        """

        # Display training information
        if self.local_solver.display and\
           self.local_solver.iter -  self.iter_old['display'] >= self.local_solver.display:
            self.logger.info(IFL + 'Iter = ' + str(self.local_solver.iter) +
                             ', train_loss = ' + str(self.local_solver.train_loss))
            self.logger.info(IFL + 'Iter = {0:d}, train_loss = {1:f}, accuracy = {2:f}'
                             .format(self.local_solver.iter, self.local_solver.train_loss,
                                     self.local_solver.train_accuracy))
	    self.log['train']['time'].append(time.time() - self._solve_start)
            self.log['train']['iter'].append(self.local_solver.iter)
            self.log['train']['loss'].append(self.local_solver.train_loss)
            self.local_solver.output_train_loss()
            self.local_solver.output_train_info()
            self.local_solver.output_learning_rate()
            self.iter_old['display'] = (self.local_solver.iter / self.local_solver.display) * self.local_solver.display

        # Test model and display test results
        if self.local_solver.test_interval and\
           self.local_solver.iter - self.iter_old['test'] >= self.local_solver.test_interval:
            self.logger.info(IFL + 'Testing...')
            start_time = time.time()
            self.local_solver.run_test()
            self.logger.info(IFL + 'Testing: Iter = ' +
                             str(self.local_solver.iter) + ', time = ' +
                             str(time.time() - start_time))
	    self.logger.info(IFL + 'Test net output #0: accuracy = ' + str(self.local_solver.val_accuracy))
            self.logger.info(IFL + 'Test net output #1: loss = ' + str(self.local_solver.val_loss))
            self.iter_old['test'] = (self.local_solver.iter / self.local_solver.test_interval) * self.local_solver.test_interval

        # Snapshot model
        if self.local_solver.snapshot_interval and\
           self.local_solver.iter - self.iter_old['snapshot'] >= self.local_solver.snapshot_interval:
            self.logger.info(IFL + 'Saving snapshot...')
            start_time = time.time()
            self.local_solver.iter -= 1
            #self.dview['local_solver.iter'] = self.local_solver.iter
            #self.dview.execute('local_solver.snapshot()', block = False)
            self.local_solver.snapshot()
            self.local_solver.iter += 1
            self.logger.info(IFL + 'Saving snapshot: Iter = ' +
                             str(self.local_solver.iter) + ', time = ' +
                             str(time.time() - start_time))
            self.iter_old['snapshot'] = (self.local_solver.iter / self.local_solver.snapshot_interval) * self.local_solver.snapshot_interval
            self.dump_log()

    def _training_log_reset(self):
        """
        Clear training log
        """
        self.log = dict(train=dict(time=[], iter=[], loss=[]))
        self.iter_old = dict()
        self.iter_old['display'] = 0
        self.iter_old['test'] = 0
        self.iter_old['snapshot'] = 0

    def dump_log(self):
        """
        Save log into file.
        """
        filename = self._cluster_config['log_prefix'] +\
            datetime.datetime.fromtimestamp(time.time()).\
            strftime('%Y-%m-%d%H:%M:%S')
        f = open(filename, 'wb')
        pickle.dump(self.log, f, -1)
        f.close()

