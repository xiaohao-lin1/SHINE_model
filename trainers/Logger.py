import importlib
import logging
import os


class Logger:
    def __init__(self, logger_path):
        log_file_name = 'log.log'
        self.log_file_name = os.path.join(logger_path, log_file_name)
        self.logger = self.get_logger()
    def get_logger(self):
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(self.log_file_name)
        c_handler.setLevel(logging.ERROR)
        f_handler.setLevel(logging.INFO)

        # Create formatters and add it to handlers
        c_format = logging.Formatter('%(filename)s - %(levelname)s - %(message)s')
        f_format = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

        return logger
    def log_keyword_arguments(self, **kwargs):
        for (k, v) in kwargs.items():
            self.logger.info(f'{k}: {v}')
    #todo: test the close function
    def close(self):
        '''
        Close the logger
        :return:
        '''
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)
    def log(self, message):
        self.logger.info(message)
if __name__ == '__main__':
    logger_path = '/home/stulinxh/korean_u_11_class/results/DCN_new_subject_independent_1_fold/logger/'
    logger = Logger(logger_path)
    logger.log_keyword_arguments(a=1, b=2, c=3)
    logger.close()
