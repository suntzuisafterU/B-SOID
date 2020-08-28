"""
Encapsulate logging for BSOID
"""

from logging.handlers import SMTPHandler
import logging


def create_generic_logger(logger_name: str, log_format: str,
                          stdout_log_level: str = None,
                          file_log_level: str = None, file_log_file_path: str = None,
                          email_log_level: str = None) -> logging.Logger:
    """
    Generic logger instantiation.

    :param logger_name: (str)
    :param log_format: (str)
    :param stdout_log_level:
    :param file_log_level:
    :param file_log_file_path:
    :param email_log_level:
    """
    valid_log_levels = {'CRITICAL', 'FATAL', 'ERROR', 'WARN', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'}
    formatter = logging.Formatter(log_format)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # Set to DEBUG in order to allow ALL messages to be sent to handlers

    # Logging to console
    if stdout_log_level is not None:
        # Check type
        if not isinstance(stdout_log_level, str):
            raise TypeError(f'`stdout_log_level` was expected to be str but found {type(stdout_log_level)}')
        # Check valid log level
        stdout_log_level = stdout_log_level.upper()
        if stdout_log_level not in valid_log_levels:
            raise ValueError(f'Invalid log level submitted for `std_log_level` (value: {stdout_log_level})')
        # Continue if no errors
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(stdout_log_level.upper())
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # Logging to file
    if file_log_level is not None:
        # Check file level type and value validity
        if not isinstance(file_log_level, str):
            raise TypeError(f'`file_log_level` was expected to be str but instead '
                            f'found: {type(file_log_level)} (value: {file_log_level}).')
        # Check file log level
        file_log_level = file_log_level.upper()
        if file_log_level not in valid_log_levels:
            raise ValueError(f'')
        # Check filepath
        if not isinstance(file_log_file_path, str):
            raise TypeError('argument `file_log_path` was expected to be type str but instead found: '
                            f'{type(file_log_file_path)} (value: {file_log_file_path}).')
        # Check that file path exists up to the folder level
        ## TODO: low

        # Continue if no errors
        file_handler = logging.FileHandler(file_log_file_path)
        file_handler.setLevel(file_log_level.upper())
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    # # Logging to email  # TODO: low: ?
    # if email_log_level:
    #     # mailhost, fromaddr, toaddrs, subject,
    #     #                  credentials=None, secure=None, timeout=5.0
    #     smtp_handler = SMTPHandler()
    #     smtp_handler.setLevel(email_log_level.upper())
    #     smtp_handler.setFormatter(formatter)
    #     logger.addHandler(smtp_handler)
    return logger

