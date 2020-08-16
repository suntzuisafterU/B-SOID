"""
Encapsulate logging for BSOID
"""


import logging


import bsoid.config as config


# def create_generic_logger(name: str,
#                           stdout_log_level: str = None,
#                           file_log_level: str = None,
#                           email_log_level: str = None):
#     """
#     Generic logger instantiation.
#     :param stdout_log_level:
#     :param file_log_level:
#     :param email_log_level:
#     """
#     formatter = config.configuration['LOGGING'][]
#     formatter = settings.LoggerVar.standard_formatter
#     logger = logging.getLogger(name)  # TODO: use __name__ ? else?
#     logger.setLevel(logging.DEBUG)
#     if stdout_log_level:
#         stream_handler = logging.StreamHandler()
#         stream_handler.setLevel(log_level_string_to_log_level[stdout_log_level.lower()])
#         stream_handler.setFormatter(formatter)
#         logger.addHandler(stream_handler)
#     if file_log_level:
#         file_handler = logging.FileHandler(settings.LoggerVar.default_logfile_path)
#         file_handler.setLevel(log_level_string_to_log_level[file_log_level.lower()])
#         file_handler.setFormatter(formatter)
#         logger.addHandler(file_handler)
#     # if email_log_level:  # TODO
#         # smtp_handler = SMTPHandler()
#         # smtp_handler.setLevel(log_level_string_to_log_level[email_log_level.lower()])
#         # smtp_handler.setFormatter(formatter)
#         # logger.addHandler(smtp_handler)
#         # pass
#     return logger