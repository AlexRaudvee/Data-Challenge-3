import logging

# define a formatter to display the messages to console (standard output)
console_formatter = logging.Formatter('%(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(console_formatter)

# define a logger for this package and attach the console handler
logger = logging.getLogger('plaspix')
# logger.handlers.clear()
logger.propagate = False
logger.addHandler(console_handler)

# set an appropriate level of logging for this package
logger.setLevel(logging.DEBUG)