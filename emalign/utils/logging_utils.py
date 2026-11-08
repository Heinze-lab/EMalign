import logging

def setup_logging(level=logging.INFO):
    '''Set up and change logging format'''
    formatter = logging.Formatter(
        fmt='%(message)s'
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)