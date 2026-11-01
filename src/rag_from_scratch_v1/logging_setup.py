import logging
import sys

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"

def setup_log(level: str = "INFO", log_file:str|None = None):
    handler = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handler.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=level, format=LOG_FORMAT, handlers=handler, force=True)

    for noisy in ("chromadb", "httpx", "httpcore", "urllib3", "httpx2", "mlflow"):
        logging.getLogger(noisy).setLevel(logging.WARNING)