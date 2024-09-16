import logging
from typing import Any
from pathlib import Path

class Logger(object):
    def __init__(self, name: str = None):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Ensure handlers are not duplicated
        if not self.logger.hasHandlers():
            self._add_console_handler()
            self._add_file_handler()

    def _add_console_handler(self):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self.formatter)
        self.logger.addHandler(console_handler)

    def _add_file_handler(self):
        log_dir = Path(r"logs")
        log_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        log_file_path = log_dir / f"{self.name}.log"
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)

    def get_logger(self):
        return self.logger
    
    def log(self, level: str, message: str, error: Any = None, params: Any = None):
        if error:
            message = f"{message} | Error: {error}"
        if params:
            message = f"{message} | Params: {params}"

        log_method = getattr(self.logger, level.lower(), None)
        if callable(log_method):
            log_method(f"{message}\n\n")
        else:
            self.logger.error(f"Invalid log level: {level}. Message: {message}")



import time
from functools import wraps

# Define the timer decorator that accepts a logger
def timer(logger=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()  # Record the start time
            result = func(*args, **kwargs)  # Call the function
            end_time = time.time()  # Record the end time
            execution_time = end_time - start_time  # Calculate execution time

            # Log the execution time using the provided logger
            if logger:
                logger.log("info", f"> Function(s): '[{func.__name__}]'. Runtime: [<OK> in {execution_time:.4f} seconds]. Mode [Sync]. Environment: [{func.__module__}].")
            else:
                print("No logger provided. Execution time will not be logged.")
                print(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds")
            
            return result
        return wrapper
    return decorator

# Set up a custom logger
logging


def async_timer(logger=None):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()  # Record the start time
            result = await func(*args, **kwargs)  # Await the async function
            end_time = time.time()  # Record the end time
            execution_time = end_time - start_time  # Calculate execution time

            # Log the execution time using the provided logger
            if logger:
                logger.log("info", f"> Function(s): '[{func.__name__}]'. Runtime: [<OK> in {execution_time:.4f} seconds]. Mode [Async]. Environment: [{func.__module__}].")
            else:
                print("No logger provided. Execution time will not be logged.")
                print(f"Async function '{func.__name__}' executed in {execution_time:.4f} seconds")
            
            return result
        return wrapper
    return decorator