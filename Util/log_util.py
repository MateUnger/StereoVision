import logging
import sys
import os


def get_logger(script_name: str) -> logging.Logger:
    """
    Configures and returns a logger that outputs to both console and file.

    Args:
        script_name (str): The name of the script for which the logger is being configured.

    Returns:
        logging.Logger: Configured logger instance.

    Usage:
        from Util.log_util import get_logger
        log = get_logger(os.path.basename(__file__).split(".")[0])
        log.info("abrakadabra")
    """

    # if there is no "Logs" folder, make one in the main directory
    if not os.path.exists("Logs"):
        os.makedirs("Logs")

    # if there is no log file for the script, create one
    # TODO: maybe find the file name dynamically
    log_filename = os.path.join("Logs", f"{script_name}.log")

    if not os.path.exists(log_filename):
        with open(log_filename, "w"):
            pass

    # Configure the logger for both console and file output
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_filename, mode="a"),  # 'a' - append, 'w' - overwrite
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Create and return the configured logger
    return logging.getLogger(script_name)
