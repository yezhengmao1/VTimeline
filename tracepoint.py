import os
import queue
import socket
import atexit
import ctypes
from pathlib import Path
import logging.handlers

################################
### Tracepoint for vtimeline ###
################################


_ROTATE_FILE_COUNT = 5
_ROTATE_FILE_MAX_SIZE = 200 * 1024 * 1024


class TracePointFormatter(logging.Formatter):
    def __init__(self):
        self.pid = os.getenv("RANK", -1)  # global rank
        self.tid = 0

    def format(self, record: logging.LogRecord):
        # the chrome trace metirc include
        #   name : event name
        #   cat  : the event categories
        #   ph   : the event type, B-begin E-end
        #   ts   : tracing clock timestamp of the event, mciro sencond
        #   pid  : pid, or global rank
        #   tid  : tid, or stream id

        # must use like below:
        #    logger.info("event_name", extra={"cat": "cat", "pid": pid, "tid": tid, ph: "B"})

        try:
            format_str = ",".join(
                [
                    str(int(record.created * 1000000)),  # microsecond
                    str(self.pid),  # global rank
                    str(self.tid),  # stream_id, default is cpu
                    record.cat,
                    record.getMessage(),
                    record.ph,
                ]
            )
        except Exception as e:
            format_str = f"error logger format : {str(e)}"

        return format_str


class TracePoint:
    def __init__(self, event_name: str, cat_name: str):
        self.logger = logging.getLogger("TracePoint")
        self.name = event_name
        self.cat = cat_name

    def begin(self):
        self.record(self.name, self.cat, "B")

    def end(self):
        self.record(self.name, self.cat, "E")

    def record(self, event_name: str, cat_name: str, ph: str):
        self.logger.info(
            event_name,
            extra={
                "cat": cat_name,
                "ph": ph,
            },
        )

    def __enter__(self):
        self.begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()
        return False


class CUDAVTimeLine:
    def __init__(
        self,
        lib_path="/usr/local/lib/libvtimeline.so",
    ):
        self.lib = ctypes.CDLL(lib_path)

        self.lib.enable_vtimeline.argtypes = []
        self.lib.enable_vtimeline.restype = ctypes.c_int

        self.lib.disable_vtimeline.argtypes = []
        self.lib.disable_vtimeline.restype = ctypes.c_int

    def enable(self):
        if self.lib.enable_vtimeline() != 0:
            print("Failed to enable CUDAVTimeline")

    def disable(self):
        self.lib.disable_vtimeline()


def __create_async_rotating_file_handler(
    log_file_path: Path, formatter: logging.Formatter
):
    # set up handler
    # rorate to 5 file, and each file 200MB
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file_path,
        mode="a",
        maxBytes=_ROTATE_FILE_MAX_SIZE,
        backupCount=_ROTATE_FILE_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    queue_buffer = queue.Queue()

    # queue_handler -> listener -> file_hander
    queue_handler = logging.handlers.QueueHandler(queue_buffer)

    queue_listener = logging.handlers.QueueListener(
        queue_buffer, file_handler, respect_handler_level=True
    )
    queue_listener.start()
    atexit.register(queue_listener.stop)

    return queue_handler


def __create_logger(log_root_dir: str, logger_name: str, formatter: logging.Formatter):
    log_dir = Path(log_root_dir) / logger_name / socket.gethostname()
    log_dir.mkdir(parents=True, exist_ok=True)

    # set logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(
        __create_async_rotating_file_handler(
            log_dir / f"{logger_name}_{os.getenv('LOCAL_RANK', -1)}.log",
        )
    )
    logger.propagate = False


def tracepoint_module_setup():
    log_dir = os.getenv("VTIMELINE_LOGGER_DIR", "/var/log")

    default_formatter = (
        logging.Formatter(
            fmt="[%(levelname)s][%(process)d][%(name)s][%(asctime)s] %(message)s"
        ),
    )

    __create_logger(log_dir, "VTimeLine", format=default_formatter)
    __create_logger(log_dir, "TracePoint", format=TracePointFormatter())


def cudavtimeline_module_setup():
    cupti = CUDAVTimeLine()
    cupti.enable()
    atexit.register(cupti.disable)
