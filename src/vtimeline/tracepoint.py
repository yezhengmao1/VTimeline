import os
import time
import queue
import atexit
import ctypes
import logging
import threading
from pathlib import Path
from typing import List, Dict
import logging.handlers

################################
### Tracepoint for vtimeline ###
################################

_ROTATE_FILE_COUNT = 5
_ROTATE_FILE_MAX_SIZE = 200 * 1024 * 1024

try:
    import torch
    import torch.distributed

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import triton

    TRITON_AVAILABLE = True
    from .triton_marker import BEGIN_KERNEL_FUNCS, END_KERNEL_FUNCS, TOTAL_MARKER_NUM

    G_TP_NAME_TO_KERNEL_IDX = {}
except ImportError:
    TRITON_AVAILABLE = False


class VTimeLineLogger(logging.Logger):
    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)

    def info_rank0(self, msg, *args, **kwargs):
        if TORCH_AVAILABLE and torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                super()._log(logging.INFO, msg, args, **kwargs)
        else:
            super()._log(logging.INFO, msg, args, **kwargs)


logging.setLoggerClass(VTimeLineLogger)
VLogger = logging.getLogger("VLog")


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


class MetricFormatter(logging.Formatter):
    def __init__(self):
        self.pid = os.getenv("RANK", -1)  # global rank

    def format(self, record: logging.LogRecord):
        try:
            format_str = ",".join(
                [
                    str(int(record.created * 1000000)),  # microsecond
                    str(self.pid),  # global rank
                    record.getMessage(),  # memory size bytes
                ]
            )
        except Exception as e:
            format_str = f"error logger format : {str(e)}"

        return format_str


class TracePoint:
    def __init__(self, event_name: str, cat_name: str, stream=None):
        self.logger = logging.getLogger("TracePoint")
        self.name = event_name
        self.cat = cat_name
        if TRITON_AVAILABLE and TORCH_AVAILABLE:
            self.gpu_stream = stream
        else:
            self.gpu_stream = None

    def begin(self):
        if (
            self.gpu_stream is not None
            and TRITON_AVAILABLE
            and TORCH_AVAILABLE
            and CUPTI.is_enable
        ):
            if self.name not in G_TP_NAME_TO_KERNEL_IDX:
                if len(G_TP_NAME_TO_KERNEL_IDX) >= TOTAL_MARKER_NUM:
                    idx = 0
                else:
                    idx = len(G_TP_NAME_TO_KERNEL_IDX)
                G_TP_NAME_TO_KERNEL_IDX[self.name] = idx
                VLogger.info(f"TracePoint {self.name} marker id {idx}")

            kernel_func = BEGIN_KERNEL_FUNCS[G_TP_NAME_TO_KERNEL_IDX[self.name]]
            with torch.cuda.stream(self.gpu_stream):
                kernel_func[(1,)]()
        self.record(self.name, self.cat, "B")

    def end(self):
        if (
            self.gpu_stream is not None
            and TRITON_AVAILABLE
            and TORCH_AVAILABLE
            and CUPTI.is_enable
        ):
            if self.name not in G_TP_NAME_TO_KERNEL_IDX:
                VLogger.warn(f"TracePoint {self.name} without BEGIN!!!")
                return

            kernel_func = END_KERNEL_FUNCS[G_TP_NAME_TO_KERNEL_IDX[self.name]]
            with torch.cuda.stream(self.gpu_stream):
                kernel_func[(1,)]()
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


class MemRecorder:
    def __init__(self):
        raise RuntimeError("Use initialize to init MemRecorder")

    @staticmethod
    def record():
        if not TORCH_AVAILABLE:
            return

        free, _ = torch.cuda.mem_get_info(torch.cuda.current_device())

        MetricRecorder.record("GPUMem", str(free))


class MetricRecorder:
    _logger_map: Dict[str, logging.Logger] = {}

    @classmethod
    def initialize(cls, logger_names: List[str]):
        for name in logger_names:
            MetricRecorder._logger_map[name] = logging.getLogger(name)

    def __init__(self):
        raise RuntimeError("Use initialize to init MetricRecorder")

    @staticmethod
    def record(logger_name: str, value: str):
        if logger_name not in MetricRecorder._logger_map:
            VLogger.warn("No MetricRecorder : {}".format(logger_name))
            return
        logger = MetricRecorder._logger_map[logger_name]
        logger.info(f"{value}")

    @staticmethod
    def record_rank0(logger_name: str, value: str):
        if TORCH_AVAILABLE and torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                MetricRecorder.record(logger_name, value)
        else:
            MetricRecorder.record(logger_name, value)


class CUPTI:
    _lib = None
    is_enable = False
    enable_times = 0

    @classmethod
    def initialize(
        cls, lib_path: str = os.path.join(os.path.dirname(__file__), "libvtimeline.so")
    ):
        if not os.path.isfile(lib_path):
            print(" >>> [vtimeline] no cupti lib to trace cuda activity.")
            cls._lib = None
            return

        cls._lib = ctypes.CDLL(lib_path)

        cls._lib.enable_vtimeline.argtypes = []
        cls._lib.enable_vtimeline.restype = ctypes.c_int

        cls._lib.disable_vtimeline.argtypes = []
        cls._lib.disable_vtimeline.restype = ctypes.c_int

        cls._lib.init_vtimeline.argtypes = []
        cls._lib.init_vtimeline.restype = ctypes.c_int

        cls._lib.deinit_vtimeline.argtypes = []
        cls._lib.deinit_vtimeline.restype = ctypes.c_int

        CUPTI._lib.init_vtimeline()

        thread = threading.Thread(target=cls._monitor_cupti_flag, daemon=True)
        thread.start()

        atexit.register(CUPTI._lib.deinit_vtimeline)

    def __init__(self):
        raise RuntimeError("Use initialize to init CUPTI")

    @staticmethod
    def enable():
        if CUPTI.enable_times <= 0 or CUPTI.is_enable:
            return

        tp = TracePoint("cupti-enable", "CUPTI")
        tp.begin()
        if CUPTI._lib is None:
            return
        if CUPTI._lib.enable_vtimeline() != 0:
            print("Failed to enable CUDAVTimeline")
        tp.end()
        CUPTI.is_enable = True
        CUPTI.enable_times -= 1

    @staticmethod
    def disable():
        if not CUPTI.is_enable:
            return

        tp = TracePoint("cupti-disable", "CUPTI")
        tp.begin()
        if CUPTI._lib is None:
            return
        CUPTI._lib.disable_vtimeline()
        tp.end()
        CUPTI.is_enable = False

    @classmethod
    def _monitor_cupti_flag(cls):
        cupti_flag_dir = Path(os.getenv("CUPTI_HOME", "/tmp"))
        last_check_time = None

        print(" >>> [vtimeline] CUPTI monitor thread started.")

        while True:
            time.sleep(5)

            if (
                not cupti_flag_dir.exists()
                or not cupti_flag_dir.is_dir()
                or CUPTI.enable_times > 0
            ):
                continue

            cupti_files = list(cupti_flag_dir.glob(".cupti.*"))

            if len(cupti_files) <= 0:
                continue

            # only get one file
            cupti_file = cupti_files[0]

            if not cupti_file.is_file() or (
                last_check_time is not None
                and cupti_file.stat().st_mtime <= last_check_time
            ):
                continue

            try:
                enable_count = int(cupti_file.name[7:])
                print(
                    f" >>> [vtimeline] CUPTI can now be enabled {enable_count} times."
                )
                CUPTI.enable_times = enable_count
                last_check_time = cupti_file.stat().st_mtime

            except ValueError:
                continue


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
    log_dir = Path(log_root_dir) / logger_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # set logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(
        __create_async_rotating_file_handler(
            log_dir / f"rank_{os.getenv('RANK', -1)}.log",
            formatter,
        )
    )
    logger.propagate = False


_vinit_initialized = False


def tracepoint_module_setup(metrics: List[str] = None):
    log_dir = os.getenv("VTIMELINE_LOGGER_DIR", "/var/log")
    metric_dir = log_dir + "/Metrics"

    default_formatter = logging.Formatter(
        fmt="[%(levelname)s][%(process)d][%(name)s][%(asctime)s] %(message)s"
    )

    __create_logger(log_dir, "VLog", default_formatter)
    __create_logger(log_dir, "TracePoint", TracePointFormatter())
    for metric in metrics:
        __create_logger(metric_dir, metric, MetricFormatter())


def vinit(metrics: List[str] = ["GPUMem", "MFU"]):
    global _vinit_initialized
    if _vinit_initialized:
        return

    tracepoint_module_setup(metrics)
    CUPTI.initialize()
    MetricRecorder.initialize(metrics)
    _vinit_initialized = True
