import collections
import contextlib
import logging
import pathlib
import time

_LOGGER = logging.getLogger("femto.timer")


class TimerSingleton:
    """Singleton class to store timing information."""

    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super(TimerSingleton, cls).__new__(cls)
            cls.__instance._initialized = False

        return cls.__instance

    def __init__(self):
        if not self._initialized:
            self._timings = collections.defaultdict(list)
            self._label_chain = []
            self._initialized = True

    @contextlib.contextmanager
    def timeit(self, label: str, extra: str | None = None):
        """Time the wrapped code."""
        extra = "" if extra is None else f" {extra.lstrip()}"

        try:
            self._label_chain.append(label)
            full_label = " > ".join(self._label_chain)

            start_time = time.time()
            yield
            end_time = time.time()

            elapsed_time = end_time - start_time

            _LOGGER.info(f"{full_label}{extra} took {elapsed_time:.4f} s.")
            self._timings[full_label].append(elapsed_time)
        finally:
            self._label_chain.pop()

    def print_statistics(self):
        import pandas

        rows = []
        for label, times in self._timings.items():
            rows.append(
                {
                    "label": label,
                    "mean": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                    "count": len(times),
                }
            )

        table = pandas.DataFrame(rows)
        print(table)


def init_timer_logging(path: pathlib.Path | None = None):
    while len(_LOGGER.handlers) > 0:
        _LOGGER.removeHandler(_LOGGER.handlers[0])

    _LOGGER.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)

    _LOGGER.addHandler(console_handler)

    if path is not None:
        file_handler = logging.FileHandler(path)
        file_handler.setLevel(logging.INFO)

        file_formatter = logging.Formatter("%(message)s")
        file_handler.setFormatter(file_formatter)

        _LOGGER.addHandler(file_handler)


timer = TimerSingleton()
