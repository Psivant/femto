import abc
import pathlib
import typing

import numpy
import tensorboardX

if typing.TYPE_CHECKING:
    from matplotlib import pyplot


class Reporter(abc.ABC):
    """Base class for reporter classes."""

    @abc.abstractmethod
    def report_figure(self, tag: str, iteration: int | None, figure: "pyplot.Figure"):
        """Report a custom plot."""

    @abc.abstractmethod
    def report_scalar(self, tag: str, iteration: int, value: float):
        """Report a scalar value."""

    @abc.abstractmethod
    def report_matrix(self, tag: str, iteration: int, value: numpy.ndarray):
        """Report a matrix of values."""

    @abc.abstractmethod
    def flush(self):
        """Flush the reporter."""

    @abc.abstractmethod
    def close(self):
        """Close the reporter."""


class NullReporter(Reporter):
    """A dummy reporter that does nothing with reported values."""

    def report_figure(self, tag: str, iteration: int | None, figure: "pyplot.Figure"):
        pass

    def report_scalar(self, tag: str, iteration: int, value: float):
        pass

    def report_matrix(self, tag: str, iteration: int, value: numpy.ndarray):
        pass

    def flush(self):
        pass

    def close(self):
        pass


class TensorboardReporter(Reporter):
    """Report statistics to ``tensorboard`` compatible event files."""

    def __init__(self, log_dir: pathlib.Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = tensorboardX.SummaryWriter(log_dir=str(log_dir))

    def report_figure(self, tag: str, iteration: int | None, figure: "pyplot.Figure"):
        self._writer.add_figure(tag, figure, iteration)

    def report_scalar(self, tag: str, iteration: int, value: float):
        self._writer.add_scalar(tag, value, iteration)

    def report_matrix(
        self,
        tag: str,
        iteration: int,
        value: numpy.ndarray,
        figure_size: tuple[int, int] | None = None,
    ):
        import seaborn
        from matplotlib import pyplot

        # a good size for tensorboard on my Mac... probably needs tweaking
        default_size = 8

        figure_size = (
            figure_size if figure_size is not None else (default_size, default_size)
        )

        figure, axis = pyplot.subplots(1, 1, figsize=figure_size)
        seaborn.heatmap(
            value,
            annot=True,
            annot_kws={"fontsize": 8},
            fmt=".2f",
            square=True,
            ax=axis,
            linewidths=0.5,
            cbar=False,
        )
        self._writer.add_figure(tag, figure, iteration)
        pyplot.close(figure)

    def flush(self):
        self._writer.flush()

    def close(self):
        self._writer.close()
