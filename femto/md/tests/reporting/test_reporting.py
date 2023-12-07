import numpy
import pytest
import seaborn
from matplotlib import pyplot

from femto.md.reporting import Reporter, TensorboardReporter


class MockReporter(Reporter):
    def report_scalar(self, tag: str, iteration: int, value: float):
        pass

    def report_matrix(self, tag: str, iteration: int, value: numpy.ndarray):
        pass

    def flush(self):
        pass

    def close(self):
        pass

    def report_figure(self, tag: str, iteration: int | None, figure: "pyplot.Figure"):
        pass


class TestTensorboardReporter:
    @pytest.fixture
    def mock_reporter(self, tmp_cwd) -> TensorboardReporter:
        return TensorboardReporter(tmp_cwd / "logs")

    def test_report_scalar(self, mock_reporter, mocker):
        mock_report = mocker.spy(mock_reporter, "report_scalar")

        mock_reporter.report_scalar("some-tag", 123, 456)
        mock_report.assert_called_with("some-tag", 123, 456)

    def test_report_figure(self, mock_reporter, mocker):
        mock_figure = pyplot.gcf()

        mock_add_figure = mocker.spy(mock_reporter._writer, "add_figure")
        mock_reporter.report_figure("some-tag", 123, mock_figure)
        mock_add_figure.assert_called_once_with("some-tag", mock_figure, 123)

    def test_report_matrix(self, mock_reporter, mocker):
        mock_report = mocker.spy(mock_reporter, "report_matrix")
        mock_heatmap = mocker.spy(seaborn, "heatmap")

        mock_reporter.report_matrix("some-tag", 123, numpy.ones((1, 1)))

        mock_report.assert_called_once()
        mock_heatmap.assert_called_once()

    def test_flush(self, mock_reporter, mocker):
        mock_flush = mocker.spy(mock_reporter, "flush")

        mock_reporter.flush()
        mock_flush.assert_called_once_with()

    def test_close(self, mock_reporter, mocker):
        mock_close = mocker.spy(mock_reporter, "close")

        mock_reporter.close()
        mock_close.assert_called_once_with()
