"""

"""

from kedro.io import AbstractDataSet

class HTMLDataSet(AbstractDataSet):
    def __init__(self, filepath):
        self._filepath = filepath

    def _load(self):
        with open(self._filepath, "r") as f:
            return f.read()

    def _save(self, data):
        with open(self._filepath, "w") as f:
            f.write(data)

    def _describe(self):
        return dict(filepath=self._filepath)
