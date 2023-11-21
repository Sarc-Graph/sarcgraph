from tempfile import TemporaryDirectory


class MockConfig:
    def __init__(self):
        self.temp_dir = TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        self.print_called = False

    def print(self):
        self.print_called = True

    def cleanup(self):
        self.temp_dir.cleanup()
