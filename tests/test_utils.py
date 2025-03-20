class MockFile:
    """Class for testing parsers that require a file.

    Usage:
    @pytest.fixture
    def file_mock(tmp_path):
       file = tmp_path / _file_name
       return MockFile(file, string_contents)
    """

    def __init__(self, file, string):
        # File object
        self.file = file
        # File contents
        self.string = string
        # Name prepended by path
        self.full_path = self.file.as_posix()
        # Write file
        self.file.write_text(self.string)
