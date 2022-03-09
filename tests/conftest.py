
def pytest_addoption(parser):
    parser.addoption('--data_path', type=str, action='store', default=None)
    parser.addoption('--imagenet_path', type=str, action='store', default=None)