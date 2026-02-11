

"""Create directories if they do not exist."""
def create_directories(*directories):
    for d in directories:
        d.mkdir(parents=True, exist_ok=True)