import os


def check_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def get_logger(path):
    """
    Returns a logger object with a Log method to write messages to the specified file.

    Args:
        path (str): Path to the log file.

    Returns:
        object: Logger with a Log method.
    """

    class Logger:
        def __init__(self, file_path):
            self.file_path = file_path
            # Ensure the directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        def Log(self, content):
            """
            Writes the given content to the log file with a newline.

            Args:
                content (str): Content to be logged.
            """
            with open(self.file_path, "a") as file:
                file.write(content + "\n")

    return Logger(path)


if __name__ == "__main__":
    logger = get_logger("lab1/part1/logs/example.log")
    logger.Log("This is a log message.")
    logger.Log("Another log entry.")
