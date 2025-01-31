import git
import datetime


class Git:
    def __init__(self,
                 repo_path: str,
                 ) -> None:
        self.__repo = git.Repo(repo_path)
        self.__sha = self.__repo.head.object.hexsha[:4]
        self.__date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.__commit = self.__repo.head.commit

    def _get_timestamp(self) -> str:
        return self.__date

    def _get_hex(self) -> str:
        return self.__sha

    def __get_details(self) -> str:
        author = str(self.__commit.author)
        commit_message = self.__commit.message

        if isinstance(commit_message, bytes):
            commit_message = commit_message.decode()

        return (f"Author: {author},"
                f"Date: {self.__commit.committed_datetime}, "
                f"Message: {commit_message}")
