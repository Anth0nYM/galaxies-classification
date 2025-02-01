import git
import datetime


class Git:
    def __init__(self,
                 repo_path: str,
                 ) -> None:
        """
        Inicializa a classe Git.

        Args:
            repo_path (str): Caminho para o repositório Git local.
        """
        self.__repo = git.Repo(repo_path)
        self.__sha = self.__repo.head.object.hexsha[:4]
        self.__date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.__commit = self.__repo.head.commit

    def _get_timestamp(self
                       ) -> str:
        """
        Retorna o timestamp atual formatado.

        Returns:
            str: Timestamp no formato 'YYYY-MM-DD_HH-MM-SS'.
        """
        return self.__date

    def _get_hex(self
                 ) -> str:
        """
        Retorna os primeiros 4 caracteres do hash SHA do commit atual.

        Returns:
            str: String representando os primeiros 4 caracteres do hash SHA.
        """
        return self.__sha

    def __get_details(self
                      ) -> str:
        """
        Retorna detalhes do commit atual.

        Returns:
            str: Informações formatadas do commit.
        """
        author = str(self.__commit.author)
        commit_message = self.__commit.message

        if isinstance(commit_message, bytes):
            commit_message = commit_message.decode()

        return (f"Author: {author},"
                f"Date: {self.__commit.committed_datetime}, "
                f"Message: {commit_message}")
