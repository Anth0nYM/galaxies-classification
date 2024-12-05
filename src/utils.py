import hashlib
import os
from tqdm import tqdm


def check_file(file_path: str) -> bool:
    """Verifica se o dataset baixado é o mesmo que o esperado.
    O dataset pode ser encontrado em: https://github.com/henrysky/Galaxy10

    Args:
        file_path (str): Caminho para o arquivo a ser verificado.

    Returns:
        bool: True se o arquivo é válido, False caso contrário.
    """
    expected = (
        '19AEFC477C41BB7F77FF07599A6B82A038DC042F889A111B0D4D98BB755C1571'
    )
    sha256_hash = hashlib.sha256()
    file_size = os.path.getsize(file_path)
    chunk_size = 4096
    with open(file_path, "rb") as f, tqdm(
        total=file_size, unit='B', unit_scale=True, desc="Verificando dataset"
    ) as progress_bar:
        for byte_block in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(byte_block)
            progress_bar.update(len(byte_block))

    calculated_hash = sha256_hash.hexdigest().upper()
    acepted = calculated_hash.upper() == expected
    if not acepted:
        print('Arquivo inválido!')
        print('Verifique o arquivo baixado')
        print(f'Hash esperado: {expected}')
        print(f'Hash calculado: {calculated_hash}')
    else:
        print('Arquivo válido!')
    return acepted
