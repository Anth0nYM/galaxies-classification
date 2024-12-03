from src import check_file

if __name__ == '__main__':
    DATA_PATH = 'Galaxy10_DECals.h5'
    if check_file(DATA_PATH):
        print('Arquivo válido!')
