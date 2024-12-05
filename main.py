from src import check_file, Galaxies

if __name__ == '__main__':
    DATA_PATH = 'data/Galaxy10_DECals.h5'
    checked = check_file(DATA_PATH)
    # checked = True
    if checked:
        dataset = Galaxies(
            path="data/Galaxy10_DECals.h5",
            transform=None,
            gray=None)

        print(f"Tamanho do dataset: {len(dataset)}")

        img, label = dataset[-1]
        print("Imagem shape:", img.shape)
        print("Label:", label)
