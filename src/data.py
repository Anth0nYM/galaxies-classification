class DataLoader:
    def __init__(self, batch_size, subset_size, as_gray, augment):
        self.batch_size = batch_size
        self.subset_size = subset_size
        self.as_gray = as_gray
        self.augment = augment

    def split(self, seed):
        pass
