import torch
import src
import torch.nn as nn
import torch.optim as optim

if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 1
    PATH = 'database/Binary_2_5_dataset.h5'
    epoch = 0

    dataloader = src.GalaxiesDataLoader(
        path=PATH,
        batch_size=BATCH_SIZE,
        size=1,
        as_gray=False)

    train, val, test = dataloader.split()
    print(f'Train size: {len(train)}')
    print(f'Validation size: {len(val)}')
    print(f'Test size: {len(test)}')

    model = src.GalaxyClassifier().to(DEVICE)
    metrics = src.Metrics()
    es = src.EsMonitor()

    loss = nn.BCELoss()

    optimizer = optim.Adam(params=model.parameters(), lr=0.001)
    lr_sched = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode='min', patience=5, cooldown=5
    )
