import torch
import src
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {DEVICE}')
    BATCH_SIZE = 1
    PATH = 'database/Binary_2_5_dataset.h5'
    epoch = 0

    dataloader = src.GalaxiesDataLoader(
        path=PATH,
        batch_size=BATCH_SIZE,
        as_gray=False,
        size=0.1)

    train, val, test = dataloader.split()

    model = src.GalaxyClassifier().to(DEVICE)
    metrics = src.Metrics(device=DEVICE)
    es = src.EsMonitor()

    loss = nn.BCELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)
    lr_sched = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode='min', patience=5, cooldown=5
    )

    while epoch < 20:
        epoch += 1

        # Train
        model.train()
        train_run_loss = []
        train_metrics: dict = {metric: [] for metric in metrics.funcs.keys()}

        for img, label in tqdm(train, desc=f"Epoch {epoch} - Train"):
            img, label = img.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()

            img = img.permute(0, 3, 1, 2)
            img = img.float() / 255.0
            label = label.view(-1, 1)
            label = label.float()
            print(img.shape)
            print(type(img))

            output = model(img)
            train_loss = loss(output, label)
            train_loss.backward()
            optimizer.step()

            # Calculate metrics for batch
            batch_metrics = metrics.report(yt=label, yp=output)
            for name, value in batch_metrics.items():
                train_metrics[name].append(value)

            train_run_loss.append(train_loss.item())

        # Log train metrics
        train_results = {
            name: np.mean(values)
            for name, values in train_metrics.items()
        }

        train_loss_mean = np.mean(train_run_loss)
        print(f"Epoch {epoch} - Train Loss: {train_loss_mean:.4f}, "
              f"Metrics: {train_results}")

        breakpoint()

        # Validação
        with torch.no_grad():
            model.eval()
            val_run_loss = []
            val_metrics: dict[str, list] = {
                metric: [] for metric in metrics.funcs.keys()
            }

            for img, label in tqdm(val, desc=f"Epoch {epoch} - Validation"):
                img, label = img.to(DEVICE), label.to(DEVICE)
                output = model(img)
                val_loss = loss(output, label)

                # Calculate metrics for batch
                batch_metrics = metrics.report(yt=label, yp=output)
                for name, value in batch_metrics.items():
                    val_metrics[name].append(value)

                val_run_loss.append(val_loss.item())

            # Log validation metrics
            val_results = {
                name: np.mean(values) for name, values in val_metrics.items()
            }

            val_loss_mean = np.mean(val_run_loss)
            print(f"Epoch {epoch} - Validation Loss: {val_loss_mean:.4f}, "
                  f"Metrics: {val_results}")

            # Early stopping and Lr adjust
            lr_sched.step(np.mean(val_run_loss))
            es(val_loss_mean)
            if es.must_stop():
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    # Teste
    with torch.no_grad():
        model.eval()
        test_metrics: dict = {metric: [] for metric in metrics.funcs.keys()}

        for img, label in tqdm(test, desc="Test"):
            img, label = img.to(DEVICE), label.to(DEVICE)
            output = model(img)

            # Calculate metrics for batch
            batch_metrics = metrics.report(yt=label, yp=output)
            for name, value in batch_metrics.items():
                test_metrics[name].append(value)

        # Log test metrics
        test_results = {
            name: np.mean(values)
            for name, values in test_metrics.items()
        }
        print(f"Test Metrics Summary: {test_results}")
