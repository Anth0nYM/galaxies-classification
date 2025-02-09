import torch
import src
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    AUGMENT = False
    denoise = src.Denoiser().mean
    GRAY = False
    denoise_name = denoise.__name__

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {DEVICE}')
    BATCH_SIZE = 64
    PATH = 'database/Binary_2_5_dataset.h5'
    EPOCH_LIMIT = 1000

    dataloader = src.GalaxiesDataLoader(
        path=PATH,
        batch_size=BATCH_SIZE,
        as_gray=GRAY,
        augment=AUGMENT,
        denoise=denoise,
        size=1)

    train, val, test = dataloader.split()

    model = src.GalaxyClassifier().to(DEVICE)
    es = src.EsMonitor()

    loss = nn.BCELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)
    lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                    mode='min',
                                                    patience=5,
                                                    cooldown=3)

    logger = src.TbLog(comment=f"_{denoise_name}_aug{AUGMENT}_gray{GRAY}")

    aug_flag = "ativado" if dataloader.is_augmented else "desativado"
    logger.log_text(f"O aumento de dados está {aug_flag}")

    for epoch in range(EPOCH_LIMIT):

        # Train
        model.train()
        train_run_loss = []
        train_run_metrics = []

        for img, label in tqdm(train, desc=f"Epoch {epoch} - Train"):
            img, label = img.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            output = model(img)
            train_loss = loss(output, label)
            train_loss.backward()
            optimizer.step()

            # Metrics
            y_true = label.cpu().detach().round()
            y_pred = output.cpu().detach().round()
            metrics_report = src.ClassificationReport(yt=y_true, yp=y_pred)
            train_run_metrics.append(metrics_report.get_report())
            train_run_loss.append(train_loss.item())

        # Logging
        avg_train_metrics = {
            key: float(np.mean(
                [
                    m[key] for m in train_run_metrics
                    ])
                ) for key in train_run_metrics[0]}

        avg_train_loss = np.mean(train_run_loss)

        logger.log_metrics(
            split="train",
            epoch=epoch,
            loss=avg_train_loss,
            metrics=avg_train_metrics)

        print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f}, ")

        # Validação
        with torch.no_grad():
            model.eval()
            val_run_loss = []
            val_run_metrics = []

            for img, label in tqdm(val, desc=f"Epoch {epoch} - Validation"):
                img, label = img.to(DEVICE), label.to(DEVICE)
                output = model(img)
                val_loss = loss(output, label)

                # Metrics
                y_true = label.cpu().detach().round()
                y_pred = output.cpu().detach().round()

                metrics_report = src.ClassificationReport(y_true, y_pred)
                val_run_metrics.append(metrics_report.get_report())
                val_run_loss.append(val_loss.item())

            # Logging
            avg_val_metrics = {
                key: float(np.mean(
                    [
                        m[key] for m in val_run_metrics
                        ])
                    ) for key in val_run_metrics[0]}

            avg_val_loss = np.mean(val_run_loss)
            logger.log_metrics(
                split="val",
                epoch=epoch,
                loss=avg_val_loss,
                metrics=avg_val_metrics)

            print(f"Validation Loss: {avg_val_loss:.4f}, ")

            # Early stopping and Lr adjust
            lr_sched.step(avg_val_loss)
            es(avg_val_loss)
            print(f"Current wait: {es.wait}")
            if es.must_stop():
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # Teste
    with torch.no_grad():
        model.eval()
        y_trues = []
        y_preds = []
        fn_samples = []
        fp_samples = []

        for img, label in tqdm(test, desc="Test"):
            img, label = img.to(DEVICE), label.to(DEVICE)
            output = model(img)

            y_trues.append(label.cpu().detach().round())
            y_preds.append(output.cpu().detach().round())

            y_true_batch = label.cpu().detach().round()
            y_pred_batch = output.cpu().detach().round()

            for i in range(img.size(0)):
                true_val = y_true_batch[i].item()
                pred_val = y_pred_batch[i].item()

                if true_val == 1 and pred_val == 0:
                    fn_samples.append((img[i].cpu(), true_val, pred_val))

                elif true_val == 0 and pred_val == 1:
                    fp_samples.append((img[i].cpu(), true_val, pred_val))

        all_y_trues = torch.cat(y_trues, dim=0)
        all_y_preds = torch.cat(y_preds, dim=0)
        cm = src.ConfusionMatrix(all_y_trues, all_y_preds).get_full_matrix()
        final_metrics_report = src.ClassificationReport(
            all_y_trues,
            all_y_preds
            ).get_report()

        print("\n🔥🔥🔥 Test Metrics Summary 🔥🔥🔥\n")
        for key, value in final_metrics_report.items():
            print(f"{key.capitalize()}: {value:.4f}")

        logger.log_cm(cm=cm, epoch=epoch)
        logger.log_metrics(
            split="test",
            epoch=epoch,
            loss=0.0,
            metrics=final_metrics_report
            )

        logger.log_imgs(fns=fn_samples,
                        fps=fp_samples,
                        epoch=epoch)
        logger.close()
