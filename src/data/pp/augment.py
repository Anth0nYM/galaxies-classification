import albumentations as A


class Augment:
    def __init__(self, p: float = 0.15) -> None:
        self.__p = p

    def __call__(self) -> A.Compose:
        return A.Compose([
            A.RandomResizedCrop(height=256, width=256, scale=(0.75, 1.0)),
            A.Rotate(limit=45, p=self.__p),
            A.HorizontalFlip(p=self.__p),
            A.VerticalFlip(p=self.__p),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0,
                rotate_limit=0,
                p=self.__p,
            )
        ])
