import pandas as pd
from PIL import Image
from typing import Tuple
from torch.utils.data import Dataset


class FlickrDataset(Dataset):
    def __init__(self, root_dir: str, labels_file: str, eos_token: str = "<|endoftext|>"):
        self.root_dir = root_dir
        self.labels_file = labels_file
        self.eos_token = eos_token
        self.images_dir = root_dir + "/images"
        self.data = pd.read_csv(self.labels_file)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> tuple[Image.Image, str]:
        img_name = self.images_dir + "/" + self.data.iloc[idx]["image_name"]
        image = Image.open(img_name)
        image = self._transform_image(image)
        comment = self._transform_comment(self.data.iloc[idx]["comment"])
        return image, comment

    def _transform_image(self, image: Image) -> Image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def _transform_comment(self, comment: str) -> str:
        return comment + self.eos_token