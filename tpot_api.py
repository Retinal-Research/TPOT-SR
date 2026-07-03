import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.utils import save_image

from model.model_LC import _NetG


IMAGE_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _list_image_files(input_dir: str) -> List[str]:
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_files = [
        file_name
        for file_name in sorted(os.listdir(input_dir))
        if os.path.isfile(os.path.join(input_dir, file_name))
        and os.path.splitext(file_name)[1].lower() in IMAGE_EXTENSIONS
    ]
    return image_files


class FolderImageDataset(Dataset):
    def __init__(self, input_dir: str, image_size: int = 256):
        self.input_dir = input_dir
        self.image_files = _list_image_files(input_dir)
        self.transform = Compose([
            Resize((image_size, image_size)),
            ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int):
        image_name = self.image_files[index]
        image_path = os.path.join(self.input_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return {"image": image, "name": image_name}


def _resolve_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("adapter_"):
            continue
        name = key[5:] if key.startswith("base.") else key
        cleaned[name] = value
    return cleaned


@dataclass
class TPOTEnhancer:
    checkpoint_path: str
    device: Optional[str] = None
    image_size: int = 256

    def __post_init__(self):
        resolved_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = str(resolved_device)
        self.torch_device = torch.device(self.device)
        self.model = self._load_model()

    def _load_model(self) -> _NetG:
        checkpoint = torch.load(self.checkpoint_path, map_location=self.torch_device)
        model = _NetG().to(self.torch_device)
        model.load_state_dict(_resolve_state_dict(checkpoint), strict=True)
        model.eval()
        return model

    def enhance_tensor_batch(self, image_batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            output = self.model(image_batch.to(self.torch_device))
            return torch.clamp(output, min=0.0, max=1.0)

    def enhance_folder(
        self,
        input_dir: str,
        output_dir: str,
        batch_size: int = 1,
        num_workers: int = 0,
    ) -> List[str]:
        dataset = FolderImageDataset(input_dir=input_dir, image_size=self.image_size)
        if len(dataset) == 0:
            os.makedirs(output_dir, exist_ok=True)
            return []

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        os.makedirs(output_dir, exist_ok=True)
        saved_paths: List[str] = []

        for batch in dataloader:
            images = batch["image"]
            names: Sequence[str] = batch["name"]
            outputs = self.enhance_tensor_batch(images).cpu()

            for index, image_name in enumerate(names):
                save_path = os.path.join(output_dir, image_name)
                save_image(outputs[index], save_path)
                saved_paths.append(save_path)

        return saved_paths


def enhance_folder(
    checkpoint_path: str,
    input_dir: str,
    output_dir: str,
    batch_size: int = 1,
    image_size: int = 256,
    num_workers: int = 0,
    device: Optional[str] = None,
) -> List[str]:
    enhancer = TPOTEnhancer(
        checkpoint_path=checkpoint_path,
        device=device,
        image_size=image_size,
    )
    return enhancer.enhance_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )
