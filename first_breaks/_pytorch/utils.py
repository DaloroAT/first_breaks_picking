from pathlib import Path
from typing import Union

from first_breaks.const import CACHE_FOLDER
from first_breaks.utils.utils import download_and_validate_file

SUPPORTED_TORCH = '2.0'


def is_torch_available() -> bool:
    try:
        import torch
        return True
    except (ModuleNotFoundError, ImportError):
        return False


def is_torch_cuda_available() -> bool:
    if is_torch_available():
        import torch
        return torch.cuda.is_available()
    else:
        return False


def raise_if_no_torch() -> None:
    if not is_torch_available():
        raise ModuleNotFoundError("To use the features of 'torch', you need to install it. "
                                  'Follow this official link: https://pytorch.org/get-started/locally/')
    else:
        import torch
        if not torch.__version__.startswith(SUPPORTED_TORCH):
            raise ImportError(f"Now the library works only with version '{SUPPORTED_TORCH}' of 'torch'. "
                              'We may loosen the requirements in the future. '
                              'Follow this official link to install supported version: '
                              'https://pytorch.org/get-started/locally/')


MODEL_TORCH_PATH = CACHE_FOLDER / "fb_torch_unet3plus_11483cfe0c10f32a4bedc8b1351054eb.pth"
MODEL_TORCH_URL = "https://oml.daloroserver.com/download/seis/fb_torch_unet3plus_11483cfe0c10f32a4bedc8b1351054eb.pth"
MODEL_TORCH_HASH = "11483cfe0c10f32a4bedc8b1351054eb"


def download_model_torch(
    fname: Union[str, Path] = MODEL_TORCH_PATH, url: str = MODEL_TORCH_URL, md5: str = MODEL_TORCH_HASH
) -> Union[str, Path]:
    return download_and_validate_file(fname=fname, url=url, md5=md5)
