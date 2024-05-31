from pathlib import Path
from typing import Optional, Union


class LastFolderManager:
    def __init__(self) -> None:
        self.last_folder: Optional[str] = None

    def get_last_folder(self) -> str:
        if self.last_folder is None:
            return str(Path.home())
        else:
            if Path(self.last_folder).exists():
                return str(Path(self.last_folder).resolve())
            else:
                return str(Path.home())

    def set_last_folder(self, path: Union[str, Path]) -> None:
        path = Path(path)
        if path.exists():
            self.last_folder = str(path.parent) if path.is_file() else str(path)
        else:
            self.last_folder = None


last_folder_manager = LastFolderManager()
