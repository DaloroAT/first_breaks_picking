from typing import Any, Optional

from tqdm.auto import tqdm

from first_breaks.picking.task import Task


class IPicker:
    def __init__(self, show_progressbar: bool = True, *args: Any, **kwargs: Any):
        self.show_progressbar = show_progressbar
        self.progressbar: Optional[tqdm] = None

    def change_settings(self, *args: Any, **kwargs: Any) -> None:
        pass

    def process_task(self, task: Task) -> Task:
        raise NotImplementedError

    def callback_processing_started(self, length: int) -> Any:
        if self.show_progressbar:
            self.progressbar = tqdm(desc="Picking", total=length)

    def callback_processing_finished(self) -> Any:
        if self.show_progressbar:
            self.progressbar.close()

    def callback_step_finished(self, finished_step: int) -> Any:
        if self.show_progressbar:
            interval = finished_step - self.progressbar.n
            self.progressbar.update(interval)
