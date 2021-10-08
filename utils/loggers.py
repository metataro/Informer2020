from abc import ABC, abstractmethod
from typing import List


class Logger(ABC):

    @abstractmethod
    def log_config(self, config) -> None: ...

    @abstractmethod
    def log_dict(self, values: dict, step=None) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class NoLogger(Logger):

    def log_config(self, config: dict) -> None:
        pass

    def log_dict(self, values: dict, step=None) -> None:
        pass

    def close(self) -> None:
        pass


class WandbLogger(Logger):
    def __init__(self,
                 name: str = None,
                 project: str = None,
                 entity: str = None,
                 tags: List[str] = None):
        super().__init__()
        import wandb
        self.run = wandb.init(name=name, project=project, entity=entity, tags=tags, reinit=True)

    def log_config(self, config: dict) -> None:
        import wandb
        wandb.config.update(config)

    def log_dict(self, values: dict, step=None) -> None:
        import wandb
        wandb.log(values, step=step)

    def close(self) -> None:
        self.run.finish()
