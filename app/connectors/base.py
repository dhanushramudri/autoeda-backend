from abc import ABC, abstractmethod
from typing import Any, Optional
import pandas as pd


class BaseConnector(ABC):
    @abstractmethod
    def connect(self, config: dict) -> Any:
        pass

    @abstractmethod
    def load_data(self, config: dict, limit: Optional[int] = None) -> pd.DataFrame:
        pass

    def preview(self, config: dict, rows: int = 100) -> pd.DataFrame:
        return self.load_data(config, limit=rows)

    def test_connection(self, config: dict) -> tuple[bool, str]:
        try:
            self.connect(config)
            return True, "Connection successful"
        except Exception as e:
            return False, str(e)
