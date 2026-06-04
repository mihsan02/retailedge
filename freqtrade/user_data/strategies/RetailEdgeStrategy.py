from freqtrade.strategy import IStrategy
import pandas as pd


class RetailEdgeStrategy(IStrategy):
    """
    Placeholder strategy — Sprint 0 only.
    Implementasi penuh di Sprint 3 (Task S3-1).

    Constraint aktif:
    - enter_long selalu 0 (tidak ada entry sampai Sprint 3)
    - Tidak ada logika apapun di sini yang boleh dipakai live
    """

    INTERFACE_VERSION = 3

    minimal_roi = {"0": 0.10}
    stoploss = -0.05
    timeframe = "15m"
    can_short = False

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["enter_long"] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["exit_long"] = 0
        return dataframe
