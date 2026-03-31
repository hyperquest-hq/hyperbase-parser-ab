from typing import List, Any

from wtpsplit import SaT


class Sentensizer:
    def __init__(self):
        self.sat = SaT('sat-3l')

    def sentensize(self, text) -> List[Any]:
        return list(self.sat.split(text))
