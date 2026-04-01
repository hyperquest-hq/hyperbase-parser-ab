from wtpsplit import SaT


class Sentensizer:
    def __init__(self) -> None:
        self.sat: SaT = SaT('sat-3l')

    def sentensize(self, text: str) -> list[str]:
        return list(self.sat.split(text))
