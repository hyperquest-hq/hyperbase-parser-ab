from wtpsplit import SaT


class Sentensizer:
    def __init__(self) -> None:
        self.sat: SaT = SaT("sat-3l")

    def sentensize(self, text: str) -> list[str]:
        return [str(sentence) for sentence in (self.sat.split(text))]
