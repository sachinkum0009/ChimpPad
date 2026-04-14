"""
Utils for the handling msgs
"""

from dataclasses import asdict, dataclass


@dataclass
class JoyMsg:
    x_axis: float = 0.0
    y_axis: float = 0.0
    z_axis: float = 0.0
    button: int = -1
    hat: tuple = (0, 0)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)
