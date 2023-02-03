import torch
import typing
import matplotlib.pyplot as plt


class LatticeModel:
    """
    Base class for lattice models
    --------------
    lattice_size: tuple[int] = torch.tensor[x_len, y_len] - size of 2D lattice 
    field_type: str = "descrete" or "continuous"
    field_values: torch.tensor - only for descrete, all field values
    boundary_conditions - None, "periodic", "twisted"

    Parameters:
    field: torch.tensor - 2D tensor with field values
    Methods:
    __init__: create class instance
    random: create random field configuration
    minimize_energy: using metropolis algorithm generate min energy configuration from self.field seed.
    """

    def __init__(self, lattice_size: torch.tensor, field_type: str = "descrete", boundary_conditions=None) -> None:
        self.field_type = field_type
        if field_type == "descrete":
            self.field = torch.ones(
                lattice_size[0].item(), lattice_size[1].item(), dtype=torch.int8)
        elif field_type == "continuous":
            self.field = torch.ones(
                lattice_size[0].item(), lattice_size[1].item(), dtype=torch.float64)
        else:
            raise TypeError("field type is either descrete or continuous")
        self._boundary_conditions = boundary_conditions

    def __getitem__(self, key):
        return self.field[key]

    def shape(self):
        return self.field.shape

    def energy(self):
        raise AttributeError("energy method not implemented")

    def minimize_energy(self) -> torch.tensor:
        raise AttributeError("Fit method not implemented")
