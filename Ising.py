from LatticeModel import LatticeModel
import random
import torch
import matplotlib.pyplot as plt
import numpy as np


class Ising(LatticeModel):
    """
    Child class of LatticeModel
    """

    def __init__(self, lattice_size: torch.tensor, field_type: str = "descrete", boundary_condition=None,  field_values=[-1, 1],
                 J=torch.tensor(1, dtype=torch.float64), h=torch.tensor(0, dtype=torch.float64)) -> None:
        super().__init__(lattice_size, field_type="descrete",
                         boundary_conditions=boundary_condition)
        if self.field_type == "continuous" and field_values != None:
            raise TypeError(
                "Can't infer descrete values for continuous field.")
        self.field_values = field_values
        self.J = J
        self.h = h

    def _randomize(self):
        x_size, y_size = self.shape()
        samples = random.choices(self.field_values, k=x_size * y_size)
        if self.field_type == "descrete":
            self.field = torch.tensor(samples).reshape(x_size, y_size)

    def energy(self) -> torch.float64:
        x_size = self.field.shape[0]
        y_size = self.field.shape[1]
        energy_val = torch.tensor(0, dtype=torch.float64)
        for x in range(x_size):
            for y in range(y_size):
                energy_val += self._local_interaction_energy(x, y) * 0.5
                energy_val += self._local_external_energy(x, y)
        return energy_val.item()

    def _local_external_energy(self, x_loc, y_loc):
        return -self.h * self.field[x_loc, y_loc]

    def _local_interaction_energy(self, x, y):
        x_size = self.field.shape[0]
        y_size = self.field.shape[1]
        return - self.J * self[x, y] * (self[x, (y - 1)] + self[x, (y + 1) % y_size] + self[x - 1, y] + self[(x + 1) % x_size, y])

    def plot(self):
        shape = self.shape()
        X, Y = np.meshgrid(range(shape[0]), range(shape[1]))
        plt.title(f"{shape[0]}x{shape[1]} ising configuration")
        plt.pcolormesh(X, Y, self.field.numpy())
        plt.show()

    def minimize_energy(self) -> torch.tensor:
        pass
