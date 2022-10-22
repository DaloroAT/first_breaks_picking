import time
from typing import Tuple, Optional

import torch


class PerfCounter:
    def __init__(self, format_string: Optional[str] = None):
        self.start_time = None
        self.end_time = None
        self.duration = None
        if format_string is not None:
            assert '{duration}', "Substring '{duration}' must be included to non-default message"
            self.format_string = format_string
        else:
            self.format_string = "Duration: {duration} seconds"

    def __enter__(self):
        self.start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        print(self.format_string.format(duration=self.duration))


class RayTracer:
    PI = torch.tensor(torch.pi)
    PI_M2 = PI * 2
    PI_M3_D2 = PI * 3 / 2
    ZERO = torch.tensor(0, dtype=torch.float32)

    RIGHT = torch.tensor(1)
    TOP = torch.tensor(2)
    LEFT = torch.tensor(3)
    BOTTOM = torch.tensor(4)
    INACTIVE = torch.tensor(0)

    # def fill_placeholders_with_active_rays(self):
    #     holder_active = self._ACTIVE_RAY[:, self._CURRENT_STEP]
    #     self._ACTIVE_RAY[holder_active, self._CURRENT_STEP] = self._active_ray
    #     self._X0 = self._x0
    #     self._Z0 = self._z0
    #     self._X1 = self._x1
    #     self._Z1 = self._z1
    #     self._ANGLES = self._angles

    def __init__(self,
                 velocity_model: torch.Tensor,
                 height: int,
                 width: int,
                 source_point: Tuple[float, float],
                 init_angles: torch.Tensor,
                 max_steps: int):
        assert velocity_model.ndim == 2
        assert init_angles.ndim == 1
        self.velocity_model = velocity_model
        self.height = height
        self.width = width
        self.source_point = source_point
        self.init_angles = init_angles
        self.max_steps = max_steps

        # Placeholders
        self._N_RAYS = len(self.init_angles)
        self._NZ, self._NX = self.velocity_model.shape
        self._H = torch.tensor(self.height / self._NZ, dtype=torch.float32)
        self._W = torch.tensor(self.width / self._NX, dtype=torch.float32)
        self._X0 = torch.zeros((self._N_RAYS, self.max_steps), dtype=torch.float32)
        self._Z0 = torch.zeros((self._N_RAYS, self.max_steps), dtype=torch.float32)
        self._X1 = torch.zeros((self._N_RAYS, self.max_steps), dtype=torch.float32)
        self._Z1 = torch.zeros((self._N_RAYS, self.max_steps), dtype=torch.float32)
        self._CELL_X = torch.zeros((self._N_RAYS, self.max_steps), dtype=torch.int32)
        self._CELL_Z = torch.zeros((self._N_RAYS, self.max_steps), dtype=torch.int32)
        self._ANGLES = torch.zeros((self._N_RAYS, self.max_steps), dtype=torch.float32)
        self._ACTIVE_RAY = torch.ones((self._N_RAYS, self.max_steps), dtype=torch.bool)

        # Init rays
        self._CURRENT_STEP = 0
        self._X0[:, self._CURRENT_STEP] = self.source_point[1]
        self._Z0[:, self._CURRENT_STEP] = self.source_point[0]
        self._CELL_X[:, self._CURRENT_STEP] = torch.floor(self.source_point[1] / self._W)
        self._CELL_Z[:, self._CURRENT_STEP] = torch.floor(self.source_point[0] / self._H)
        self._ANGLES[:, self._CURRENT_STEP] = self.init_angles

        # States for active rays on current step
        self._x0 = None
        self._z0 = None
        self._x1 = None
        self._z1 = None
        self._angles = None
        self._cell_x = None
        self._cell_z = None
        self._next_border = None
        self._next_slowness = None
        self._active_ray = None

        self.set_active_rays()

    def set_active_rays(self):
        self._active_ray = self._ACTIVE_RAY[:, self._CURRENT_STEP]
        self._x0 = self._X0[self._active_ray, self._CURRENT_STEP]
        self._z0 = self._Z0[self._active_ray, self._CURRENT_STEP]
        self._x1 = self._X1[self._active_ray, self._CURRENT_STEP]
        self._z1 = self._Z1[self._active_ray, self._CURRENT_STEP]
        self._angles = self._ANGLES[self._active_ray, self._CURRENT_STEP]
        self._cell_x = self._CELL_X[self._active_ray, self._CURRENT_STEP]
        self._cell_z = self._CELL_Z[self._active_ray, self._CURRENT_STEP]

    def step(self):
        self.set_active_rays()
        self.find_next_points_borders_slowness()
        print(sum(self._next_border == self.INACTIVE), len(self._next_border))

    def calc_borders_angles(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        right_bottom = torch.arctan(torch.divide(self._W - self._x0, self._H - self._z0))
        right_top = self.PI - torch.arctan(torch.divide(self._W - self._x0, self._z0))
        left_top = self.PI_M3_D2 - torch.arctan(torch.divide(self._z0, self._x0))
        left_bottom = self.PI_M2 - torch.arctan(torch.divide(self._x0, self._H - self._z0))
        return right_bottom, right_top, left_top, left_bottom

    @staticmethod
    def calc_x1(z, x0, z0, angles) -> torch.Tensor:
        return x0 + (z - z0) * torch.tan(angles)

    @staticmethod
    def calc_z1(x, x0, z0, angles) -> torch.Tensor:
        return z0 + (x - x0) / torch.tan(angles)

    def find_next_points_borders_slowness(self) -> None:
        right_bottom, right_top, left_top, left_bottom = self.calc_borders_angles()

        num_rays = len(self._x0)
        border = self.INACTIVE * torch.ones(num_rays, dtype=torch.int32)
        x = torch.zeros(num_rays)
        z = torch.zeros(num_rays)

        right_mask = torch.logical_and(right_bottom < self._angles, self._angles <= right_top)
        border[right_mask] = self.RIGHT
        x[right_mask] = self._W
        z[right_mask] = self.calc_z1(x[right_mask], self._x0[right_mask], self._z0[right_mask], self._angles[right_mask])

        top_mask = torch.logical_and(right_top < self._angles, self._angles <= left_top)
        border[top_mask] = self.TOP
        z[top_mask] = self.ZERO
        x[top_mask] = self.calc_x1(z[top_mask], self._x0[top_mask], self._z0[top_mask], self._angles[top_mask])

        left_mask = torch.logical_and(left_top < self._angles, self._angles <= left_bottom)
        border[left_mask] = self.LEFT
        x[left_mask] = self.ZERO
        z[left_mask] = self.calc_z1(x[left_mask], self._x0[left_mask], self._z0[left_mask], self._angles[left_mask])

        bottom_mask = torch.logical_or(torch.logical_and(left_bottom < self._angles, self._angles <= self.PI_M2),
                                       torch.logical_and(0 <= self._angles, self._angles <= right_bottom))
        border[bottom_mask] = self.RIGHT
        z[bottom_mask] = self._H
        x[bottom_mask] = self.calc_x1(z[bottom_mask], self._x0[bottom_mask], self._z0[bottom_mask], self._angles[bottom_mask])

    #     self._x1 = x
    #     self._z1 = z
    #     self._next_border = border
    #
    # def get_next_slowness(self):





if __name__ == '__main__':
    NX = 300
    NZ = 200
    N_RAYS = 500
    SX = 0.5
    SZ = 0.5
    HEIGHT = 200
    WIDTH = 300
    V_CONST = 1000
    V_VAR = 100
    MAX_STEPS = 10

    SP = (SZ, SX)
    VELOCITY = V_CONST + V_VAR * torch.rand(NZ, NX)
    ANGLES = torch.linspace(0, torch.deg2rad(torch.tensor(360)), N_RAYS)

    with PerfCounter():
        tracer = RayTracer(velocity_model=VELOCITY,
                           height=HEIGHT,
                           width=WIDTH,
                           source_point=SP,
                           init_angles=ANGLES,
                           max_steps=MAX_STEPS)

    with PerfCounter():
        tracer.step()




