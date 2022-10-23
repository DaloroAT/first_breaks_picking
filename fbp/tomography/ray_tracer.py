import time
from typing import Tuple, Optional

import torch
from tqdm import tqdm


class PerfCounter:
    def __init__(self, message: str = 'Duration'):
        self.start_time = None
        self.end_time = None
        self.duration = None
        self.message = message

    def __enter__(self):
        self.start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        print(f"{self.message}: {self.duration} seconds")


def composed_and(*values: Tuple[torch.Tensor, ...]):
    assert len(values) > 1
    res = values[0]
    for v in values[1:]:
        res = torch.logical_and(res, v)
    return res


class RayTracer:
    PI = torch.tensor(torch.pi)
    PI_M2 = PI * 2
    PI_D2 = PI / 2
    PI_M3_D2 = PI * 3 / 2
    PI_M3 = PI * 3
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
        self._CELL_X = torch.zeros((self._N_RAYS, self.max_steps), dtype=torch.long)
        self._CELL_Z = torch.zeros((self._N_RAYS, self.max_steps), dtype=torch.long)
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
        self._next_cell_x = None
        self._next_cell_z = None
        self._next_angles = None
        self._border = None
        self._v1 = None
        self._v2 = None
        self._active_ray = None

        self.set_active_rays()

    def get_num_active_rays(self):
        return sum(self._ACTIVE_RAY[:, self._CURRENT_STEP])

    def set_active_rays(self):
        self._active_ray = self._ACTIVE_RAY[:, self._CURRENT_STEP]
        self._x0 = self._X0[self._active_ray, self._CURRENT_STEP]
        self._z0 = self._Z0[self._active_ray, self._CURRENT_STEP]
        self._x1 = self._X1[self._active_ray, self._CURRENT_STEP]
        self._z1 = self._Z1[self._active_ray, self._CURRENT_STEP]
        self._angles = self._ANGLES[self._active_ray, self._CURRENT_STEP]
        self._cell_x = self._CELL_X[self._active_ray, self._CURRENT_STEP]
        self._cell_z = self._CELL_Z[self._active_ray, self._CURRENT_STEP]
        self._v1 = self.velocity_model[self._cell_z, self._cell_x]

    def step(self):
        self.set_active_rays()
        self.find_next_points_borders_slowness()
        self.get_next_v()
        self.get_next_angles()
        self.fill_states_to_placeholders()

    def run(self):
        pbar = tqdm(range(self.max_steps), desc='Rays calculation')
        for _ in tqdm(range(self.max_steps)):
            self.step()
            pbar.set_postfix_str(f'Active rays: {self.get_num_active_rays()}/{self._N_RAYS}')

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
        x1 = torch.zeros(num_rays, dtype=torch.float32)
        z1 = torch.zeros(num_rays, dtype=torch.float32)

        right_mask = torch.logical_and(right_bottom < self._angles, self._angles <= right_top)
        border[right_mask] = self.RIGHT
        x1[right_mask] = self._W
        z1[right_mask] = self.calc_z1(x1[right_mask], self._x0[right_mask], self._z0[right_mask], self._angles[right_mask])

        top_mask = torch.logical_and(right_top < self._angles, self._angles <= left_top)
        border[top_mask] = self.TOP
        z1[top_mask] = self.ZERO
        x1[top_mask] = self.calc_x1(z1[top_mask], self._x0[top_mask], self._z0[top_mask], self._angles[top_mask])

        left_mask = torch.logical_and(left_top < self._angles, self._angles <= left_bottom)
        border[left_mask] = self.LEFT
        x1[left_mask] = self.ZERO
        z1[left_mask] = self.calc_z1(x1[left_mask], self._x0[left_mask], self._z0[left_mask], self._angles[left_mask])

        bottom_mask = torch.logical_or(torch.logical_and(left_bottom < self._angles, self._angles <= self.PI_M2),
                                       torch.logical_and(0 <= self._angles, self._angles <= right_bottom))
        border[bottom_mask] = self.BOTTOM
        z1[bottom_mask] = self._H
        x1[bottom_mask] = self.calc_x1(z1[bottom_mask], self._x0[bottom_mask], self._z0[bottom_mask], self._angles[bottom_mask])

        self._x1 = x1
        self._z1 = z1
        self._border = border

    def get_next_v(self):
        v2 = torch.nan * torch.ones(len(self._border), dtype=torch.float32)

        right_mask = self._border == self.RIGHT
        v2[right_mask] = self.velocity_model[self._cell_z[right_mask], self._cell_x[right_mask] + 1]

        top_mask = self._border == self.TOP
        v2[top_mask] = self.velocity_model[self._cell_z[top_mask] - 1, self._cell_x[top_mask]]

        left_mask = self._border == self.LEFT
        v2[left_mask] = self.velocity_model[self._cell_z[left_mask], self._cell_x[left_mask] - 1]

        bottom_mask = self._border == self.BOTTOM
        v2[bottom_mask] = self.velocity_model[self._cell_z[bottom_mask] + 1, self._cell_x[bottom_mask]]

        self._v2 = v2

    def get_next_angles(self):
        num_rays = len(self._border)
        next_cell_x = torch.zeros(num_rays, dtype=torch.long)
        next_cell_z = torch.zeros(num_rays, dtype=torch.long)

        right_mask = self._border == self.RIGHT
        top_mask = self._border == self.TOP
        left_mask = self._border == self.LEFT
        bottom_mask = self._border == self.BOTTOM

        border_shift = torch.zeros(num_rays, dtype=torch.float32)
        border_shift[right_mask] = self.PI_D2
        border_shift[top_mask] = self.PI_M2
        border_shift[left_mask] = self.PI_M3_D2

        inc_angles = self._angles - border_shift
        out_angles = torch.asin(self._v2 / self._v1 * torch.sin(inc_angles))
        inner_reflection_mask = out_angles.isnan()

        right_top_submask = self._z1[right_mask] <= self._z0[right_mask]
        out_angles[right_mask][right_top_submask] = self.PI_D2 + out_angles[right_mask][right_top_submask]
        out_angles[right_mask][~right_top_submask] = self.PI_D2 - out_angles[right_mask][~right_top_submask]
        right_reflection_mask = torch.logical_and(right_mask, inner_reflection_mask)
        out_angles[right_reflection_mask] = self.PI_M2 - self._angles[right_reflection_mask]
        next_cell_x[right_mask] = self._cell_x[right_mask] + 1
        next_cell_z[right_mask] = self._cell_z[right_mask]
        next_cell_x[right_reflection_mask] = self._cell_x[right_reflection_mask]

        top_left_submask = self._x1[top_mask] <= self._x0[top_mask]
        out_angles[top_mask][top_left_submask] = self.PI + out_angles[top_mask][top_left_submask]
        out_angles[top_mask][~top_left_submask] = self.PI - out_angles[top_mask][~top_left_submask]
        top_reflection_mask = torch.logical_and(top_mask, inner_reflection_mask)
        out_angles[top_reflection_mask] = self.PI_D2 - self._angles[top_reflection_mask]
        next_cell_x[top_mask] = self._cell_x[top_mask]
        next_cell_z[top_mask] = self._cell_z[top_mask] - 1
        next_cell_z[top_reflection_mask] = self._cell_z[top_reflection_mask]

        left_bottom_submask = self._z1[left_mask] >= self._z0[left_mask]
        out_angles[left_mask][left_bottom_submask] = self.PI_M3_D2 + out_angles[left_mask][left_bottom_submask]
        out_angles[left_mask][~left_bottom_submask] = self.PI_M3_D2 - out_angles[left_mask][~left_bottom_submask]
        left_reflection_mask = torch.logical_and(left_mask, inner_reflection_mask)
        out_angles[left_reflection_mask] = self.PI_M2 - self._angles[left_reflection_mask]
        next_cell_x[left_mask] = self._cell_x[left_mask] - 1
        next_cell_z[left_mask] = self._cell_z[left_mask]
        next_cell_x[left_reflection_mask] = self._cell_x[left_reflection_mask]

        bottom_left_submask = self._x1[bottom_mask] <= self._x0[bottom_mask]
        out_angles[bottom_mask][bottom_left_submask] = self.PI_M2 - out_angles[bottom_mask][bottom_left_submask]
        bottom_reflection_mask = torch.logical_and(bottom_mask, inner_reflection_mask)
        bottom_angles = self._angles[bottom_reflection_mask]
        left_bottom_angles_mask = bottom_angles > self.PI
        bottom_angles[left_bottom_angles_mask] = self.PI_M2 - bottom_angles[left_bottom_angles_mask]
        out_angles[bottom_reflection_mask] = self.PI - bottom_angles
        next_cell_x[bottom_mask] = self._cell_x[bottom_mask]
        next_cell_z[bottom_mask] = self._cell_z[bottom_mask] + 1
        next_cell_z[bottom_reflection_mask] = self._cell_z[bottom_reflection_mask]

        self._next_angles = out_angles
        self._next_cell_x = next_cell_x
        self._next_cell_z = next_cell_z

    def fill_states_to_placeholders(self):
        holder_mask = self._ACTIVE_RAY[:, self._CURRENT_STEP]
        self._X1[holder_mask, self._CURRENT_STEP] = self._x1
        self._Z1[holder_mask, self._CURRENT_STEP] = self._z1

        if self._CURRENT_STEP < self.max_steps - 1:
            next_active_ray = composed_and(self._border != self.INACTIVE,
                                           self._next_cell_x >= 0,
                                           self._next_cell_x < self._NX,
                                           self._next_cell_z >= 0,
                                           self._next_cell_z < self._NZ)
            next_step = self._CURRENT_STEP + 1

            self._ACTIVE_RAY[holder_mask, next_step] = next_active_ray
            self._X0[holder_mask, next_step] = self._x1
            self._Z0[holder_mask, next_step] = self._z1
            self._CELL_X[holder_mask, next_step] = self._next_cell_x
            self._CELL_Z[holder_mask, next_step] = self._next_cell_z
            self._ANGLES[holder_mask, next_step] = self._next_angles

            self._CURRENT_STEP += 1


if __name__ == '__main__':
    torch.manual_seed(0)

    NX = 300
    NZ = 200
    N_RAYS = 1000
    SX = 15.5
    SZ = 15.5
    HEIGHT = 200
    WIDTH = 300
    V_CONST = 1000
    V_VAR = 100
    MAX_STEPS = 5

    SP = (SZ, SX)
    VELOCITY = V_CONST + V_VAR * torch.rand(NZ, NX)
    ANGLES = torch.linspace(0, torch.deg2rad(torch.tensor(360)), N_RAYS)

    # with PerfCounter():
    tracer = RayTracer(velocity_model=VELOCITY,
                       height=HEIGHT,
                       width=WIDTH,
                       source_point=SP,
                       init_angles=ANGLES,
                       max_steps=MAX_STEPS)

    # with PerfCounter():
    #     tracer.step()

    # with PerfCounter():
    tracer.run()
