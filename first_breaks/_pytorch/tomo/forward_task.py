import numpy as np
import eikonalfm
import matplotlib.pyplot as plt


XMAX = 1000
ZMAX = 1000

DX = 1
DZ = 1

NX = int(XMAX / DX)
NZ = int(ZMAX / DZ)

MIN_VELOCITY = 1000
MAX_VELOCITY = 4000
VELOCITY = np.linspace(MIN_VELOCITY, MAX_VELOCITY, NZ)[:, None]
VELOCITY = np.tile(VELOCITY, (1, NX))

plt.imshow(VELOCITY)
plt.show()

SOURCE = (0, 0)
DELTA = (DZ, DX)
ORDER = 2

tau_fm = eikonalfm.fast_marching(VELOCITY, SOURCE, DELTA, ORDER)
print(tau_fm.max())
print(NZ, NX, tau_fm.shape)
# plt.contourf(tau_fm, levels=np.arange(0, 400, 20))
plt.contourf(tau_fm)
plt.ylim(NZ, 0)
plt.show()



