import scipy.io as io
from pathlib import Path

base_path = Path(__file__).parent
file_path = (base_path / 'ex4weights.mat').resolve()
mat = io.loadmat(file_path)
for key,value in mat.items():
    if key == 'Theta1':
        Theta1 = value
        pass
    else:
        Theta2 = value
        pass
file_path = (base_path / 'ex4data1.mat').resolve()
mat = io.loadmat(file_path)
for key,value in mat.items():
    if key == 'X':
        X = value
        pass
    else:
        y = value
        pass