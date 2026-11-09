import numpy as np

def read_data_from_file(filename= str) -> np.ndarray:
    if filename == "" or filename is None:
        raise ValueError("Filename cannot be empty")
    path = f"../../../hyperelasticity/calibration/{filename}.txt"


    with open(path, "r") as f:
        num_lines = sum(1 for _ in f)

    deformation = np.empty(shape=(num_lines,3,3), dtype=np.float64)
    stress = np.empty(shape=(num_lines,3,3), dtype=np.float64)
    energy_density = np.empty(shape=num_lines, dtype=np.float64)
    with open(path, "r") as file:
        for i, line in enumerate(file):
            F11, F12, F13, F21, F22, F23, F31, F32, F33, P11, P12, P13, P21, P22, P23, P31, P32, P33, W  = line.split(" ")
            deformation[i] = np.array([[F11, F12, F13], [F21, F22, F23], [F31, F32, F33]], dtype=np.float64)
            stress[i] = np.array([[P11, P12, P13], [P21, P22, P23], [P31, P32, P33]], dtype=np.float64)
            energy_density[i] = float(W)

    return deformation, stress, energy_density
