import numpy as np
import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
from data_import import read_data_from_file


def compute_hyperelastic_potential(tensor: np.ndarray, output_invs = False):
    """Computes the invariants I1 - I5 for a single given 3x3 deformation gradient tensor."""
    assert tensor.shape == (3, 3)

    C = tensor.T @ tensor
    G_ti = jnp.diag(jnp.array([4, 0.5, 0.5]))

    inv_1 = jnp.trace(C)
    inv_2 = jnp.linalg.det(tensor)
    inv_3 = jnp.linalg.det(C)
    inv_4 = jnp.trace(C @ G_ti)
    inv_5 = jnp.trace(inv_3 * jnp.linalg.inv(C.T)@G_ti)

    pot = 8 * inv_1 + 10 *inv_2**2 - 56 * jnp.log(inv_2) + 0.2 *(inv_4**2 + inv_5**2) - 44

    if output_invs:
        return pot, inv_1, inv_2, inv_3, inv_4, inv_5
    else:
        return pot


file = "biaxial"
deformation_grad, stress, energy_density = read_data_from_file(file)

n_rows = deformation_grad.shape[0]
matrix_size = deformation_grad.shape[1:]

invariants = np.zeros(shape=(n_rows, 5), dtype=float)
hyperelastic_potential = np.zeros(shape=n_rows, dtype=float)
piola_kirchhoff = np.zeros(shape=(n_rows, 3, 3), dtype=float)

piola_kirchhoff_fun = jax.grad(compute_hyperelastic_potential)


for i in range(n_rows):
    hyperelastic_potential[i], invariants[i, 0], invariants[i, 1], invariants[i, 2], invariants[i, 3], invariants[i, 4] \
        = compute_hyperelastic_potential(deformation_grad[i], output_invs=True)
    piola_kirchhoff[i] = piola_kirchhoff_fun(deformation_grad[i])




reference_path = f"../../../hyperelasticity/invariants/I_{file}.txt"

with open(reference_path, 'r') as f:
    reference_data = np.loadtxt(f, delimiter=' ')

atol = 1e-5
np.testing.assert_allclose(invariants[:, 0], reference_data[:, 0], atol=atol) # I_1
np.testing.assert_allclose(invariants[:, 1], reference_data[:, 1], atol=atol) # J
np.testing.assert_allclose(invariants[:, 3], reference_data[:, 2], atol=atol) # I_4
np.testing.assert_allclose(invariants[:, 4], reference_data[:, 3], atol=atol) # I_5

np.testing.assert_allclose(piola_kirchhoff, stress, atol=10*atol)

fig, ax = plt.subplots()
ax.plot(energy_density, label="reference")
ax.plot(hyperelastic_potential, label="hyperelastic potential")
ax.legend()
plt.savefig("invariants.png")

