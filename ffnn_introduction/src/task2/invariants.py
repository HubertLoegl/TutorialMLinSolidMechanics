import numpy as np
import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
from data_import import read_data_from_file


def compute_invariants(deformation_grad):

    C = np.einsum('...ji,...jk->...ik',deformation_grad, deformation_grad.transpose(0,2,1))
    # C = deformation_grad @ np.transpose(deformation_grad, axes=(0, 2, 1))
    G_ti = np.diag(np.array([4, 0.5, 0.5]))
    inv1 = np.linalg.trace(C)
    inv_J = np.zeros(deformation_grad.shape[0])
    inv3 = np.zeros(deformation_grad.shape[0])
    inv4 = np.empty(deformation_grad.shape[0])
    inv5 = np.zeros(deformation_grad.shape[0])
    for i in range(deformation_grad.shape[0]):
        inv_J[i] = np.linalg.det(deformation_grad[i])
        inv3[i] = np.linalg.det(C[i])
        inv4[i] = np.trace(C[i]@G_ti)
        inv5[i] = np.trace(inv3[i] * np.linalg.inv(np.transpose(C[i]))@G_ti)

    hyper_potential = 8 * inv1 + 10 *inv_J**2 - 56 * np.log(inv_J) + 0.2 *(inv4**2 + inv5**2) - 44
    # piola_kirchhoff = jax.grad(hyper_potential)

    return hyper_potential #, inv1, inv_J, inv3, inv4, inv5


file = "biaxial"
deformation_grad, stress, energy_density = read_data_from_file(file)

# hyper_potential, inv1, inv2, inv3, inv4, inv5= compute_invariants(deformation_grad)
hyper_potential= compute_invariants(deformation_grad)
piola_kirchhoff = jax.grad(compute_invariants)

np.testing.assert_allclose(hyper_potential, energy_density, atol=5e-5)
np.testing.assert_allclose(piola_kirchhoff, stress, atol=5e-5)

fig, ax = plt.subplots()
ax.plot(energy_density, label="from file")
ax.plot(hyper_potential, label="hyper potential")
ax.legend()
plt.savefig("invariants.png")


# reference_path = f"../../../hyperelasticity/invariants/I_{file}.txt"

# with open(reference_path, 'r') as f:
#     reference_data = np.loadtxt(f, delimiter=' ')

# np.testing.assert_allclose(inv1, reference_data[:, 0], atol=1e-5)
# np.testing.assert_allclose(inv2, reference_data[:, 1], atol=1e-5)
# np.testing.assert_allclose(inv4, reference_data[:, 2], atol=1e-5)
# np.testing.assert_allclose(inv5, reference_data[:, 3], atol=1e-5)
