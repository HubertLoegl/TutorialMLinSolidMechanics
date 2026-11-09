import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from data_import import read_data_from_file

file = "biaxial"
deformation, stress, energy_density = read_data_from_file(file)

# case uniaxial
unity =  np.broadcast_to(np.identity(deformation.shape[1]), deformation.shape)
displacement = deformation - unity
strain = 1/2 * (displacement + np.transpose(displacement, axes=(0, 2,1)))

print(strain.shape)

for i in range(stress.shape[1]):
    fig, ax = plt.subplots()
    ax.scatter(strain[:,i,i], stress[:,i,i])
    # ax.plot(strain[:,i,i], stress[:,i,i], marker="x", markevery=15)
    ax.set_ylabel(r"stress in $\sigma$" +str(i+1)+str(i+1) + "-direction")
    ax.set_xlabel(r"strain in $\epsilon$" +str(i+1)+str(i+1) + "-direction")
    plt.savefig(f"images/{file}_stress_in_" + str(i) + ".png")

fig, ax = plt.subplots()
ax.plot(np.arange(energy_density.shape[0]), energy_density)
ax.set_ylabel("Energy density")
plt.savefig(f"images/{file}_energy_density.png")
