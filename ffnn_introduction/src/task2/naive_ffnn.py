from models import FFNN
import jax
import jax.numpy as jnp
import jax.random as jrandom
import time
import datetime
import klax
import matplotlib.pyplot as plt
from data_import import read_data_from_file
from tmlsm.losses import MSE

# model parameters
input_dim = 6
output_dim = 9
nodes_per_layer = 16
number_hidden_layers = 2
activation_fct = jax.nn.softplus
batch_size = 32
number_training_steps = 50_000

# model key
key = jrandom.PRNGKey(time.time_ns())
keys = jrandom.split(key, 2)

# create FFNN
model = FFNN(input_dim, output_dim, nodes_per_layer, number_hidden_layers, activation_fct, key=keys[0])

# load data
file = "biaxial"
deformation, piola_kirchhoff_stress , _ = read_data_from_file(file)

cauchy_green = deformation.transpose(0,2,1) @ deformation

# 6 independent components of symmetric C (Voigt order: 11, 22, 33, 23, 13, 12) -> shape (199, 6)
input_data = jnp.stack((cauchy_green[:,0,0], cauchy_green[:,1,1], cauchy_green[:,2,2],
                        cauchy_green[:,1,2], cauchy_green[:,0,2], cauchy_green[:,0,1]), axis=-1)
# all 9 components of P (row-major: P11, P12, P13, P21, P22, P23, P31, P32, P33) -> shape (199, 9)
output_data = jnp.asarray(piola_kirchhoff_stress.reshape(-1, 9))

# Calibrate model
t1 = datetime.datetime.now()
print(t1)

model, history = klax.fit(
    model,
    (input_data, output_data),
    batch_size=batch_size,
    steps=number_training_steps,
    loss_fn=MSE(),
    history=klax.HistoryCallback(log_every=1),
    key=keys[1],
)

t2 = datetime.datetime.now()
print("it took", t2 - t1, "(sec) to calibrate the model")

history.plot()

model = klax.finalize(model)


# evaluate model interpolation
# compare FFNN-Piola-Kirchhof stress to stress from calibration file
model_stress = jax.vmap(model)(input_data)

for i in range(piola_kirchhoff_stress.shape[1]):
    for j in range(piola_kirchhoff_stress.shape[2]):
        fig, ax = plt.subplots()
        ax.scatter(cauchy_green[:,i,j], piola_kirchhoff_stress[:,i,j], label="calibration data", s=2)
        ax.scatter(cauchy_green[:,i,j], model_stress[:,i*piola_kirchhoff_stress.shape[1]+j], label="model data", s=2, marker="s")
        ax.set_xlabel(r"strain in $\epsilon$" +str(i+1)+str(j+1) + "-direction")
        ax.set_ylabel(r"stress in $\sigma$" +str(i+1)+str(j+1) + "-direction")
        ax.set_title(str(i*piola_kirchhoff_stress.shape[1]+j))
        ax.legend()
        plt.savefig(f"images/naive_ffnn_interpol_{file}_stress_in_" + str(i)+ str(j+1) + ".png")
#
# fig, ax = plt.subplots()
# ax.plot(np.arange(energy_density.shape[0]), energy_density)
# ax.set_ylabel("Energy density")
# plt.savefig(f"images/{file}_energy_density.png")
#
#
# plt.figure(2)
# plt.scatter(, c="green", label="calibration data")
# plt.plot(x, y, c="black", linestyle="--", label="bathtub function")
# plt.plot(x, jax.vmap(model_)(x), label="model", color="red")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.title("ICNN")
# plt.savefig("images/ICNN2.png",
#             facecolor='white',   # figure background
#             transparent=False)
# plt.show()


# compare model to unseen data from hyperelasticity/test/...
if file =="biaxial":
    test_data_path = "../../../hyperelasticity/test/biax_test.txt"
elif file == "":
    test_data_path = "../../../hyperelasticity/test/mixed_test.txt"
else:
    plt.show()

test_deformation, test_pk_stress, _ = read_data_from_file(test_data_path, is_path=True)
test_c = test_deformation.transpose(0,2,1) @ test_deformation

input_test_data = jnp.stack((test_c[:,0,0], test_c[:,1,1], test_c[:,2,2],
                        test_c[:,1,2], test_c[:,0,2], test_c[:,0,1]), axis=-1)
model_stress = jax.vmap(model)(input_test_data)

for i in range(test_pk_stress.shape[1]):
    for j in range(test_pk_stress.shape[2]):
        fig, ax = plt.subplots()
        ax.scatter(test_c[:,i,j], test_pk_stress[:,i,j], label="calibration data", s=2)
        ax.scatter(test_c[:,i,j], model_stress[:,i*test_pk_stress.shape[1]+j], label="model data", s=2, marker="s")
        ax.set_xlabel(r"strain in $\epsilon$" +str(i+1)+str(j+1) + "-direction")
        ax.set_ylabel(r"stress in $\sigma$" +str(i+1)+str(j+1) + "-direction")
        ax.set_title("Unseen data" + str(i*test_pk_stress.shape[1]+j))
        ax.legend()
        plt.savefig(f"images/naive_ffnn_unseen_data_{file}_stress_in_" + str(i+1) + str(j+1) + ".png")



plt.show()