import numpy as np
import theano.tensor as T

from machine_learning_acds.regressors.nn import MLPNetwork, layer_functions, DaD
from machine_learning_acds.dataset import TrajectoryDataset, IdentityDataset, StateActionDatasetWrapper
from machine_learning_acds.systems.theano import PhysicalSystem
from lasagne import updates, nonlinearities

Xs, Us = [], []
for i in range(50):
    arr = np.genfromtxt("data_{}".format(i+1), delimiter=",").T
    Xs.append(arr[:,:2])
    Us.append(arr[:,[2]])

STATE_DIM, CONTROL_DIM, DYNAMIC_INDICES, DT = 2, 1, [1], 0.1
traj_dataset = TrajectoryDataset(2, 1, [1], 0.1)
traj_dataset.add_data(Xs, Us)

X, U, Y = traj_dataset.get_data_as_one_array_with_state_action()
one_step_dataset = StateActionDatasetWrapper(IdentityDataset(), STATE_DIM, CONTROL_DIM)
one_step_dataset.add_data(X, U, Y)

layers = [layer_functions.inputlayer(STATE_DIM+CONTROL_DIM),
        layer_functions.fullyconnected(8, nonlinearity=nonlinearities.tanh),
        layer_functions.fullyconnected(8, nonlinearity=nonlinearities.tanh),
        layer_functions.fullyconnected(len(DYNAMIC_INDICES), nonlinearity=nonlinearities.linear)]
network = MLPNetwork(layers)

update_fn = lambda loss, params : updates.adam(loss, params, learning_rate=1e-3)
training_kwargs = dict(n_epochs=1000, printevery=1, minibatchsize=100, update_fn=update_fn)
network.fit(one_step_dataset, **training_kwargs)
network.eval_model(one_step_dataset, show_scores=True)
network.save("one_step_model")

def config_mapping(Q):
    angles = Q[:,0]
    angles_mapping = T.arctan2(T.sin(angles), T.cos(angles))
    return T.stack([angles_mapping], axis=1)

dynamical_system_constructor = lambda acc_sym : PhysicalSystem(acc_sym, config_mapping,
        PhysicalSystem.DEFAULT_INVERSE_CONFIG_MAPPING, None, STATE_DIM, 1, 1, CONTROL_DIM, DT)

dad = DaD()
dad.learn(network, traj_dataset, 30, dynamical_system_constructor, dad_iters=5,
        **training_kwargs)
dad.min_val_error_model.save("multistep_model")
