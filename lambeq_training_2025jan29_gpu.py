from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn.functional as F
from lambeq import BobcatParser, IQPAnsatz, RemoveCupsRewriter, AtomicType
from sympy import default_sort_key
import jax
from jax import numpy as np
import numpy as numpy_backend
from jax import value_and_grad
import warnings
import time

warnings.filterwarnings('ignore')
from lambeq.backend.numerical_backend import set_backend
set_backend('jax')
numpy_backend.random.seed(0)
key = jax.random.PRNGKey(0)
print("JAX devices:", jax.devices())

model = SentenceTransformer("all-MiniLM-L6-v2")
def read_data(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
    data = [ln.strip() for ln in lines]
    return data
train_data = read_data('train.txt')
valid_data = read_data('dev.txt')
train_data_size = len(train_data)
valid_data_size = len(valid_data)
print(f"Train data size: {train_data_size}, Validation data size: {valid_data_size}")

with torch.no_grad():
    embeddings = model.encode(train_data, convert_to_tensor=True, device='gpu')
    embeddings_valid = model.encode(valid_data, convert_to_tensor=True, device='gpu')

similarity_matrix_np = F.cosine_similarity(embeddings.unsqueeze(0), embeddings.unsqueeze(1), dim=2).numpy()
similarity_matrix_valid_np = F.cosine_similarity(embeddings_valid.unsqueeze(0), embeddings_valid.unsqueeze(1), dim=2).numpy()

parser = BobcatParser(verbose='suppress')
train_diagrams = parser.sentences2diagrams(train_data)
valid_diagrams = parser.sentences2diagrams(valid_data)
N = AtomicType.NOUN
S = AtomicType.SENTENCE
remove_cups = RemoveCupsRewriter()
iqp_ansatz = IQPAnsatz({N: 1, S: 2}, n_layers=1)
train_circuits = [iqp_ansatz(remove_cups(d)) for d in train_diagrams]
valid_circuits = [iqp_ansatz(remove_cups(d)) for d in valid_diagrams]

vocab = sorted({sym for circ in train_circuits for sym in circ.free_symbols}, key=default_sort_key)
print(f"Vocabulary size: {len(vocab)}")

def loss_fn(params_array, circuits_list, target_similarity_matrix_np):

    try:
        np_circuits = [c.lambdify(*vocab)(*params_array) for c in circuits_list]
        out_states_list = [np.asarray(c.eval()) for c in np_circuits]
    except Exception as e:
        print(f"Error: {e}")
        return np.inf

    for i, state in enumerate(out_states_list):
        if state is None: 
             print(f"Warning:{i}")
             return np.inf
        if np.any(np.isnan(state)):
            print(f"Warning:{i}")
            return np.inf

    norms = np.array([np.linalg.norm(state) for state in out_states_list])
    safe_norms = np.where(norms == 0, 1e-9, norms)
    out_states_list_normal = [state / norm for state, norm in zip(out_states_list, safe_norms)]

    try:
        shapes = {s.shape for s in out_states_list_normal}
        if len(shapes) > 1:
           print(f"Warning: {shapes}")
           return np.inf
        out_states_list_normal_vertical = [state.reshape((-1, 1)) for state in out_states_list_normal]
        out_states_array = np.concatenate(out_states_list_normal_vertical, axis=1)
    except Exception as e:
        print(f"Error: {e}")
        return np.inf

    fidelity_mat = np.conjugate(out_states_array.T) @ out_states_array
    target_similarity_matrix_jax = np.asarray(target_similarity_matrix_np)

    cost = np.sum(np.square(np.abs(fidelity_mat) - target_similarity_matrix_jax)) / (len(circuits_list) ** 2)

    return cost

loss_and_grad_fn = value_and_grad(loss_fn, argnums=0)

learning_rate = 0.1
epochs = 1000
key, subkey = jax.random.split(key)
params_array = np.asarray(numpy_backend.random.rand(len(vocab)))

training_losses = []
validation_losses = []

start_time = time.time()

for epoch in range(epochs):
    epoch_start_time = time.time()
    current_params_array = params_array

    try:
        train_loss_value, gr = loss_and_grad_fn(current_params_array, train_circuits, similarity_matrix_np)

        if np.isnan(gr).any() or np.isinf(gr).any() or np.isnan(train_loss_value) or np.isinf(train_loss_value):
            print(f"Warning:{epoch+1}")
            if training_losses:
                training_losses.append(training_losses[-1])
            else:
                training_losses.append(float('nan'))
            if validation_losses:
                 validation_losses.append(validation_losses[-1])
            else:
                 validation_losses.append(float('nan'))
            continue
        else:
            params_array = current_params_array - gr * learning_rate
            training_losses.append(float(train_loss_value))

    except Exception as e:
        print(f"Error{epoch+1}: {e}")
        continue 

    try:
        valid_loss_value = loss_fn(params_array, valid_circuits, similarity_matrix_valid_np)

        if np.isnan(valid_loss_value) or np.isinf(valid_loss_value):
             print(f"Warning: NaN/Inf validation loss detected at epoch {epoch+1}.")
             if validation_losses:
                 validation_losses.append(validation_losses[-1])
             else:
                 validation_losses.append(float('nan'))
        else:
            validation_losses.append(float(valid_loss_value))
    except Exception as e:
        print(f"Error{epoch+1}: {e}")

    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time

    if (epoch + 1) % 10 == 0:
        train_loss_disp = training_losses[-1] if training_losses else float('nan')
        valid_loss_disp = validation_losses[-1] if validation_losses else float('nan')
        print(f"Epoch {epoch + 1}/{epochs} - "
              f"Train Loss: {train_loss_disp:.4f} - "
              f"Valid Loss: {valid_loss_disp:.4f} - "
              f"Duration: {epoch_duration:.2f}s")

end_time = time.time()
total_duration = end_time - start_time
print(f"\nTraining finished in {total_duration:.2f} seconds.")