# filename: quantum_layer.py
import torch
import torch.nn as nn
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

class QuantumLayer(nn.Module):
    def __init__(self, n_qubits: int, n_outputs: int = None, n_layers: int = 1, seed: int = 42):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_outputs = n_outputs if n_outputs is not None else n_qubits
        self.n_layers = n_layers
        
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.x_params = ParameterVector("x", n_qubits) 
        self.theta_params = ParameterVector("θ", n_qubits * n_layers)

        qc = QuantumCircuit(n_qubits)

        # Feature Map
        for i in range(n_qubits):
            qc.ry(self.x_params[i] * np.pi, i)

        # Ansatz
        theta_index = 0
        for layer in range(n_layers):
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)
            
            # Uncomment the next line if you want circular entanglement
            # if n_qubits > 2: qc.cx(n_qubits - 1, 0) 

            for i in range(n_qubits):
                qc.ry(self.theta_params[theta_index], i)
                theta_index += 1

        self.qc = qc 

        observables = []
        for i in range(self.n_outputs):
            pauli_string = ["I"] * n_qubits
            pauli_string[n_qubits - 1 - i] = "Z"
            observables.append(SparsePauliOp.from_list([("".join(pauli_string), 1.0)]))

        qnn = EstimatorQNN(
            circuit=qc,
            input_params=list(self.x_params),
            weight_params=list(self.theta_params),
            observables=observables,
            estimator=StatevectorEstimator()
        )

        self.q_layer = TorchConnector(qnn)

    def forward(self, x: torch.Tensor):
        return self.q_layer(x)

    def draw(self, output="text"):
        return self.qc.draw(output=output)