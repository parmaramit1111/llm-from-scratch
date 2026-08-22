from llm_from_scratch.core.activation import Activation
from llm_from_scratch.core.embedding import Embedding
from llm_from_scratch.core.layer import Layer
from llm_from_scratch.core.language_training import LanguageTrainer
from llm_from_scratch.core.network import Network
from llm_from_scratch.core.neuron import Neuron
from llm_from_scratch.core.optimizer import GradientDescent
from llm_from_scratch.core.softmax_cross_entropy import SoftmaxCrossEntropy
from llm_from_scratch.language.embedded_sequence import EmbeddedSequence


def test_language_trainer_train_step():
    network = Network([
        Layer([
            Neuron(
                weights=[0.5],
                bias=0.0,
                activation=Activation("linear"),
            ),
            Neuron(
                weights=[-0.5],
                bias=0.0,
                activation=Activation("linear"),
            ),
        ]),
    ])

    trainer = LanguageTrainer(
        network=network,
        loss_function=SoftmaxCrossEntropy(),
        optimizer=GradientDescent(
            learning_rate=0.01,
        ),
    )

    original_weights = [
        neuron.weights[0]
        for neuron in network.layers[0].neurons
    ]

    loss = trainer.train_step(
        input_values=[1.0],
        target_index=0,
    )

    updated_weights = [
        neuron.weights[0]
        for neuron in network.layers[0].neurons
    ]

    assert loss > 0.0
    assert updated_weights != original_weights

def test_language_trainer_updates_embedding_and_network():
    embedding = Embedding(
        vocabulary_size=4,
        embedding_size=2,
    )

    embedding.weights = [
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
        [0.7, 0.8],
    ]

    embedded_sequence = EmbeddedSequence(
        embedding=embedding,
    )

    network = Network([
        Layer([
            Neuron(
                weights=[0.1, 0.2, 0.3, 0.4],
                bias=0.0,
                activation=Activation("linear"),
            ),
            Neuron(
                weights=[-0.1, -0.2, -0.3, -0.4],
                bias=0.0,
                activation=Activation("linear"),
            ),
            Neuron(
                weights=[0.2, 0.1, 0.4, 0.3],
                bias=0.0,
                activation=Activation("linear"),
            ),
            Neuron(
                weights=[-0.2, -0.1, -0.4, -0.3],
                bias=0.0,
                activation=Activation("linear"),
            ),
        ]),
    ])

    trainer = LanguageTrainer(
        network=network,
        loss_function=SoftmaxCrossEntropy(),
        optimizer=GradientDescent(
            learning_rate=0.1,
        ),
        embedded_sequence=embedded_sequence,
    )

    original_embedding_weight = embedding.weights[1][0]
    original_network_weight = network.layers[0].neurons[0].weights[0]

    loss = trainer.train_step(
        input_values=[1.0, 2.0],
        target_index=3,
    )

    assert loss > 0.0

    assert embedding.weights[1][0] != original_embedding_weight
    assert network.layers[0].neurons[0].weights[0] != original_network_weight