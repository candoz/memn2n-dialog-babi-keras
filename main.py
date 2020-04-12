from time import time
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model
from data_utils import *
from custom_network_sections import *
"""
Trains an End-to-End Memory Network on the Dialog bAbI dataset.
References:
  - Antoine Bordes, Y-Lan Boureau, Jason Weston, "Learning End-to-End Goal-Oriented Dialog"
    https://arxiv.org/abs/1605.07683
  - Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus, "End-To-End Memory Networks"
    http://arxiv.org/abs/1503.08895
  - The bAbI project by Facebook research
    https://research.fb.com/downloads/babi/
Implementation details:
  - Sentences are encoded as bags of words as described in the paper.
    Another more efficient solution could be implemented by encoding the words as indices,
    exploiting TensorFlow's 'embedding_lookup' functionality, and finally sum the resulting
    embedded words vectors in order to obtain an equivalent sentence encoding.
    The Keras equivalent for 'embedding_lookup' (the Embedding Layer) isn't suitable for the task
    because it only accepts 2D tensor as input (including the batch size), while it's crucial
    to accept a 3D tensor for the context, with shape (batch size, memory size, vocabulary size).
  - In order to at least reduce some of the space occupied by the bag of words I used the 'uint8' datatype.
    This solution made it more manageable loading all the dataset in memory (and caching it on disk),
    reducing the vectorized dataset's size by a factor of 4 (scaling down from 32bits to 8bit datatypes).
    The only drawback is the necessity to cast every batch of tensors to a Keras supported type
    (float16, float32 or float64) every time the batch enters the neural network...
    to do so I used the 'K.cast_to_floatx', which converts tensors data types to the Keras default 'float32'.
  - It could be handy to define a maximum memory size, and then
    take in consideration only the latest context's facts as memory inputs.
  - Regarding the multi-hops weight tying, I opted for the Layer-wise method (with the H matrix).
  - Match type feature is not implemented.
"""

EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.01
EMBEDDING_SIZE = 32  # try from 32 to 128
HOPS = 3             # try from 1 to 4
TASK_NUMBER = 1      # 1 to 5

"""
Enable/disable caching for preprocessed data, vectorized as numpy arrays from the original textual dataset.
The data will be saved with pickle inside the 'cached-data' folder and, when deserialized later,
will be ready to enter the neural network.
This will save some time, especially if the data is stored in a fast hard drive.
Note: Pickled pre-processed data encoded as bag of words can be demanding space-wise though;
      therefore, set cache data to True only if you have some GBs of free hard drive space.
"""
USE_CACHED_DATA = True

""" Retrieve all the needed data for the selected task. """
data: UsefulData = get_everything_you_need(TASK_NUMBER, USE_CACHED_DATA)

how_many_candidates = data.candidates_matrix.shape[1]     # candidates_matrix shape: (vocab size, candidates size)
print('Candidates size: {}'.format(how_many_candidates))  # the number of answers the bot can choose from
print('Vocab size: {}'.format(data.vocabulary_size))      # all the words used in every dataset


# ======================================================================================================================
# MODEL COMPOSITION
# ======================================================================================================================

memories_tensor = Input(name='memories_tensor',
                        shape=(data.context_size,      # the memory size: the max number of facts we need to remember
                               data.vocabulary_size))  # the size of the vocabulary, including all the words for every task
memories_tensor = K.cast_to_floatx(memories_tensor)

utterance_tensor = Input(name='user_utterance_tensor',
                         shape=(data.vocabulary_size, ))
utterance_tensor = K.cast_to_floatx(utterance_tensor)

A = EmbeddingMatrixA(output_dim=EMBEDDING_SIZE)  # A shape: (embedding size, vocabulary size)
embedded_m = A(memories_tensor)         # embedded_m shape: (batch size, memory size, embedding size)
embedded_u = A(utterance_tensor)        # embedded_q shape: (batch size, embedding size)

for _ in range(HOPS):
    embedded_u = MatchingSection(output_dim=EMBEDDING_SIZE)([embedded_m, embedded_u])

answer_tensor = FinalSection(output_dim=how_many_candidates,
                             candidates_matrix=K.cast_to_floatx(data.candidates_matrix)
                             )(embedded_u)

model = Model(inputs=[memories_tensor, utterance_tensor], outputs=answer_tensor)
model.compile(loss='sparse_categorical_crossentropy',  # 'categorical_crossentropy' if candidates as one-hot vectors
              optimizer=RMSprop(learning_rate=LEARNING_RATE),
              metrics=['accuracy'])

tensorboard = TensorBoard(log_dir='logs/{}'.format(int(time())))


# ======================================================================================================================
# MODEL REPRESENTATION
# ======================================================================================================================

""" Uncomment the next line to print a model summary on the console """
# model.summary()

"""
Uncomment the following line to plot the model schema and save it as a png image file in the project folder.
NB: It requires "graphviz" installed system-wide!
    Sometimes model plotting doesn't work when main.py is run from the IDE launcher;
    therefore launch it from the system console if necessary.
"""
# plot_model(model, to_file='model-schema-{}hops.png'.format(HOPS), show_shapes=True, show_layer_names=True)


# ======================================================================================================================
# MODEL TRAINING
# ======================================================================================================================

train_memories_as_bags, train_user_utterances_as_bags, train_bot_responses_as_indices = data.vectorized_training
test_memories_as_bags, test_user_utterances_as_bags, test_bot_responses_as_indices = data.vectorized_test

model.fit([train_memories_as_bags, train_user_utterances_as_bags],
          train_bot_responses_as_indices,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_data=([test_memories_as_bags, test_user_utterances_as_bags],
                           test_bot_responses_as_indices),
          callbacks=[tensorboard])
