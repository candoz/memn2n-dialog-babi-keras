import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
"""
Contains my Neural Network custom sections.
The Embedding matrix 'A' will be used to embed both the last user's utterance and the context's facts saved in memory.
The 'Matching section' will be called N times, where N is the number of Hops.
The final section will be used to select the most relevant response between all the candidates.
"""


class EmbeddingMatrixA(Layer):
    """
    Embedding matrix A of size dxV, where:
    - d is the embedding size
    - V is the vocabulary size
    It's used to embed both the user utterance and the facts inside the memory (the context).
    """

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(EmbeddingMatrixA, self).__init__(**kwargs)

    def build(self, input_shape):

        # Note: The vocab size will be the last dimension for both the kind of tensors embedded with A:
        #       - the memories tensor will have the shape: (batch size, memory size, vocab size)
        #       - the user's utterance tensor will have the shape: (batch size, vocab size)
        vocab_size = input_shape[-1]

        self.A = self.add_weight(name='A',
                                 shape=(vocab_size, self.output_dim),
                                 initializer='random_normal',
                                 trainable=True)
        super(EmbeddingMatrixA, self).build(input_shape)

    def call(self, input, mask=None):
        return K.dot(input, self.A)


class MatchingSection(Layer):
    """
    Neural Network section that matches the embedded question with the embedded memories.
    """

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MatchingSection, self).__init__(**kwargs)

    def build(self, input_shape):
        embedded_memories_shape = input_shape[0]  # (batch size, memory size, embedding size)
        embedded_question_shape = input_shape[1]  # (batch size, embedding size)
        embedding_size = embedded_memories_shape[2]
        assert (embedding_size == embedded_question_shape[1] == self.output_dim)

        # Matrix weights of size dxd
        self.R = self.add_weight(name='R',
                                 shape=(embedding_size, embedding_size),
                                 initializer='random_normal',
                                 trainable=True)

        super(MatchingSection, self).build(input_shape)

    def call(self, inputs, mask=None):
        # Tip: Print K.int_shape(your_tensor) to see its dimensions:
        #      don't know why, but some debuggers have problems in this code section.

        embedded_memories = inputs[0]
        embedded_user_utterance = inputs[1]

        inner_product = K.batch_dot(embedded_memories, embedded_user_utterance)
        weights = K.softmax(inner_product)
        weighted_embedded_memories = K.batch_dot(weights, embedded_memories)

        o = K.dot(weighted_embedded_memories, self.R)

        return embedded_user_utterance + o


class FinalSection(Layer):
    """
    Neural Network section used to choose the final response, it:
      - multiplies the input for a W matrix of size dxV, then
      - multiplies it for the candidates matrix (with all the candidates encoded as bags of words), and finally
      - applies a softmax.
    """

    def __init__(self, output_dim, candidates_matrix, **kwargs):
        self.output_dim = output_dim
        self.candidates_matrix = candidates_matrix
        super(FinalSection, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]  # remember that input_shape[0] is the batch size
        self.W = self.add_weight(name='W',
                                 shape=(input_dim, self.candidates_matrix.shape[0]),
                                 initializer='random_normal',
                                 trainable=True)

        super(FinalSection, self).build(input_shape)

    def call(self, input, mask=None):
        w_output = K.dot(input, self.W)
        matched_candidates = K.dot(w_output, self.candidates_matrix)
        return K.softmax(matched_candidates)
