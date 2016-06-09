import tensorflow as tf
from tensorflow.models.rnn import rnn_cell

from nltk.corpus import brown

batch_size = 100
num_steps = 10
lstm_size = 100
vocab_size = 10000
size = 1000


# Placeholder for the inputs in a given iteration.
words = tf.placeholder(tf.int32, [batch_size, num_steps])

lstm = rnn_cell.BasicLSTMCell(lstm_size)
# Initial state of the LSTM memory.
initial_state = state = tf.zeros([batch_size, lstm.state_size])
targets = tf.placeholder(tf.int32, [batch_size, num_steps])

for i in range(len(num_steps)):
    # The value of state is updated after processing each batch of words.
    output, state = lstm(words[:, i], state)

    # The rest of the code.
    # ...

final_state = state

# functions defined
softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
softmax_b = tf.get_variable("softmax_b", [vocab_size])
logits = tf.matmul(output, softmax_w) + softmax_b
loss = tf.nn.seq2seq.sequence_loss_by_example(
    [logits],
    [tf.reshape(targets, [-1])],
    [tf.ones([batch_size * num_steps])])

# A numpy array holding the state of LSTM after each batch of words.
numpy_state = initial_state.eval()
total_loss = 0.0
for fileid in brown.filesid():
    current_batch_of_words = brown.words()
    numpy_state, current_loss = tf.session.run([final_state, loss],
                                                feed_dict={initial_state: numpy_state, words: current_batch_of_words})
    total_loss += current_loss


