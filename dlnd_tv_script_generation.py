
# coding: utf-8

# # TV Script Generation
# In this project, you'll generate your own [Simpsons](https://en.wikipedia.org/wiki/The_Simpsons) TV scripts using RNNs.  You'll be using part of the [Simpsons dataset](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data) of scripts from 27 seasons.  The Neural Network you'll build will generate a new TV script for a scene at [Moe's Tavern](https://simpsonswiki.com/wiki/Moe's_Tavern).
# ## Get the Data
# The data is already provided for you.  You'll be using a subset of the original dataset.  It consists of only the scenes in Moe's Tavern.  This doesn't include other versions of the tavern, like "Moe's Cavern", "Flaming Moe's", "Uncle Moe's Family Feed-Bag", etc..

# In[1]:


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper

data_dir = './data/simpsons/moes_tavern_lines.txt'
text = helper.load_data(data_dir)
# Ignore notice, since we don't use it for analysing the data
text = text[81:]


# ## Explore the Data
# Play around with `view_sentence_range` to view different parts of the data.

# In[2]:


view_sentence_range = (0, 10)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))
scenes = text.split('\n\n')
print('Number of scenes: {}'.format(len(scenes)))
sentence_count_scene = [scene.count('\n') for scene in scenes]
print('Average number of sentences in each scene: {}'.format(np.average(sentence_count_scene)))

sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
print('Number of lines: {}'.format(len(sentences)))
word_count_sentence = [len(sentence.split()) for sentence in sentences]
print('Average number of words in each line: {}'.format(np.average(word_count_sentence)))

print()
print('The sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))


# ## Implement Preprocessing Functions
# The first thing to do to any dataset is preprocessing.  Implement the following preprocessing functions below:
# - Lookup Table
# - Tokenize Punctuation
# 
# ### Lookup Table
# To create a word embedding, you first need to transform the words to ids.  In this function, create two dictionaries:
# - Dictionary to go from the words to an id, we'll call `vocab_to_int`
# - Dictionary to go from the id to word, we'll call `int_to_vocab`
# 
# Return these dictionaries in the following tuple `(vocab_to_int, int_to_vocab)`

# In[3]:


import numpy as np
import problem_unittests as tests

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    # TODO: Implement Function
    vocab = set(text)
    vocab_to_int = {c: i for i, c in enumerate(vocab)}
    int_to_vocab = dict(enumerate(vocab))
    return (vocab_to_int, int_to_vocab)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_create_lookup_tables(create_lookup_tables)


# ### Tokenize Punctuation
# We'll be splitting the script into a word array using spaces as delimiters.  However, punctuations like periods and exclamation marks make it hard for the neural network to distinguish between the word "bye" and "bye!".
# 
# Implement the function `token_lookup` to return a dict that will be used to tokenize symbols like "!" into "||Exclamation_Mark||".  Create a dictionary for the following symbols where the symbol is the key and value is the token:
# - Period ( . )
# - Comma ( , )
# - Quotation Mark ( " )
# - Semicolon ( ; )
# - Exclamation mark ( ! )
# - Question mark ( ? )
# - Left Parentheses ( ( )
# - Right Parentheses ( ) )
# - Dash ( -- )
# - Return ( \n )
# 
# This dictionary will be used to token the symbols and add the delimiter (space) around it.  This separates the symbols as it's own word, making it easier for the neural network to predict on the next word. Make sure you don't use a token that could be confused as a word. Instead of using the token "dash", try using something like "||dash||".

# In[4]:


def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    # TODO: Implement Function
#     punctuation = (',','.','"',';','!','?','(',')','--','\n')
#     i = "||"
#     punc_to_token = {c : i for c in punctuation}
#     return punc_to_token
    punc_dict = {'.' :  '||Period||',
                 ',' :  '||Comma||',
                 '"' :  '||Quotation||',
                 ';' :  '||Semicolon||', 
                 '!' :  '||Exclamation||', 
                 '?' :  '||Question||', 
                 '(' :  '||Left||', 
                 ')' :  '||Right||',
                 '--' : '||Dash||', 
                 '\n' : '||Return||'}
    return punc_dict

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_tokenize(token_lookup)


# ## Preprocess all the data and save it
# Running the code cell below will preprocess all the data and save it to file.

# In[5]:


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)


# # Check Point
# This is your first checkpoint. If you ever decide to come back to this notebook or have to restart the notebook, you can start from here. The preprocessed data has been saved to disk.

# In[6]:


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper
import numpy as np
import problem_unittests as tests

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()


# ## Build the Neural Network
# You'll build the components necessary to build a RNN by implementing the following functions below:
# - get_inputs
# - get_init_cell
# - get_embed
# - build_rnn
# - build_nn
# - get_batches
# 
# ### Check the Version of TensorFlow and Access to GPU

# In[7]:


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


# ### Input
# Implement the `get_inputs()` function to create TF Placeholders for the Neural Network.  It should create the following placeholders:
# - Input text placeholder named "input" using the [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder) `name` parameter.
# - Targets placeholder
# - Learning Rate placeholder
# 
# Return the placeholders in the following tuple `(Input, Targets, LearningRate)`

# In[8]:


def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    # TODO: Implement Function
    
# get_inputs 函数实现正确，成功返回张量占位符（TF Placeholder）✅。
# 代码实现上注意下input为python内建函数，变量命名的时候建议避开python内建函数名，使用内建函数名命名变量会导致该内建函数被overshadow，
# 从而在当前代码范围内不可用。
# 可以看下这里的内建函数列表：https://docs.python.org/3.5/library/functions.html
    
    input = tf.placeholder(tf.int32,[None, None], name = 'input')
    targets = tf.placeholder(tf.int32,[None, None], name = 'targets')
    learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')
    return (input, targets, learning_rate)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_inputs(get_inputs)


# ### Build RNN Cell and Initialize
# Stack one or more [`BasicLSTMCells`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/BasicLSTMCell) in a [`MultiRNNCell`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell).
# - The Rnn size should be set using `rnn_size`
# - Initalize Cell State using the MultiRNNCell's [`zero_state()`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell#zero_state) function
#     - Apply the name "initial_state" to the initial state using [`tf.identity()`](https://www.tensorflow.org/api_docs/python/tf/identity)
# 
# Return the cell and initial state in the following tuple `(Cell, InitialState)`

# In[9]:


def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    # TODO: Implement Function
     ### Build the LSTM Cell
    # Use a basic LSTM cell
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    num_layers = 2
    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell([lstm] * num_layers)
    initial_state = cell.zero_state(batch_size, tf.float32)
    initial_state = tf.identity(initial_state, name = 'initial_state')
    return (cell, initial_state)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_init_cell(get_init_cell)


# ### Word Embedding
# Apply embedding to `input_data` using TensorFlow.  Return the embedded sequence.

# In[10]:


def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    # TODO: Implement Function
#     n_vocab = len(int_to_vocab)
#     n_embedding = 200 # Number of embedding features 
    
# get_embed函数实现正确，get_embed函数相当于创建一个全链接层，
# 函数实现里将embedding weights(或者说embedding matrix)初始化为-1到1之间的uniform random numbers，实现的非常棒。

    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, input_data)
    return embed


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_embed(get_embed)


# ### Build RNN
# You created a RNN Cell in the `get_init_cell()` function.  Time to use the cell to create a RNN.
# - Build the RNN using the [`tf.nn.dynamic_rnn()`](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn)
#  - Apply the name "final_state" to the final state using [`tf.identity()`](https://www.tensorflow.org/api_docs/python/tf/identity)
# 
# Return the outputs and final_state state in the following tuple `(Outputs, FinalState)` 

# In[11]:


def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    # TODO: Implement Function
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype = tf.float32)
    final_state = tf.identity(final_state, name = 'final_state')
    return (outputs, final_state)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_build_rnn(build_rnn)


# ### Build the Neural Network
# Apply the functions you implemented above to:
# - Apply embedding to `input_data` using your `get_embed(input_data, vocab_size, embed_dim)` function.
# - Build RNN using `cell` and your `build_rnn(cell, inputs)` function.
# - Apply a fully connected layer with a linear activation and `vocab_size` as the number of outputs.
# 
# Return the logits and final state in the following tuple (Logits, FinalState) 

# In[12]:


def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Logits, FinalState)
    """
    # TODO: Implement Function
    embedding_layer = get_embed(input_data, vocab_size, embed_dim)
    outputs, final_state = build_rnn(cell, embedding_layer)
    
#     build_nn 函数实现正确，你的函数实现里注意到将全链接层的激活函数设为None以保持线性激活，非常棒！
    logits = tf.layers.dense(outputs, vocab_size, activation= None, use_bias=True)
    return (logits, final_state)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_build_nn(build_nn)


# ### Batches
# Implement `get_batches` to create batches of input and targets using `int_text`.  The batches should be a Numpy array with the shape `(number of batches, 2, batch size, sequence length)`. Each batch contains two elements:
# - The first element is a single batch of **input** with the shape `[batch size, sequence length]`
# - The second element is a single batch of **targets** with the shape `[batch size, sequence length]`
# 
# If you can't fill the last batch with enough data, drop the last batch.
# 
# For exmple, `get_batches([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 2, 3)` would return a Numpy array of the following:
# ```
# [
#   # First Batch
#   [
#     # Batch of Input
#     [[ 1  2  3], [ 7  8  9]],
#     # Batch of targets
#     [[ 2  3  4], [ 8  9 10]]
#   ],
#  
#   # Second Batch
#   [
#     # Batch of Input
#     [[ 4  5  6], [10 11 12]],
#     # Batch of targets
#     [[ 5  6  7], [11 12 13]]
#   ]
# ]
# ```

# In[13]:


def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    # TODO: Implement Function
#     n_batches = len(int_text)
    
#     # only full batches
#     words = np.array(int_text[:n_batches * (batch_size  * seq_length)])
#     for idx in range(0, len(int_text), batch_size):
#         x, y = [], []
#         batch = words[idx:idx+batch_size]
#         for ii in range(len(batch)):
#             batch_x = batch[ii]
#             batch_y = get_target(batch, ii, seq_length)
#             y.extend(batch_y)
#             x.extend([batch_x]*len(batch_y))
#             x = np.array(x)
#             y = np.array(y)
#         yield x, y
    
#     return x

# get_batches 函数实现的非常棒👍，在这里int_text是一个序列数据，在这里是剧本里单词按语意排列层的句子，
# 我们要把数据分成很多小序列，然后让神经网络在这些小序列上并行训练。
# 比如把序列 [1 2 3 4 5 6 7 8 9 10 11 12] 分成 [1 2 3 4 5 6] 和 [7 8 9 10 11 12]，
# 然后如果我们以2的step size训练，我们就要从分好的序列里分别取2个步长，
# 第一个batch: Batch of Input传入[[1 2], [7 8]], Batch of targets为input batch后移一位，即[[2 3], [8 9]]。
# 第二个batch: Batch of Input传入[[3 4], [9 10]]，Batch of targets同样为input batch后移一位。
 # Calculate the number of batches
    num_batches = len(int_text) // (batch_size  * seq_length)
    # Drop long batches. Transform into a numpy array and reshape it for our purposes
    np_text = np.array(int_text[:num_batches * (batch_size  * seq_length)])
    # Reshape the data to give us the inputs sequence.
    in_text = np_text.reshape(-1, seq_length)
    # Roll (shift) and reshape to get target sequences (maybe not optimal)
    tar_text = np.roll(np_text, -1).reshape(-1, seq_length)
    output = np.zeros(shape=(num_batches, 2, batch_size, seq_length), dtype=np.int)
    # Prepare the output
    for idx in range(0, in_text.shape[0]):
        jj = idx % num_batches
        ii = idx // num_batches
        output[jj,0,ii,:] = in_text[idx,:]
        output[jj,1,ii,:] = tar_text[idx,:]
    return output


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_batches(get_batches)


# ## Neural Network Training
# ### Hyperparameters
# Tune the following parameters:
# 
# - Set `num_epochs` to the number of epochs.
# - Set `batch_size` to the batch size.
# - Set `rnn_size` to the size of the RNNs.
# - Set `embed_dim` to the size of the embedding.
# - Set `seq_length` to the length of sequence.
# - Set `learning_rate` to the learning rate.
# - Set `show_every_n_batches` to the number of batches the neural network should print progress.

# In[14]:


# Epoch的选择主要是需要使神经网络的训练 loss 接近或达到最小，一般选择当给出更多训练次数神经网络的结果不再有太大的改善时附近的值，
# 从运行结构看，loss值128的迭代次数下loss值降低的效果比较好，也可以适当提高到150左右，看下继续训练下去loss值是否会进一步降低。

# batch size的选择主要是满足使神经网络能够高效地训练，同时也不能过大否则内存会不够。
# batch size 往往受你 GPU 显存的影响， batch size一般设为128，256，512都是不错的选择。

# RNN size（即隐藏层中节点的数量）需要选的足够大，使之能够很好的拟合数据， RNN size 越大，模型的学习能力越强，
# 但是同时也注意，增大到一定值后模型开始overfitting，128, 256, 512是常用的典型值。

# embed_dim是配合总单词量一起创建embedding lookup table即embedding weights的，相当于决定embedding layer隐藏单元权重的个数，
# 这个值跟 rnn size 一样，取的越大模型的能力会越强，但是代价是训练速度的下降。
# 取的过大的话相当于需要计算的weights过多，导致神经网络难以训练。这个参数选择的时候建议以级数 (32,64,128,256)方式优化调整而不是线形调整，
# 这样神经网络的表现会更偏向指数特性而不是线形特性。对这个项目一般推荐取200左右以跟词汇量相匹配，这里设为128或256都是比较合适。

# seq length 与long term的数据结构有关，应当与给出数据的结构相匹配，The network learns through backprop only over the length you have defined，
# 项目推荐应大约是你希望生成句子的长度， 对单词来说，一个句子大概15到20个单词，通常大一点的值会使模型的能力也跟强一些，这个参数设置合理。

# 学习速率选的不错，对这个项目来说，0.01,0.001都是不错的选择。

# show_every_n_batches 为训练过程中打印进程频率，每show_every_n_batches 打印一次训情况。

# Number of Epochs
num_epochs = 128
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 512
# Embedding Dimension Size
embed_dim = 256
# Sequence Length
seq_length = 16
# Learning Rate
learning_rate = 0.001
# Show stats for every n number of batches
show_every_n_batches = 64

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
save_dir = './save'


# ### Build the Graph
# Build the graph using the neural network you implemented.

# In[15]:


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from tensorflow.contrib import seq2seq

train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')

    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)


# ## Train
# Train the neural network on the preprocessed data.  If you have a hard time getting a good loss, check the [forms](https://discussions.udacity.com/) to see if anyone is having the same problem.

# In[16]:


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
batches = get_batches(int_text, batch_size, seq_length)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # Show every <show_every_n_batches> batches
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')


# ## Save Parameters
# Save `seq_length` and `save_dir` for generating a new TV script.

# In[17]:


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Save parameters for checkpoint
helper.save_params((seq_length, save_dir))


# # Checkpoint

# In[18]:


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import tensorflow as tf
import numpy as np
import helper
import problem_unittests as tests

_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
seq_length, load_dir = helper.load_params()


# ## Implement Generate Functions
# ### Get Tensors
# Get tensors from `loaded_graph` using the function [`get_tensor_by_name()`](https://www.tensorflow.org/api_docs/python/tf/Graph#get_tensor_by_name).  Get the tensors using the following names:
# - "input:0"
# - "initial_state:0"
# - "final_state:0"
# - "probs:0"
# 
# Return the tensors in the following tuple `(InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)` 

# In[19]:


def get_tensors(loaded_graph):
    """
    Get input, initial state, final state, and probabilities tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
    """
    # TODO: Implement Function
    input_tensor = loaded_graph.get_tensor_by_name("input:0")
    initial_state_tensor = loaded_graph.get_tensor_by_name("initial_state:0")
    final_state_tensor = loaded_graph.get_tensor_by_name("final_state:0")
    probs_tensor = loaded_graph.get_tensor_by_name("probs:0")
    return (input_tensor, initial_state_tensor, final_state_tensor, probs_tensor)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_tensors(get_tensors)


# ### Choose Word
# Implement the `pick_word()` function to select the next word using `probabilities`.

# In[20]:


def pick_word(probabilities, int_to_vocab):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """
    # TODO: Implement Function
    
#     pick_word 函数实现正确，
#     这里可以考虑在实现的时候加入些选词的随机性（slight randomness ），如使用np.random.choice()函数，
#     因为选词的时候概率最大的不代表是最合适的选择。
    
    argmax_pr = np.argmax(probabilities)
    next_word = int_to_vocab[argmax_pr]
    return next_word


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_pick_word(pick_word)


# ## Generate TV Script
# This will generate the TV script for you.  Set `gen_length` to the length of TV script you want to generate.

# In[21]:


gen_length = 200
# homer_simpson, moe_szyslak, or Barney_Gumble
prime_word = 'moe_szyslak'

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # Get Tensors from loaded model
    input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

    # Sentences generation setup
    gen_sentences = [prime_word + ':']
    prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

    # Generate sentences
    for n in range(gen_length):
        # Dynamic Input
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])

        # Get Prediction
        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input_text: dyn_input, initial_state: prev_state})
        
        pred_word = pick_word(probabilities[dyn_seq_length-1], int_to_vocab)

        gen_sentences.append(pred_word)
    
    # Remove tokens
    tv_script = ' '.join(gen_sentences)
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        tv_script = tv_script.replace(' ' + token.lower(), key)
    tv_script = tv_script.replace('\n ', '\n')
    tv_script = tv_script.replace('( ', '(')
        
    print(tv_script)


# # The TV Script is Nonsensical
# It's ok if the TV script doesn't make any sense.  We trained on less than a megabyte of text.  In order to get good results, you'll have to use a smaller vocabulary or get more data.  Luckly there's more data!  As we mentioned in the begging of this project, this is a subset of [another dataset](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data).  We didn't have you train on all the data, because that would take too long.  However, you are free to train your neural network on all the data.  After you complete the project, of course.
# # Submitting This Project
# When submitting this project, make sure to run all the cells before saving the notebook. Save the notebook file as "dlnd_tv_script_generation.ipynb" and save it as a HTML file under "File" -> "Download as". Include the "helper.py" and "problem_unittests.py" files in your submission.
