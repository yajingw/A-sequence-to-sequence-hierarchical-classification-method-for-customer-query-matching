import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy
import pickle
from sklearn.metrics import  average_precision_score,precision_recall_fscore_support
from tensorflow.keras import layers,metrics

class StructureAttention(tf.keras.layers.Layer):
    def __init__(self, vector_matrix,max_node_num):
        super(StructureAttention, self).__init__()
        self.vector_matrix = tf.constant(vector_matrix, tf.float32)
        # self.vector_matrix = tf.pad(self.vector_matrix, [[0, max_node_num - vector_matrix.shape[0]],
        #                                                  [0, max_node_num - vector_matrix.shape[1]]])
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
    def call(self, inputs):
        father,son = inputs[0],inputs[1]
        score = tf.matmul(father, self.vector_matrix)
        # structed_son = self.layer_norm1(score * son)
        # structed_son = tf.math.sqrt(score * son) #相比较于HC版本增加了sqrt的部分 normalize 一下就好
        structed_son = score * son
        return structed_son

class treeDecoder(tf.keras.layers.Layer):
    def __init__(self, units,embedding_matrix,max_node_num):
        super( treeDecoder, self).__init__()
        self.units = units
        self.vector_matrix = tf.constant(embedding_matrix, tf.float32)# (max_node_num, 1024)
        # self.vector_matrix = tf.pad(self.vector_matrix, [[0, max_node_num - embedding_matrix.shape[0]],[0, 0]]) 预处理已经pad过了
        self.W_q = layers.Dense(max_node_num,use_bias=False)
        # self.W_k = layers.Dense(self.units,use_bias=False)

    def call(self, inputs):
        question = inputs # TensorShape([B, 1024])
        q = self.W_q(question)#  [B, max_node_num]
        # k = self.W_k(self.vector_matrix) #  # (max_node_num, units )
        similarity_score = tf.matmul(q, self.vector_matrix, transpose_b=False) # (B, max_node_num)
        return similarity_score

class AttentionEncoder(tf.keras.layers.Layer):
    def __init__(self, units, bert_dim):
        super(AttentionEncoder, self).__init__()
        self.units = units # 128
        self.bert_dim = bert_dim
        self.W_q = layers.Dense(self.units,use_bias=False)
        self.W_k = layers.Dense(self.units,use_bias=False)
        self.W_v= layers.Dense(self.units,use_bias=False)

    def call(self, query, context):
        q = self.W_q(query) # [22, 1, 1024] [1024, 128] -> [22, 1, 128]
        k =  self.W_k(context) # [22, 106, 1024] [1024, 128] -> [22, 106, 128]
        attention_weights = tf.nn.softmax(tf.matmul(q, k, transpose_b=True)) # [22,1,106]
        v = self.W_v(context) # [22, 106, 1024] [1024, 128] -> [22, 106, 128]
        output = tf.matmul(attention_weights, v)
        return output # TensorShape([22, 1, 128])

class Attention_lstm_model(tf.keras.Model):

  def __init__(self,d_bert, batch_size,latent_dim,node_num, tree_matrix,tree_embedding):
    super(Attention_lstm_model, self).__init__()
    self.level_num = len(tree_embedding)
    self.d_bert = d_bert
    self.batch_size = batch_size
    self.latent_dim = latent_dim  # 128有点over fitting了
    self.node_num = node_num

    self.encoder = AttentionEncoder(self.d_bert,self.d_bert)
    self.encoder_lstm = layers.LSTM(self.latent_dim, return_sequences=True, return_state=True, name='encoder_lstm')
    self.decoder_lstm = layers.LSTM(self.latent_dim, return_state=True, name='decoder_lstm')
    
    self.treeDecoders = [treeDecoder(self.latent_dim, tree_embedding[i], node_num) for i in range(0, len(tree_embedding))]
    self.decoder_denses = [layers.Dense(self.node_num,activation='sigmoid',name='decoder_dense'+str(i)) for i in range(0, len(tree_embedding))]
    self.structures = [StructureAttention(tree_matrix[i],node_num) for i in range(0, len(tree_embedding))]
      
  def call(self, inputs):
    query = inputs[0]
    context = inputs[1]
    context_vector = self.encoder(query, context)
    question = tf.concat([query,context_vector], axis=-1) # TensorShape([2, 1, 2048])
    encoder_outputs, encoder_state_h, encoder_state_c = self.encoder_lstm(question) # [22, 1, 128],[22, 128],[22, 128]


    all_outputs = []
    decoder_inputs = numpy.zeros((self.batch_size, self.node_num)).astype(numpy.float32)
    decoder_inputs[:, 0] = 1
  # 循环在每一层解码
    decoder_states = [encoder_state_h, encoder_state_c]
    for i in range(0,self.level_num):
         outputs1, state_h, state_c = self.decoder_lstm(tf.expand_dims(decoder_inputs, 1), initial_state=decoder_states) 
         outputs1 = self.treeDecoders[i](outputs1)
         outputs1 = self.decoder_denses[i](outputs1)
         outputs1 = self.structures[i]([decoder_inputs, outputs1])
         # 更新状态
         all_outputs.append(outputs1)
         decoder_inputs = outputs1
         decoder_states = [state_h, state_c]
    all_outputs = layers.Lambda(lambda x: tf.concat(x, axis=1))(all_outputs)
    return all_outputs