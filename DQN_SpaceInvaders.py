import random
import gym
import tensorflow as tf
import numpy as np
import progressbar 
import time
import collections
import shutil
import matplotlib.pyplot as plt



#################################################################################################################################################################################
# %% Initialize code

shutil.rmtree('/home/ryan/Documents/Python/OpenAI Tutorial/logs', ignore_errors=True)

###########################################################################################################################
# %% Classes

class DQN():
    def __init__(self,env,batch_size,memory_size):
        self.env = env
        self.sess = tf.Session()

        self.input_height = self.env.observation_space.shape[0]
        self.input_width = self.env.observation_space.shape[1] 
        self.action_size = self.env.action_space.n
        self.mini_batch_size = 20
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.mini_batch_len = self.batch_size//self.mini_batch_size
        self.cells = 2
        self.cell_size = 20

        self.lr = 10**-4
        self.gamma = .95
        self.epsilon_min = .01
        self.epsilon_decay = .995
        self.i = 0
        self.keep_prob = .8
        
        self.memory = collections.deque(maxlen = self.memory_size)
        
        with tf.name_scope('Inputs'):
            self.s0 = tf.placeholder(tf.float32,[None,self.input_height,self.input_width,3],name='s_0')
            self.s1 = tf.placeholder(tf.float32,[None,self.input_height,self.input_width,3],name='s_1')
            self.target = tf.placeholder(tf.float32,[None,self.action_size],name='Target')
        
        with tf.variable_scope('Q_Net') as model_vars:
            with tf.name_scope('Prediction_Net'):
                self.Q_pred = self.CNN(self.s0)
            with tf.name_scope('Target_Net'):
                model_vars.reuse_variables()
                self.Q_target = self.CNN(self.s1)
            self.model_vars = [var for var in tf.global_variables() if var.name.startswith(model_vars.name)]
        
 
        
        with tf.name_scope('Losses'):
            self.losses()
        
        with tf.variable_scope('Optimize') as opt_vars:
            self.optimize()
        self.opt_vars = [var for var in tf.global_variables() if var.name.startswith(opt_vars.name)]
        

        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('logs/', self.sess.graph)
        
    # ===================================================================================================#
    # Tensorflow Functions
    # ===================================================================================================#
               
    def Initialize(self):
        init_model_vars = tf.variables_initializer(self.model_vars)
        init_opt_vars = tf.variables_initializer(self.opt_vars)
        self.sess.run(init_model_vars)
        self.sess.run(init_opt_vars)
        
    # ===================================================================================================#
    # Environment Functions
    # ===================================================================================================#
    
    def reset(self):
        self.current_state = self.env.reset()
        
    def move(self):
        if np.random.rand() <= self.epsilon:
            self.action = random.randrange(self.action_size)
        else:
            feed_dict = {self.s0: np.reshape(self.current_state,[1,self.input_height,self.input_width,3])}
            self.action = np.argmax(self.sess.run(self.Q_pred,feed_dict = feed_dict))

    def step_env(self):
        self.next_state,self.reward,self.done,self.info = self.env.step(self.action)


    # ===================================================================================================#
    # Loss Function
    # ===================================================================================================#
    
    def losses(self):
        self.loss = tf.losses.mean_squared_error(self.target,self.Q_pred)
        tf.summary.scalar('Loss',self.loss)



    # ===================================================================================================#
    # Optimization
    # ===================================================================================================#
    
    def optimize(self):
        if self.i == 0:
            self.epsilon = 1
        grad = tf.train.AdamOptimizer(self.lr).compute_gradients(self.loss,var_list=self.model_vars)
        for i,_ in enumerate(grad):
            tf.summary.histogram('{0}_Grad'.format(self.model_vars[i].name),grad[i])
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
         
    # ===================================================================================================#
    # Deep Q Learning Functions
    # ===================================================================================================#
    
    def remember(self):
        self.memory.append((self.current_state,self.action,self.reward,self.next_state,self.done))
        
    def learn(self):
        
        batch= random.sample(self.memory,self.batch_size)
        for mini_batch in range(self.mini_batch_size):
            s_0 = np.array([batch[mini_batch*self.mini_batch_len:mini_batch*self.mini_batch_len+self.mini_batch_len][i][0] for i in range(self.mini_batch_len)])
            s_1 = np.array([batch[mini_batch*self.mini_batch_len:mini_batch*self.mini_batch_len+self.mini_batch_len][i][3] for i in range(self.mini_batch_len)])

            r = np.array([batch[mini_batch*self.mini_batch_len:mini_batch*self.mini_batch_len+self.mini_batch_len][i][2] for i in range(self.mini_batch_len)])
            a = np.array([batch[mini_batch*self.mini_batch_len:mini_batch*self.mini_batch_len+self.mini_batch_len][i][1] for i in range(self.mini_batch_len)])
            done = 1-np.array([batch[mini_batch*self.mini_batch_len:mini_batch*self.mini_batch_len+self.mini_batch_len][i][4] for i in range(self.mini_batch_len)])*1
            
            feed_dict_target = {self.s1: s_1}
            Q_target = self.sess.run(self.Q_target,feed_dict = feed_dict_target)
            target = r+ self.gamma*np.amax(Q_target,axis = 1)*done
            
            feed_dict_pred = {self.s0: s_0}
            target_f = self.sess.run(self.Q_pred,feed_dict = feed_dict_pred)
            target_f[(np.eye(len(a))[a]).astype(bool)] = target

            feed_dict = {self.s0: s_0, self.s1: s_1,self.target: target_f}
            self.sess.run(self.train_step,feed_dict = feed_dict)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.results = self.sess.run(self.merged,feed_dict = feed_dict)  
        self.writer.add_summary(self.results,self.i)
        self.i += 1
        
        
    #===================================================================================================#
    # RNN Functions 
    #===================================================================================================#
    
    def rnn_cell(self,):
        with tf.name_scope('RRN_cell'):
            rnn_cell = tf.contrib.rnn.BasicRNNCell(self.cell_size)
            with tf.name_scope('Dropout'):
                rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=self.keep_prob)
            return rnn_cell
    
    def lstm_cell(self,):
        with tf.name_scope('LSTM_cell'):
            lstm_cell = tf.contrib.rnn.LSTMCell(self.cell_size, forget_bias = 1,initializer= tf.contrib.layers.xavier_initializer())
            with tf.name_scope('Dropout'):
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
            return lstm_cell

    def add_rnn_layer(self,x,single_output,n_layer):
        
        if self.LSTM:    
            layer_name = 'LSTM_Layer_%s' % n_layer
        else:
            layer_name = 'RNN_Layer_%s' % n_layer
            
        with tf.variable_scope(layer_name):
            x = tf.reshape(x,(self.batch_size,1,1))
            if self.LSTM:
                stacked_cell = tf.contrib.rnn.OutputProjectionWrapper(self.lstm_cell(),self.action_num)
                stacked_cell = tf.contrib.rnn.MultiRNNCell([stacked_cell]*self.cells)
            else:
                stacked_cell = tf.contrib.rnn.OutputProjectionWrapper(self.rnn_cell(),self.action_num)
                stacked_cell = tf.contrib.rnn.MultiRNNCell([stacked_cell]*self.cells)
          
            with tf.name_scope('Output'):
                output, state = tf.nn.dynamic_rnn(stacked_cell,x,dtype=tf.float32)
                tf.summary.histogram('state',state)
                tf.summary.histogram('output',output)
            
                if single_output:
                    output = tf.transpose(output, [1, 0, 2])
                    output = tf.gather(output, int(output.get_shape()[0]) - 1)
                
        return output 
    
    #===================================================================================================#
    # CNN Functions 
    #===================================================================================================#
    
    def CNN(self,x):
        with tf.name_scope('CNN'):
            o1 = self.add_conv_layer(x,3,32,8,0,4,1,tf.nn.relu,False,False)
            o2 = self.add_conv_layer(o1,32,64,4,0,2,2,tf.nn.relu,False,False)
            o3 = self.add_conv_layer(o2,64,64,1,2,2,3,tf.nn.relu,False,False)
            o3 = tf.reshape(o3,[-1,14*10*64])
            o4 = self.add_fc_layer(o3,o3.get_shape()[-1].value,512,4,tf.nn.relu,True,False)
            o5 = self.add_fc_layer(o4,o4.get_shape()[-1].value,self.action_size,5,None,False,False)
        return o5
    
    def add_conv_layer(self,inputs,feature_old,feature_new,patch,kernal,strides,
                       n_layer,activation_function=None,pool = False,batch_normalization = False):
        
        layer_name = 'Conv_Layer_%s' % n_layer        
        with tf.variable_scope(layer_name):
            with tf.name_scope('Weights'):
                W = self.weights([patch,patch,feature_old, feature_new], name='W')
            tf.summary.histogram('Weights',W)
            with tf.name_scope('Biases'):
                b = self.bias([feature_new], name='b')
            tf.summary.histogram('Biases',b)
            with tf.name_scope('Convolution'):
                conv = tf.nn.conv2d(inputs,W,strides = [1,strides,strides,1],padding = 'SAME')
                tf.summary.histogram( 'Output', conv)
            with tf.name_scope('Conv_plus_b'):
                if activation_function is None:
                    conv_plus_b = conv+ b
                else:
                    conv_plus_b = activation_function(conv+b)
                    tf.summary.histogram( 'Output', conv_plus_b)
            if pool == True:
                with tf.name_scope('Pooling'):
                    pooling = tf.nn.max_pool(conv_plus_b,ksize = [1,kernal, kernal,1],
                            strides = [1,kernal,kernal,1],padding = 'SAME') 
                    tf.summary.histogram('Output', pooling)
            else:       
                pooling = conv_plus_b
            if batch_normalization == True:
                with tf.name_scope('Batch_Normalization'):
                    o1 = tf.contrib.layers.batch_norm(pooling)
                    tf.summary.histogram('output',o1)
            else:       
                o1 = pooling   
                    
        return o1            
            
        
    # ===================================================================================================#
    # NN Functions
    # ===================================================================================================#

    def add_fc_layer(self, inputs, in_size, out_size, n_layer, activation_function=None,
                     keep_prob_=False, batch_normalization=False):

        layer_name = 'FC_Layer_%s' % n_layer
        with tf.variable_scope(layer_name):
            with tf.name_scope('Weights'):
                W = self.weights([in_size, out_size], name='W')
            tf.summary.histogram('Weights', W)
            with tf.name_scope('Biases'):
                b = self.bias([out_size], name='b')
            tf.summary.histogram('Biases', b)
            with tf.name_scope('Wx_plus_b'):
                Wx_plus_b = tf.matmul(inputs, W)+ b
                if activation_function is not None:
                    Wx_plus_b = activation_function(Wx_plus_b)
                tf.summary.histogram('Output', Wx_plus_b)
            if keep_prob_ == True:
                with tf.name_scope('Dropout'):
                    o1 = tf.nn.dropout(Wx_plus_b, self.keep_prob)
                    tf.summary.histogram('Output', o1)
            else:
                o1 = Wx_plus_b
            if batch_normalization == True:
                with tf.name_scope('Batch_Normalization'):
                    o2 = tf.contrib.layers.batch_norm(o1)
                    tf.summary.histogram('output', o2)
            else:
                o2 = o1

        return o2

    # ===================================================================================================#
    # Parameter Variables
    # ===================================================================================================#

    def weights(self, shape, name):
        initializer = tf.contrib.layers.xavier_initializer()
        #initializer = tf.truncated_normal()
        #initializer = tf.truncated_normal_initializer(stddev=.01)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def bias(self, shape, name):
        initializer = tf.constant_initializer(0)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)


#################################################################################################################################################################################    
# %% Functions
        
def render(env,step,epoch,epsilon,title):
    plt.figure(2)
    plt.clf()
    plt.imshow(env.render(mode='rgb_array'))
    plt.title('%(x)s, Step = %(y)s, Epoch = %(z)s, Epsilon = %(q)s'  % {'x': title,
              'y': step, 'z': epoch, 'q': epsilon})
    plt.axis('off')
    plt.pause(.01)
    plt.draw()
    
    
###########################################################################################################################
# %% Run Function

def run():

    title = 'SpaceInvaders-v0'
    epoch_bar = progressbar.ProgressBar()
    
    #parameters
    epoch_len = 10**3*5
    memory_size = 30000
    batch_size = 5000
    time_max = 2000
    render_epoch = 50
    
    #create model and environment
    env = gym.make(title)
    tf.reset_default_graph()
    model = DQN(env,batch_size,memory_size)
    
    #initialize variables
    model.Initialize()
    
    #run for epochs
    for epoch in epoch_bar(range(epoch_len)):
        time.sleep(.02)
        model.reset()
        
        #run game
        for t in range(time_max):
            
            #render game if at render epoch
            if epoch % render_epoch == 0: #and epoch != 0: 
                render(model.env,t,epoch,round(model.epsilon,3),title)
            
            #move bot and grab environment features
            model.move()
            model.step_env()
            
            #store environment features
            model.remember()
            
            #set current state to next state
            model.current_state = model.next_state
            
            #if game ended
            if model.done:                   
                break
            
        #optimize
        if len(model.memory) >= batch_size and epoch % render_epoch != 0:
            model.learn()
###########################################################################################################################
# %% Main

if __name__ == '__main__':
    y = run()



