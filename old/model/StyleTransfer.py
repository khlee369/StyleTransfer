import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from six.moves import cPickle
from tensorflow.contrib.layers import xavier_initializer
import shutil

vgg_w = cPickle.load(open('./data/vgg19_weights.pkl', 'rb'))
vgg_w.keys()

class StyleTransfer:
    '''
    <Configuration info>
    ID : Model ID
    n_iter : Total # of iterations
    n_prt : Loss print cycle
    input_h : Image height
    input_w : Image width
    input_ch : Image channel (e.g. RGB)
    n_save : Model save cycle
    n_history : Train/Test loss save cycle
    LR : Learning rate
    
    <Configuration example>
    config = {
        'ID' : 'test_CNN',
        'n_iter' : 5000,
        'n_prt' : 100,
        'input_h' : 28,
        'input_w' : 28,
        'input_ch' : 1,
        'style' : style2,
        'content' : content2,
        's1' : 'conv2_1',
        's2' : 'conv3_1',
        'c1' : 'conv4_2',
        'c2' : 'conv5_1',
        'alpha' : 1,
        'beta' : 100,
        'n_save' : 1000,
        'n_history' : 50,
        'LR' : 0.0001
    }
    '''
    def __init__(self, config):
        self.ID = config['ID']
        self.n_iter = config['n_iter']
        self.n_prt = config['n_prt']
        self.input_h = config['input_h']
        self.input_w = config['input_w']
        self.input_ch = config['input_ch']
        self.n_save = config['n_save']
        self.n_history = config['n_history']
        self.LR = config['LR']
        
        self.style = config['style']
        self.content = config['content']
        
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.noise_weight = config['noise_weight']
        
        self.init_img = config['init_img']
        
        self.s1 = config['s1']
        self.s2 = config['s2']
        self.c1 = config['c1']
        self.c2 = config['c2']
        
        self.history = {
            'style' : [],
            'content' : []
        }
        self.checkpoint = 0
        self.path = './{}'.format(self.ID)
        try: 
            os.mkdir(self.path)
            os.mkdir('{0}/{1}'.format(self.path, 'checkpoint'))
        except FileExistsError:
            msg = input('[FileExistsError] Will you remove directory? [Y/N] ')
            if msg == 'Y': 
                shutil.rmtree(self.path)
                os.mkdir(self.path)
                os.mkdir('{0}/{1}'.format(self.path, 'checkpoint'))
            else: 
                print('Please choose another ID')
                assert 0
        
        # VGG Net Graph
        graph_vgg_transfer = tf.Graph()
        with graph_vgg_transfer.as_default() as g:
            with g.name_scope('vgg_transfer'):
                self.weights_transfer = {
                    'conv1_1' : tf.constant(vgg_w['conv1_1'][0]), # (3, 3, 3, 64)
                    'conv1_2' : tf.constant(vgg_w['conv1_2'][0]), # (3, 3, 64, 64)

                    'conv2_1' : tf.constant(vgg_w['conv2_1'][0]), # (3, 3, 64, 128)
                    'conv2_2' : tf.constant(vgg_w['conv2_2'][0]), # (3, 3, 128, 128)

                    'conv3_1' : tf.constant(vgg_w['conv3_1'][0]), # (3, 3, 128, 256)
                    'conv3_2' : tf.constant(vgg_w['conv3_2'][0]), # (3, 3, 256, 256)
                    'conv3_3' : tf.constant(vgg_w['conv3_3'][0]), # (3, 3, 256, 256)
                    'conv3_4' : tf.constant(vgg_w['conv3_4'][0]), # (3, 3, 256, 256)

                    'conv4_1' : tf.constant(vgg_w['conv4_1'][0]), # (3, 3, 256, 512)
                    'conv4_2' : tf.constant(vgg_w['conv4_2'][0]), # (3, 3, 512, 512)
                    'conv4_3' : tf.constant(vgg_w['conv4_3'][0]), # (3, 3, 512, 512)
                    'conv4_4' : tf.constant(vgg_w['conv4_4'][0]), # (3, 3, 512, 512)

                    'conv5_1' : tf.constant(vgg_w['conv5_1'][0]), # (3, 3, 512, 512)
                    'conv5_2' : tf.constant(vgg_w['conv5_2'][0]), # (3, 3, 512, 512)
                    'conv5_3' : tf.constant(vgg_w['conv5_3'][0]), # (3, 3, 512, 512)
                    'conv5_4' : tf.constant(vgg_w['conv5_4'][0]), # (3, 3, 512, 512)
                }
                self.biases_transfer = {
                    'conv1_1' : tf.constant(vgg_w['conv1_1'][1]), # (64,)
                    'conv1_2' : tf.constant(vgg_w['conv1_2'][1]), # (64,)

                    'conv2_1' : tf.constant(vgg_w['conv2_1'][1]), # (128,)
                    'conv2_2' : tf.constant(vgg_w['conv2_2'][1]), # (128,)

                    'conv3_1' : tf.constant(vgg_w['conv3_1'][1]), # (256,)
                    'conv3_2' : tf.constant(vgg_w['conv3_2'][1]), # (256,)
                    'conv3_3' : tf.constant(vgg_w['conv3_3'][1]), # (256,)
                    'conv3_4' : tf.constant(vgg_w['conv3_4'][1]), # (256,)

                    'conv4_1' : tf.constant(vgg_w['conv4_1'][1]), # (512,)
                    'conv4_2' : tf.constant(vgg_w['conv4_2'][1]), # (512,)
                    'conv4_3' : tf.constant(vgg_w['conv4_3'][1]), # (512,)
                    'conv4_4' : tf.constant(vgg_w['conv4_4'][1]), # (512,)

                    'conv5_1' : tf.constant(vgg_w['conv5_1'][1]), # (512,)
                    'conv5_2' : tf.constant(vgg_w['conv5_2'][1]), # (512,)
                    'conv5_3' : tf.constant(vgg_w['conv5_3'][1]), # (512,)
                    'conv5_4' : tf.constant(vgg_w['conv5_4'][1]), # (512,)
                }
                self.input = tf.placeholder(tf.float32, [1, self.input_h, self.input_w, self.input_ch])
                self.art = self.vgg_net(self.input, self.weights_transfer, self.biases_transfer)
                self.img = self.vgg_net(self.input, self.weights_transfer, self.biases_transfer)
                
                self.shape_s1 = self.art[self.s1].get_shape().as_list()
                self.shape_s2 = self.art[self.s2].get_shape().as_list()
              
                self.F_s1 = tf.reshape(self.art[self.s1], [np.prod(self.shape_s1[0:3]), self.shape_s1[3]])
                self.F_s2 = tf.reshape(self.art[self.s2], [np.prod(self.shape_s2[0:3]), self.shape_s2[3]])
                
                self.gram_F1 = tf.matmul(tf.transpose(self.F_s1), self.F_s1)
                self.gram_F2 = tf.matmul(tf.transpose(self.F_s2), self.F_s2)
                
                
                
             
        with tf.Session(graph=graph_vgg_transfer, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # init = tf.global_variables_initializer()
            # sess.run(init)
            # self.art = sess.run(self.art ,feed_dict={self.input : self.style})
            self.gram_F1 = sess.run(self.gram_F1, feed_dict={self.input : self.style})
            self.gram_F2 = sess.run(self.gram_F2, feed_dict={self.input : self.style})
            # self.gram_s2 = sess.run(self.gram_s2, feed_dict={self.input : self.style})
            # sess.run(init)
            self.img_eval = sess.run(self.img, feed_dict={self.input : self.content})
        
        tf.reset_default_graph()
        graph_vgg_x = tf.Graph()
        with graph_vgg_x.as_default() as g:
            with g.name_scope('vgg_x'):
                self.weights_x = {
                    'conv1_1' : tf.constant(vgg_w['conv1_1'][0]), # (3, 3, 3, 64)
                    'conv1_2' : tf.constant(vgg_w['conv1_2'][0]), # (3, 3, 64, 64)

                    'conv2_1' : tf.constant(vgg_w['conv2_1'][0]), # (3, 3, 64, 128)
                    'conv2_2' : tf.constant(vgg_w['conv2_2'][0]), # (3, 3, 128, 128)

                    'conv3_1' : tf.constant(vgg_w['conv3_1'][0]), # (3, 3, 128, 256)
                    'conv3_2' : tf.constant(vgg_w['conv3_2'][0]), # (3, 3, 256, 256)
                    'conv3_3' : tf.constant(vgg_w['conv3_3'][0]), # (3, 3, 256, 256)
                    'conv3_4' : tf.constant(vgg_w['conv3_4'][0]), # (3, 3, 256, 256)

                    'conv4_1' : tf.constant(vgg_w['conv4_1'][0]), # (3, 3, 256, 512)
                    'conv4_2' : tf.constant(vgg_w['conv4_2'][0]), # (3, 3, 512, 512)
                    'conv4_3' : tf.constant(vgg_w['conv4_3'][0]), # (3, 3, 512, 512)
                    'conv4_4' : tf.constant(vgg_w['conv4_4'][0]), # (3, 3, 512, 512)

                    'conv5_1' : tf.constant(vgg_w['conv5_1'][0]), # (3, 3, 512, 512)
                    'conv5_2' : tf.constant(vgg_w['conv5_2'][0]), # (3, 3, 512, 512)
                    'conv5_3' : tf.constant(vgg_w['conv5_3'][0]), # (3, 3, 512, 512)
                    'conv5_4' : tf.constant(vgg_w['conv5_4'][0]), # (3, 3, 512, 512)
                }
                self.biases_x = {
                    'conv1_1' : tf.constant(vgg_w['conv1_1'][1]), # (64,)
                    'conv1_2' : tf.constant(vgg_w['conv1_2'][1]), # (64,)

                    'conv2_1' : tf.constant(vgg_w['conv2_1'][1]), # (128,)
                    'conv2_2' : tf.constant(vgg_w['conv2_2'][1]), # (128,)

                    'conv3_1' : tf.constant(vgg_w['conv3_1'][1]), # (256,)
                    'conv3_2' : tf.constant(vgg_w['conv3_2'][1]), # (256,)
                    'conv3_3' : tf.constant(vgg_w['conv3_3'][1]), # (256,)
                    'conv3_4' : tf.constant(vgg_w['conv3_4'][1]), # (256,)

                    'conv4_1' : tf.constant(vgg_w['conv4_1'][1]), # (512,)
                    'conv4_2' : tf.constant(vgg_w['conv4_2'][1]), # (512,)
                    'conv4_3' : tf.constant(vgg_w['conv4_3'][1]), # (512,)
                    'conv4_4' : tf.constant(vgg_w['conv4_4'][1]), # (512,)

                    'conv5_1' : tf.constant(vgg_w['conv5_1'][1]), # (512,)
                    'conv5_2' : tf.constant(vgg_w['conv5_2'][1]), # (512,)
                    'conv5_3' : tf.constant(vgg_w['conv5_3'][1]), # (512,)
                    'conv5_4' : tf.constant(vgg_w['conv5_4'][1]), # (512,)
                }
                if self.init_img == None:
                    self.input_x = tf.Variable(tf.random_normal([1, self.input_h, self.input_w, self.input_ch], stddev=0.1))
                else:
                    self.input_x = tf.Variable(dtype=tf.float32, initial_value=self.init_img)
                self.vgg_x = self.vgg_net(self.input_x, self.weights_x, self.biases_x)
                
                self.art = self.vgg_net(self.input, self.weights_transfer, self.biases_transfer)
                self.img = self.vgg_net(self.input, self.weights_transfer, self.biases_transfer)
              
                self.A_s1 = tf.reshape(self.vgg_x[self.s1], [np.prod(self.shape_s1[0:3]), self.shape_s1[3]])
                self.A_s2 = tf.reshape(self.vgg_x[self.s2], [np.prod(self.shape_s2[0:3]), self.shape_s2[3]])
                
                self.gram_A1 = tf.matmul(tf.transpose(self.A_s1), self.A_s1)
                self.gram_A2 = tf.matmul(tf.transpose(self.A_s2), self.A_s2)
                
                self.gram_F1 = tf.constant(self.gram_F1)
                self.gram_F2 = tf.constant(self.gram_F2)
                
                self.img_c1 = tf.constant(self.img_eval[self.c1])
                self.img_c2 = tf.constant(self.img_eval[self.c2])
                
                # lagrangian loss to clip 0~255
                self.const_0 = -tf.minimum(tf.reduce_min(self.input_x), 0)
                self.const_255 = tf.maximum(tf.reduce_max(self.input_x), 255)
                                                
                self.loss_content1 = tf.reduce_sum(tf.square(tf.subtract(self.vgg_x[self.c1], self.img_c1)), axis=1)
                self.loss_content1 = tf.reduce_mean(self.loss_content1)
                self.loss_content2 = tf.reduce_sum(tf.square(tf.subtract(self.vgg_x[self.c2], self.img_c2)), axis=1)
                self.loss_content2 = tf.reduce_mean(self.loss_content2)
                self.loss_content = tf.add(self.loss_content1, self.loss_content2)
                self.loss_content = tf.add(self.loss_content, self.const_0)
                self.loss_content = tf.add(self.loss_content, self.const_255)
                
                self.loss_style1 = tf.reduce_sum(tf.square(tf.subtract(self.gram_A1, self.gram_F1)), axis=1)
                self.loss_style1 = tf.reduce_mean(self.loss_style1)
                self.loss_style2 = tf.reduce_sum(tf.square(tf.subtract(self.gram_A2, self.gram_F2)), axis=1)
                self.loss_style2 = tf.reduce_mean(self.loss_style2)
                self.loss_style = tf.add(self.loss_style1, self.loss_style2)
                self.loss_style = tf.add(self.loss_style, self.const_0)
                self.loss_style = tf.add(self.loss_style, self.const_255)
                
                # denoising
                self.noise_x = self.input_x[:, 1:, :, :] - self.input_x[:, :-1, :, :]
                self.noise_x = self.noise_weight * tf.nn.l2_loss(self.noise_x) / self._tensor_size(self.input_x[:, 1:, :, :])
                
                self.noise_y = self.input_x[:, :, 1:, :] - self.input_x[:, :, :-1, :]
                self.noise_y = self.noise_weight * tf.nn.l2_loss(self.noise_y) / self._tensor_size(self.input_x[:, :, 1:, :])
                
                self.loss_noise = self.noise_weight * 2 * (self.noise_x + self.noise_y)
                
                self.loss_content = tf.multiply(self.alpha, self.loss_content)
                self.loss_style = tf.multiply(self.beta, self.loss_style)
                
                self.loss = self.loss_content + self.loss_style + self.loss_noise
                
                self.optm_style = tf.train.AdamOptimizer(learning_rate=self.LR).minimize(self.loss_style)
                self.optm_content = tf.train.AdamOptimizer(learning_rate=self.LR).minimize(self.loss_content)
                self.optm = tf.train.AdamOptimizer(learning_rate=self.LR).minimize(self.loss)
                
                self.init = tf.global_variables_initializer()
                self.saver = tf.train.Saver(max_to_keep=None)
        
        self.sess = tf.Session(graph=graph_vgg_x, config=tf.ConfigProto(allow_soft_placement=True))
        self.sess.run(self.init)
                             

    def vgg_net(self, x, weights, biases):
        # conv1
        conv1_1 = tf.add(tf.nn.conv2d(x, weights['conv1_1'], strides=[1, 1, 1, 1], padding='SAME'), biases['conv1_1'])
        conv1_1 = tf.nn.relu(conv1_1, name='conv1_1')
        conv1_2 = tf.add(tf.nn.conv2d(conv1_1, weights['conv1_2'], strides=[1, 1, 1, 1], padding='SAME'), biases['conv1_2'])
        conv1_2 = tf.nn.relu(conv1_2, name='conv1_2')
        avgp1 = tf.nn.avg_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # conv2
        conv2_1 = tf.add(tf.nn.conv2d(avgp1, weights['conv2_1'], strides=[1, 1, 1, 1], padding='SAME'), biases['conv2_1'])
        conv2_1 = tf.nn.relu(conv2_1, name='conv2_1')
        conv2_2 = tf.add(tf.nn.conv2d(conv2_1, weights['conv2_2'], strides=[1, 1, 1, 1], padding='SAME'), biases['conv2_2'])
        conv2_2 = tf.nn.relu(conv2_2, name='conv2_2')
        avgp2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # conv3
        conv3_1 = tf.add(tf.nn.conv2d(avgp2, weights['conv3_1'], strides=[1, 1, 1, 1], padding='SAME'), biases['conv3_1'])
        conv3_1 = tf.nn.relu(conv3_1, name='conv3_1')
        conv3_2 = tf.add(tf.nn.conv2d(conv3_1, weights['conv3_2'], strides=[1, 1, 1, 1], padding='SAME'), biases['conv3_2'])
        conv3_2 = tf.nn.relu(conv3_2, name='conv3_2')
        conv3_3 = tf.add(tf.nn.conv2d(conv3_2, weights['conv3_3'], strides=[1, 1, 1, 1], padding='SAME'), biases['conv3_3'])
        conv3_3 = tf.nn.relu(conv3_3, name='conv3_3')
        conv3_4 = tf.add(tf.nn.conv2d(conv3_3, weights['conv3_4'], strides=[1, 1, 1, 1], padding='SAME'), biases['conv3_4'])
        conv3_4 = tf.nn.relu(conv3_4, name='conv3_4')
        avgp3 = tf.nn.max_pool(conv3_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # conv4
        conv4_1 = tf.add(tf.nn.conv2d(avgp3, weights['conv4_1'], strides=[1, 1, 1, 1], padding='SAME'), biases['conv4_1'])
        conv4_1 = tf.nn.relu(conv4_1, name='conv4_1')
        conv4_2 = tf.add(tf.nn.conv2d(conv4_1, weights['conv4_2'], strides=[1, 1, 1, 1], padding='SAME'), biases['conv4_2'])
        conv4_2 = tf.nn.relu(conv4_2, name='conv4_2')
        conv4_3 = tf.add(tf.nn.conv2d(conv4_2, weights['conv4_3'], strides=[1, 1, 1, 1], padding='SAME'), biases['conv4_3'])
        conv4_3 = tf.nn.relu(conv4_3, name='conv4_3')
        conv4_4 = tf.add(tf.nn.conv2d(conv4_3, weights['conv4_4'], strides=[1, 1, 1, 1], padding='SAME'), biases['conv4_4'])
        conv4_4 = tf.nn.relu(conv4_4, name='conv4_4')
        avgp4 = tf.nn.max_pool(conv4_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # conv5
        conv5_1 = tf.add(tf.nn.conv2d(avgp4, weights['conv5_1'], strides=[1, 1, 1, 1], padding='SAME'), biases['conv5_1'])
        conv5_1 = tf.nn.relu(conv5_1, name='conv5_1')
        conv5_2 = tf.add(tf.nn.conv2d(conv5_1, weights['conv5_2'], strides=[1, 1, 1, 1], padding='SAME'), biases['conv5_2'])
        conv5_2 = tf.nn.relu(conv5_2, name='conv5_2')
        conv5_3 = tf.add(tf.nn.conv2d(conv5_2, weights['conv5_3'], strides=[1, 1, 1, 1], padding='SAME'), biases['conv5_3'])
        conv5_3 = tf.nn.relu(conv5_3, name='conv5_3')
        conv5_4 = tf.add(tf.nn.conv2d(conv5_3, weights['conv5_4'], strides=[1, 1, 1, 1], padding='SAME'), biases['conv5_4'])
        conv5_4 = tf.nn.relu(conv5_4, name='conv5_4')
        avgp5 = tf.nn.max_pool(conv5_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        return {
            'x' : x,
            'conv1_1' : conv1_1,
            'conv1_2' : conv1_2,

            'conv2_1' : conv2_1,
            'conv2_2' : conv2_1,
            'conv2_3' : conv2_1,
            'conv2_4' : conv2_1,

            'conv3_1' : conv3_1,
            'conv3_2' : conv3_2,
            'conv3_3' : conv3_3,
            'conv3_4' : conv3_4,

            'conv4_1' : conv4_1,
            'conv4_2' : conv4_2,
            'conv4_3' : conv4_3,
            'conv4_4' : conv4_4,

            'conv5_1' : conv5_1,
            'conv5_2' : conv5_2,
            'conv5_3' : conv5_3,
            'conv5_4' : conv5_4,
        }
    
    def fit(self, verbose=True):
        for step in range(1, self.n_iter+1):
            # self.sess.run(self.optm_content)
            # self.sess.run(self.optm_style)
            self.sess.run(self.optm)

            if step % self.n_prt == 0:
                # self.sess.run(self.optm)
                lc = self.sess.run(self.loss_content)
                ls = self.sess.run(self.loss_style)
                ln = self.sess.run(self.loss_noise)
                print('\nContent loss ({0}/{1}) : {2}'.format(step, self.n_iter, lc))
                print('Style loss ({0}/{1}) : {2}'.format(step, self.n_iter, ls))
                print('Noise loss ({0}/{1}) : {2}\n'.format(step, self.n_iter, ln))
                
                ustyle = self.style.astype(np.uint8)
                ucontent = self.content.astype(np.uint8)
                
                my_img = self.sess.run(self.input_x)
                my_img = my_img.astype(np.uint8)
                my_img = np.clip(my_img[0], 0, 255).astype(np.uint8)
                
                fig, axs = plt.subplots(1,3, figsize = (15, 5))
                axs[0].imshow(ustyle[0])
                axs[0].axis('off')
                axs[1].imshow(ucontent[0])
                axs[1].axis('off')
                axs[2].imshow(my_img)
                axs[2].axis('off')
                fig.savefig("{0}/{1}{2}.png".format(self.path, "my_img", step))
                if verbose:
                    plt.show()
                
            if step % self.n_save == 0:
                self.checkpoint += self.n_save
                self.save('{0}/{1}/{2}_{3}'.format(self.path, 'checkpoint', self.ID, self.checkpoint))

            if step % self.n_history == 0:
                self.history['content'].append(self.loss_content)
                self.history['style'].append(self.loss_style)
                
    def save(self, path):
        self.saver.save(self.sess, path)
    
    def load(self, path):
        self.saver.restore(self.sess, path)
        checkpoint = path.split('_')[-1]
        self.checkpoint = int(checkpoint)
        print('Model loaded from file : {}'.format(path))
        
    def _tensor_size(self, tensor):
        from operator import mul
        from functools import reduce
        return reduce(mul, (d.value for d in tensor.get_shape()), 1)
    
#     def gram_matrix(self, input_tensor):
#         flatten_shape = np.prod(input_tensor.get_shape().as_list()[1:])
#         flatten = tf.reshape(input_tensor, [-1, flatten_shape])
#         flatten_T = tf.transpose(flatten)
#         return tf.matmul(flatten_T, flatten)
