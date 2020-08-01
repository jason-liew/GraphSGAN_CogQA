import tensorflow as tf
from tensorflow import keras
import pdb
from functional import  LinearWeightNorm,Linear
class Discriminator(tf.keras.Model):
    def __init__(self, input_dim = 28 ** 2, output_dim = 10):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.layers_hidden = [
            LinearWeightNorm(input_dim, 500),
            LinearWeightNorm(500, 500),
            LinearWeightNorm(500, 250),
            LinearWeightNorm(250, 250),
            LinearWeightNorm(250, 250)]
        self.final = LinearWeightNorm(250, output_dim)
    def call(self, x, feature = False, cuda = False, first = False):
#        pdb.set_trace()
     #   x = x.view(-1, self.input_dim)
        x = tf.reshape(x,[-1,self.input_dim])
        noise = tf.random.uniform(x.shape) * 0.05
        if cuda:
            noise = noise.cuda()
        x = x + tf.Variable(noise, trainable = False)
        if first:
            return self.layers[0](x)
        for i in range(len(self.layers_hidden)):
            m = self.layers[i]
            x_f = tf.nn.elu(m(x))
            noise = tf.random.uniform(x_f.shape) * 0.5 
            if cuda:
                noise = noise.cuda()
            x = (x_f + tf.Variable(noise, trainable = False))
        if feature:
            return x_f, self.final(x)
        return self.final(x)


class Generator(tf.keras.Model):
    def __init__(self, z_dim, output_dim = 28 ** 2):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.fc1 = Linear(z_dim, 500)
        self.bn1 = tf.keras.layers.BatchNormalization(trainable= False, epsilon=1e-6, momentum= 0.5)
        self.fc2 = Linear(500, 500)
        self.bn2 = tf.keras.layers.BatchNormalization(trainable= False, epsilon=1e-6, momentum= 0.5)
        self.fc3 = LinearWeightNorm(500, output_dim)
        self.bn1_b = tf.Variable(tf.zeros(500))
        self.bn2_b = tf.Variable(tf.zeros(500))

    def call(self, batch_size, cuda = False, seed = -1):
        x = tf.Variable(tf.random.uniform(shape=[batch_size, self.z_dim]), trainable = False)
        if cuda:
            x = x.cuda()
        x = tf.nn.elu(self.bn1(self.fc1(x)) + self.bn1_b)
        x = tf.nn.elu(self.bn2(self.fc2(x)) + self.bn2_b)
        x = tf.math.tanh(self.fc3(x))
        return x

