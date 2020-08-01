# -*- coding:utf-8 -*-
from __future__ import print_function 
import numpy as np
from functional import log_sum_exp, pull_away_term
import sys
import tensorflow as tf
import argparse
from Nets import Generator, Discriminator
import os
import random
from FeatureGraphDataset import FeatureGraphDataset
import pickle as pkl
class GraphSGAN(object):
    def __init__(self, G, D, dataset, args):
        self.G = G
        self.D = D
        # self.embedding_layer = nn.Embedding(dataset.n, dataset.d)
        self.embedding_layer = tf.constant(dataset.embbedings) 
        self.dataset = dataset
        self.Doptim = tf.keras.optimizers.Adam(learning_rate=args.lr, beta_1 = args.momentum, beta_2 = 0.999)
        self.Goptim = tf.keras.optimizers.Adam(learning_rate=args.lr, beta_1 = args.momentum, beta_2 = 0.999)
        self.args = args

    def trainD(self, idf_label, y, idf_unlabel):
        x_label, x_unlabel, y = self.make_input(*idf_label), self.make_input(*idf_unlabel), y
        output_label, (mom_un, output_unlabel), output_fake = self.D(x_label, cuda=self.args.cuda), self.D(x_unlabel, cuda=self.args.cuda, feature = True), self.D(tf.reshape(self.G(x_unlabel.shape[0], cuda = self.args.cuda),x_unlabel.shape), cuda=self.args.cuda)
        logz_label, logz_unlabel, logz_fake = log_sum_exp(output_label), log_sum_exp(output_unlabel), log_sum_exp(output_fake) # log ∑e^x_i
        prob_label = tf.gather(output_label, 1, tf.expand_dims(y,1)) # log e^x_label = x_label 
        loss_supervised = -tf.reduce_mean(prob_label) + tf.reduce_mean(logz_label)
        loss_unsupervised = 0.5 * (-tf.reduce_mean(logz_unlabel) + tf.reduce_mean(tf.math.softplus(logz_unlabel))  + # real_data: log Z/(1+Z)
                           tf.reduce_mean(tf.math.softplus(logz_fake)) ) # fake_data: log 1/(1+Z)
        entropy = -tf.reduce_mean(tf.nn.softmax(output_unlabel, axis = 1) * tf.nn.log_softmax(output_unlabel, axis = 1))
        pt = pull_away_term(mom_un)
        loss = loss_supervised + self.args.unlabel_weight * loss_unsupervised + entropy + pt
        ac = tf.reduce_mean(tf.cast(tf.argmax(input=output_label,axis = 1) == y,dtype=tf.float32))
        return loss_supervised.numpy(), loss_unsupervised.numpy() , loss , ac
    
    def trainG(self, idf_unlabel):
        x_unlabel = self.make_input(*idf_unlabel)
        fake = tf.reshape(self.G(x_unlabel.shape[0], cuda = self.args.cuda),x_unlabel.shape)
        mom_gen, output_fake = self.D(fake, feature=True, cuda=self.args.cuda)
        mom_unlabel, output_unlabel = self.D(x_unlabel, feature=True, cuda=self.args.cuda)
        loss_pt = pull_away_term(mom_gen)
        mom_gen = tf.raw_ops.Mean(input=mom_gen, axis = 0)
        mom_unlabel = tf.raw_ops.Mean(input=mom_unlabel, axis = 0) 
        loss_fm = tf.reduce_mean(tf.abs(mom_gen - mom_unlabel))
        loss = loss_fm + loss_pt 

        return loss

    def make_input(self, ids, feature, volatile = False):
        '''Concatenate feature and embeddings

        Args:
            feature: Size=>[batch_size, dataset.k], Type=>FloatTensor
            ids: Size=>[batch_size], Type=>LongTensor
        '''
       # embedding = self.embedding_layer(Variable(ids, volatile = volatile)).detach() # detach temporarily
        embedding = tf.nn.embedding_lookup(self.embedding_layer,ids)
        return tf.concat([feature, embedding], axis = 1)
    def train(self):
        gn = 0
        NUM_BATCH = 100
        for epoch in range(self.args.epochs
        ):
          #  self.G.train()
          #  self.D.train()
            self.D.turn = epoch
            loss_supervised = loss_unsupervised = loss_gen = accuracy = 0.
            for batch_num in range(NUM_BATCH):
                # extract batch from dataset
                idf_unlabel1 = self.dataset.unlabel_batch(self.args.batch_size)
                idf_unlabel2 = self.dataset.unlabel_batch(self.args.batch_size)
                id0, xf, y = self.dataset.label_batch(self.args.batch_size)
              
                # train D
                with tf.GradientTape() as tape:
                    ls, lu, ld, ac = self.trainD((id0, xf), y, idf_unlabel1)
                grads = tape.gradient(ld, self.D.trainable_weights)
                self.Doptim.apply_gradients(zip(grads, self.D.trainable_weights))
                
                loss_supervised += ls
                loss_unsupervised += lu
                accuracy += ac

                # train G on unlabeled data
                with tf.GradientTape() as tape:
                    lg = self.trainG(idf_unlabel2)
                grads = tape.gradient(lg, self.G.trainable_weights)
                self.Goptim.apply_gradients(zip(grads, self.G.trainable_weights))
                
                loss_gen += lg
            
            # calculate average loss at the end of an epoch
            batch_num += 1
            loss_supervised /= batch_num
            loss_unsupervised /= batch_num
            loss_gen /= batch_num
            accuracy /= batch_num
            print("Iteration %d, loss_supervised = %.4f, loss_unsupervised = %.4f, loss_gen = %.4f train acc = %.4f" % (epoch, loss_supervised, loss_unsupervised, loss_gen, accuracy))
            sys.stdout.flush()
   
    def predict(self, x):
        '''predict label in volatile mode

        Args:
            x: Size=>[batch_size, self.dataset.k + self.dataset.d], Type=>Variable(FloatTensor), volatile
        '''
        return tf.max(self.D(x, cuda=self.args.cuda), 1)[1].data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch GraphS GAN')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                        help='learning rate (default: 0.003)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='CUDA training')
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--eval-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before evaling training status')
    parser.add_argument('--unlabel-weight', type=float, default=0.5, metavar='N',
                        help='scale factor between labeled and unlabeled data')
    parser.add_argument('--logdir', type=str, default='./logfile', metavar='LOG_PATH', help='logfile path, tensorboard format')
    parser.add_argument('--savedir', type=str, default='./models', metavar='SAVE_PATH', help = 'saving path, pickle format')
    args = parser.parse_args()
    args.cuda = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    # but we load the example of cora
    with open('cora.dataset', 'rb') as fdata:
        dataset = pkl.load(fdata,fix_imports=True, encoding="latin1")
    gan = GraphSGAN(Generator(200, dataset.k + dataset.d), Discriminator(dataset.k + dataset.d, dataset.m), dataset, args)
    gan.train() 
    
