# -*- coding: utf-8 -*-
"""
Using SDNE
"""
# import matplotlib.pyplot as plt

# from gem.utils import graph_util, plot_util
from gem.utils import graph_util
# from gem.evaluation import visualize_embedding as viz
from gem.evaluation import evaluate_graph_reconstruction as gr
from time import time

# from gem.embedding.gf import GraphFactorization
# from gem.embedding.hope import HOPE
# from gem.embedding.lap import LaplacianEigenmaps
# from gem.embedding.lle import LocallyLinearEmbedding
# from gem.embedding.node2vec import node2vec
from gem.embedding.sdne import SDNE

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

__author__ = 'Min'

if __name__ == "__main__":

    parser = ArgumentParser(
        description='SDNE takes embedding dimension (d), seen edge reconstruction weight (beta), first order proximity weight (alpha), lasso regularization coefficient (nu1), ridge regreesion coefficient (nu2), number of hidden layers (K), size of each layer (n_units), number of iterations (n_ite), learning rate (xeta), size of batch (n_batch), location of modelfile and weightfile save (modelfile and weightfile) as inputs',
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-data',  type=str, help="The path of dataset.", required=True)
    parser.add_argument("-d", type=int, help="dimension of the embedding.", default=2)
    parser.add_argument("-beta", type=float,
                        help="penalty parameter in matrix B of 2nd order objective.", default=5)
    parser.add_argument("-alpha", type=float,
                        help="weighing hyperparameter for 1st order objective.", default=1e-5)
    parser.add_argument("-nu1", type=float, help="L1-reg hyperparameter.", default=1e-6)
    parser.add_argument("-nu2", type=float, help="L2-reg hyperparameter.", default=1e-6)
    parser.add_argument(
        "-k", type=int, help="number of hidden layers in encoder/decoder.", default=3)
    parser.add_argument(
        "-nunits", nargs='+', type=int, help="vector of length K-1 containing #units in hidden layers of encoder/decoder, not including the units in the embedding layer.", default=[50, 15])
    parser.add_argument("-niter", type=float,
                        help="number of sgd iterations for first embedding (const).", default=50)
    parser.add_argument("-xeta", type=float, help="sgd step size parameter.", default=0.01)
    parser.add_argument("-nbatch", type=float, help="minibatch size for SGD.", default=500)

    args = parser.parse_args()
    # File that contains the edges. Format: source target
    # Optionally, you can add weights as third column: source target weight
    # edge_f = 'karate.edgelist'
    edge_f = args.data
    # Specify whether the edges are directed
    isDirected = True

    # Load graph
    G = graph_util.loadGraphFromEdgeListTxt(edge_f, directed=isDirected)
    G = G.to_directed()

    models = []
    # You can comment out the methods you don't want to run
    # GF takes embedding dimension (d), maximum iterations (max_iter), learning rate (eta), regularization coefficient (regu) as inputs
    # models.append(GraphFactorization(d=2, max_iter=100000, eta=1*10**-4, regu=1.0))
    # HOPE takes embedding dimension (d) and decay factor (beta) as inputs
    # models.append(HOPE(d=4, beta=0.01))
    # LE takes embedding dimension (d) as input
    # models.append(LaplacianEigenmaps(d=2))
    # LLE takes embedding dimension (d) as input
    # models.append(LocallyLinearEmbedding(d=2))
    # node2vec takes embedding dimension (d),  maximum iterations (max_iter), random walk length (walk_len), number of random walks (num_walks), context size (con_size), return weight (ret_p), inout weight (inout_p) as inputs
    # models.append(node2vec(d=2, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=1, inout_p=1))
    # SDNE takes embedding dimension (d), seen edge reconstruction weight (beta), first order proximity weight (alpha), lasso regularization coefficient (nu1), ridge regreesion coefficient (nu2), number of hidden layers (K), size of each layer (n_units), number of iterations (n_ite), learning rate (xeta), size of batch (n_batch), location of modelfile and weightfile save (modelfile and weightfile) as inputs
    ''' Initialize the SDNE class

        Args:
            d: dimension of the embedding
            beta: penalty parameter in matrix B of 2nd order objective
            alpha: weighing hyperparameter for 1st order objective
            nu1: L1-reg hyperparameter
            nu2: L2-reg hyperparameter
            K: number of hidden layers in encoder/decoder
            n_units: vector of length K-1 containing #units in hidden layers
                     of encoder/decoder, not including the units in the
                     embedding layer
            rho: bounding ratio for number of units in consecutive layers (< 1)
            n_iter: number of sgd iterations for first embedding (const)
            xeta: sgd step size parameter
            n_batch: minibatch size for SGD
            modelfile: Files containing previous encoder and decoder models
            weightfile: Files containing previous encoder and decoder weights
    '''
    models.append(
        SDNE(d=args.d, beta=args.beta, alpha=args.alpha, nu1=args.nu1, nu2=args.nu2,
             K=args.k, n_units=args.nunits, n_iter=args.niter, xeta=args.xeta, n_batch=args.nbatch,
             modelfile=['enc_model.json', 'dec_model.json'],
             weightfile=['enc_weights.hdf5', 'dec_weights.hdf5']))

    for embedding in models:
        print('Num nodes: %d, num edges: %d' % (G.number_of_nodes(),
                                                G.number_of_edges()))
        t1 = time()
        # Learn embedding - accepts a networkx graph or file with edge list
        Y, t = embedding.learn_embedding(
            graph=G, edge_f=None, is_weighted=True, no_python=True)
        print(embedding._method_name +
              ':\n\tTraining time: %f' % (time() - t1))
        # Evaluate on graph reconstruction
        MAP, prec_curv, err, err_baseline = gr.evaluateStaticGraphReconstruction(
            G, embedding, Y, None)
        # ---------------------------------------------------------------------------------
        print(("\tMAP: {} \t precision curve: {}\n\n\n\n" + '-' * 100).format(
            MAP, prec_curv[:5]))
        # ---------------------------------------------------------------------------------
        # Visualize
        # viz.plot_embedding2D(embedding.get_embedding(), di_graph=G, node_colors=None)
        # plt.show()
