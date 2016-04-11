# VI network using THEANO, takes batches of state input
from NNobj import *
from theano_utils import *
import myparameters as prm
import wiki
import scipy.sparse as SS
import qp
import h5py


class vin(NNobj):
    "Class for a neural network that does k iterations of value iteration"
    def __init__(self, model="valIterWiki", N = 6072, D = 279, emb_dim = 300,
                 dropout=False, devtype="cpu",
                 grad_check=False, reg=0, k=10, A = 400,
                 maxhops=4, batchsize=100):
        self.N = N                            # Number of pages
        self.D = D + 1                        # Number of max outgoing links per page + 1 (including self)
        self.emb_dim = emb_dim                # Dimension of word embedding
        self.model = model
        self.reg = reg                        # regularization (currently not implemented)
        self.k = k                            # number of VI iterations
        self.A = A                            # Actions to take
        self.batchsize = batchsize            # batch size for training
        self.maxhops = 1    # for convenience, we assume statebatchsize = 1
        #self.maxhops = maxhops+1              # number of state inputs for every query,
                                              #     here simply the number of hops per query + 1 (including stop)
        np.random.seed(0)
        print(model)
        theano.config.blas.ldflags = "-L/usr/local/lib -lopenblas"

        # initial self.wk, self.idx, self.rev_idx, self.edges, self.page_emb, self.q
        print 'load graph data ...'
        self.load_graph()

        # X input : l=2 stacked images: obstacle map and reward function prior
        #self.X = T.ftensor4(name="X")
        # S1,S2 input : state position (vertical and horizontal position)
        #self.S1 = T.bmatrix("S1")  # state first dimension * statebatchsize
        #self.S2 = T.bmatrix("S2")  # state second dimension * statebatchsize
        #self.y = T.bvector("y")    # output action * statebatchsize

        # query input: input embedding vector of query
        self.Q_in = T.fmatrix('Q_in')  # batchsize * emb_dim
        # S input: state position
        self.S_in = T.imatrix("S_in")  # batchsize * maxhops
        # output action
        self.y = T.ivector("y")        # batchsize * maxhops

        #l = 2   # channels in input layer
        #l_h = 150  # channels in initial hidden layer
        #l_q = 10   # channels in q layer (~actions)

        print 'building model ...'

        self.vin_net = VinBlockWiki(Q_in=self.Q_in, S_in=self.S_in,
                                    N = self.N, D = self.D, emb_dim = self.emb_dim,
                                    page_emb = self.page_emb, edges = self.edges,
                                    batchsize=self.batchsize, maxhops=self.maxhops, 
                                    k=self.k, A=self.A)
        self.p_of_y = self.vin_net.output
        self.params = self.vin_net.params
        # Total 1910 parameters ?????

        self.cost = -T.mean(T.log(self.p_of_y)[T.arange(self.y.shape[0]),
                                               self.y.flatten()], dtype=theano.config.floatX)
        self.y_pred = T.argmax(self.p_of_y, axis=1)
        self.err = T.mean(T.neq(self.y_pred, self.y.flatten()), dtype=theano.config.floatX)

        self.computeloss = theano.function(inputs=[self.Q_in, self.S_in, self.y],
                                           outputs=[self.err, self.cost])
        self.y_out = theano.function(inputs=[self.Q_in, self.S_in], outputs=[self.y_pred])

    def load_graph(self):  
        """
        :type page_emb: np.fmatrix
        :param page_emb: input data, embedding for each page, of shape [emb_dim, N_pages]

        :type edges: scipy.sparse.csc_matrix
        :param edges: adjacency matrix, of shape [N_pages, N_pages * D], column sparse
        """
        f = h5py.File(prm.pages_emb_path, 'r', driver='core')
        self.page_emb = np.zeros((self.emb_dim, self.N), dtype=theano.config.floatX)
        for i in range(self.N):
            self.page_emb[:, i] = f['emb'][i]
        f.close()

        self.wk = wiki.Wiki(prm.pages_path)
        self.idx = self.wk.get_titles_pos()

        ptr = 0
        row_idx = []
        col_idx = []
        self.rev_idx = []
        for i in range(self.N):
            urls = self.wk.get_article_links(i)
            if (not i in urls):
                urls.append(i)
            rev = {}
            for x, y in enumerate(urls):
                rev[y] = x
            self.rev_idx.append(rev)
            n = len(urls)
            col_idx += range(ptr, ptr + n)
            row_idx += urls
            ptr += self.D
        n = len(col_idx)
        dat_arr = np.ones(n, dtype=theano.config.floatX)     
        self.edges = SS.csc_matrix((dat_arr, (row_idx, col_idx)), shape=(self.N, self.N * self.D), dtype=theano.config.floatX)

        self.q = qp.QP(prm.curr_query_path)


    def run_training(self, stepsize=0.01, epochs=10, output='None',
                     grad_check=True,
                     profile=False):

        print 'Training Start ...'

        batch_size = self.batchsize

        train_queries = self.q.get_train_queries()
        valid_queries = self.q.get_valid_queries()
        test_queries = self.q.get_test_queries()
        train_paths = self.q.get_train_paths()
        valid_paths = self.q.get_valid_paths()
        test_paths = self.q.get_test_paths()

        train_entry = self.q.get_tuples(train_paths, self.rev_idx)
        valid_entry = self.q.get_tuples(valid_paths, self.rev_idx)
        test_entry = self.q.get_tuples(test_paths, self.rev_idx)

        train_n = len(train_entry)
        valid_n = len(valid_entry)
        test_n = len(test_entry)

        self.updates = rmsprop_updates_T(self.cost, self.params, stepsize=stepsize)
        self.train = theano.function(inputs=[self.Q_in, self.S_in, self.y], outputs=[], updates=self.updates)

        Q_dat = np.zeros((batch_size,self.emb_dim), dtype = theano.config.floatX)
        S_dat = np.zeros((batch_size, 1), dtype = np.int32) # for convinence, maxhops = 1
        y_dat = np.zeros(batch_size*1, dtype = np.int32) # for convinence, maxhops = 1
        
        print fmt_row(10, ["Epoch", "Train NLL", "Train Err", "Test NLL", "Test Err", "Epoch Time"])
        for i_epoch in xrange(int(epochs)):
            tstart = time.time()
            # shuffle training index
            inds = np.random.permutation(train_n)
            train_n_curr = train_n
            if (prm.select_subset_data > 0):
                train_n_curr = train_n / prm.select_subset_data
            # do training
            for start in xrange(0, train_n_curr, batch_size):
                end = start+batch_size
                if end <= train_n_curr:
                    # prepare training data
                    for i in xrange(start, end):
                        q_i, s_i, y_i = train_entry[inds[i]]
                        Q_dat[i-start, :] = train_queries[q_i, :]
                        S_dat[i-start, 0] = s_i
                        y_dat[i-start] = y_i
                    
                    self.train(Q_dat, S_dat, y_dat)
                    
            elapsed = time.time() - tstart
            # compute losses
            trainerr = 0.
            trainloss = 0.
            testerr = 0.
            testloss = 0.
            num = 0
            for start in xrange(0, test_n, batch_size):
                end = start+batch_size
                if end <= test_n:  # assert(text_n <= train_n)
                    num += 1
                    # prepare training data
                    for i in xrange(start, end):
                        q_i, s_i, y_i = train_entry[inds[i]]
                        Q_dat[i-start, :] = train_queries[q_i, :]
                        S_dat[i-start, 0] = s_i
                        y_dat[i-start] = y_i
                    trainerr_, trainloss_ = self.computeloss(Q_dat, S_dat, y_dat)
                    # prepare testing data
                    for i in xrange(start, end):
                        q_i, s_i, y_i = test_entry[i]
                        Q_dat[i-start, :] = test_queries[q_i, :]
                        S_dat[i-start, 0] = s_i
                        y_dat[i-start] = y_i
                    testerr_, testloss_ = self.computeloss(Q_dat, S_dat, y_dat)
                    trainerr += trainerr_
                    trainloss += trainloss_
                    testerr += testerr_
                    testloss += testloss_
            print fmt_row(10, [i_epoch, trainloss/num, trainerr/num, testloss/num, testerr/num, elapsed])

    # TODO
    def predict(self, input):
        # NN output for a single input, read from file
        matlab_data = sio.loadmat(input)
        im_data = matlab_data["im_data"]
        im_data = (im_data - 1)/255  # obstacles = 1, free zone = 0
        state_data = matlab_data["state_xy_data"]
        value_data = matlab_data["value_data"]
        xim_test = im_data.astype(theano.config.floatX)
        xim_test = xim_test.reshape(-1, 1, self.im_size[0], self.im_size[1])
        xval_test = value_data.astype(theano.config.floatX)
        xval_test = xval_test.reshape(-1, 1, self.im_size[0], self.im_size[1])
        x_test = np.append(xim_test, xval_test, axis=1)
        s_test = state_data.astype('int8')
        s1_test = s_test[:, 0].reshape([1, 1])
        s2_test = s_test[:, 1].reshape([1, 1])
        out = self.y_out(x_test, s1_test, s2_test)
        return out[0][0]

    def load_weights(self, infile="weight_dump.pk"):
        dump = pickle.load(open(infile, 'r'))
        [n.set_value(p) for n, p in zip(self.params, dump)]

    def save_weights(self, outfile="weight_dump.pk"):
        pickle.dump([n.get_value() for n in self.params], open(outfile, 'w'))

class VinBlockWiki(object):
    """VIN block for wiki-school dataset"""
    def __init__(self, Q_in, S_in, N, D, emb_dim,
                 page_emb, edges,
                 batchsize, maxhops,
                 k, A):
        """
        Allocate a VIN block with shared variable internal parameters.

        :type Q_in: theano.tensor.fmatrix
        :param Q_in: symbolic input query embeding, of shape [batchsize, emb_dim]

        :type S_in: theano.tensor.imatrix
        :param S_in: symbolic input batches of positions, of shape [batchsize, maxhops]

        :type page_emb: np.fmatrix
        :param page_emb: input data, embedding for each page, of shape [emb_dim, N_pages]

        :type edges: scipy.sparse.csc_matrix
        :param edges: adjacency matrix, of shape [N_pages, N_pages * D], column sparse

        :type N: int32
        :param N: number of pages

        :type D: int32
        :param D: max degree for each page

        :type emb_dim: int32
        :param emb_dim: dimension of word embedding
        
        :type batchsize: int32
        :param batchsize: batch size

        :type maxhops: int32
        :param maxhops: number of state inputs for each sample

        :type k: int32
        :param k: number of VI iterations (actually, real number of iterations is k+1)

        """

        self.params = []
        if (not prm.query_map_linear):
            print 'Now we only support linear transformation over query embedding'
        # Q_in * W
        if (prm.query_weight_diag):
            self.W = init_weights_T(1, emb_dim);
            self.params.append(self.W)
            self.W = T.extra_ops.repeat(self.W, batchsize, axis = 0)
            self.q = Q_in * self.W
        else:
            self.W = init_weights_T(emb_dim, emb_dim)
            self.params.append(self.W)
            self.q = T.dot(Q_in, self.W)
        # add bias
        self.q_bias = init_weights_T(emb_dim)
        self.params.append(self.q_bias)
        self.q = self.q + self.q_bias.dimshuffle('x', 0) # batch * emb_dim
        # non-linear transformation
        if (prm.query_tanh):
            self.q = T.tanh(self.q)

        
        # create reword: R: [batchsize, N_pages]
        #   q: [batchsize, emb_dim]
        #   page_emb: [emb_dim, N_pages]
        self.R = T.nnet.softmax(T.dot(self.q, page_emb))
        # initial value
        self.V = self.R

        # transition param
        if (prm.diagonal_action_mat): # use simple diagonal matrix for transition
            A = D
            #self.w = T.eye(D, D, dtype = theano.config.floatX)
        else:
            self.w = init_weights_T(1, D, A)
            self.params.append(self.w)
        
        self.w_local = init_weights_T(1, 1, A)
        self.params.append(self.w_local)

        #self.full_w = self.w.dimshuffle('x', 0, 1);
        if (not prm.diagonal_action_mat):
            self.full_w = T.extra_ops.repeat(self.w, batchsize, axis = 0) # batchsize * D * A

        #self.full_w_local = self.w_local.dimshuffle('x', 'x', 0);
        self.full_w_local = T.extra_ops.repeat(self.w_local, batchsize, axis = 0) # batchsize * 1 * A

        self.R_full = self.R.dimshuffle(0, 1, 'x') # batchsize * N * 1
        self.add_R = T.batched_dot(self.R_full, self.full_w_local) # batchsize * N * A
        
        self.dense_q = T.zeros(batchsize * N * D, dtype = theano.config.floatX)
        # Value Iteration
        for i in range(k - 1):
            self.tq = TS.basic.structured_dot(self.V, edges) # batchsize * (N * D)
            self.nq = T.set_subtensor(self.dense_q[:], self.tq.flatten())
            self.q = T.reshape(self.nq, (batchsize, N, D)) # batchsize * N * D
            if (not prm.diagonal_action_mat):
                self.q = T.batched_dot(self.q, self.full_w) # batchsize * N * A
            self.q = self.q + self.add_R
            self.V = T.max(self.q, axis=2, keepdims=False) # batchsize * N

        # Do last Conv Step
        self.tq = TS.basic.structured_dot(self.V, edges) # batchsize * (N * D)
        self.nq = T.set_subtensor(self.dense_q[:], self.tq.flatten())
        self.q = T.reshape(self.nq, (batchsize, N, D)) # batchsize * N * D
        if (not prm.diagonal_action_mat):
            self.q = T.batched_dot(self.q, self.full_w) # batchsize * N * A
        self.q = self.q + self.add_R
	

        # fetch values for each state in S_in
        # (B * H) * A
        self.q_out = self.q[T.extra_ops.repeat(T.arange(batchsize), maxhops), S_in.flatten(), :]

        if (prm.diagonal_action_mat): # simple diagonal transition matrix
            if (prm.final_scale):
                self.w_o = init_weights_T(1, D)
                self.params.append(self.w_o)
                self.full_w_o = T.extra_ops.repeat(self.w_o, batchsize * maxhops, axis = 0) # (B * H) * A
                self.q_out = self.q_out * self.full_w_o
            self.output = T.nnet.softmax(self.q_out) 
        else:
            # softmax output weights
            self.w_o = init_weights_T(A, D)
            self.params.append(self.w_o)
            # (Batch * Hops) * D
            self.output = T.nnet.softmax(T.dot(self.q_out, self.w_o)) 

