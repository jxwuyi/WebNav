# VI network using THEANO, takes batches of state input
from NNobj import *
from theano_utils import *
import myparameters as prm
import wiki
import scipy.sparse as SS
import qp
import h5py


class vin_web(NNobj):
    "Class for a neural network that does k iterations of value iteration"
    def __init__(self, model="WikiPreCompute", N = 6072, D = 279, emb_dim = 500,
                 dropout=False, devtype="cpu",
                 grad_check=False, reg=0, k=10, seed = 0, batchsize = 32,
                 report_gap = 50000, data_select = 1):
        self.N = N                            # Number of pages
        #self.D = D + 1                        # Number of max outgoing links per page + 1 (including self)
        self.emb_dim = emb_dim                # Dimension of word embedding
        self.model = model
        self.reg = reg                        # regularization (currently not implemented)
        self.k = k                            # number of VI iterations
        self.report_gap = report_gap
        self.data_select = data_select
        # We assume BatchSize = 1
        self.batchsize = batchsize            # batch size for training
        #self.maxhops = maxhops+1              # number of state inputs for every query,
                                              #     here simply the number of hops per query + 1 (including stop)
        np.random.seed(seed)
        print(model)
        #theano.config.blas.ldflags = "-L/usr/local/lib -lopenblas"

        # initial self.wk, self.idx, self.rev_idx, self.edges, self.page_emb, self.q
        print 'load graph data ...'
        self.load_graph()

        # query input: input embedding vector of query
        self.Q_in = T.fmatrix('Q_in')  # batchsize * emb_dim

        print 'building value function pre-proc model ...'

        self.vin_net = VinBlockWiki(Q_in=self.Q_in, batchsize = self.batchsize, 
                                    N = self.N, emb_dim = self.emb_dim,
                                    page_emb = self.school_emb,
                                    adj_mat = self.adj_mat, shft_mat = self.shft_mat,
                                    k=self.k)
        
        self.vin_params = self.vin_net.vin_params
        
        self.V = self.vin_net.output # output value function for every wiki-school page, batchsize * N

        self.compute = theano.function(inputs=[self.Q_in],
                                        outputs=[self.V])
        

    def load_graph(self):  
        """
        :type page_emb: np.fmatrix
        :param page_emb: input data, embedding for each page, of shape [emb_dim, N_pages]

        :type edges: scipy.sparse.csc_matrix
        :param edges: adjacency matrix, of shape [N_pages, N_pages * D], column sparse
        """
        tstart = time.time()
        
        fs = h5py.File(prm.school_pages_emb_path, 'r')
        self.school_emb = np.zeros((self.emb_dim, self.N), dtype=theano.config.floatX)
        for i in range(self.N):
            self.school_emb[:, i] = fs['emb'][i]
        fs.close()

        self.wk = wiki.Wiki(prm.school_pages_path)
        self.idx = self.wk.get_titles_pos()

        self.adj_mat = np.zeros((self.N, self.N), dtype = theano.config.floatX)
        self.shft_mat = -10 * np.ones((self.N, self.N), dtype = theano.config.floatX)
        for i in range(self.N):
            self.adj_mat[i,i] = 1
            urls = self.wk.get_article_links(i)
            for j in urls:
                self.adj_mat[j, i] = 1
                self.sht_mat[j, i] = 0
        
        self.q = qp.QP(prm.curr_query_path) # query for webnav task

        elap = time.time() - tstart
        print ' >>> time elapsed: %f' % (elap)

    def precompute(self,epochs=10,output='query_value.hdf5'):

        print 'Precompute Value Function ...'
        
        batch_size = self.batchsize

        train_queries = self.q.get_train_queries()
        #valid_queries = self.q.get_valid_queries()
        test_queries = self.q.get_test_queries()

        train_n = train_queries.shape[0]
        test_n = test_queries.shape[0]

        dim = self.emb_dim
        

        f = h5py.File(output, 'w')
        v_train = f.create_dataset('train',(train_n, self.N),dtype='float32')
        v_test = f.create_dataset('test',(test_n, self.N),dtype='float32')

	Q_dat = np.zeros((batch_size,self.emb_dim), dtype = theano.config.floatX) # batchsize * emb_dim
        V_dat = np.zeros((batch_size,self.N), dtype = theano.config.floatX) # batchsize * emb_dim

      
        print 'train_n = %d ...' % (train_n)
        print 'test_n = %d ...' % (test_n)

        print '>> Training Queries ...'

        tstart = time.time()
        ptr = 0
        cnt = 0
        while (ptr < train_n):
            end = min(train_n, ptr + batch_size)
            det = end - ptr
            for i in xrange(det):
                Q_dat[i, :] = train_queries[ptr + i, :]
            V_dat = self.compute(Q_dat)[0]
            for i in xrange(det):
                v_train[ptr + i, :] = V_dat[i, :]
            ptr += batch_size
            if (ptr > cnt * self.report_gap):
                cnt += 1
                print '   >> finished samples %d / %d (%f percent)... elapsed = %f' % (end, train_n, (100.0 * end) / train_n, time.time()-tstart)             	

        print '    ----> OK! Total Time Elapsed: %f' % (time.time() - tstart)

        print '>> Testing Queries ...'

        tstart = time.time()
        ptr = 0
        cnt = 0
        while (ptr < test_n):
            end = min(test_n, ptr + batch_size)
            det = end - ptr
            for i in xrange(det):
                Q_dat[i, :] = test_queries[ptr + i, :]
            V_dat = self.compute(Q_dat)[0]
            for i in xrange(det):
                v_test[ptr + i, :] = V_dat[i, :]
            ptr += batch_size
            if (ptr > cnt * self.report_gap):
                cnt += 1
                print '   >> finished samples %d / %d (%f percent)... elapsed = %f' % (end, test_n, (100.0 * end) / test_n, time.time()-tstart)             	

        print '    ----> OK! Total Time Elapsed: %f' % (time.time() - tstart)

        f.close()

    def load_pretrained(self, vin_file="../pretrain/WebNavVIN-map.pk"):
        dump_vin = pickle.load(open(vin_file, 'r'))
        [n.set_value(p) for n, p in zip(self.vin_params, dump_vin)]


class VinBlockWiki(object):
    """VIN block for wiki-school dataset"""
    def __init__(self, Q_in, N, emb_dim, batchsize,
                 page_emb, adj_mat, shft_mat,
                 k):
        """
        Allocate a VIN block with shared variable internal parameters.

        :type Q_in: theano.tensor.fmatrix
        :param Q_in: symbolic input query embedding, of shape [batchsize, emb_dim]

        :type page_emb: np.fmatrix
        :param page_emb: input data, embedding for each page, of shape [emb_dim, N_pages]

        :type adj_mat: np.fmatrix
        :param edges: adjacency matrix, of shape [N_pages, N_pages], each entry is either 1.0 or 0.0
                      ad_mat[j, i] = 1 means there is a link from i to j

        :type N: int32
        :param N: number of pages

        :type emb_dim: int32
        :param emb_dim: dimension of word embedding

        :type k: int32
        :param k: number of VI iterations (actually, real number of iterations is k+1)

        """

        self.page_emb = theano.sandbox.cuda.var.float32_shared_constructor(page_emb)
        self.adj_mat = theano.sandbox.cuda.var.float32_shared_constructor(adj_mat)
        self.adj_mat = self.adj_mat.dimshuffle('x', 0, 1) # x * N * N
        self.shft_mat = theano.sandbox.cuda.var.float32_shared_constructor(shft_mat)
        self.shft_mat = self.shft_mat.dimshuffle('x', 0, 1) # x * N * N

        self.vin_params = []
        
        # Q_in * W
        self._W = init_weights_T(1, emb_dim);
        #self.params.append(self.W)
        self.vin_params.append(self._W)
        self.W = T.extra_ops.repeat(self._W, batchsize, axis = 0) # batchsize * emb_dim
        self.q = Q_in * self.W

        # add bias
        self.q_bias = init_weights_T(emb_dim)
        #self.params.append(self.q_bias)
        self.vin_params.append(self.q_bias)
        self.q = self.q + self.q_bias.dimshuffle('x', 0) # batch * emb_dim

        self.R = T.dot(self.q, self.page_emb)
	self.R = T.tanh(self.R) # [batchsize * N_pages]
        # initial value
        self.V = self.R  # [batchsize * N_pages]

        # Value Iteration
        for i in xrange(k):
            self.tV = self.V.dimshuffle(0, 'x', 1) # batchsize * x * N
            #self.tV = T.extra_ops.repeat(self.V, N, axis = 0)  # N * N
            self.q = self.tV * self.adj_mat + self.shft_mat # batchsize * N * N
            #self.q = self.q + self.add_R
            self.V = T.max(self.q, axis=1, keepdims=False) # batchsize * N
	    self.V = self.V + self.R # batchsize * N

        self.output = self.V
        
