# VI network using THEANO, takes batches of state input
from NNobj import *
from theano_utils import *
import myparameters as prm
import wiki
import scipy.sparse as SS
#import qp
import qp_bsl_modify as qp
import h5py


class vin(NNobj):
    "Class for a neural network that does k iterations of value iteration"
    def __init__(self, model="WikiCombineSANJoint", N = 6072, D = 279, emb_dim = 300, k = 0,
                 dropout=False, devtype="cpu",
                 grad_check=False, reg=0, seed = 0, batchsize = 32,
                 report_gap = 1200000, data_select = 1):
        self.N = N                            # Number of pages
        #self.D = D + 1                       # Number of max outgoing links per page + 1 (including self)
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
        # np.random.seed(seed)
        print(model)
        #theano.config.blas.ldflags = "-L/usr/local/lib -lopenblas"

        # initial self.wk, self.idx, self.rev_idx, self.edges, self.page_emb, self.q
        print 'load graph data ...'
        self.load_graph()
        np.random.seed(seed) # load_graph resets seed to 0

        # query input: input embedding vector of query
        self.Q_in = T.fmatrix('Q_in')  # 1 * emb_dim
        # S input: embedding for the current state
        self.S_in = T.fmatrix("S_in")  # 1 * emb_dim
        # A input: embedding for adjacent pages to the current state
        self.A_in = T.fmatrix('A_in')  # emb_dim * max_degree
        # output action
        self.y = T.ivector("y")        # actuall 1 * 1

        #l = 2   # channels in input layer
        #l_h = 150  # channels in initial hidden layer
        #l_q = 10   # channels in q layer (~actions)

        print 'building combined model with VIN on approx graph (indexing pages)...'

        self.vin_net = VinBlockWiki(Q_in=self.Q_in, S_in = self.S_in, A_in=self.A_in,
                                    N = self.M, emb_dim = self.emb_dim,
                                    page_emb = self.school_emb,
                                    adj_mat = self.adj_mat,
                                    adj_shft = self.adj_shft,
                                    k = self.k)
        self.p_of_y = self.vin_net.output
        self.params = self.vin_net.params
        self.vin_params = self.vin_net.vin_params
        self.bsl_params = self.vin_net.bsl_params
        # Total 1910 parameters ?????

        self.cost = -T.mean(T.log(self.p_of_y)[T.arange(self.y.shape[0]),
                                               self.y.flatten()], dtype=theano.config.floatX)
        self.y_pred = T.argmax(self.p_of_y, axis=1)
        self.y_inc_order = T.argsort(self.p_of_y, axis = 1)
        
        self.err = T.mean(T.neq(self.y_pred, self.y.flatten()), dtype=theano.config.floatX)

        self.computeloss = theano.function(inputs=[self.Q_in, self.S_in, self.A_in, self.y],
                                           outputs=[self.err, self.cost])
        self.y_out = theano.function(inputs=[self.Q_in, self.S_in, self.A_in], outputs=[self.y_pred])

        self.y_full_out = theano.function(inputs=[self.Q_in, self.S_in, self.A_in], outputs=[self.y_inc_order])
        

    def load_graph(self):  
        """
        :type page_emb: np.fmatrix
        :param page_emb: input data, embedding for each page, of shape [emb_dim, N_pages]

        :type edges: scipy.sparse.csc_matrix
        :param edges: adjacency matrix, of shape [N_pages, N_pages * D], column sparse
        """
        tstart = time.time()

        self.wk = wiki.Wiki(prm.pages_path)
        self.idx = self.wk.get_index_pages(6072)
        self.M = len(self.idx)
        self.idx_rank = {}
        for i in xrange(self.M):
            self.idx_rank[self.idx[i]]=i

        fs = h5py.File(prm.pages_emb_path, 'r')
        self.full_page_emb = np.zeros((self.emb_dim, self.N), dtype=theano.config.floatX)
        for i in range(self.N):
            self.full_page_emb[:, i] = fs['emb'][i]

        fs.close()

        # Approximation Graph with only indexing pages        
        self.adj_mat = np.zeros((self.M, self.M), dtype = theano.config.floatX)
        self.adj_shft = np.ones((self.M, self.M), dtype = theano.config.floatX) * -100 # - infinity
        self.school_emb = np.zeros((self.emb_dim, self.M), dtype=theano.config.floatX)
        for i in range(self.M):
            self.school_emb[:, i] = self.full_page_emb[:,self.idx[i]]
            self.adj_mat[i,i] = 1
            self.adj_shft[i,i] = 0
            urls = self.wk.get_article_links(self.idx[i])
            for j in urls:
                if (j in self.idx_rank):
                    self.adj_mat[self.idx_rank[j], i] = 1
                    self.adj_shft[self.idx_rank[j], i] = 0

        
        
        self.q = qp.QP(prm.curr_query_path) # query for webnav task
        
        elap = time.time() - tstart
        print ' >>> time elapsed: %f' % (elap)

    def run_training(self, stepsize=0.01, epochs=10, output='None',
                     grad_check=True,
                     profile=False):

        best = 1
        print 'Prepare Training Data ...'

        tmp_tstart = time.time()
        
        batch_size = self.batchsize

        train_queries = self.q.get_train_queries()
        #valid_queries = self.q.get_valid_queries()
        test_queries = self.q.get_test_queries()
        train_paths = self.q.get_train_paths()
        #valid_paths = self.q.get_valid_paths()
        test_paths = self.q.get_test_paths()

        train_entry = self.q.get_tuples(train_paths)
        #train_entry = self.q.get_tuples_raw(train_paths)
        #valid_entry = self.q.get_tuples_raw(valid_paths)
        test_entry = self.q.get_tuples(test_paths)
        #test_entry = self.q.get_tuples_raw(test_paths)

        train_n = len(train_entry)
        cnt = {}
        ver_pos = {}
        m = 0
        for x in train_entry:
            if (x[1] in cnt):
                cnt[x[1]] += 1
            else:
                cnt[x[1]] = 1
		ver_pos[x[1]] = m
                m += 1

        #full_wk = wiki.Wiki(prm.pages_path)
        full_wk = self.wk


	
	Q_dat = np.zeros((batch_size,self.emb_dim), dtype = theano.config.floatX) # batchsize * emb_dim
        Q_sig = np.zeros((1,self.emb_dim), dtype = theano.config.floatX) # 1 * emb_dim
        S_dat = np.zeros((1,self.emb_dim), dtype = theano.config.floatX) # 1 * emb_dim
        y_sig = np.zeros(1, dtype = np.int32) # 1

	tmp_elap = time.time() - tmp_tstart
        print ' >>> time elapsed: %f' % (tmp_elap)


        print 'Compiling ...'

        #valid_n = len(valid_entry)
        if (prm.only_predict):
            test_n = len(test_entry)
        else:
            test_n = len(test_entry) / 2 # to make things faster

        self.updates = rmsprop_updates_T(self.cost, self.params, stepsize=stepsize)
        #self.updates = rmsprop_updates_T(self.cost, self.bsl_params, stepsize=stepsize)
        self.train = theano.function(inputs=[self.Q_in, self.S_in, self.A_in, self.y], outputs=[], updates=self.updates)

        #self.school_emb = np.zeros((self.emb_dim, self.N), dtype=theano.config.floatX)
        #for i in range(self.N):
        #    self.school_emb[:, i] = fs['emb'][i]
        
        print 'Training Starts ...'
        print 'train_n = %d ...' % (train_n)
        print 'test_n = %d ...' % (test_n)

        train_order = np.random.permutation(len(train_entry))
        test_order = np.random.permutation(len(test_entry))
         
        print fmt_row(10, ["Epoch", "Train NLL", "Train Err", "Test NLL", "Test Err", "Epoch Time"])
        for i_epoch in xrange(int(epochs)):
            tstart = time.time()
            # shuffle training index
            ver_prior = np.random.permutation(m)
            inds = sorted(np.random.permutation(train_n), key=lambda x:ver_prior[ver_pos[train_entry[x][1]]])
            
	    print ' >> sort time : %f s' %(time.time() - tstart)

            train_n_curr = train_n
            if (self.data_select > 0):
                train_n_curr = train_n / self.data_select
            # do training
            if (not prm.only_predict): # we do need to perform training
                start = 0
                total_proc = 0
                total_out = 0
                
                while (start < train_n_curr):
                    s = train_entry[inds[start]][1]
                    S_dat[0, :] = self.full_page_emb[:,s]
                    links_dat = full_wk.get_article_links(s)
                    deg = len(links_dat)
                    A_dat = np.zeros((self.emb_dim, deg), dtype = theano.config.floatX)
                    adj_ind = {}
                    for _k, _v in enumerate(links_dat):
                        adj_ind[_v] = _k
                        A_dat[:, _k] = self.full_page_emb[:,_v]
                    
                    n = cnt[s]
                    end = min(start + n, train_n_curr)
                    ptr = start
                    while(ptr < end):
                        det = min(end - ptr, batch_size)
                        y_dat = np.zeros(det, dtype = np.int32)
                        if (det == batch_size):
                            Q_now = Q_dat
                        else:
                            Q_now = np.zeros((det,self.emb_dim), dtype = theano.config.floatX)
                        for i in xrange(det):
                            q_i, _, y_i = train_entry[inds[ptr + i]]
                            Q_now[i, :] = train_queries[q_i, :]
                            y_dat[i] = adj_ind[y_i]

                        self.train(Q_now, S_dat, A_dat, y_dat)
                        total_proc += det
                        if ((self.report_gap > 0)
                                and (total_proc > total_out * self.report_gap)):
                            total_out += 1
                            print '>> finished samples %d / %d (%f percent)... elapsed = %f' % (total_proc, train_n_curr, (100.0 * total_proc) / train_n_curr, time.time()-tstart)             	
                        
                        ptr += batch_size
                    start = end
                
            # compute losses
            trainerr = 0.
            trainloss = 0.
            testerr = 0.
            testloss = 0.
            num = 0

            ##############
            if (prm.perform_full_inference):
                total_trial = len(test_paths)
                test_fail = [len(test_paths[j])-1 for j in xrange(total_trial)]
                test_success = 0
            ##############

            for start in xrange(0, test_n):  # assume batch_size = 1
                end = start+1   #batch_size = 1
                if end <= test_n:  # assert(text_n <= train_n)
                    num += 1
                    # prepare training data
                    q_i, s_i, y_i = train_entry[train_order[start]]
                    Q_sig[0, :] = train_queries[q_i, :]
                    S_dat[0, :] = self.full_page_emb[:,s_i]
                    links_dat = full_wk.get_article_links(s_i)
                    deg = len(links_dat)
                    A_dat = np.zeros((self.emb_dim, deg), dtype = theano.config.floatX)
                    for _k, _v in enumerate(links_dat):
                        if (_v == y_i):
                            k_i = _k
                            y_sig[0] = _k
                        A_dat[:, _k] = self.full_page_emb[:,_v]         
                    trainerr_, trainloss_ = self.computeloss(Q_sig, S_dat, A_dat, y_sig)
                    if (prm.top_k_accuracy != 1):  # compute top-k accuracy
                        y_full = self.y_full_out(Q_sig, S_dat, A_dat)[0]
                        tmp_err = 1
                        if (k_i in y_full[0][-prm.top_k_accuracy:]):
                            tmp_err = 0 
                        trainerr_ = tmp_err * 1.0
                    
                    # prepare testing data
                    q_i, s_i, y_i = test_entry[test_order[start]]
                    Q_sig[0, :] = test_queries[q_i, :]
                    S_dat[0, :] = self.full_page_emb[:,s_i]
                    links_dat = full_wk.get_article_links(s_i)
                    deg = len(links_dat)
                    A_dat = np.zeros((self.emb_dim, deg), dtype = theano.config.floatX)
                    for _k, _v in enumerate(links_dat):
                        if (_v == y_i):
                            k_i = _k
                            y_sig[0] = _k
                        A_dat[:, _k] = self.full_page_emb[:,_v]         
                    testerr_, testloss_ = self.computeloss(Q_sig, S_dat, A_dat, y_sig)
                    if (prm.top_k_accuracy != 1): # compute top-k accuracy
                        y_full = self.y_full_out(Q_sig, S_dat, A_dat)[0]
                        tmp_err = 1
                        if (k_i in y_full[0][-prm.top_k_accuracy:]):
                            tmp_err = 0
                            if (prm.perform_full_inference):
                                test_fail[q_i] -= 1
                                if (test_fail[q_i] == 0):
                                    test_success += 1
   
                        testerr_ = tmp_err * 1.0
                    
                    trainerr += trainerr_
                    trainloss += trainloss_
                    testerr += testerr_
                    testloss += testloss_
            
            if (prm.perform_full_inference):
                print 'total sucess trails = %d over %d (percent: %f) ...' %(test_success, total_trial, (test_success*1.0 / total_trial))

            if (testerr/num < best):
                best = testerr / num
                self.save_weights(self.model + '_best.pk')
            
            elapsed = time.time() - tstart
            print fmt_row(10, [i_epoch, trainloss/num, trainerr/num, testloss/num, testerr/num, elapsed])

    #TODO
    def load_pretrained(self, vin_file="../pretrain/WebNavVIN-map.pk",
                              bsl_file="../pretrain/WebNavCMB_SAN.pk"):
        dump_vin = pickle.load(open(vin_file, 'r'))
        [n.set_value(p) for n, p in zip(self.vin_params, dump_vin)]
        dump_bsl = pickle.load(open(bsl_file, 'r'))
        [n.set_value(p) for n, p in zip(self.bsl_params, dump_bsl)]
        #pass

    def load_pretrained_vin(self, vin_file="../pretrain/WebNavVIN-map.pk"):
        dump_vin = pickle.load(open(vin_file, 'r'))
        [n.set_value(p) for n, p in zip(self.vin_params, dump_vin)]

    def load_pretrained_bsl(self, bsl_file="../pretrain/WebNavCMB_SAN.pk"):
        dump_bsl = pickle.load(open(bsl_file, 'r'))
        [n.set_value(p) for n, p in zip(self.bsl_params, dump_bsl)]

    def load_weights(self, infile="weight_dump.pk"):
        dump = pickle.load(open(infile, 'r'))
        [n.set_value(p) for n, p in zip(self.params, dump)]

    def save_weights(self, outfile="weight_dump.pk"):
        pickle.dump([n.get_value() for n in self.params], open(outfile, 'w'))

class VinBlockWiki(object):
    """VIN block for wiki-school dataset"""
    def __init__(self, Q_in, S_in, A_in, N, emb_dim,
                 page_emb, adj_mat, adj_shft, k):
        """
        Allocate a VIN block with shared variable internal parameters.

        :type Q_in: theano.tensor.fmatrix
        :param Q_in: symbolic input query embedding, of shape [batchsize, emb_dim]

        :type S_in: theano.tensor.fmatrix
        :param S_in: symbolic input current page embedding, of shape [1, emb_dim]

        :type A_in: theano.tensor.fmatrix
        :param A_in: symbolic input embedding of adjacent pages, of shape [emb_dim, 1~max_deg]

        :type page_emb: np.fmatrix
        :param page_emb: input data, embedding for each page, of shape [emb_dim, N_pages]

        :type N: int32
        :param N: number of pages

        :type emb_dim: int32
        :param emb_dim: dimension of word embedding

        """

        self.page_emb = theano.sandbox.cuda.var.float32_shared_constructor(page_emb)
        #self.title_emb = theano.sandbox.cuda.var.float32_shared_constructor(title_emb)
        if (k > 0):
            self.adj_mat = theano.sandbox.cuda.var.float32_shared_constructor(adj_mat)
            self.adj_mat = self.adj_mat.dimshuffle('x', 0, 1) # x * N * N
            self.adj_shft = theano.sandbox.cuda.var.float32_shared_constructor(adj_shft) # ensure -infinity
            self.adj_shft = self.adj_shft.dimshuffle('x', 0, 1) # x * N * N

        self.params = []
        self.vin_params = []
        self.bsl_params = []
        
        if (not prm.query_map_linear):
            print 'Now we only support linear transformation over query embedding'
        # Q_in * W
        self._W = init_weights_one_T(1, emb_dim);
        self.params.append(self._W)
        self.vin_params.append(self._W) # mapping params
        
        self.W = T.extra_ops.repeat(self._W, Q_in.shape[0], axis = 0)
        self.q = Q_in * self.W

        # add bias
        self.q_bias = init_weights_T(emb_dim)
        self.params.append(self.q_bias)
        self.vin_params.append(self.q_bias)
        self.q = self.q + self.q_bias.dimshuffle('x', 0) # batch * emb_dim

        
        self.R = T.dot(self.q, self.page_emb)
        # self.R = T.tanh(self.R) # T.nnet.softmax(5*self.R)  
        # initial value
        self.V = self.R  # [batchsize * N_pages]
    

        # Value Iteration
        if (k > 0):
            for i in range(k):
                self.tV = self.V.dimshuffle(0, 'x', 1) # batchsize * x * N
                #self.tV = T.extra_ops.repeat(self.V, N, axis = 0)  # N * N
                self.q = self.tV * self.adj_mat + self.adj_shft # batchsize * N * N
                #self.q = self.q + self.add_R
                self.V = T.max(self.q, axis=1, keepdims=False) # batchsize * N
                self.V = self.V + self.R # batchsize * N

        # compute mapping from wiki_school reward to page reward
        self.p_W = init_weights_one_T(emb_dim); 
        self.params.append(self.p_W);
        self.vin_params.append(self.p_W)
        
        #self.page_W = T.extra_ops.repeat(self.p_W, A_in.shape[1], axis = 0) # deg * emb_dim
        self.page_W = self.p_W.dimshuffle(0, 'x')  # emb_dim * &
        self.coef_A = A_in * self.page_W   # emb_dim * deg
        
        self.p_bias = init_weights_T(emb_dim)
        self.params.append(self.p_bias)
        self.vin_params.append(self.p_bias)
        
        self.coef_A = self.coef_A + self.p_bias.dimshuffle(0, 'x') # emb_dim * deg
        
        self.page_map = T.dot(self.coef_A.T, self.page_emb)  # deg * N
	self.page_map = T.nnet.sigmoid(self.page_map) # T.nnet.softmax(5*self.page_map) 
	
        self.page_R = T.dot(self.V,self.page_map.T) # batchsize * deg

        # tanh layer for local information
        #self.S = T.extra_ops.repeat(S_in, Q_in.shape[0], axis = 0) # batchsize * dim
        # combined vector for query and local page, batchsize * (emb_dim * 2)
        #self.H = T.concatenate([Q_in, self.S], axis = 1) 

        # now only a single tanh layer
        
        #self.H_W = init_weights_T(2 * emb_dim, emb_dim + 1)
        #self.H_W = init_weights_T(2 * emb_dim, emb_dim)
        #self.params.append(self.H_W)
        #self.bsl_params.append(self.H_W)
      
        #self.H_bias = init_weights_T(1, emb_dim + 1)
        #self.H_bias = init_weights_T(1, emb_dim)
        #self.params.append(self.H_bias)
        #self.bsl_params.append(self.H_bias)

        #self.beta_W = init_weights_T(2 * emb_dim)
        #self.beta_bias = init_weights_T(1)
        #self.params.append(self.beta_W)
        #self.params.append(self.beta_bias)

        #self.H_bias_full = T.extra_ops.repeat(self.H_bias, Q_in.shape[0], axis = 0) # batchsize * emb_dim
        #self.beta_bias_full = T.extra_ops.repeat(self.beta_bias, Q_in.shape[0], axis = 0) # batchsize * 1
        #self.H_proj_full = T.tanh(T.dot(self.H, self.H_W) + self.H_bias_full) # batchsize * emb_dim
        #self.H_proj = self.H_proj_full #[:, 1:] # batchsize * emb_dim
        #self.beta = T.tanh(T.dot(self.H, self.beta_W) + self.beta_bias_full) # batchsize * 1
        # self.beta = self.H_proj_full[:, 0] # batchsize
        # do we need one more layer here???

        #self.orig_R = T.dot(self.H_proj, A_in)  # batchsize * deg

        #self.beta_full = self.beta.dimshuffle(0, 'x') # batchsize * deg

        # compute final reward for every function
        #self.reward = self.orig_R + self.beta_full * self.page_R
        self.reward = self.page_R

        self.output = T.nnet.softmax(self.reward)
        
