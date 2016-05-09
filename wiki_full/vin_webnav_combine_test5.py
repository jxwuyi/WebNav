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
    def __init__(self, model="WikiCombine_Test3", N = 6072, D = 279, emb_dim = 300,
                 dropout=False, devtype="cpu",
                 grad_check=False, reg=5, k=4, seed = 0, batchsize = 128,
                 report_gap = 50000, data_select = 1):
        self.N = N                            # Number of pages
        #self.D = D + 1                        # Number of max outgoing links per page + 1 (including self)
        self.emb_dim = emb_dim                # Dimension of word embedding
        self.model = model
        self.reg = reg                        # regularization constant
        self.k = k                            # number of VI iterations
        self.report_gap = report_gap
        self.data_select = data_select
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
        self.Q_in = T.fmatrix('Q_in')  # 1 * emb_dim
        # S input: embedding for the current state
        self.S_in = T.fmatrix("S_in")  # 1 * emb_dim
        # A input: embedding for adjacent pages to the current state
        self.A_in = T.fmatrix('A_in')  # emb_dim * max_degree
        # V input: value function computed for the query on the wikiSchool pages
        self.V_in = T.fmatrix('V_in')
        # output action
        self.y = T.ivector("y")        # actuall 1 * 1

        #l = 2   # channels in input layer
        #l_h = 150  # channels in initial hidden layer
        #l_q = 10   # channels in q layer (~actions)

        print 'building Full VIN model ...'

        self.vin_net = VinBlockWiki(Q_in=self.Q_in, S_in=self.S_in, A_in=self.A_in, V_in = self.V_in,
                                    N = self.N, emb_dim = self.emb_dim,
                                    page_emb = self.school_emb, k=self.k, reg = self.reg)
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

        self.computeloss = theano.function(inputs=[self.Q_in, self.S_in, self.A_in, self.V_in, self.y],
                                           outputs=[self.err, self.cost])
        self.y_out = theano.function(inputs=[self.Q_in, self.S_in, self.A_in, self.V_in], outputs=[self.y_pred])

        self.y_full_out = theano.function(inputs=[self.Q_in, self.S_in, self.A_in, self.V_in], outputs=[self.y_inc_order])
        

    def load_graph(self, value_file = "../pretrain/WebNavVAL_PreCalc_New.hdf5"):  
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

        self.fval = h5py.File(value_file, 'r')

        self.wk = wiki.Wiki(prm.school_pages_path)
        self.idx = self.wk.get_titles_pos()

        #self.adj_mat = np.zeros((self.N, self.N), dtype = theano.config.floatX)
        #for i in range(self.N):
        #    self.adj_mat[i,i] = 1
        #    urls = self.wk.get_article_links(i)
        #    for j in urls:
        #        self.adj_mat[j, i] = 1
        
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
        #valid_entry = self.q.get_tuples(valid_paths)
        test_entry = self.q.get_tuples(test_paths)

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

        full_wk = wiki.Wiki(prm.pages_path)


	fs = h5py.File(prm.pages_emb_path, 'r')

        tmp_elap = time.time() - tmp_tstart
        print ' >>> time elapsed: %f' % (tmp_elap)

	
	print 'Allocate Memory ...'
	tmp_tstart = time.time()

	Q_dat = np.zeros((batch_size,self.emb_dim), dtype = theano.config.floatX) # batchsize * emb_dim
        Q_sig = np.zeros((1,self.emb_dim), dtype = theano.config.floatX) # 1 * emb_dim
        S_dat = np.zeros((1,self.emb_dim), dtype = theano.config.floatX) # 1 * emb_dim
        y_sig = np.zeros(1, dtype = np.int32) # 1
        V_dat = np.zeros((batch_size, self.N), dtype = theano.config.floatX) # batchsize * N
        V_sig = np.zeros((1, self.N), dtype = theano.config.floatX) # 1 * N

	tmp_elap = time.time() - tmp_tstart
        print ' >>> time elapsed: %f' % (tmp_elap)

        test_ind = np.random.permutation(len(test_entry))
        train_ind = np.random.permutation(len(train_entry))

        #valid_n = len(valid_entry)
        if (prm.only_predict):
            test_n = len(test_entry)
        else:
            test_n = len(test_entry) / 20 # to make things faster

        self.updates = rmsprop_updates_T(self.cost, self.params, stepsize=stepsize)
        self.train = theano.function(inputs=[self.Q_in, self.S_in, self.A_in, self.V_in, self.y], outputs=[], updates=self.updates)

        #self.school_emb = np.zeros((self.emb_dim, self.N), dtype=theano.config.floatX)
        #for i in range(self.N):
        #    self.school_emb[:, i] = fs['emb'][i]
        
        print 'Training on full VIN with Pretrained Weights ...'
        print 'train_n = %d ...' % (train_n)
        print 'test_n = %d ...' % (test_n)
         
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
                    S_dat[0, :] = fs['emb'][s]
                    links_dat = full_wk.get_article_links(s)
                    deg = len(links_dat)
                    A_dat = np.zeros((self.emb_dim, deg), dtype = theano.config.floatX)
                    adj_ind = {}
                    for _k, _v in enumerate(links_dat):
                        adj_ind[_v] = _k
                        A_dat[:, _k] = fs['emb'][_v]
                    
                    n = cnt[s]
                    end = min(start + n, train_n_curr)
                    ptr = start
                    while(ptr < end):
                        det = min(end - ptr, batch_size)
                        y_dat = np.zeros(det, dtype = np.int32)
                        if (det == batch_size):
                            Q_now = Q_dat
                            V_now = V_dat
                        else:
                            Q_now = np.zeros((det,self.emb_dim), dtype = theano.config.floatX)
                            V_now = np.zeros((det,self.N), dtype = theano.config.floatX)
                        for i in xrange(det):
                            q_i, _, y_i = train_entry[inds[ptr + i]]
                            Q_now[i, :] = train_queries[q_i, :]
                            V_now[i, :] = self.fval['train'][q_i, :]
                            y_dat[i] = adj_ind[y_i]

                        self.train(Q_now, S_dat, A_dat, V_now, y_dat)
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
                    q_i, s_i, y_i = train_entry[train_ind[start]]
                    Q_sig[0, :] = train_queries[q_i, :]
                    V_sig[0, :] = self.fval['train'][q_i, :]
                    S_dat[0, :] = fs['emb'][s_i]
                    links_dat = full_wk.get_article_links(s_i)
                    deg = len(links_dat)
                    A_dat = np.zeros((self.emb_dim, deg), dtype = theano.config.floatX)
                    for _k, _v in enumerate(links_dat):
                        if (_v == y_i):
                            k_i = _k
                            y_sig[0] = _k
                        A_dat[:, _k] = fs['emb'][_v]         
                    trainerr_, trainloss_ = self.computeloss(Q_sig, S_dat, A_dat, V_sig, y_sig)
                    if (prm.top_k_accuracy != 1):  # compute top-k accuracy
                        y_full = self.y_full_out(Q_sig, S_dat, A_dat, V_sig)[0]
                        tmp_err = 1
                        if (k_i in y_full[0][-prm.top_k_accuracy:]):
                            tmp_err = 0 
                        trainerr_ = tmp_err * 1.0
                    
                    # prepare testing data
                    q_i, s_i, y_i = test_entry[test_ind[start]]
                    Q_sig[0, :] = test_queries[q_i, :]
                    V_sig[0, :] = self.fval['test'][q_i, :]
                    S_dat[0, :] = fs['emb'][s_i]
                    links_dat = full_wk.get_article_links(s_i)
                    deg = len(links_dat)
                    A_dat = np.zeros((self.emb_dim, deg), dtype = theano.config.floatX)
                    for _k, _v in enumerate(links_dat):
                        if (_v == y_i):
                            k_i = _k
                            y_sig[0] = _k
                        A_dat[:, _k] = fs['emb'][_v]         
                    testerr_, testloss_ = self.computeloss(Q_sig, S_dat, A_dat, V_sig, y_sig)
                    if (prm.top_k_accuracy != 1): # compute top-k accuracy
                        y_full = self.y_full_out(Q_sig, S_dat, A_dat, V_sig)[0]
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

            if (testerr / num < best):
                best = testerr / num
                self.save_weights(self.model + '_R' + str(int(self.reg))+'_best.pk')
                
            elapsed = time.time() - tstart
            print fmt_row(10, [i_epoch, trainloss/num, trainerr/num, testloss/num, testerr/num, elapsed])

        fs.close()
        self.fval.close()
   
    def load_pretrained(self, vin_file="../pretrain/WebNavVIN-map-k4.pk",
                              bsl_file="../pretrain/WebNavBSL_best.pk"):
        dump_vin = pickle.load(open(vin_file, 'r'))
        for n, p in zip(self.vin_params, dump_vin):
            n.set_value(p)
        dump_bsl = pickle.load(open(bsl_file, 'r'))
        for n, p in zip(self.bsl_params, dump_bsl):
            n.set_value(p)

    def load_weights(self, infile="weight_dump.pk"):
        dump = pickle.load(open(infile, 'r'))
        [n.set_value(p) for n, p in zip(self.params, dump)]

    def save_weights(self, outfile="weight_dump.pk"):
        pickle.dump([n.get_value() for n in self.params], open(outfile, 'w'))

class VinBlockWiki(object):
    """VIN block for wiki-school dataset"""
    def __init__(self, Q_in, S_in, A_in, V_in, N, emb_dim,
                 page_emb, k, reg):
        """
        Allocate a VIN block with shared variable internal parameters.

        :type Q_in: theano.tensor.fmatrix
        :param Q_in: symbolic input query embedding, of shape [batchsize, emb_dim]

        :type S_in: theano.tensor.fmatrix
        :param S_in: symbolic input current page embedding, of shape [1, emb_dim]

        :type A_in: theano.tensor.fmatrix
        :param A_in: symbolic input embedding of adjacent pages, of shape [emb_dim, 1~max_deg]

        :type V_in: theano.tensor.fmatrix
        :param V_in: symbolic input values of the query on the wiki-school pages, of shape [batchsize, N]

        :type page_emb: np.fmatrix
        :param page_emb: input data, embedding for each page, of shape [emb_dim, N_pages]

        :type N: int32
        :param N: number of pages

        :type emb_dim: int32
        :param emb_dim: dimension of word embedding

        :type k: int32
        :param k: number of VI iterations

        :type reg: float32
        :param reg: regularization constant to ensure the value has the same scale as local decision weight

        """

        self.page_emb = theano.sandbox.cuda.var.float32_shared_constructor(page_emb)

        self.params = []
        self.vin_params = []
        self.bsl_params = []
  
        #self._W = init_weights_T(1, emb_dim);
        #self.params.append(self.W)
        #self.vin_params.append(self._W)
        #self.vin_params.append(None)
        #self.W = T.extra_ops.repeat(self._W, Q_in.shape[0], axis = 0)
        #self.q = Q_in * self.W

        # add bias
        #self.q_bias = init_weights_T(emb_dim)
        #self.params.append(self.q_bias)
        #self.vin_params.append(self.q_bias)
        #self.vin_params.append(None)
        #self.q = self.q + self.q_bias.dimshuffle('x', 0) # batch * emb_dim

        #self.R = T.dot(self.q, self.page_emb)# + self.alpha_full * T.dot(self.q_t, self.title_emb)
        #self.R = T.dot(self.q_t, title_emb)
	#self.R = T.tanh(self.R) # [batchsize * N_pages]
        # initial value
        #self.V = self.R  # [batchsize * N_pages]

        # Value Iteration
        #for i in xrange(k):
        #    self.tV = self.V.dimshuffle(0, 'x', 1) # batchsize * x * N
        #    #self.tV = T.extra_ops.repeat(self.V, N, axis = 0)  # N * N
        #    self.q = self.tV * self.adj_mat  # batchsize * N * N
        #    #self.q = self.q + self.add_R
        #    self.V = T.max(self.q, axis=1, keepdims=False) # batchsize * N
	#    self.V = self.V + self.R # batchsize * N

        # Do regularization here!!
	#    Note: reg is already inversed
	if (reg < 0.000000001):
            self.V = V_in
        else:
            self.coef = reg / (1.0 + k)
            self.V = V_in * self.coef

        # compute mapping from wiki_school reward to page reward
        self.p_W = init_weights_T(emb_dim); 
        #self.params.append(self.p_W);
        self.vin_params.append(self.p_W)
        
        #self.page_W = T.extra_ops.repeat(self.p_W, A_in.shape[1], axis = 0) # deg * emb_dim
        self.page_W = self.p_W.dimshuffle(0, 'x')  # emb_dim * &
        self.coef_A = A_in * self.page_W   # emb_dim * deg
        self.p_bias = init_weights_T(emb_dim)
        #self.params.append(self.p_bias)
        self.vin_params.append(self.p_bias)
        
        self.coef_A = self.coef_A + self.p_bias.dimshuffle(0, 'x') # emb_dim * deg
        
        self.page_map = T.dot(self.coef_A.T, self.page_emb)  # deg * N
	self.page_map = T.nnet.sigmoid(self.page_map)
	
        self.page_R = T.dot(self.V,self.page_map.T) # batchsize * deg

        # tanh layer for local information
        self.S = T.extra_ops.repeat(S_in, Q_in.shape[0], axis = 0) # batchsize * deg
        # combined vector for query and local page, batchsize * (emb_dim * 2)
        self.H = T.concatenate([Q_in, self.S], axis = 1) 

        # now only a single tanh layer
        self.proj_dim = emb_dim # probably larger proj dim??????
        
        self.H_W = init_weights_T(2 * emb_dim, emb_dim)
        #self.params.append(self.H_W)
        self.bsl_params.append(self.H_W)
        
        self.H_bias = init_weights_T(1, emb_dim)
        #self.params.append(self.H_bias)
        self.bsl_params.append(self.H_bias)
        
        self.H_bias_full = T.extra_ops.repeat(self.H_bias, Q_in.shape[0], axis = 0) # batchsize * emb_dim
        self.H_proj = T.tanh(T.dot(self.H, self.H_W) + self.H_bias_full) # batchsize * emb_dim
        #self.H_proj = self.H_proj_full[:, 1:] # batchsize * emb_dim

        self.beta_W = init_weights_T(2 * emb_dim, 1)
        self.params.append(self.beta_W)
        self.beta_bias = init_weights_T(1)
        self.params.append(self.beta_bias)
        self.beta_bias_full = self.beta_bias.dimshuffle('x', 0) # batchsize * 1
        self.beta = T.nnet.relu(T.dot(self.H, self.beta_W) + self.beta_bias_full) # batchsize * 1
        # do we need one more layer here???

        self.beta_flat = self.beta.flatten() # batchsize

        self.orig_R = T.dot(self.H_proj, A_in)  # batchsize * deg

        self.beta_full = self.beta_flat.dimshuffle(0, 'x') # batchsize * deg

        # compute final reward for every function
        # .... Do we need an extra scalar??????
        self.reward = self.orig_R + self.beta_full * self.page_R
        #self.reward = self.page_R

        self.output = T.nnet.softmax(self.reward)
        
