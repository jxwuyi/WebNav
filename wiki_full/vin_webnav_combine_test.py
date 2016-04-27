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
    def __init__(self, model="WikiCombine_Test", N = 6072, D = 279, emb_dim = 300,
                 dropout=False, devtype="cpu",
                 grad_check=False, reg=0, k=10, seed = 0, batchsize = 32,
                 report_gap = 100):
        self.N = N                            # Number of pages
        #self.D = D + 1                        # Number of max outgoing links per page + 1 (including self)
        self.emb_dim = emb_dim                # Dimension of word embedding
        self.model = model
        self.reg = reg                        # regularization (currently not implemented)
        self.k = k                            # number of VI iterations
        self.report_gap = report_gap
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

        print 'building Full VIN model ...'

        self.vin_net = VinBlockWiki(Q_in=self.Q_in, S_in=self.S_in, A_in=self.A_in,
                                    N = self.N, emb_dim = self.emb_dim,
                                    page_emb = self.school_emb, title_emb = self.school_title_emb,
                                    adj_mat = self.adj_mat,
                                    k=self.k)
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
        
        fs = h5py.File(prm.school_pages_emb_path, 'r')
        self.school_emb = np.zeros((self.emb_dim, self.N), dtype=theano.config.floatX)
        for i in range(self.N):
            self.school_emb[:, i] = fs['emb'][i]
        fs.close()

        self.wk = wiki.Wiki(prm.school_pages_path)
        self.idx = self.wk.get_titles_pos()

        self.adj_mat = np.zeros((self.N, self.N), dtype = theano.config.floatX)
        for i in range(self.N):
            self.adj_mat[i,i] = 1
            urls = self.wk.get_article_links(i)
            for j in urls:
                self.adj_mat[j, i] = 1
        
        self.q = qp.QP(prm.curr_query_path) # query for webnav task
        self.school_title_emb = np.zeros((self.emb_dim, self.N), dtype=theano.config.floatX)
        for i in range(self.N):
            self.school_title_emb[:, i] = self.q.get_content_embed(self.wk.get_article_content(i))

        elap = time.time() - tstart
        print ' >>> time elapsed: %f' % (elap)

    def reward_checking(self, queries, paths, page_emb):
        """
        queries: M * emb_dim
        page_emb: emb_dim * N
        """
        reward = np.dot(queries, page_emb) # M * N
        target = np.argmax(reward, axis = 1) # M * array
        n = len(paths)
        correct = 0
        for i in xrange(n):
            cur = target[i]
            ans = paths[i][-1]
            if (cur == ans):
                correct += 1
        print " >>> Result: accuracy = %d / %d (%f percent) ..." % (correct, n, correct * 100.0 / n)

    def run_training(self, stepsize=0.01, epochs=10, output='None',
                     grad_check=True,
                     profile=False):

        
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

	tmp_elap = time.time() - tmp_tstart
        print ' >>> time elapsed: %f' % (tmp_elap)


        #valid_n = len(valid_entry)
        if (prm.only_predict):
            test_n = len(test_entry)
        else:
            test_n = len(test_entry) / 10 # to make things faster

        self.updates = rmsprop_updates_T(self.cost, self.params, stepsize=stepsize)
        self.train = theano.function(inputs=[self.Q_in, self.S_in, self.A_in, self.y], outputs=[], updates=self.updates)

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
            if (prm.select_subset_data > 0):
                train_n_curr = train_n / prm.select_subset_data
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
                        else:
                            Q_now = np.zeros((det,self.emb_dim), dtype = theano.config.floatX)
                        for i in xrange(det):
                            q_i, _, y_i = train_entry[inds[ptr + i]]
                            Q_now[i, :] = train_queries[q_i, :]
                            y_dat[i] = adj_ind[y_i]

                        self.train(Q_now, S_dat, A_dat, y_dat)
                        total_proc += det
                        if ((prm.report_elap_gap > 0)
                                and (total_proc > total_out * prm.report_elap_gap)):
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
                    q_i, s_i, y_i = train_entry[inds[start]]
                    Q_sig[0, :] = train_queries[q_i, :]
                    S_dat[0, :] = fs['emb'][s_i]
                    links_dat = full_wk.get_article_links(s_i)
                    deg = len(links_dat)
                    A_dat = np.zeros((self.emb_dim, deg), dtype = theano.config.floatX)
                    for _k, _v in enumerate(links_dat):
                        if (_v == y_i):
                            k_i = _k
                            y_sig[0] = _k
                        A_dat[:, _k] = fs['emb'][_v]         
                    trainerr_, trainloss_ = self.computeloss(Q_sig, S_dat, A_dat, y_sig)
                    if (prm.top_k_accuracy != 1):  # compute top-k accuracy
                        y_full = self.y_full_out(Q_sig, S_dat, A_dat)[0]
                        tmp_err = 1
                        if (k_i in y_full[0][-prm.top_k_accuracy:]):
                            tmp_err = 0 
                        trainerr_ = tmp_err * 1.0
                    
                    # prepare testing data
                    q_i, s_i, y_i = test_entry[start]
                    Q_sig[0, :] = test_queries[q_i, :]
                    S_dat[0, :] = fs['emb'][s_i]
                    links_dat = full_wk.get_article_links(s_i)
                    deg = len(links_dat)
                    A_dat = np.zeros((self.emb_dim, deg), dtype = theano.config.floatX)
                    for _k, _v in enumerate(links_dat):
                        if (_v == y_i):
                            k_i = _k
                            y_sig[0] = _k
                        A_dat[:, _k] = fs['emb'][_v]         
                    testerr_, testloss_ = self.computeloss(Q_sig, S_dat, A_dat, y_sig)
                    if (prm.top_k_accuracy != 1): # compute top-k accuracy
                        y_full = self.y_full_out(Q_sig, S_dat)[0]
                        tmp_err = 1
                        if (k_i in y_full[0][-prm.top_k_accuracy:]):
                            tmp_err = 0
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
                
            elapsed = time.time() - tstart
            print fmt_row(10, [i_epoch, trainloss/num, trainerr/num, testloss/num, testerr/num, elapsed])

        fs.close()
        

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

    def load_pretrained(self, vin_file="../pretrain/WikiVIN-sanity.pk",
                              bsl_file="../pretrain/WebNavBSL.pk"):
        dump_vin = pickle.load(open(vin_file, 'r'))
        [n.set_value(p) for n, p in zip(self.vin_params, dump_vin)]
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
                 page_emb, title_emb, adj_mat,
                 k):
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

        self.page_emb = T.as_tensor(page_emb)
        self.title_emb = T.as_tensor(title_emb)
        self.adj_mat = T.as_tensor(adj_mat)
        self.adj_mat = self.adj_mat.dimshuffle('x', 0, 1) # x * N * N

        self.params = []
        self.vin_params = []
        self.bsl_params = []
        
        if (not prm.query_map_linear):
            print 'Now we only support linear transformation over query embedding'
        # Q_in * W
        if (prm.query_weight_diag):
            self.W = init_weights_T(1, emb_dim);
            #self.params.append(self.W)
            self.vin_params.append(self.W)
            self.W = T.extra_ops.repeat(self.W, Q_in.shape[0], axis = 0)
            self.q = Q_in * self.W

            ###########################
            self.W_t = init_weights_T(1, emb_dim);
            #self.params.append(self.W_t)
            self.vin_params.append(self.W_t)
            self.W_t = T.extra_ops.repeat(self.W_t, Q_in.shape[0], axis = 0)
            self.q_t = Q_in * self.W_t
        else:
            #######
            print 'currently we only support diagonal matrix ...'
            self.W = init_weights_T(emb_dim, emb_dim)
            self.params.append(self.W)
            self.q = T.dot(Q_in, self.W)
        # add bias
        self.q_bias = init_weights_T(emb_dim)
        #self.params.append(self.q_bias)
        self.vin_params.append(self.q_bias)
        self.q = self.q + self.q_bias.dimshuffle('x', 0) # batch * emb_dim

        # self.q_t = self.q
        self.q_t_bias = init_weights_T(emb_dim)
        #self.params.append(self.q_t_bias)
        self.vin_params.append(self.q_t_bias)
        self.q_t = self.q_t + self.q_t_bias.dimshuffle('x', 0) # batch * emb_dim

        # non-linear transformation
        #if (prm.query_tanh):
        #    self.q = T.tanh(self.q)

        
        # create reword: R: [batchsize, N_pages]
        #   q: [batchsize, emb_dim]
        #   page_emb: [emb_dim, N_pages]
        self.alpha = theano.shared((np.random.random((1, 1)) * 0.1).astype(theano.config.floatX))
	#self.params.append(self.alpha)
        self.vin_params.append(self.alpha)
	self.alpha_full = T.extra_ops.repeat(self.alpha.flatten(), N, axis = 0)
	self.alpha_full = self.alpha_full.dimshuffle('x', 0) # x * N
        self.R = T.dot(self.q, self.page_emb) + self.alpha_full * T.dot(self.q_t, self.title_emb)
        #self.R = T.dot(self.q_t, title_emb)
	self.R = T.nnet.softmax(self.R) # [batchsize * N_pages]
        # initial value
        self.V = self.R  # [batchsize * N_pages]

        # transition param
        #if (prm.diagonal_action_mat): # use simple diagonal matrix for transition
        #    A = D
        #    #self.w = T.eye(D, D, dtype = theano.config.floatX)
        #else:
        #    self.w = init_weights_T(1, D, A)
        #    self.params.append(self.w)
        
        #self.w_local = theano.shared((np.ones((1, 1, A))*0.1).astype(theano.config.floatX))#init_weights_T(1, 1, A)
        #self.params.append(self.w_local)

        #self.full_w = self.w.dimshuffle('x', 0, 1);
        #if (not prm.diagonal_action_mat):
        #    self.full_w = T.extra_ops.repeat(self.w, batchsize, axis = 0) # batchsize * D * A

        #self.full_w_local = self.w_local.dimshuffle('x', 'x', 0);
        #self.full_w_local = T.extra_ops.repeat(self.w_local, batchsize, axis = 0) # batchsize * 1 * A

        #self.R_full = self.R.dimshuffle(0, 1, 'x') # batchsize * N * 1
        #self.add_R = T.batched_dot(self.R_full, self.full_w_local) # batchsize * N * A
	#self.add_R = T.extra_ops.repeat(self.R_full, A, axis = 2)        

        # Value Iteration
        for i in range(k):
            self.tV = self.V.dimshuffle(0, 'x', 1) # batchsize * x * N
            #self.tV = T.extra_ops.repeat(self.V, N, axis = 0)  # N * N
            self.q = self.tV * self.adj_mat  # batchsize * N * N
            #self.q = self.q + self.add_R
            self.V = T.max(self.q, axis=1, keepdims=False) # batchsize * N
	    self.V = self.V + self.R # batchsize * N

        # compute mapping from wiki_school reward to page reward
        self.p_W = init_weights_T(emb_dim); 
        self.params.append(self.p_W);
        #self.page_W = T.extra_ops.repeat(self.p_W, A_in.shape[1], axis = 0) # deg * emb_dim
        self.page_W = self.p_W.dimshuffle(0, 'x')  # emb_dim * &
        self.coef_A = A_in * self.page_W   # emb_dim * deg
        self.p_bias = init_weights_T(emb_dim)
        self.params.append(self.p_bias)
        self.coef_A = self.coef_A + self.p_bias.dimshuffle(0, 'x') # emb_dim * deg
        
        self.page_R = T.dot(self.coef_A.T, self.page_emb)  # deg * N
        self.page_R = T.nnet.softmax(self.page_R)  # deg * N
        self.page_R = T.dot(self.V,self.page_R.T) # batchsize * deg

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
        # do we need one more layer here???

        self.orig_R = T.dot(self.H_proj, A_in)  # batchsize * deg


        # compute beta
        self.beta_dim = 20
        self.beta_W = init_weights_T(2 * emb_dim, self.beta_dim)
        self.params.append(self.beta_W)
        self.beta_bias = init_weights_T(1, self.beta_dim)
        self.params.append(self.beta_bias)
        self.b_hid = T.nnet.relu(T.dot(self.H, self.beta_W) + self.beta_bias)  # 1 * alpha_dim
        #  another layer to a scalar
        self.beta_W2 = init_weights_T(self.beta_dim, 1)
        self.params.append(self.beta_W2)
        self.beta_bias2 = init_weights_T(1, 1)
        self.params.append(self.beta_bias2)
        self.beta = T.nnet.sigmoid(T.dot(self.b_hid, self.beta_W2) + self.beta_bias2) # 1 * 1
        self.beta = self.beta.flatten()
        # repeat
        self.beta_full = self.beta.dimshuffle(0,'x') # 1 * x
        #T.extra_ops.repeat(self.alpha, A_in.shape[1], axis = 1) # 1 * deg
        self.page_extra = self.page_R * self.beta_full


        # compute final reward for every function
        # .... Do we need an extra scalar??????
        self.reward = self.orig_R + self.page_extra

        self.output = T.nnet.softmax(self.reward)
        
