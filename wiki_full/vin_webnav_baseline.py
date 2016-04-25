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
    def __init__(self, model="valIterWebNavBL", emb_dim = 300,
                 dropout=False, devtype="cpu", batchsize = 128, 
                 grad_check=False, reg=0, seed = 0):
        self.emb_dim = emb_dim                # Dimension of word embedding
        self.batchsize = batchsize            # maximum batchsize
        self.model = model
        self.reg = reg                        # regularization (currently not implemented)

        np.random.seed(seed)
        print(model)
        #theano.config.blas.ldflags = "-L/usr/local/lib -lopenblas"

        # initial self.wk, self.idx, self.rev_idx, self.edges, self.page_emb, self.q
        print 'load data ...'
        self.load_data()

        # query input: input embedding vector of query
        self.Q_in = T.fmatrix('Q_in')  # batchsize * emb_dim
        # S input: embedding for the current state
        self.S_in = T.fmatrix("S_in")  # 1 * emb_dim
        # A input: embedding for adjacent pages to the current state
        self.A_in = T.fmatrix('A_in')  # emb_dim * max_degree
        # output action
        self.y = T.ivector("y")        # batchsize

        print 'building Baseline model ...'

        self.vin_net = BaseLineBlockWiki(
                                    Q_in=self.Q_in, S_in=self.S_in, A_in=self.A_in,
                                    emb_dim = self.emb_dim)
        
        self.p_of_y = self.vin_net.output
        self.params = self.vin_net.params
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
        

    def load_data(self):  
        """
        :type page_emb: np.fmatrix
        :param page_emb: input data, embedding for each page, of shape [emb_dim, N_pages]

        :type edges: scipy.sparse.csc_matrix
        :param edges: adjacency matrix, of shape [N_pages, N_pages * D], column sparse
        """
        tstart = time.time()
        
        self.q = qp.QP(prm.curr_query_path) # query for webnav task

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
                ver_pos[x[1]] = m
                m += 1
            else:
                cnt[x[1]] = 1

        full_wk = wiki.Wiki(prm.pages_path)

        tmp_elap = time.time() - tmp_tstart
        print ' >>> time elapsed: %f' % (tmp_elap)

        print 'Training on baseline wiki model starts ...'

        #valid_n = len(valid_entry)
        if (prm.only_predict):
            test_n = len(test_entry)
        else:
            test_n = len(test_entry) / 10 # to make things faster

        self.updates = rmsprop_updates_T(self.cost, self.params, stepsize=stepsize)
        self.train = theano.function(inputs=[self.Q_in, self.S_in, self.A_in, self.y], outputs=[], updates=self.updates)

        Q_dat = np.zeros((batch_size,self.emb_dim), dtype = theano.config.floatX) # batchsize * emb_dim
        Q_sig = np.zeros((1,self.emb_dim), dtype = theano.config.floatX) # 1 * emb_dim
        S_dat = np.zeros((1,self.emb_dim), dtype = theano.config.floatX) # 1 * emb_dim
        y_sig = np.zeros(1, dtype = np.int32) # 1

        fs = h5py.File(prm.pages_emb_path, 'r', driver='core')
        #self.school_emb = np.zeros((self.emb_dim, self.N), dtype=theano.config.floatX)
        #for i in range(self.N):
        #    self.school_emb[:, i] = fs['emb'][i]
        
        print 'train_n = %d ...' % (train_n)
        print 'test_n = %d ...' % (test_n)
         
        print fmt_row(10, ["Epoch", "Train NLL", "Train Err", "Test NLL", "Test Err", "Epoch Time"])
        for i_epoch in xrange(int(epochs)):
            tstart = time.time()
            # shuffle training index
            ver_prior = np.random.permutation(m)
            inds = sorted(np.random.permutation(train_n), key=lambda x:ver_prior[ver_pos[train_entry[x][1]]])
            
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
                        adj[_v] = _k
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

    def load_weights(self, infile="weight_dump.pk"):
        dump = pickle.load(open(infile, 'r'))
        [n.set_value(p) for n, p in zip(self.params, dump)]

    def save_weights(self, outfile="weight_dump.pk"):
        pickle.dump([n.get_value() for n in self.params], open(outfile, 'w'))

class BaseLineBlockWiki(object):
    """VIN block for wiki-school dataset"""
    def __init__(self, Q_in, S_in, A_in, emb_dim):
        """
        Allocate a baseline model with shared variable internal parameters.

        :type Q_in: theano.tensor.fmatrix
        :param Q_in: symbolic input query embedding, of shape [batchsize, emb_dim]

        :type S_in: theano.tensor.fmatrix
        :param S_in: symbolic input current page embedding, of shape [1, emb_dim]

        :type A_in: theano.tensor.fmatrix
        :param A_in: symbolic input embedding of adjacent pages, of shape [emb_dim, 1~max_deg]

        :type emb_dim: int32
        :param emb_dim: dimension of word embedding
        """

        self.params = []

        self.S = T.extra_ops.repeat(S_in, Q_in.shape[0], axis = 0) # batchsize * emb_dim

        # tanh layer for local information 
        self.H = T.concatenate([Q_in, S_in], axis = 1) # combined vector for query and local page, batchsize * (emb_dim * 2)

        # now only a single tanh layer
        self.proj_dim = emb_dim # probably larger proj dim??????
        
        self.H_W = init_weights_T(2 * emb_dim, emb_dim)
        self.params.append(self.H_W)
        self.H_bias = init_weights_T(1, emb_dim)
        self.params.append(self.H_bias)

        self.bias = T.extra_ops.repeat(self.H_bias, Q_in.shape[0], axis = 0) # batchsize * emb_dim
        self.H_proj = T.tanh(T.dot(self.H, self.H_W) + self.bias) # batchsize * emb_dim
        # do we need one more layer here???

        self.R = T.dot(self.H_proj, A_in)  # batchsize * deg

        self.output = T.nnet.softmax(self.reward)
        
