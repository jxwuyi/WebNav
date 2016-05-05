'''
Class to access queries and paths stored in the hdf5 file.
'''
import h5py
import utils
import myparameters as prm
import cPickle as pkl

class QP():

    def __init__(self, path):
        self.f = h5py.File(path, 'r')
        self.wemb = pkl.load(open(prm.wordemb_path, 'rb'))
        outs = self.get_query_embed()
        self.q_train = outs[0]
        self.q_valid = outs[1]
        self.q_test = outs[2]
        outs = self.get_paths()
        self.path_train = outs[0]
        self.path_valid = outs[1]
        self.path_test = outs[2]
        self.query_texts = self.get_queries()

    def get_train_queries(self):
        return self.q_train

    def get_train_query_texts(self):
        return self.query_texts[0]

    def get_valid_queries(self):
        return self.q_valid

    def get_test_queries(self):
        return self.q_test

    def get_train_paths(self):
        return self.path_train

    def get_valid_paths(self):
        return self.path_valid

    def get_test_paths(self):
        return self.path_test

    def get_queries(self, dset=['train', 'valid', 'test']):
        '''
        Return the queries.
        'dset' is the list of datasets to be returned ('train', 'valid' and/or 'test').
        '''
        outs = []
        for name in dset:
            outs.append(self.f['queries_'+name][:])

        return outs

    def get_query_embed(self, dset=['train', 'valid', 'test']):
        '''
        Return the embedding of queries.
        'dset' is the list of datasets to be returned ('train', 'valid' and/or 'test').
        '''
        outs = []
        for name in dset:
            curr = self.f['queries_'+name][:]
            q_emb = utils.Word2Vec_encode(curr, self.wemb)
            outs.append(q_emb)

        return outs   # 3 * n_queries * n_dim

    def get_content_embed(self, text):
        return utils.Sent2Vec_encode(text, self.wemb)
    

    def get_paths(self, dset=['train', 'valid', 'test']):
        '''
        Return the paths (as a list of articles' ids) to reach the query,
        starting from the root page.
        'dset' is the list of datasets to be returned ('train', 'valid' and/or 'test').
        '''
        outs = []
        for name in dset:
            out = []
            for item in self.f['paths_'+name]:
                out.append(self.tolist(item))
            outs.append(out)

        return outs  # 3 * n_queries * n_length

    def get_tuples(self, paths, inv_idx):
        outs = []
        for i in xrange(len(paths)):
            n = len(paths[i])
            for j in range(n - 1):
                if (j == n - 1):
                    outs.append((i, paths[i][j], 0))
                else:
                    x = paths[i][j]
                    y = paths[i][j + 1]
                    outs.append((i, x, y))
        return outs

    def tolist(self, text):
        '''
        Convert a string whose elements are separated by a space to a list of integers.
        '''
        return [int(a) for a in text.strip().split(' ')]
