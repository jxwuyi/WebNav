######################
# Dataset parameters #
######################
n_samples = [[1e8,1000,1000],[1e8,20000,20000],[1e8,20000,20000]] # maximum number of samples for the training, validation and test sets per hop
max_hops = [2,4,8] # Maximum number of hops to be visited to extract queries.
max_hops_pages = 10 # Maximum number of hops.
max_sents = 5 # Maximum number of query sentences to be extracted in each page.
max_links = 300 # Maximum number of links a page can have. If it has more links than max_links, it is not included in the dataset. Set it to None if you want all pages to be included in the dataset.
min_words_query = 10 # Minimum number of words a query can have.
max_words_query = 30 # Maximum number of words a query can have.
n_words = 374000 # words for the vocabulary
n_consec = 4 # maximum number of consecutive sentences to form a query
root_page = 'category:main topic classifications'
school_root_page = 'http://schools-wikipedia.org/wp/index/subject.htm'
vocab_path = '../data/D_cbow_pdw_8B.pkl' # Path to the python dictionary containing the vocabulary.
wordemb_path = '../data/D_cbow_pdw_8B.pkl' # Path to the python dictionary containing the word embeddings.
idf_path = '../data/wiki_idf.pkl' # Path to the IDF dictionary.
school_pages_path = '../data/wiki_school.hdf5' # Path to save the articles and links.
pages_path = '../../Downloads/wiki.hdf5' # Path to save the articles and links.
school_pages_emb_path = '../data/wiki_school_emb.hdf5' # Path to save articles embeddings (set to None to not compute it).
pages_emb_path = '../../Downloads/wiki_emb.hdf5' # Path to save articles embeddings (set to None to not compute it).
qp_path_pre = '../data/qp_school.hdf5' # Path to save queries and paths.

###################################
# Wikipedia for school parameters #
###################################
content_path = 'content.txt' # file with all the titles and urls
curr_query_path = '../../Downloads/queries_paths_4hops_4sentences.hdf5'
total_pages = 6072
max_links_per_page = 279

#########################
# Wikipedia  parameters #
#########################
compute_page_pos = False # Compute or not the page positions in the Wikipedia dump file
dump_path = '../data/enwiki-latest-pages-articles.xml' # Path to the wikipedia dump file.
page_pos_path = '../../Downloads/page_pos.pkl' # Path to save the dictionary that stores each article position in the wikipedia dump file.
cat_pages_path = '../../Downloads/cat_pages.pkl' # Path to save the dictionary that stores the pages in each wikipedia category.



####################
#  VIN parameters  #
####################
query_map_linear = True
query_weight_diag = True
query_weight_rank = 10
query_tanh = False
diagonal_action_mat = True
final_scale = False
select_subset_data = 0
output_wrong_predict = False
only_predict = False
top_k_accuracy = 1
perform_full_inference = False



####################
# Model parameters #
####################
qp_path = '../data/qp_171_24810_2hops.hdf5' # Path to load queries and paths.
dim_proj=500  # LSTM number of hidden units.
dim_emb=500  # word embedding dimension.
patience=100  # Number of epochs to wait before early stop if no progress.
max_epochs=5000  # The maximum number of epochs to run.
dispFreq=10  # Display to stdout the training progress every N updates.
decay_c=0.  # Weight decay for the classifier applied to the U weights.
lrate=0.0002  # Learning rate for sgd (not used for adadelta and rmsprop).
erate = 0.001  # multiplier for the entropy regularization.
saveto='model.npz'  # The best model will be saved there.
validFreq=5000  # Compute the validation error after this number of update.
saveFreq=5000  # Save the parameters after every saveFreq updates.
batch_size_train=16  # The batch size during training.
batch_size_pred=4  # The batch size during training.
max_hops_train = 2 # maximum number of pages to be visited before giving up - training.
max_hops_pred = 4 # maximum number of pages to be visited before giving up - prediction.
supervised=True # Use supervised learning. False, True. For supervised > 1, Supervised and RL will alternate between training.
# in this case, <supervised> define the decay rate for supervised mode.
encoder='LSTM' # valid options are 'LSTM' or 'RNN'.
#reload_model='/scratch/rfn216/test/run-7599482/model.npz'  # Path to a saved model we want to start from.
reload_model=False  # Path to a saved model we want to start from.
idb=False # use input-dependent baseline.
train_size=1000  # If >0, we keep only this number of train examples when measuring accuracy.
valid_size=1000  # If >0, we keep only this number of valid examples when measuring accuracy.
test_size=1000  # If >0, we keep only this number of test examples when measuring accuracy.
outpath = "out.log" # where to save the logs file.
normalize = False # If True, normalize the embeddings.
fixed_wemb = True # set to true if you don't want to learn the word embedding weights.
k = 4 # beam search width. Used in prediction only.
attention = True # if True, use attention mechanism.
