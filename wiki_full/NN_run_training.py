from NNobj import *
import vin_webnav as vn
import vin_webnav_fast as vnf
import vin_webnav_baseline as bsl
import vin_webnav_combine as cmb
import vin_webnav_combine_test as cmbt
import myparameters as prm


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="None")
    parser.add_argument("--output", default="None")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--dropout", action="store_true")
    parser.add_argument("--stepsize", type=float, default=.0002)
    parser.add_argument("--model",
                        choices=["dense1", "dense2", "dense3", "conv", "WikiBaseLine", "WikiCombine", "WikiCombineTest","valIterWiki", "valIterWebNav","valIterWebNavFast", "valIterBatch", "CBvalIterBatch", "valIterMars", "valIterMarsSingle"],
                        default="valIterWebNav")
    parser.add_argument("--unittest", action="store_true")
    parser.add_argument("--grad_check", action="store_true")
    parser.add_argument("--devtype", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--warmstart", default="None")
    parser.add_argument("--reg", type=float, default=.0)
    parser.add_argument("--imsize", type=int, default=28)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--A", type=int, default=300)
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--maxhops", type=int, default=1)
    parser.add_argument("--stepdecreaserate", type=float, default=1.0)
    parser.add_argument("--stepdecreasetime", type=int, default=10000)
    parser.add_argument("--sanity_check", default="None")
    args = parser.parse_args()

    if (args.model == "valIterWebNav"):
        # VI network
        my_nn = vn.vin_web(model=args.model, N = prm.total_pages, D=prm.max_links_per_page,
                    emb_dim = prm.dim_emb, dropout=args.dropout,
                    devtype=args.devtype, grad_check=args.grad_check, reg=args.reg,
                    k=args.k, seed = args.seed)
    elif (args.model == "valIterWebNavFast"):
        my_nn = vnf.vin_web(model=args.model, N = prm.total_pages, D=prm.max_links_per_page,
                    emb_dim = prm.dim_emb, dropout=args.dropout,
                    devtype=args.devtype, grad_check=args.grad_check, reg=args.reg,
                    k=args.k, seed = args.seed)
    elif (args.model == "WikiBaseLine"):
        my_nn = bsl.vin_web(model=args.model,
                    emb_dim = prm.dim_emb, dropout=args.dropout,
                    devtype=args.devtype, grad_check=args.grad_check, reg=args.reg,
                    seed = args.seed, batchsize = batchsize)
    elif (args.model == "WikiCombine"):
        my_nn = cmb.vin_web(model=args.model, N = prm.total_pages, D=prm.max_links_per_page,
                    emb_dim = prm.dim_emb, dropout=args.dropout,
                    devtype=args.devtype, grad_check=args.grad_check, reg=args.reg,
                    k=args.k, seed = args.seed)
        my_nn.load_pretrained()
    elif (args.model == "WikiCombineTest"):
        my_nn = cmbt.vin_web(model=args.model, N = prm.total_pages,
                    emb_dim = prm.dim_emb, dropout=args.dropout,
                    devtype=args.devtype, grad_check=args.grad_check, reg=args.reg,
                    k=args.k, seed = args.seed, batchsize = batchsize)
        my_nn.load_pretrained()
        
        
        
        
    if args.warmstart != "None":
        print('warmstarting...')
        my_nn.load_weights(args.warmstart)

    try:
    
        my_nn.run_training(stepsize=args.stepsize, epochs=args.epochs,
                           grad_check=args.grad_check)
    except KeyboardInterrupt:
        print "Training interupted!!!"

    print "Saving Weights ..."
        
    my_nn.save_weights(outfile=str(args.output))

if __name__ == "__main__":
    main()
