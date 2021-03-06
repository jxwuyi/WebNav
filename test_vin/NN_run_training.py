from NNobj import *
import vin_wiki_sanity as vn_san
import vin_wiki_fast as vn
import vin_baseline as bsl
import vin_baseline2 as bsl2
import vin_wiki_approx as vn_apx
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
                        choices=["dense1", "dense2", "dense3", "conv",
                                 "WikiBaseline","WikiBaseline2", "WikiApprox","valIterWiki",
                                 "valIterMultiBatch", "valIterBatch", "CBvalIterBatch",
                                 "valIterMars", "valIterMarsSingle"],
                        default="valIterWiki")
    parser.add_argument("--unittest", action="store_true")
    parser.add_argument("--grad_check", action="store_true")
    parser.add_argument("--devtype", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--warmstart", default="None")
    parser.add_argument("--reg", type=float, default=.0)
    parser.add_argument("--imsize", type=int, default=28)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--A", type=int, default=300)
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--maxhops", type=int, default=1)
    parser.add_argument("--stepdecreaserate", type=float, default=1.0)
    parser.add_argument("--stepdecreasetime", type=int, default=10000)
    parser.add_argument("--reportgap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sanity_check", default="None")
    parser.add_argument("--pretrain_vin", default="None")
    parser.add_argument("--pretrain_bsl", default="None")
    args = parser.parse_args()

    if (args.sanity_check != "None" and args.model == "valIterWiki"):
        # VI network Sanity Check
        my_nn = vn_san.vin(model=args.model, N = prm.total_pages, D=prm.max_links_per_page,
                    emb_dim = prm.dim_emb, dropout=args.dropout,
                    devtype=args.devtype, grad_check=args.grad_check, reg=args.reg,
                    k=args.k, A=args.A,
                    batchsize=args.batchsize, maxhops=args.maxhops)
    elif (args.model == "valIterWiki"):
        # VI network
        my_nn = vn.vin(model=args.model, N = prm.total_pages, D=prm.max_links_per_page,
                    emb_dim = prm.dim_emb, dropout=args.dropout,
                    devtype=args.devtype, grad_check=args.grad_check, reg=args.reg,
                    k=args.k, A=args.A,
                    batchsize=args.batchsize, maxhops=args.maxhops)
    elif (args.model == "WikiBaseline"):
        # Baseline model
        my_nn = bsl.vin(model=args.model, emb_dim=prm.dim_emb, devtype=args.devtype,
                        batchsize=args.batchsize, seed=args.seed,
                        report_gap = args.reportgap)
    elif (args.model == "WikiBaseline2"):
        # Baseline model with augumented training/testing data
        my_nn = bsl2.vin(model=args.model, emb_dim=prm.dim_emb, devtype=args.devtype,
                        batchsize=args.batchsize, seed=args.seed,
                        report_gap = args.reportgap)
    elif (args.model == "WikiApprox"):
        # Baseline model with augumented training/testing data
        my_nn = vn_apx.vin(model=args.model, emb_dim=prm.dim_emb, devtype=args.devtype,
                           batchsize=args.batchsize, seed=args.seed,
                           report_gap = args.reportgap, k = args.k)
        
    if args.pretrain_vin != "None":
        print('loading pretrained vin weights')
        my_nn.load_pretrained_vin(args.pretrain_vin)
    if args.pretrain_bsl != "None":
        print('loading pretrained baseline weights')
        my_nn.load_pretrained_bsl(args.pretrain_bsl)
    if args.warmstart != "None":
        print('warmstarting...')
        my_nn.load_weights(args.warmstart)

    try:
        if args.sanity_check != "None":
            my_nn.run_training_sanity_check(stepsize=args.stepsize,  epochs=args.epochs)
        else:
            my_nn.run_training(stepsize=args.stepsize, epochs=args.epochs,
                           grad_check=args.grad_check)

    except KeyboardInterrupt:
        print "Training interupted!!!"
    
    my_nn.save_weights(outfile=str(args.output))

if __name__ == "__main__":
    main()
