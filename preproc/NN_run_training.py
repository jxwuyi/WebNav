from NNobj import *
import vin_preproc as vin
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
                        choices=["dense1", "dense2", "dense3", "conv", "WikiBaseLine", "WikiProj","WikiProjAtt","WikiProjAtt2","WikiProjSanityFull","WikiProjSanity","WikiProjSanityBias","WikiProjSanity3","WikiProjSanity2","WikiProjSG", "WikiCombine","WikiCombineJoint","WikiCombineSanity", "WikiCombineTest","WikiCombineTest2","valIterWiki", "valIterWebNav","valIterWebNavFast", "valIterBatch", "CBvalIterBatch", "valIterMars", "valIterMarsSingle"],
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
    parser.add_argument("--reportgap", type=int, default=100)
    args = parser.parse_args()

    my_nn = vin.vin_web(N = prm.total_pages, D=prm.max_links_per_page,
                    emb_dim = prm.dim_emb, 
                    devtype=args.devtype, 
                    k=args.k, report_gap=args.reportgap)
    my_nn.load_pretrained()
    try:
    
        my_nn.precompute(epochs=args.epochs, output = str(args.output))
        
    except KeyboardInterrupt:
        print "Training interupted!!!"

if __name__ == "__main__":
    main()
