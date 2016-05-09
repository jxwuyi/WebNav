from NNobj import *
import vin_webnav as vn
import vin_webnav_baseline as bsl
import vin_webnav_combine_sanity as cmb_san
import vin_webnav_combine_test2 as cmbt2
import vin_webnav_combine_test2_new as cmbt2n
import vin_webnav_combine_test4 as cmbt4
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
                        choices=["dense1", "dense2", "dense3", "conv", "WikiBaseLine", "WikiProj","WikiProjAtt","WikiProjAtt2","WikiProjSanityFull","WikiProjSanity","WikiProjSanityBias","WikiProjSanity3","WikiProjSanity2","WikiProjSG", "WikiCombine","WikiCombineJoint","WikiCombineSanity", "WikiCombineTest","WikiCombineTest2","WikiCombineTest2New","WikiCombineTest3","WikiCombineTest4","valIterWiki", "valIterWebNav","valIterWebNavFast", "valIterBatch", "CBvalIterBatch", "valIterMars", "valIterMarsSingle"],
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
    parser.add_argument("--reportgap", type=int, default=50000)
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
                    seed = args.seed, batchsize = args.batchsize,
                    reportgap=args.reportgap)
    elif (args.model == "WikiCombine"):
        my_nn = cmb.vin_web(model=args.model, N = prm.total_pages, D=prm.max_links_per_page,
                    emb_dim = prm.dim_emb, dropout=args.dropout,
                    devtype=args.devtype, grad_check=args.grad_check, reg=args.reg,
                    k=args.k, seed = args.seed)
        my_nn.load_pretrained()
    elif (args.model == "WikiCombineSanity"):
        my_nn = cmb_san.vin_web(model=args.model, N = prm.total_pages, D=prm.max_links_per_page,
                    emb_dim = prm.dim_emb, dropout=args.dropout,
                    devtype=args.devtype, grad_check=args.grad_check, reg=args.reg,
                    seed = args.seed)
        my_nn.load_pretrained()
    elif (args.model == "WikiCombineJoint"):
        my_nn = cmb_jt.vin_web(model=args.model, N = prm.total_pages, D=prm.max_links_per_page,
                    emb_dim = prm.dim_emb, dropout=args.dropout,
                    devtype=args.devtype, grad_check=args.grad_check, reg=args.reg,
                    seed = args.seed)
        if (args.warmstart == "None"):
            my_nn.load_pretrained()
    elif (args.model == "WikiCombineTest"):
        my_nn = cmbt.vin_web(model=args.model, N = prm.total_pages,
                    emb_dim = prm.dim_emb, dropout=args.dropout,
                    devtype=args.devtype, grad_check=args.grad_check, reg=args.reg,
                    k=args.k, seed = args.seed, batchsize = args.batchsize)
        if (args.warmstart == "None"):
            my_nn.load_pretrained()
        else:
            my_nn.load_pretrained(bsl_file = 'NA')
    elif (args.model == "WikiCombineTest2"):
        my_nn = cmbt2.vin_web(model=args.model, N = prm.total_pages,
                    emb_dim = prm.dim_emb, dropout=args.dropout,
                    devtype=args.devtype, grad_check=args.grad_check, reg=args.reg,
                    k=args.k, seed = args.seed, batchsize = args.batchsize,
                    report_gap = args.reportgap)
        my_nn.load_pretrained()
    elif (args.model == "WikiCombineTest2New"):
        my_nn = cmbt2n.vin_web(model=args.model, N = prm.total_pages,
                    emb_dim = prm.dim_emb, dropout=args.dropout,
                    devtype=args.devtype, grad_check=args.grad_check, reg=args.reg,
                    k=args.k, seed = args.seed, batchsize = args.batchsize,
                    report_gap = args.reportgap)
        my_nn.load_pretrained()
    elif (args.model == "WikiCombineTest3"):
        my_nn = cmbt3.vin_web(model=args.model, N = prm.total_pages,
                    emb_dim = prm.dim_emb, dropout=args.dropout,
                    devtype=args.devtype, grad_check=args.grad_check, reg=args.reg,
                    k=args.k, seed = args.seed, batchsize = args.batchsize,
                    report_gap = args.reportgap)
    elif (args.model == "WikiCombineTest4"):
        my_nn = cmbt4.vin_web(model=args.model, N = prm.total_pages,
                    emb_dim = prm.dim_emb, dropout=args.dropout,
                    devtype=args.devtype, grad_check=args.grad_check, reg=args.reg,
                    k=args.k, seed = args.seed, batchsize = args.batchsize,
                    report_gap = args.reportgap)
    elif (args.model == "WikiProjSG"):
        my_nn = prj_sg.vin_web(model=args.model, N = prm.total_pages,
                    emb_dim = prm.dim_emb, dropout=args.dropout,
                    devtype=args.devtype, grad_check=args.grad_check, reg=args.reg,
                    k=args.k, seed = args.seed)
        my_nn.load_pretrained()
    elif (args.model == "WikiProj"):
        my_nn = prj.vin_web(model=args.model, N = prm.total_pages,
                    emb_dim = prm.dim_emb, dropout=args.dropout,
                    devtype=args.devtype, grad_check=args.grad_check, reg=args.reg,
                    k=args.k, seed = args.seed, batchsize = args.batchsize)
        my_nn.load_pretrained()
    elif (args.model == "WikiProjSanity"):
        my_nn = prj_san.vin_web(model=args.model, N = prm.total_pages,
                    emb_dim = prm.dim_emb, dropout=args.dropout,
                    devtype=args.devtype, grad_check=args.grad_check, reg=args.reg,
                    seed = args.seed, batchsize = args.batchsize)
    elif (args.model == "WikiProjSanityBias"):
        my_nn = prj_san_b.vin_web(model=args.model, N = prm.total_pages,
                    emb_dim = prm.dim_emb, dropout=args.dropout,
                    devtype=args.devtype, grad_check=args.grad_check, reg=args.reg,
                    seed = args.seed, batchsize = args.batchsize)
    elif (args.model == "WikiProjSanity2"):
        my_nn = prj_san2.vin_web(model=args.model, N = prm.total_pages,
                    emb_dim = prm.dim_emb, dropout=args.dropout,
                    devtype=args.devtype, grad_check=args.grad_check, reg=args.reg,
                    seed = args.seed, batchsize = args.batchsize)
    elif (args.model == "WikiProjSanity3"):
        my_nn = prj_san3.vin_web(model=args.model, N = prm.total_pages,
                    emb_dim = prm.dim_emb, dropout=args.dropout,
                    devtype=args.devtype, grad_check=args.grad_check, reg=args.reg,
                    seed = args.seed, batchsize = args.batchsize)
    elif (args.model == "WikiProjSanityFull"):
        my_nn = prj_sanF.vin_web(model=args.model, N = prm.total_pages,
                    emb_dim = prm.dim_emb, dropout=args.dropout,
                    devtype=args.devtype, grad_check=args.grad_check, reg=args.reg,
                    seed = args.seed, batchsize = args.batchsize)
    elif (args.model == "WikiProjAtt"):
        my_nn = prj_att.vin_web(model=args.model, N = prm.total_pages,
                    emb_dim = prm.dim_emb, dropout=args.dropout,
                    devtype=args.devtype, grad_check=args.grad_check, reg=args.reg,
                    seed = args.seed, batchsize = args.batchsize)
    elif (args.model == "WikiProjAtt2"):
        my_nn = prj_att2.vin_web(model=args.model, N = prm.total_pages,
                    emb_dim = prm.dim_emb, dropout=args.dropout,
                    devtype=args.devtype, grad_check=args.grad_check, reg=args.reg,
                    seed = args.seed, batchsize = args.batchsize)
        
        
        
        
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
