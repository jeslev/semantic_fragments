import os
import json

openfile="../_experiments"

configs = {
    "rob_raw": "", #rob_raw
    "35k.s1.e1.o1.t1": "/gpfsscratch/rech/evd/uwe77wt/neurips/roberta-large/35k.s1.e1.o1.t1/lot_exp/model/model.tar.gz", #rob neurips
    #sh2_rnd
    "35k.s3.e3.o1.t1.sh2": "/gpfsscratch/rech/evd/uwe77wt/neurips/roberta-large/35k.s3.e3.o1.t1/sh2_rnd/model/model.tar.gz",
    "35k.s3.e3.o1.t3.sh2": "/gpfsscratch/rech/evd/uwe77wt/neurips/roberta-large/35k.s3.e3.o1.t3/sh2_rnd/model/model.tar.gz",
    "35k.s1.e1.o1.t1.sh2": "/gpfsscratch/rech/evd/uwe77wt/neurips/roberta-large/35k.s1.e1.o1.t1/sh2_rnd/model/model.tar.gz",
    #sh3_rnd_t 
    "35k.s1.e4.o2.t1.sh3t.p7": "/gpfsscratch/rech/evd/uwe77wt/neurips/roberta-large/35k.s1.e4.o2.t1/sh3_rnd_t/p7/model/model.tar.gz", # p7
    "35k.s3.e1.o1.t1.sh3t.p7": "/gpfsscratch/rech/evd/uwe77wt/neurips/roberta-large/35k.s3.e1.o1.t1/sh3_rnd_t/p7/model/model.tar.gz",# -> p7
    "35k.s1.e1.o1.t1.sh3t.p3": "/gpfsscratch/rech/evd/uwe77wt/neurips/roberta-large/35k.s1.e1.o1.t1/sh3_rnd_t/p3/model/model.tar.gz",#
    "35k.s1.e1.o1.t1.sh3t.p6": "/gpfsscratch/rech/evd/uwe77wt/neurips/roberta-large/35k.s1.e1.o1.t1/sh3_rnd_t/p6/model/model.tar.gz",#-> p3, p6
}

for k,v in configs.items():
    tmp_res = []
    for ds in ["definitions", "hypernymy", "hyponymy", "synonymy", "dictionary_qa"]:
        datafile = os.path.join(openfile,k,"result_"+ds+".jsonl")
        cnt, pos = 0, 0
        print("Reading:", datafile)
        try:
            with open(datafile,"r") as f:
                for line in f.readlines():
                    _json = json.loads(line)
                    ans, cans = _json["answer_index"], _json["correct_answer_index"]
                    if ans == cans:
                        pos+=1
                    cnt+=1
                print(pos,cnt)
                result =  round(100.0*pos/cnt,2)
                tmp_res.append(result)
        except:
            print(datafile, " not found")
    print(tmp_res)

