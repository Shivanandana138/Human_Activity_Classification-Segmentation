import argparse
from pathlib import Path
from data import load_data, load_features, group_by_subject

p=argparse.ArgumentParser()
p.add_argument("--data_dir",required=True)
a=p.parse_args()
(trX,try_,trs),(teX,tey,tes)=load_data(a.data_dir)
features=load_features(a.data_dir)
print("TRAIN:",trX.shape, try_.shape, trs.shape)
print("TEST :",teX.shape, tey.shape, tes.shape)
print("FEATURES:",len(features))
print("Subjects train:",sorted(set(trs.tolist())))
print("Subjects test :",sorted(set(tes.tolist())))
print("First 10 feature names:")
for x in features[:10]: print(" ",x)
for split,(X,y,s) in [("train",(trX,try_,trs)),("test",(teX,tey,tes))]:
    seqs=group_by_subject(X,y,s)
    print(split,"subject 1 frames:",len(seqs[0]["y"]))
    print(split,"subject 1 first labels:",seqs[0]["y"][:20].tolist())
