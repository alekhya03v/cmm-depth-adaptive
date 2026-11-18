import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows=[l.rstrip("\n").rsplit(",",6) for l in open("results/paper/tables/T9b_strength_sweep_with_best.csv")][1:]
U=[];D=[]
for v,st,avg,acc,vs,bwt,n in rows:
    a=float(acc)
    if v.startswith("Ours"):
        lo,hi=[float(x) for x in st.replace("→"," ").split()]
        D.append((round(hi-lo,2),a,f"{lo:.1f}→{hi:.1f}","slide" in v))
    else: U.append((float(avg),a))
B,O,K="#2a78d6","#eb6834","#52514e"
fig,(a,b)=plt.subplots(1,2,figsize=(11,4.8),dpi=200)
a.plot([x for x,_ in U],[y for _,y in U],"-o",color=B,lw=2,label="Uniform strength")
for t,y,l,s in D:
    a.scatter([1.8],[y],marker="s",s=70,zorder=3,facecolor="white" if s else O,edgecolor=O,lw=2)
    a.annotate(l,(1.8,y),xytext=(-10,3) if s else (8,-3),ha="right" if s else "left",textcoords="offset points",fontsize=9,color=K)
a.set_xlabel("Average strength s"); a.set_ylabel("Accuracy, %"); a.legend(frameon=False,loc="lower right")
a.set_title("Same average, higher accuracy when depth-aware",loc="left",fontsize=10.5)
u18=[y for x,y in U if x==1.8][0]; pts=[(0,u18,False)]+sorted([(t,y,s) for t,y,_,s in D])
b.plot([p[0] for p in pts],[p[1] for p in pts],"-",color=O,lw=2)
for x,y,s in pts:
    b.scatter([x],[y],marker="s",s=60,zorder=3,facecolor="white" if s else O,edgecolor=O,lw=2)
    b.annotate(f"{y:.2f}",(x,y),xytext=(0,9),textcoords="offset points",ha="center",fontsize=9,color=K)
b.set_xlabel("Tilt = s(layer 11) - s(layer 0), average 1.8"); b.set_ylabel("Accuracy, %")
b.set_title("A steeper slope helps, then flattens",loc="left",fontsize=10.5)
for ax in (a,b): ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",color="#e6e5e0"); ax.set_axisbelow(True); ax.set_ylim(74.8,78.9)
fig.suptitle("Strength sweep, ViT-B/32, 8 tasks, mean of 3 orders (T9)",x=0.06,ha="left",fontsize=12)
fig.tight_layout(rect=(0,0,1,0.95)); fig.savefig("results/paper/figs/F12_strength_sweep.png")
print(U,D)
