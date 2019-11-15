import os 

def outcfg(shellpath,nc=1, kmeans_path=None):
    if kmeans_path!=None:
        with open(kmeans_path,"r") as f:
            line =  f.readline()
        print(line)
        ll =  line.split(',')
        for idx, l in enumerate(ll):
            ll[idx] = ll[idx].strip()
        out = ','
        out = out.join(ll)
        os.system('{} {} {}'.format(shellpath,nc,out))
    else:
        os.system('{} {}'.format(shellpath, nc))