"""NumPy BPR-MF and NCF with dense Adam and finite-difference gradient checks."""
import numpy as np
from scipy.special import expit

class Network:
    def __init__(self, nu, ni, kind, seed=42, dtype='float32', layout='io', sigmoid='expit', label64=False, combine_user=False):
        self.kind=kind;self.layout=layout;self.dtype=np.dtype(dtype);self.sigmoid=sigmoid;self.label64=label64;self.combine_user=combine_user
        self.dim=32 if kind=='BPR-MF' else 16
        rng=np.random.default_rng(seed);std=.05 if kind=='BPR-MF' else 1.
        self.p={'U':rng.normal(0,std,(nu,self.dim)).astype(dtype),
                'V':rng.normal(0,std,(ni,self.dim)).astype(dtype)}
        if kind=='NCF':
            for j,(a,b) in enumerate([(32,64),(64,32),(32,1)]):
                shape=(a,b) if layout=='io' else (b,a)
                w=rng.uniform(-1/np.sqrt(a),1/np.sqrt(a),shape).astype(dtype)
                self.p[f'W{j}']=w if layout=='io' else w.T.copy()
                self.p[f'b{j}']=rng.uniform(-1/np.sqrt(a),1/np.sqrt(a),b).astype(dtype)

    def loss_grad(self, u, pos, neg):
        p=self.p;n=len(u);g={k:np.zeros_like(v) for k,v in p.items()}
        if self.kind=='BPR-MF':
            ue=p['U'][u];vi=p['V'][pos];vj=p['V'][neg]
            delta=np.sum(ue*(vi-vj),axis=1)
            loss=float(np.logaddexp(0,-delta).mean())
            factor=(-expit(-delta)/n)[:,None]
            np.add.at(g['U'],u,factor*(vi-vj))
            np.add.at(g['V'],pos,factor*ue);np.add.at(g['V'],neg,-factor*ue)
        else:
            uu=np.concatenate([u,u]);ii=np.concatenate([pos,neg])
            y=np.concatenate([np.ones(n),np.zeros(n)]).astype('float64' if self.label64 else self.dtype)
            x=np.concatenate([p['U'][uu],p['V'][ii]],axis=1)
            z0=x@p['W0']+p['b0'];h0=np.maximum(z0,0)
            z1=h0@p['W1']+p['b1'];h1=np.maximum(z1,0)
            logit=(h1@p['W2']+p['b2']).ravel()
            loss=float((np.logaddexp(0,logit)-y*logit).mean())
            prob=expit(logit) if self.sigmoid=='expit' else 1/(1+np.exp(-np.clip(logit,-40,40)))
            dz=((prob-y)/(2*n))[:,None]
            g['W2']=h1.T@dz;g['b2']=dz.sum(axis=0)
            dz1=(dz@p['W2'].T)*(z1>0)
            g['W1']=h0.T@dz1;g['b1']=dz1.sum(axis=0)
            dz0=(dz1@p['W1'].T)*(z0>0)
            g['W0']=x.T@dz0;g['b0']=dz0.sum(axis=0)
            dx=dz0@p['W0'].T
            if self.combine_user:np.add.at(g['U'],u,dx[:n,:self.dim]+dx[n:,:self.dim])
            else:np.add.at(g['U'],uu,dx[:,:self.dim])
            np.add.at(g['V'],ii,dx[:,self.dim:])
        return loss,g

    def score(self, uid, start=0, end=None):
        p=self.p;end=len(p['V']) if end is None else end
        if self.kind=='BPR-MF':return np.sum(p['U'][uid]*p['V'][start:end],axis=1)
        x=np.concatenate([np.broadcast_to(p['U'][uid],(end-start,self.dim)),p['V'][start:end]],axis=1)
        h=np.maximum(x@p['W0']+p['b0'],0);h=np.maximum(h@p['W1']+p['b1'],0)
        return (h@p['W2']+p['b2']).ravel()

def fit(network, arr, ni, epochs=2, seed=42, batch_size=4096, callback=None, square_first=False, state64=False):
    p=network.p;rng=np.random.default_rng(seed)
    seen={}
    for u,i in arr:seen.setdefault(int(u),set()).add(int(i))
    first={k:np.zeros_like(v,dtype='float64' if state64 else v.dtype) for k,v in p.items()};second={k:np.zeros_like(v,dtype='float64' if state64 else v.dtype) for k,v in p.items()}
    t=0;logs=[];wd=1e-5 if network.kind=='BPR-MF' else 0.
    for ep in range(epochs):
        order=rng.permutation(len(arr));total=0.
        for st in range(0,len(arr),batch_size):
            b=arr[order[st:st+batch_size]];u,pos=b[:,0],b[:,1]
            neg=rng.integers(ni,size=len(u));bad=np.array([int(i) in seen[int(a)] for a,i in zip(u,neg)])
            while bad.any():
                neg[bad]=rng.integers(ni,size=int(bad.sum()));bad=np.array([int(i) in seen[int(a)] for a,i in zip(u,neg)])
            loss,g=network.loss_grad(u,pos,neg);total+=loss*len(b);t+=1
            for k in p:
                grad=g[k]+wd*p[k] if wd else g[k]
                first[k]*=.9;first[k]+=.1*grad
                second[k]*=.999;second[k]+=.001*(grad*grad) if square_first else .001*grad*grad
                p[k]-=.001*(first[k]/(1-.9**t))/(np.sqrt(second[k]/(1-.999**t))+1e-8)
        log={'epoch':ep+1,'mean_loss':total/len(arr)};logs.append(log)
        if callback:callback(log)
    return logs

def gradient_checks():
    rng=np.random.default_rng(21);results=[]
    for kind in ['BPR-MF','NCF']:
        m=Network(4,7,kind,dtype='float64')
        u=np.array([0,1,1]);pos=np.array([0,2,3]);neg=np.array([4,5,6])
        loss,g=m.loss_grad(u,pos,neg);worst=0.
        for k in m.p:
            for j in rng.choice(m.p[k].size,size=min(8,m.p[k].size),replace=False):
                idx=np.unravel_index(j,m.p[k].shape);v=m.p[k][idx];eps=1e-6
                m.p[k][idx]=v+eps;a=m.loss_grad(u,pos,neg)[0]
                m.p[k][idx]=v-eps;b=m.loss_grad(u,pos,neg)[0]
                m.p[k][idx]=v
                error=abs((a-b)/(2*eps)-g[k][idx]);worst=max(worst,error)
        assert worst<1e-6,(kind,worst)
        results.append({'model':kind,'maximum_absolute_gradient_error':worst})
    return results

if __name__=='__main__':print(gradient_checks())
