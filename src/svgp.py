
"""
Sparse Variational Gaussian Process (SVGP) regression benchmark.

This is a self-contained PyTorch implementation of the inducing-point
variational GP of Hensman et al. It is included as the additional modern
benchmark requested by Reviewer 1.

Model:
    f ~ GP(0, k_ARD-RBF)
    y | f ~ N(f, sigma_n^2)
    q(u) = N(m, S),  u = f(Z)

The Gaussian-likelihood ELBO is evaluated analytically. Inducing locations,
kernel hyperparameters, likelihood noise, and q(u) are optimized jointly.

This benchmark is intentionally independent of GPyTorch so the revision code
runs in a minimal PyTorch environment.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG2PI = math.log(2.0*math.pi)

def inv_softplus(x):
    x=torch.as_tensor(x,dtype=torch.get_default_dtype())
    return torch.log(torch.expm1(x))

def sqdist_ard(x1,x2,lengthscale):
    x1=torch.as_tensor(x1,dtype=lengthscale.dtype,device=lengthscale.device)
    x2=torch.as_tensor(x2,dtype=lengthscale.dtype,device=lengthscale.device)
    a=x1/lengthscale
    b=x2/lengthscale
    return torch.cdist(a,b,p=2.0)**2

def rbf_kernel(x1,x2,lengthscale,outputscale):
    return outputscale*torch.exp(-0.5*sqdist_ard(x1,x2,lengthscale))

def safe_chol(K,jitter=1e-6,max_tries=8):
    eye=torch.eye(K.shape[-1],dtype=K.dtype,device=K.device)
    j=jitter
    for _ in range(max_tries):
        L,info=torch.linalg.cholesky_ex(K+j*eye)
        if int(info.max().item())==0:
            return L,j
        j*=10
    raise RuntimeError("SVGP Cholesky failure")

class SparseVariationalGP(nn.Module):
    def __init__(self, inducing_points, jitter=1e-6):
        super().__init__()
        Z=torch.as_tensor(inducing_points,dtype=torch.get_default_dtype())
        self.Z=nn.Parameter(Z.clone())
        M,D=Z.shape

        self.raw_lengthscale=nn.Parameter(inv_softplus(torch.full((D,),0.35)))
        self.raw_outputscale=nn.Parameter(inv_softplus(torch.tensor(1.0)))
        self.raw_noise=nn.Parameter(inv_softplus(torch.tensor(0.15)))

        self.q_mu=nn.Parameter(torch.zeros(M))
        # Cholesky parameter of q(u); initialize near prior scale.
        self.q_L_raw=nn.Parameter(torch.eye(M)*0.10)
        self.jitter=jitter

    @property
    def lengthscale(self):
        return F.softplus(self.raw_lengthscale)+1e-4

    @property
    def outputscale(self):
        return F.softplus(self.raw_outputscale)+1e-5

    @property
    def noise(self):
        return F.softplus(self.raw_noise)+1e-5

    def q_chol(self):
        A=torch.tril(self.q_L_raw)
        diag=F.softplus(torch.diagonal(A))+1e-4
        A=A-torch.diag(torch.diagonal(A))+torch.diag(diag)
        return A

    def prior_terms(self):
        Kmm=rbf_kernel(self.Z,self.Z,self.lengthscale,self.outputscale)
        Lmm,_=safe_chol(Kmm,self.jitter)
        return Kmm,Lmm

    def qf_marginal(self,x):
        Kmm,Lmm=self.prior_terms()
        Knm=rbf_kernel(x,self.Z,self.lengthscale,self.outputscale)
        # A = K_nm K_mm^{-1}
        A=torch.cholesky_solve(Knm.T,Lmm).T

        Lq=self.q_chol()
        S=Lq@Lq.T
        mean=A@self.q_mu

        # diag[K_nn + A(S-Kmm)A^T]
        prior_diag=self.outputscale.expand(x.shape[0])
        correction=torch.sum((A@(S-Kmm))*A,dim=1)
        var=torch.clamp(prior_diag+correction,min=1e-8)
        return mean,var

    def kl_qu_pu(self):
        Kmm,Lmm=self.prior_terms()
        Lq=self.q_chol()
        S=Lq@Lq.T
        Kinv_m=torch.cholesky_solve(self.q_mu[:,None],Lmm).squeeze(1)
        quad=torch.dot(self.q_mu,Kinv_m)
        Kinv_S=torch.cholesky_solve(S,Lmm)
        trace=torch.trace(Kinv_S)
        logdetK=2*torch.log(torch.diagonal(Lmm)).sum()
        logdetS=2*torch.log(torch.diagonal(Lq)).sum()
        M=self.q_mu.numel()
        return 0.5*(trace+quad-M+logdetK-logdetS)

    def negative_elbo(self,y,x):
        mean,var=self.qf_marginal(x)
        noise2=self.noise**2
        ell=-0.5*torch.sum(
            LOG2PI+torch.log(noise2)+((y-mean)**2+var)/noise2
        )
        kl=self.kl_qu_pu()
        return -(ell-kl), {
            "ell":float(ell.detach()),
            "kl":float(kl.detach()),
            "noise":float(self.noise.detach()),
            "outputscale":float(self.outputscale.detach()),
            "mean_lengthscale":float(self.lengthscale.mean().detach()),
        }

    @torch.no_grad()
    def predictive_samples(self,x,n_draws=1000):
        mean,var=self.qf_marginal(x)
        total_var=var+self.noise**2
        eps=torch.randn(n_draws,x.shape[0],dtype=x.dtype,device=x.device)
        return (mean[None,:]+torch.sqrt(total_var)[None,:]*eps).cpu().numpy()

def choose_inducing(x,m,seed=1):
    """Deterministic k-means-like subset selection without sklearn dependency."""
    rng=np.random.default_rng(seed)
    X=np.asarray(x)
    m=min(m,len(X))
    # Farthest-point initialization gives good spatial coverage.
    chosen=[int(rng.integers(len(X)))]
    d2=np.sum((X-X[chosen[0]])**2,axis=1)
    for _ in range(1,m):
        j=int(np.argmax(d2))
        chosen.append(j)
        d2=np.minimum(d2,np.sum((X-X[j])**2,axis=1))
    return X[chosen]

def fit_svgp(model,y,x,iterations=1500,lr=0.01,print_every=0,
             early_stop_min_iter=900,early_stop_window=200,
             early_stop_check_every=100,early_stop_rel_slope=1e-4,
             early_stop_cv=0.03):
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    milestones=[max(1,int(.5*iterations)),max(2,int(.8*iterations))]
    sched=torch.optim.lr_scheduler.MultiStepLR(opt,milestones=milestones,gamma=.25)
    trace=[]
    for it in range(1,iterations+1):
        opt.zero_grad(set_to_none=True)
        loss,info=model.negative_elbo(y,x)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"SVGP non-finite loss at {it}")
        loss.backward()
        gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),50.0))
        opt.step();sched.step()
        trace.append({
            "iteration":it,"negative_elbo":float(loss.detach()),
            "grad_norm":gn,"learning_rate":opt.param_groups[0]["lr"],**info
        })
        if print_every and it%print_every==0:
            print(f"SVGP iter={it} negELBO={float(loss.detach()):.3f}")
        if it>=early_stop_min_iter and it%early_stop_check_every==0:
            w=min(early_stop_window,len(trace))
            z=np.array([r["negative_elbo"] for r in trace[-w:]],dtype=float)
            if len(z)>=max(50,early_stop_window//2):
                x_idx=np.arange(len(z),dtype=float)
                slope=float(np.polyfit(x_idx,z,1)[0])
                mean=float(np.mean(z)); sd=float(np.std(z,ddof=1))
                rel=abs(slope)/(abs(mean)+1e-12)
                cv=sd/(abs(mean)+1e-12)
                if rel<early_stop_rel_slope and cv<early_stop_cv:
                    trace[-1]["early_stopped"]=1
                    break
    return trace
