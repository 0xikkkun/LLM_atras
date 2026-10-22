"""CPU geometry primitives. Importing this module does not import PyTorch."""
import numpy as np

def upper(D):
    return D[np.triu_indices(len(D),1)]

def normalize(S):
    scale = upper(S).mean()
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Distance scale must be positive and finite")
    return S/scale

def pairwise_from_features(X):
    from scipy.spatial.distance import pdist, squareform
    return squareform(pdist(np.asarray(X)))

def cka_distance(features):
    grams=[]
    for X in features:
        X=X-X.mean(0)
        K=X@X.T
        assert np.linalg.norm(K)>0
        grams.append((K/np.linalg.norm(K)).ravel())
    return pairwise_from_features(grams)

def graph_distances(X, metrics, k):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components, dijkstra
    from scipy.spatial.distance import cdist
    X=np.asarray(X, dtype=float); metrics=np.asarray(metrics, dtype=float)
    if X.ndim != 2 or len(X) < 2 or not 1 <= k < len(X):
        raise ValueError("Expected a 2D feature array and 1 <= k < n")
    if metrics.shape != (len(X), X.shape[1], X.shape[1]):
        raise ValueError("Local metric shape must be (n, d, d)")
    E=cdist(X,X); np.fill_diagonal(E,np.inf)
    ix=np.argsort(E,axis=1)[:,:k]
    A=np.zeros(E.shape,dtype=bool)
    A[np.arange(len(X))[:,None],ix]=True
    A|=A.T
    ncomp=connected_components(csr_matrix(A),directed=False)[0]
    if ncomp!=1:
        return None, ncomp
    a,b=np.where(A)
    delta=X[a]-X[b]
    q=np.einsum('ni,nij,nj->n',delta,(metrics[a]+metrics[b])/2,delta)
    weights={'euclidean':np.linalg.norm(delta,axis=1), 'fisher':np.sqrt(np.maximum(q,0))}
    result={}
    for name,w in weights.items():
        assert np.all(w>0) and np.isfinite(w).all()
        S=dijkstra(csr_matrix((w,(a,b)),shape=A.shape),directed=False)
        assert np.isfinite(S).all() and np.allclose(S,S.T)
        result[name]=dict(S=S, a=a, b=b, weights=w)
    return result, ncomp

def rank_distance(matrices):
    from scipy.stats import rankdata
    rows=[]
    for S in matrices:
        v=rankdata(upper(S)); v-=v.mean()
        assert np.linalg.norm(v)>0
        rows.append(v/np.linalg.norm(v))
    return pairwise_from_features(rows)

def knn_distance(matrices, k=10):
    k = min(k, len(matrices[0]) - 1)
    if k < 1:
        raise ValueError("At least two samples and a positive k are required")
    vectors=[]
    for S in matrices:
        T=S.copy(); np.fill_diagonal(T,np.inf)
        ix=np.argsort(T,axis=1)[:,:k]
        A=np.zeros(T.shape,dtype=np.float64)
        A[np.arange(len(T))[:,None],ix]=1
        vectors.append(A.ravel())
    # 1-overlap equals normalized L1 on fixed-size adjacency sets.
    from scipy.spatial.distance import pdist,squareform
    return squareform(pdist(vectors,metric='cityblock'))/(2*len(matrices[0])*k)

def netlsd_vector(graph):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import laplacian
    a,b,w=graph['a'],graph['b'],graph['weights']
    scale=np.median(w)
    affinity=np.exp(-0.5*(w/scale)**2)
    A=csr_matrix((affinity,(a,b)),shape=graph['S'].shape)
    L=laplacian(A,normed=True).toarray()
    eigen=np.maximum(np.linalg.eigvalsh(L),0)
    times=np.logspace(-2,2,128)
    return np.exp(-times[:,None]*eigen[None,:]).mean(1)

def classical(D):
    J=np.eye(len(D))-np.ones(D.shape)/len(D)
    B=-0.5*J@D**2@J
    val,vec=np.linalg.eigh(B); order=np.argsort(val)[::-1][:2]
    return vec[:,order]*np.sqrt(np.maximum(val[order],0))

def align(X,Y,rotation_only=False):
    X=X-X.mean(0); Y=Y-Y.mean(0)
    U,s,Vt=np.linalg.svd(X.T@Y)
    correction=np.eye(2)
    if rotation_only and np.linalg.det(U@Vt)<0:
        correction[-1,-1]=-1
    R=U@correction@Vt
    Z=X@R
    assert np.allclose(pairwise_from_features(X),pairwise_from_features(Z),atol=1e-8)
    return Z,R,float(np.sqrt(np.mean((Z-Y)**2)))

def assess(D,Y):
    from scipy.spatial.distance import cdist
    from scipy.stats import spearmanr
    Z=cdist(Y,Y); d,z=upper(D),upper(Z)
    scale=float(d@z/(z@z)) if z@z else 1.
    stress=float(np.sqrt(np.sum((d-scale*z)**2)/np.sum(d*d)))
    k=min(3,len(D)-1)
    a=D.copy();b=Z.copy();np.fill_diagonal(a,np.inf);np.fill_diagonal(b,np.inf)
    ia=np.argsort(a,axis=1)[:,:k];ib=np.argsort(b,axis=1)[:,:k]
    overlap=np.mean([len(set(x)&set(y))/k for x,y in zip(ia,ib)])
    return dict(stress_scale_fitted=stress,spearman=float(spearmanr(d,z).statistic),
                model_knn_k=k,model_knn_overlap=float(overlap))

def metric_audit(D):
    errors=D[:,:,None]-D[:,None,:]-D.T[None,:,:]
    mask=np.ones(errors.shape,dtype=bool)
    for i in range(len(D)):
        mask[i,i,:]=False;mask[i,:,i]=False;mask[:,i,i]=False
    violations=errors[mask]>1e-8*max(1,float(D.max()))
    return dict(finite=bool(np.isfinite(D).all()),symmetric=bool(np.allclose(D,D.T)),
        nonnegative=bool(D.min()>=-1e-10),diagonal_max=float(abs(np.diag(D)).max()),
        triangle_violations=int(violations.sum()),triangle_tests=int(mask.sum()),
        triangle_max_excess=float(max(0,errors.max())))

