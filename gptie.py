import matplotlib.pyplot as plt
import numpy as np
from numpy import fft
from scipy.io import loadmat
from scipy.optimize import fsolve
from pycromanager import Dataset

class GPTIE:
    def __init__(self,datapath,analpath,prefix):
        self.datapath = datapath
        self.analpath = analpath
        self.prefix = prefix
        dataset = Dataset(self.datapath+self.prefix)
        X = dataset.as_array(stitched=False,axes=['z','channel','row','column'])
        nz,nc,nt,_,nx,ny = X.shape
        z_vec = np.arange(-4,5)*1e-6
        lambd = 520e-9
        ps = 108.3e-9
        zfocus = 4
        Nsl = 100
        tile_sz = 512
        for m in range(nt):
            for n in range(nt):
                ch3t = X[:,-1,m,n,:,:]
                ch3t = np.moveaxis(ch3t,0,-1)
                ch3t = ch3t[:512,:512,:]
                phase = self.apply(ch3t,z_vec,lambd,ps,zfocus,Nsl=Nsl)
                fig, ax = plt.subplots(1,2)
                ax[0].imshow(ch3t[:,:,zfocus],cmap='gray')
                ax[1].imshow(phase)
                plt.show()
                
    def blockshaped(self,arr,nrows,ncols):
        h, w = arr.shape
        assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
        assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
        return (arr.reshape(h//nrows, nrows, -1, ncols)
                   .swapaxes(1,2)
                   .reshape(-1, nrows, ncols))
    def apply(self,Ividmeas, z_vec, lambd, ps, zfocus, Nsl=100, eps1=1, eps2=1, reflect=False):
        if len(z_vec.shape) == 1:
            z_vec = z_vec[:, None]
        if isinstance(ps, float):
            ps = np.array([[ps]])
        elif len(ps.shape) == 1:
            ps = ps[:, None]
        RePhase1 = self.RunGaussianProcess(
            Ividmeas, zfocus, z_vec, lambd, ps, Nsl, eps1, eps2, reflect
        )
        RePhase1 = RePhase1 / np.mean(Ividmeas)
        return RePhase1
    def RunGaussianProcess(self,Ividmeas, zfocus, z_vec, lambd, ps, Nsl, eps1, eps2, reflect):
        (Nx, Ny, Nz) = Ividmeas.shape
        I0 = Ividmeas[:, :, zfocus]
        zfocus = z_vec[zfocus]
        freqs = self.CalFrequency(Ividmeas[:, :, 0], lambd, ps, 1)
        max_freq = np.max(freqs)
        max_freq = np.sqrt(max_freq / (lambd / 2))
        freq_cutoff = np.linspace(0, 1, Nsl) * max_freq
        freq_cutoff = freq_cutoff ** 2 * lambd / 2
        SigmafStack = np.zeros((Nsl, 1))
        SigmanStack = np.zeros((Nsl, 1))
        SigmalStack = np.zeros((Nsl, 1))
        freq_to_sc = np.linspace(1.2, 1.1, Nsl)
        p = Nz / (np.max(z_vec) - np.min(z_vec))
        for k in range(Nsl):
            Sigman = 10.0 ** -9
            Sigmaf = 1.0
            f1 = freq_cutoff[k]
            sc = f1 * freq_to_sc[k]
            a = sc ** 2 * 2 * np.pi ** 2
            b = np.log((p * (2 * np.pi) ** 0.5) / Sigman)
            def fu2(x):
                return a * np.exp(x) - 0.5 * x - b
            x = fsolve(fu2, 5)
            Sigmal = np.exp(x)
            SigmafStack[k] = Sigmaf
            SigmanStack[k] = Sigman
            SigmalStack[k] = Sigmal
        dIdzStack = np.zeros((Nx, Ny, Nsl))
        CoeffStack = np.zeros((Nz, Nsl))
        Coeff2Stack = np.zeros((Nz, Nsl))
        for k in range(Nsl):
            Sigmal = SigmalStack[k]
            Sigman = SigmanStack[k]
            Sigmaf = SigmafStack[k]
            dIdz, Coeff, Coeff2 = self.GPRegression(
                Ividmeas, zfocus, z_vec, Sigmaf, Sigmal, Sigman
            )
            dIdzStack[:, :, k] = 2 * np.pi / lambd * ps ** 2 * dIdz
            CoeffStack[:, k] = Coeff
            Coeff2Stack[:, k] = Coeff2
        dIdzC = self.CombinePhase(dIdzStack, freq_cutoff, freqs, CoeffStack, Coeff2Stack)
        Del2_Psi_xy = (-2 * np.pi / lambd) * dIdzC
        N = dIdzC.shape[0]
        Psi_xy = self.poisson_solve(Del2_Psi_xy, ps, eps1, 0, reflect)
        Grad_Psi_x, Grad_Psi_y = np.gradient(Psi_xy / ps)
        Grad_Psi_x = Grad_Psi_x / (I0 + eps2)
        Grad_Psi_y = Grad_Psi_y / (I0 + eps2)
        grad2x, _ = np.gradient(Grad_Psi_x / ps)
        _, grad2y = np.gradient(Grad_Psi_y / ps)
        Del2_Psi_xy = grad2x + grad2y
        Phi_xy = self.poisson_solve(Del2_Psi_xy, ps, eps1, 1, reflect)
        dcval = (
            np.sum(Phi_xy[:, 0])
            + np.sum(Phi_xy[0, :])
            + np.sum(Phi_xy[N - 1, :])
            + np.sum(Phi_xy[:, N - 1])
        ) / (4 * N)
        RePhase = -1 * (Phi_xy - dcval)
        return RePhase
    def CalFrequency(self,img, lambd, ps, dz):
        (nx, ny) = img.shape
        dfx = 1 / nx / ps
        dfy = 1 / ny / ps
        (Kxdown, Kydown) = np.mgrid[-nx // 2 : nx // 2, -ny // 2 : ny // 2]
        Kxdown = Kxdown * dfx
        Kydown = Kydown * dfy
        freqs = lambd * np.pi * (Kxdown ** 2 + Kydown ** 2)
        freqs = freqs * dz / (2 * np.pi)
        return freqs
    def CombinePhase(self,dIdzStack, Frq_cutoff, freqs, CoeffStack, Coeff2Stack):
        def F(x):
            return fft.ifftshift(fft.fft2(fft.fftshift(x)))
        def Ft(x):
            return fft.ifftshift(fft.ifft2(fft.fftshift(x)))
        Nx, Ny, Nsl = dIdzStack.shape
        dIdzC_fft = np.zeros((Nx, Ny))
        Maskf = np.zeros((Nx, Ny))
        f0 = 0
        f1 = 1
        for k in range(Nsl):
            dIdz = dIdzStack[:, :, k]
            dIdz_fft = F(dIdz)
            f1 = Frq_cutoff[k]
            Maskf = np.zeros((Nx, Ny))
            Maskf[np.argwhere((freqs <= f1) & (freqs > f0))] = 1
            f0 = f1
            dIdzC_fft = dIdzC_fft + (dIdz_fft * Maskf)
        return np.real(Ft(dIdzC_fft))
    def poisson_solve(self,func, ps, eps, symm, reflect):
        N = len(func)
        if reflect != 0:
            N = N * 2
            func = np.hstack([func, np.fliplr(func)])
            func = np.vstack([func, np.flipud(func)])
        wx = 2 * np.pi * np.arange(0, N, 1) / N
        fx = 1 / (2 * np.pi * ps) * (wx - np.pi * (1 - N % 2 / N))
        [Fx, Fy] = np.meshgrid(fx, fx)
        func_ft = np.fft.fftshift(np.fft.fft2(func))
        Psi_ft = func_ft / (-4 * np.pi ** 2 * (Fx ** 2 + Fy ** 2 + eps))
        if symm:
            Psi_xy = np.fft.irfft2(np.fft.ifftshift(Psi_ft)[:, 0 : N // 2 + 1])
        else:
            Psi_xy = np.fft.ifft2(np.fft.ifftshift(Psi_ft))
        if reflect != 0:
            N = N // 2
            Psi_xy = np.array(Psi_xy)[:N, :N]
        return Psi_xy
    def mrdivide(self,A, B):
        return A.dot(np.linalg.pinv(B))
    def GPRegression(self,Ividmeas, zfocus, z, Sigmaf, Sigmal, Sigman):
        Nx, Ny, Nz = Ividmeas.shape
        ones = np.ones((Nz, 1))
        KZ = ones.dot(z.T) - z.dot(ones.T)
        K = Sigmaf * (np.exp(-1 / 2 / Sigmal * (KZ ** 2)))
        L = np.linalg.cholesky(K + (Sigman * np.eye(Nz))).T
        z2 = zfocus
        Nz2 = len(z2)
        ones2 = np.ones((Nz2, 1))
        KZ2 = ones * (z2.T) - z * (ones2.T)
        D = Sigmaf * (np.exp((-1 / 2 / Sigmal) * (KZ2 ** 2))) / -Sigmal * KZ2
        Coeff = self.mrdivide(self.mrdivide(D.T, L), L.T)[0]  # D.T/L/L.T
        D2 = Sigmaf * (np.exp((-1 / 2 / Sigmal) * (KZ2 ** 2)))
        Coeff2 = self.mrdivide(self.mrdivide(D2.T, L), L.T)  # D2.T/L/L.T
        dIdz = np.zeros((Nx, Ny))
        for k in range(Nz):
            dIdz = dIdz + Ividmeas[:, :, k]*Coeff[k]
        return dIdz, Coeff, Coeff2
    

