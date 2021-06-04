"""
TODO:
    creat a debelnding class
    
    
Created on Tue Apr 13 10:26:43 2021

@author: Junhai Cao
@email: j.cao@tudelft.nl /junhaicao1990@163.com
@ Delft university of Technology


=================================================
Tasks: April 20, 2021
    - 1. finish the blending classes (yes)
    - 2. complete the mkblcode function (yes)
    - 3. test the algorithm Pbl = P*Gamma and plot the data (checking)
    - 4. write the ghost operator G (undergoing)

"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import os


def is_defined(v):
    return False if v is None else True

# ===============================================================
# read data from bin or su file
# ===============================================================


def from_bin(filename, shape):
    if not isinstance(filename, (list, tuple)):
        filename = [filename]

    filename = [f for f in filename if not os.path.isdir(f)]
    outs = []
    for file in filename:
        print(f'< {file} > np.array({shape})')

        data_type = np.dtype('float32').newbyteorder('<')
        vs = np.fromfile(file, dtype=data_type)
        try:
            vs = vs.reshape(*shape)
            vs = np.transpose(vs, (1, 0, 2))
            outs = vs
        except ValueError:
            print(f'\tFailed to reshape {vs.shape} to ({shape})')
    return outs


class Cube():
    """
    TODO:	- For the overloaded operators make sure that the class properties are COPIED.
            - Make plot methods for common offset / common receiver gathers.
            - Harmonize all plot methods to use the new plot_gather() method.
    """

    def __init__(self, data, **kwargs):
        self.data = np.array(data)

        # source and receiver info
        optargin = self.extract_argument(**kwargs)

        self.dt = optargin.get('dt', 0.004)
        self.fmt = optargin.get('fmt', 'xt')

        # source # and reciver #  and sampling #
        self.nrx = np.int32(optargin.get('nrx', data.shape[0]))
        self.nsx = np.int32(optargin.get('nsx', data.shape[1]))
        self.nt = np.int32(optargin.get('nt', data.shape[2]))

        self.sx = optargin.get('sx', np.arange(self.nsx))
        self.rx = optargin.get('rx', np.arange(self.nrx))

        # assin sx, rx, dt parameters
        # for keyname in optargin:
        # #   self.keyname = optargin.get(keyname)

        self.sdx = self.sx[1] - self.sx[0]
        self.rdx = self.rx[1] - self.rx[0]

        # freq
        self.nf = np.int32(np.floor(self.nt/2) + 1)
        self.df = 1/(self.nt*self.dt)

    # ===============================================================
    # transform to frequency or time domain
    # ===============================================================
    def freq(self):
        # xt---> xf domain
        if not self.is_xf():
            data_Freq = np.fft.rfft(self.data, axis=2)
            C = Cube(data_Freq, fmt='xf', sx=self.sx,\
                rx=self.rx, dt=self.dt, nt=self.nt)
        else:
            C = self

        return C

    def timeF(self):
        # xf--->xt domain; return in time domain
        if not self.is_xt():
            data_time = np.fft.irfft(self.data, axis=2)
            C = Cube(data_time, fmt='xt', sx=self.sx, rx=self.rx, dt=self.dt)
        else:
            C = self
        return C

    def padzero(self, padsize):
        # pad zeros
        # padsize [pad_rec,pad_src,pad_t]
        data_new = np.pad(
            self.data, ((0, padsize[0]), (0, padsize[1]), (0, padsize[2])), 'constant')

        fmt = 'xt'
        nt = self.nt + padsize[2]
        sx = np.append(self.sx, self.sx[-1] + (np.arange(padsize[1])+1)*self.sdx)
        rx = np.append(self.rx, self.rx[-1] + (np.arange(padsize[0])+1)*self.rdx)
        dt = self.dt
        C = Cube(data_new, fmt=fmt, sx=sx, rx=rx, dt=dt, nt=nt)

        return C

    # ===================================================
    # plot functions
    # ===================================================
    def extract_csg(self, id):
        if id > self.nsx or id < 0:
            raise ValueError(f'cube:IdNotValid---The CSG id is not valid. Id should be in range 0~{self.nsx}.')

        plot_index = id-1
        csg_data = np.squeeze(self.data[:, plot_index, :]).T
        shot_coords = self.sx[plot_index]
        recv_coords = self.rx[:]
        offsets = recv_coords - shot_coords

        return csg_data, shot_coords, offsets

    def extract_crg(self, id):
        if id > self.nrx or id < 0:
            raise ValueError('cube:IdNotValid----', 'The CRG id is not valid. Id should be in range 0~{self.nrx}.')
        plot_index = id-1
        crg_data = np.squeeze(self.data[plot_index, :, :]).T
        shot_coords = self.sx[:]
        recv_coords = self.rx[plot_index]
        offsets = shot_coords - recv_coords

        return crg_data, recv_coords, offsets

    def plot_csg(self, id, **kwargs):

        if not self.is_xt():
            raise Exception('Cube must be in xt format for plotting.')

        optargin = self.extract_argument(**kwargs)

        csg_data, shot_coords, offsets = self.extract_csg(id)
        clip = optargin.get('clip', 0.8)

        # Fig = optargin.get('fig', plt.figure(figsize=(5, 7)))
        # ax = optargin.get('axes', Fig.add_subplot())

        if optargin.get('domain') == 'xt':
            
            fylabel = optargin.get('ylabel','Times (s)')
            fcmap = optargin.get('cmap','gray')
            ftitle = optargin.get('title',f'CSG {id} at ({shot_coords},0) ')
            fytickdata = optargin.get('ytickdata',np.array([0, self.nt-1])*(self.dt)) 
            
            vmax = optargin.get('vmax',clip*np.amax(csg_data))
            vmin = optargin.get('vmin',clip*np.amin(csg_data))

            if optargin.get('xaxis') == 'lateral':
                
                fxlabel = optargin.get('xlabel','Detector lacation (km)')
                fxtickdata =  optargin.get('xtickdata',np.array([self.rx[0], self.rx[-1]])/1000)
                
                self.plot_gather(csg_data, xlabel=fxlabel, ylabel=fylabel, title=ftitle,\
                    cmap=fcmap, xtickdata=fxtickdata, ytickdata=fytickdata, clip=clip, vmax=vmax, vmin=vmin)
            elif optargin.get('xaxis') == 'offset':
                
                fxlabel = 'Offset (km)'
                fxtickdata = optargin.get('xtickdata', np.array([offsets[0], offsets[-1]])/1000)
                
                self.plot_gather(csg_data, xlabel=fxlabel, ylabel=fylabel, title=ftitle,\
                    cmap=fcmap, xtickdata=fxtickdata, ytickdata=fytickdata, clip=clip, vmax=vmax, vmin=vmin)

        elif optargin.get('domain') == 'xf':
            
            fxlabel = 'Offset (km)'
            fylabel = 'Frequency (Hz)'
            
            ftitle = optargin.get('title',f'CSG {id} at ({shot_coords},0) in xf domain')
            fxtickdata = np.array([offsets[0], offsets[-1]])/1000
            fytickdata = np.array([0, self.nf-1])/(self.dt*self.nt)
            fcmap = optargin.get('cmap','jet')
            
            plot_data = np.absolute(np.fft.rfft(csg_data, axis=0))
            vmax = optargin.get('vmax',clip*np.amax(plot_data))
            vmin = optargin.get('vmin',clip*np.amin(plot_data))

            self.plot_gather(plot_data,\
                xlabel=fxlabel, ylabel=fylabel, title=ftitle,\
                    cmap=fcmap, xtickdata=fxtickdata, ytickdata=fytickdata, clip=clip, vmax=vmax, vmin=vmin)
                
        elif optargin.get('domain') == 'fk':
            
            fxlabel = 'Wavenumber (1/m)'
            fylabel = 'Frequency (Hz)'
            
            ftitle = optargin.get('title', f'CSG {id} at ({shot_coords},0) in fk domain')
            
            if (self.nrx % 2) == 0:
                nk = self.nrx/2
                fxtickdata = np.array([-nk, nk-1])/(self.nrx*self.rdx)
            else:
                nk = np.floor(self.nrx/2)
                fxtickdata = np.array([-nk, nk])/(self.nrx*self.rdx)

            fytickdata = np.array([0, self.nf-1])/(self.nt*self.dt)
            fcmap = optargin.get('cmap','jet')
            
            plot_data = np.absolute(np.fft.fftshift(np.fft.fft(np.fft.rfft(csg_data, axis=0), axis=1), axes=1))
            vmax = optargin.get('vmax',clip*np.amax(plot_data))
            vmin = optargin.get('vmin',clip*np.amin(plot_data))

            self.plot_gather(plot_data,\
                xlabel=fxlabel, ylabel=fylabel, title=ftitle,\
                    cmap=fcmap, xtickdata=fxtickdata, ytickdata=fytickdata, clip=clip, vmax=vmax, vmin=vmin)
        else:
            raise Exception('cube:NotSupported---',
                            'Plot domain not supported currently.')

    def plot_crg(self, id, **kwargs):

        if not self.is_xt():
            raise Exception('Cube must be in xt format for plotting.')

        optargin = self.extract_argument(**kwargs)

        crg_data, recv_coords, offsets = self.extract_crg(id)
        clip = optargin.get('clip', 0.8)

        # Fig = optargin.get('fig', plt.figure(figsize=(5, 7)))
        # ax = optargin.get('axes', Fig.add_subplot())

        if optargin.get('domain') == 'xt':
            
            fylabel = optargin.get('ylabel','Times (s)')
            fcmap = optargin.get('cmap','gray')
            ftitle = optargin.get('title',f'CRG {id} at ({recv_coords},0) ')
            fytickdata = optargin.get('ytickdata',np.array([0, self.nt-1])*(self.dt)) 
            
            vmax = optargin.get('vmax',clip*np.amax(crg_data))
            vmin = optargin.get('vmin',clip*np.amin(crg_data))

            if optargin.get('xaxis') == 'lateral':
                fxlabel = 'Source lacation (km)'
                fxtickdata = np.array([self.sx[0], self.sx[-1]])/1000

                self.plot_gather(crg_data, xlabel=fxlabel, ylabel=fylabel, title=ftitle, cmap=fcmap,\
                xtickdata=fxtickdata, ytickdata=fytickdata, clip=clip, vmax=vmax, vmin=vmin)

            elif optargin.get('xaxis') == 'offset':

                fxlabel = 'Offset (km)'
                fxtickdata = [offsets[0], offsets[-1]]/1000
                self.plot_gather(crg_data, xlabel=fxlabel, ylabel=fylabel, title=ftitle, cmap=fcmap,\
                    xtickdata=fxtickdata, ytickdata=fytickdata, clip=clip, vmax=vmax, vmin=vmin)

        elif optargin.get('domain') == 'xf':
            
            fxlabel = 'Offset (km)'
            fylabel = 'Frequency (Hz)'
            ftitle = optargin.get('title',f'CRG {id} at ({recv_coords},0) in xf domain')
            
            fxtickdata = np.array([offsets[0], offsets[-1]])/1000
            fytickdata = np.array([0, self.nf-1])/(self.dt*self.nt)
            fcmap = 'jet'

            plot_data = np.absolute(np.fft.rfft(crg_data, axis=0))
            vmax = optargin.get('vmax',clip*np.amax(plot_data))
            vmin = optargin.get('vmin',clip*np.amin(plot_data))

            self.plot_gather(plot_data, xlabel=fxlabel, ylabel=fylabel, title=ftitle,\
                    cmap=fcmap, xtickdata=fxtickdata, ytickdata=fytickdata, clip=clip, vmax=vmax, vmin=vmin)

        elif optargin.get('domain') == 'fk':
            fxlabel = 'Wavenumber (1/m)'
            fylabel = 'Frequency (Hz)'
            
            ftitle = optargin.get('title',f'CRG {id} at ({recv_coords},0) in fk domain')
            if (self.nsx % 2) == 0:
                nk = self.nsx/2
                fxtickdata = np.array([-nk, nk-1])/(self.nsx*self.sdx)
            else:
                nk = np.floor(self.nsx/2)
                fxtickdata = np.array([-nk, nk])/(self.nsx*self.sdx)

            fytickdata = np.array([0, self.nf-1])/(self.nt*self.dt)
            fcmap = optargin.get('cmap','jet')
            
            plot_data = np.absolute(np.fft.fftshift(np.fft.fft(np.fft.rfft(crg_data, axis=0), axis=1), axes=1))
            vmax = optargin.get('vmax',clip*np.amax(plot_data))
            vmin = optargin.get('vmin',clip*np.amin(plot_data))

            self.plot_gather(plot_data,xlabel=fxlabel, ylabel=fylabel, title=ftitle,\
                    cmap=fcmap, xtickdata=fxtickdata, ytickdata=fytickdata, clip=clip, vmax=vmax, vmin=vmin)
        else:
            raise Exception('cube:NotSupported---',
                            'Plot domain not supported currently.')

    def plot_ctg(self, id, **kwargs):
        if id > self.nt or id < 0:
            raise ValueError('cube:IdNotValid----', 'The CSG id is not valid.')

        plot_index = id-1
        ctg_data = np.squeeze(self.data[:, :, plot_index])

        tindex = (id-1)*self.dt
        clip = 0.95
        
        vmax = clip*np.amax(ctg_data)
        vmin = clip*np.amin(ctg_data)
        xmin = self.sx[0]
        xmax = self.sx[-1]
        ymin = self.rx[0]
        ymax = self.rx[-1]

        plt.figure(figsize=(6, 6))
        plt.imshow(ctg_data, extent=[xmin, xmax, ymax, ymin],\
            vmin=vmin, vmax=vmax, cmap='gray', aspect='auto')
        plt.xlabel('Source locations (m)')
        plt.ylabel('Detector locations (m)')
        plt.title(f'Time slice at {tindex}s')
        plt.show()

    def plot_cfg(self, id, **kwargs):
        if id > self.nf or id < 0:
            raise ValueError('cube:IdNotValid----', 'The CSG id is not valid.')
        plot_index = id-1
        freq_index = plot_index*self.df

        if self.is_xt():
            fdata = np.fft.rfft(self.data, axis=2)
            cfg_data = np.abs(np.squeeze(fdata[:, :, plot_index]))
        elif self.is_xf():
            cfg_data = np.abs(np.squeeze(self.data[:, :, plot_index]))

        clip = 0.95
        vmax = clip*np.amax(cfg_data)
        vmin = clip*np.amin(cfg_data)
        xmin = self.sx[0]
        xmax = self.sx[-1]
        ymin = self.rx[0]
        ymax = self.rx[-1]

#         plt.figure(figsize=(6, 6))
        plt.imshow(cfg_data, extent=[xmin, xmax, ymax, ymin],\
            vmin=vmin, vmax=vmax, cmap='jet', aspect='auto')
        plt.xlabel('Source locations (m)')
        plt.ylabel('Detector locations (m)')
        plt.title(f'Frequency slice at {freq_index} Hz')
        plt.show()

    # ===============================================================
    def plot_gather(self, gather_data, **kwargs):
        optargin = self.extract_argument(**kwargs)

        fxlabel = optargin.get('xlabel')
        fylabel = optargin.get('ylabel')
        ftitle = optargin.get('title')

        # Fig = optargin.get('fig')
        # ax = optargin.get('axes')
        # print(ax)

        if 'ytickdata' in optargin:
            ymin, ymax = optargin.get('ytickdata')
        else:
            ymin = 0
            ymax = gather_data.shape[0]

        if 'xtickdata' in optargin:
            xmin, xmax = optargin.get('xtickdata')
        else:
            xmin = 1
            xmax = gather_data.shape[1]
        
        cmap = optargin.get('cmap','gray')
        # clip = optargin.get('clip',0.8)

        vmax = optargin.get('vmax')
        vmin = optargin.get('vmin')

        plt.figure(figsize=(5, 7))
        plt.imshow(gather_data, extent=[xmin, xmax, ymax, ymin], \
            vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')
        plt.xlabel(fxlabel)
        plt.ylabel(fylabel)
        plt.title(ftitle)
        plt.colorbar()
#         plt.show()

    def extract_argument(self, **kwargs):
        optargin = dict()
        optargin['domain'] = 'xt'
        optargin['xaxis'] = 'lateral'
        # optargin['circshift'] = False
        optargin['colormap'] = 'gray'

        for key, value in kwargs.items():
            optargin[key.lower()] = value

        return optargin

    # ============ State-checking methods ===============

    def is_xt(self):
        if self.fmt == 'xt':
            b = True
        else:
            b = False

        return b

    def is_xf(self):
        if self.fmt == 'xf':
            b = True
        else:
            b = False

        return b

    def copydata(self):
        C = Cube(self.data, fmt=self.fmt, sx=self.sx, rx=self.rx, dt=self.dt)
        return C

    def minValue(self):
        return self.data.min()

    def maxValue(self):
        return self.data.max()

    def extrema(self):
        y = [self.minValue(), self.maxValue()]
        return y

    # ============ Overload operators ===============
    def __add__(self,other):
        # should check the Two Cube objects are same property
        C = Cube(self.data + other.data , fmt=self.fmt, sx=self.sx, rx=self.rx, dt=self.dt,nt=self.nt)
        return C
    
    def __sub__(self,other):
        # should check the Two Cube objects are same property
        C = Cube(self.data - other.data , fmt=self.fmt, sx=self.sx, rx=self.rx, dt=self.dt,nt=self.nt)
        return C
    
    def conj(self):
        C = Cube(np.conj(self.data), fmt=self.fmt, sx=self.sx, rx=self.rx, dt=self.dt,nt=self.nt)
        return C

    def ctranspose(self):
        C = Cube(np.conj(np.transpose(self.data, (1, 0, 2))), fmt=self.fmt, sx=self.rx, rx=self.sx, dt=self.dt,nt=self.nt)
        return C
    
    def transpose(self):
        C = Cube(np.transpose(self.data, (1, 0, 2)), fmt=self.fmt, sx=self.rx, rx=self.sx, dt=self.dt,nt=self.nt)
        return C

    def mulcube(self,B):    
        data = np.zeros((self.nrx,B.nsx,self.nf),dtype=complex)
        data = np.einsum('ijl,jkl->ikl',self.data,B.data)
        print('Done with einsum')
        C = Cube(data, fmt=self.fmt, sx=B.sx, rx=self.rx, dt=self.dt,nt=self.nt)
        return C



def extract_input(**kwargs):
        optargin = dict()
        for key, value in kwargs.items():
            optargin[key.lower()] = value
        return optargin

''' !!!!!!! check this later '''
def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind

# index to sub index, same for matlab ind2sub function
def ind2sub(array_shape, ind):
    ind = np.array(ind)
    rows = (ind.astype("int32") // array_shape[0])
    cols = (ind.astype("int32") % array_shape[0])
    return (np.int32(np.array(rows)), np.int32(np.array(cols)))


class Sourceghost():
    """
    Public methods:
        forward:        add source ghost process
        adjoint:        apply complex-conjugate 
    """
    
    def __init__(self, P,**kwargs):
        
        # get the input arguments
        optargin = extract_input(**kwargs)
        c = optargin.get('c',1500.0)
        depth = optargin.get('depth',30)
        is_normalize = optargin.get('optargin',False)
        self.options = dict()
        self.options['focal_points'] = optargin.get('focal_points',np.arange(P.nsx)*P.sdx)
        focal_points = self.options['focal_points']

        dx = P.sdx
        nx = P.nsx
        fmax = P.nf
        df = P.df

        # construct f,k and kx-axis
        f = np.arange(0,fmax)*df
        k = 2.0*np.pi*f/c
        dkx = 2.0*np.pi/(dx*nx)

        # 
        kx = np.zeros((1,nx))
        if (nx % 2) == 0:
            kxmid = int(nx/2) + 1
            kx[:,0:kxmid:1] = np.arange(kxmid)*dkx
            kx[:,kxmid:nx:1] = -np.arange(kxmid-2,0,-1)*dkx
        else:
            kxmid = int((nx +1)/2)
            kx[:,0:kxmid] = np.arange(kxmid)*dkx
            kx[:,kxmid:nx] = -np.arange(kxmid-1,0,-1)*dkx
            
        # print('kx is ',kx)

        # create W matrix 
        nfp = len(focal_points)
        nsp = len(P.sx)

        kx2D = np.tile(kx, (nfp,1))
        dz2D = np.tile(depth,(nfp,nx))
        
        # print('kx2D is',kx2D)
        # print('dz2D is ',dz2D)

        self.W = Cube(np.zeros((nfp,nsp,P.nf),dtype=complex),fmt='xf',nt=P.nt,dt=P.dt,rx=focal_points,sx=P.sx)
        self.G = Cube(np.zeros((nsp,nsp,P.nf),dtype=complex),fmt='xf',nt=P.nt,dt=P.dt,rx=P.sx,sx=P.sx)

        for I in range(1,P.nf): # range(1,P.nf)
            kzdz = np.conj(np.sqrt(np.complex64(k[I]**2 - np.square(kx2D))))*dz2D
            # print(f'freq slice {I}\n')
            # print(kzdz[0,:])
            
            cop = np.exp(-1j*kzdz)
            cop = np.fft.ifft(cop,axis=1)
            
            # print(cop[0,:])
            self.W.data[:,:,I] =  np.transpose(np.array(list(map(lambda k: np.roll(np.transpose(cop[k,:]),k), np.arange(cop.shape[1])))))

        # create taper for mask
        A = np.ones((nsp,nsp))
        mask = np.zeros((nsp,nsp))

        half_index = int(np.floor(nsp/2))
        for I in range(-half_index,half_index,1):
            mask += np.diag(np.diag(A,I),I)
            
        # plt.figure()
        # plt.imshow(mask)
        # plt.title('Mask')
        # plt.show()

        # tmp = self.G.copydata()
        # tmp.data = np.tile(np.complex64(np.eye(nsp)),(1,1,P.nf))
        tmp_eye = np.complex64(np.eye(nsp))
        

        for I in range(P.nf):
            tmp_squee = np.squeeze(self.W.data[:,:,I])
            self.G.data[:,:,I] = tmp_eye - np.einsum('ij,jk->ik', mask*np.transpose(tmp_squee), mask*tmp_squee)
            
        # fig, ax = plt.subplots()
        # im = ax.imshow(np.abs(Sg.W.data[:,:,50]), cmap=plt.cm.jet)
        # im.set_clim(0,0.1734)
        # fig.colorbar(im, ax=ax)

        # del tmp
        
        self.forward = self.ghost
        self.adjoint = self.deghost

    def ghost(self,D_in):

        D_out = D_in.mulcube(self.G)
        return D_out

    def deghost(self,D_in):

        D_out = D_in.mulcube(self.G.ctranspose())
        
        return D_out

class Detectorghost():
    """
    Public methods:
        forward:        add detector ghost process
        adjoint:        apply complex-conjugate 
    """
    
    def __init__(self, P,**kwargs):
        
        # get the input arguments
        optargin = extract_input(**kwargs)
        c = optargin.get('c',1500.0)
        depth = optargin.get('depth',30)
        is_normalize = optargin.get('optargin',False)
        self.options = dict()
        self.options['focal_points'] = optargin.get('focal_points',np.arange(P.nrx)*P.rdx)
        focal_points = self.options['focal_points']

        dx = P.rdx
        nx = P.nrx
        fmax = P.nf
        df = P.df

        # construct f,k and kx-axis
        f = np.arange(0,fmax)*df
        k = 2.0*np.pi*f/c
        dkx = 2.0*np.pi/(dx*nx)

        # 
        kx = np.zeros((1,nx))
        if (nx % 2) == 0:
            kxmid = int(nx/2) + 1
            kx[:,0:kxmid:1] = np.arange(kxmid)*dkx
            kx[:,kxmid:nx:1] = -np.arange(kxmid-2,0,-1)*dkx
        else:
            kxmid = int((nx +1)/2)
            kx[:,0:kxmid] = np.arange(kxmid)*dkx
            kx[:,kxmid:nx] = -np.arange(kxmid-1,0,-1)*dkx
            
        # print('kx is ',kx)

        # create W matrix 
        nfp = len(focal_points)
        nrp = len(P.rx)

        kx2D = np.tile(kx, (nfp,1))
        dz2D = np.tile(depth,(nfp,nx))
        
        # print('kx2D is',kx2D)
        # print('dz2D is ',dz2D)

        self.W = Cube(np.zeros((nfp,nrp,P.nf),dtype=complex),fmt='xf',nt=P.nt,dt=P.dt,rx=focal_points,sx=P.rx)
        self.G = Cube(np.zeros((nrp,nrp,P.nf),dtype=complex),fmt='xf',nt=P.nt,dt=P.dt,rx=P.rx,sx=P.rx)

        for I in range(1,P.nf): # range(1,P.nf)
            kzdz = np.conj(np.sqrt(np.complex64(k[I]**2 - np.square(kx2D))))*dz2D
            # print(f'freq slice {I}\n')
            # print(kzdz[0,:])
            
            cop = np.exp(-1j*kzdz)
            cop = np.fft.ifft(cop,axis=1)
            
            # print(cop[0,:])
            self.W.data[:,:,I] =  np.transpose(np.array(list(map(lambda k: np.roll(np.transpose(cop[k,:]),k), np.arange(cop.shape[1])))))

        # create taper for mask
        A = np.ones((nrp,nrp))
        mask = np.zeros((nrp,nrp))

        half_index = int(np.floor(nrp/2))
        for I in range(-half_index,half_index,1):
            mask += np.diag(np.diag(A,I),I)
            
        # plt.figure()
        # plt.imshow(mask)
        # plt.title('Mask')
        # plt.show()

        # tmp = self.G.copydata()
        # tmp.data = np.tile(np.complex64(np.eye(nsp)),(1,1,P.nf))
        tmp_eye = np.complex64(np.eye(nrp))
        

        for I in range(P.nf):
            tmp_squee = np.squeeze(self.W.data[:,:,I])
            self.G.data[:,:,I] = tmp_eye - np.einsum('ij,jk->ik', mask*np.transpose(tmp_squee), mask*tmp_squee)
            
        # fig, ax = plt.subplots()
        # im = ax.imshow(np.abs(Sg.W.data[:,:,50]), cmap=plt.cm.jet)
        # im.set_clim(0,0.1734)
        # fig.colorbar(im, ax=ax)

        # del tmp
        
        self.forward = self.ghost
        self.adjoint = self.deghost

    def ghost(self,D_in):

        D_out = self.G.mulcube(D_in)
        return D_out

    def deghost(self,D_in):
        
        Gtranp = self.G.ctranspose()
        D_out = Gtranp.mulcube(D_in)
        
        return D_out


if __name__ == "__main__":
    # Test this modules
    filename = 'shots.pluto.bin'

    # The number of receivers and sources
    nrec = 512
    nsrc = 512

    # Spatial sampling
    sdx = 5
    rdx = 5
    sx = np.arange(nsrc)*sdx
    rx = np.arange(nrec)*rdx

    # Time samples and sampling
    nt = 512
    dt = 0.004

    # Read in seismic data and reshape to matrix notation (binary format)
    data = from_bin(filename, [nsrc, nrec, nt])

    # Apply Fourier transform
    P = Cube(data, fmt='xt', sx=sx, rx=rx, dt=dt, nt=nt)
    plot_shot = 256
    
    P = P.padzero([0,0,128])
    
    # plot function
    # P.plot_csg(plot_shot, clip=0.1, domain='xt',vmin=-1.5,vmax=1.5)
    # P.plot_csg(plot_shot, clip=0.95, domain='fk',vmin=0,vmax=4200)
    
    # P.plot_crg(plot_shot, clip=0.1, domain='xt',vmin=-1.5,vmax=1.5)
    # P.plot_crg(plot_shot, clip=0.95, domain='fk',vmin=0,vmax=4200)
    
    # np.random.seed(0)
    # blcode = mkblcode(P,blending_factor=2,min_tau=0.1,max_tau=0.5)
    # Gbl = Blending(blcode)
    Pf = P.freq()
    
    # Pbl = Gbl.blend(Pf)
    # Pps = Gbl.pseudodeblend(Pbl)
    
    # Pblt = Pbl.timeF()
    # Ppst = Pps.timeF()
    
    # plot_bl = 1
    # Pblt.plot_csg(plot_bl, clip=0.1, domain='xt',vmin=-1.5,vmax=1.5,title=f'Pbl CSG {plot_bl}')
    # Ppst.plot_csg(plot_bl, clip=0.1, domain='xt',vmin=-1.5,vmax=1.5,title=f'Pps CSG {plot_bl}')

    # print('Add source ghost...')

    # Sg = Sourceghost(Pf)
    # Pg = Sg.ghost(Pf)
    # Pgt = Pg.timeF()

    # Pgt.plot_crg(plot_shot, clip=0.1, domain='xt',title=f'CRG {plot_shot} with ghost',vmin=-1.5,vmax=1.5)
    # Pgt.plot_crg(plot_shot, clip=0.95, domain='fk',title=f' CRG {plot_shot} with ghost',vmin=0,vmax=4200)
    
    print('Add detector ghost...')

    Dg = Detectorghost(Pf)
    Pdg = Dg.ghost(Pf)
    Pdgt = Pdg.timeF()

    Pdgt.plot_csg(plot_shot, domain='xt',title=f'CSG {plot_shot} with ghost',vmin=-1.5,vmax=1.5)
    Pdgt.plot_csg(plot_shot, domain='fk',title=f' CSG {plot_shot} with ghost',vmin=0,vmax=4200)

