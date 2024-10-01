import numpy as np
import random
import io
import os
from itertools import islice
import torch

class Modulator(object):
    def encode(self, data, bitlen=None):
        raise NotImplemented()

    def decode(self, vector):
        raise NotImplemented()

    # interface compatible with commpy
    def modulate(self, coded_bits):
        data = self.bvec_to_data(coded_bits)
        vec = self.encode(data, bitlen=len(coded_bits))
        return vec

    def demodulate(self, modulated, demod_type='hard', **kwargs):
        
        if demod_type == 'hard':
            data = self.decode(modulated.real)
            bits = self.data_to_bvec(data)
            return np.asarray(bits, dtype=np.int8)
        else: # soft
            data = self.decode_soft(modulated.real)
            # reorder bits
            data = np.asarray([data[(x // 8) * 8 + (7-(x & 7))] for x in range(len(data))])
            return data

    def data_to_bvec(self, data):
        x = int.from_bytes(data, byteorder="little")
        input_vector = np.asarray([int(x // 2**i) & 1 for i in range(256)])
        return input_vector

    def bvec_to_data(self, bvec):
        x = sum([int(d) * int(2)**i for i,d in enumerate(bvec)])
        data = int.to_bytes(x, (len(bvec) + 7) // 8, byteorder="little")
        return data

    def compute_errors(self, data_a, data_b, bitlen=None):
        # it's ok to decode more random bits
        data_b = data_b[:len(data_a)]
        data_a = data_a[:len(data_b)]
        if bitlen is None:
            bitlen = 8 * len(data_a)
            padding = 0
        else:
            padding = 8 * len(data_a) - bitlen

        x = int.from_bytes(data_a, byteorder="big")
        y = int.from_bytes(data_b, byteorder="big")

        ### HAMMING DISTANCE
        return sum([(int(x // 2**i) & 1) ^ (int(y // 2**i) & 1) for i in range(padding, 256)])

class OrthogonalModulator(Modulator):

    def __init__(self, key, use_gpu=True):
        self.D = 256
        self.key = key
        # using cupy if available to use gpu for fft
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(key)
        self.dtype = torch.float32
        self.L = self.D
        self.basis = self.gen_basis()

    def gen_mod1(self, n):
        # Generate random uniform samples
        u = torch.rand(size=((n-2)//2,), dtype=self.dtype, generator = self.generator, device = self.generator.device)
        Y = torch.zeros(n, dtype=torch.complex32 if self.dtype != torch.float32 else torch.complex64, device=self.device)

        c = torch.cos(2 * torch.pi * u)
        Y[1:(n-2)//2 + 1].real = c
        Y[(n-2)//2 + 2:] = c.flip(dims=(0,))
        s = torch.sin(2 * torch.pi * u)
        Y[1:(n-2)//2 + 1].imag = s
        Y[(n-2)//2 + 2:].imag = -s.flip(dims=(0,))

        # Perform iFFT using cuFFT (cuPy)
        y = torch.fft.ifft(Y)

        # Return the real part as a NumPy array
        return torch.real(y)

    def gen_classical(self, n):
        w = torch.randn(n, dtype=self.dtype, generator = self.generator, device = self.generator.device) * (1/np.sqrt(n))
        return w / torch.sqrt(torch.dot(w, w))

    def gen_basis(self):
        basis = []
        for i in range(self.L):
            # generate carrier
            carrier = self.gen_classical(self.D)

            # orthogonalize wrt to previous carriers to minimize interference
            for bvec in basis:
                # Project carrier on basis and remove component
                c = torch.dot(carrier, bvec)
                proj = (c / torch.dot(bvec, bvec)) * bvec
                carrier -= proj
                carrier /= torch.sqrt(torch.dot(carrier, carrier))

            basis.append(carrier)
        basis = torch.stack(basis)
        return basis

    def encode(self, data, bitlen=None):
        self.generator.manual_seed(self.key)

        if bitlen is None:
            bitlen = 8*len(data)

        # convert data to bitstream
        value = int.from_bytes(data, byteorder="big", signed=False)

        bitstream = bin(value)[2:] # strip "0b"
        # 0-pad prefix
        bitstream = '0' * (8*len(data) - len(bitstream)) + bitstream
        # cut to bitlen
        bitstream = bitstream[:bitlen]

        # encode the necessary bits using the first components
        w = 0
        basis = self.basis
        for i, bit in enumerate(bitstream):
            carrier = basis[i]
            if bit == '1':
                # BPSK
                carrier = -carrier

            w += carrier


        # normalize to power 1
        w = w / torch.sqrt(torch.dot(w,w))

        return w.float().cpu().numpy()

    def decode(self, w):

        # Convert w to a tensor
        w = torch.tensor(w, dtype=self.dtype, device=self.device)

        basis = self.basis
        seq = []
        for i in range(self.L):
            # BPSK
            bit = w.dot(basis[i]) < 0
            seq.append('1' if bit else '0')

        # convert data to bitstream
        value = int(''.join(seq), 2)
        data = int.to_bytes(value, self.L // 8, byteorder="big", signed=False)
        return data

class CyclicModulator(Modulator):
    def __init__(self, key, use_gpu=True, use_fp16=False, use_flip=True, use_sign=True, use_mod1=False, direct=False, M=4, N=3):
        self.D = 256
        self.BITS = 8 * N - 2
        self.use_sign = use_sign
        self.use_flip = use_flip
        if not self.use_sign:
            self.BITS +=1
        if not self.use_flip:
            self.BITS += 1
        #self.FACTORS = [0.83767842, 0.45881537, 0.25130353, 0.13764461, 0.07539106]
        self.key = key

        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(key)
        if use_fp16 and not use_gpu:
            print("FP16 not supported on CPU")
            use_fp16 = False
        if use_fp16:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        self.M = M
        self.N = N
        self.gen_carrier = self.gen_mod1 if use_mod1 else self.gen_classical
        self.L = max(self.D, 2**self.BITS)
        self.direct = direct

        # precompute carriers, store in CPU and upload to GPU on demand, to reduce VRAM usage
        self.basis = []
        self.generator.manual_seed(self.key)
        for i in range(self.M):  # 96-bit max
            self.basis.append(self.gen_carrier(self.L).cpu())

        # precompute inverse norms
        self.invnorms = []
        for i in range(self.M):
            carrier = self.basis[i]
            carrier2 = carrier**2
            norms = carrier2.clone()
            for j in range(1, self.D):
                norms += carrier2.roll(-j)
            invnorms = 1 / torch.sqrt(norms)
            self.invnorms.append(invnorms)

    def gen_mod1(self, n):

        u = torch.rand(size=(n//2,), dtype=self.dtype, generator = self.generator, device = self.generator.device)
        Y = torch.zeros(n//2+1, dtype=torch.complex64, device=self.device)
        c = torch.cos(2 * torch.pi * u)
        s = torch.sin(2 * torch.pi * u)
        Y[0].real = torch.sign(c[-1])
        Y[n//2].real = torch.sign(s[-1])
        Y[1:n//2].real = c[:-1]
        Y[1:n//2].imag = s[:-1]

        zp = torch.fft.irfft(Y)

        return zp.to(dtype=self.dtype)

    def gen_classical(self, n):
        w = torch.randn(n, dtype=self.dtype, generator = self.generator, device = self.generator.device) * (1/np.sqrt(n))
        return w / torch.sqrt(torch.dot(w, w))

    def encode(self, data, bitlen=None, return_components=False):
        def batched_with_padding(iterable, batch_size, fill_value=None):
            it = iter(iterable)
            for batch in iter(lambda: list(islice(it, batch_size)), []):
                yield batch + [fill_value] * (batch_size - len(batch))
        # chunk into 24-bit sequences and convert to numbers
        self.generator.manual_seed(self.key)
        w = 0

        basis = []
        for i, chunk in enumerate(batched_with_padding(data, self.N, 0)):
            if i >= self.M:
                break
            # generate carrier
            carrier = self.basis[i].to(device=self.device, dtype=self.dtype)

            chunk = bytes(chunk)
            value = int.from_bytes(chunk, byteorder="big", signed=self.use_sign)

            # negate carrier to code sign
            if value < 0:
                carrier = -carrier
                value = -value
                
            if self.use_flip:
                # flip carrier to code MSB
                flip = (value & (1 << self.BITS))
                value &= ~(1 << self.BITS)
                if flip:
                    #print("flip carrier to code MSB")
                    carrier = carrier.flip(dims=(0,))
                    value = -value - 1 # cycle in the other direction

            # cyclic permutation to encode the rest of the message
            carrier = torch.roll(carrier, -value)

            # truncate to 256 D
            carrier = carrier[:self.D]

            w += carrier

            basis.append(carrier)

        if return_components:
            return [b.float().cpu() for b in basis]
        else:
            # normalize
            w = w / torch.sqrt(torch.dot(w,w))
            return w.float().cpu().numpy()

    def lowmem_absmax(self, CZ):
        v0p = torch.argmax(CZ)
        c0p = CZ[v0p]

        if not self.use_sign:
            return c0p, v0p

        v0n = torch.argmin(CZ)
        c0n = CZ[v0n]

        if abs(c0n) > abs(c0p):
            c0 = c0n
            v0 = v0n
        else:
            c0 = c0p
            v0 = v0p
        return c0, v0

    def decode_plain(self, w):
        return self.decode_direct(w)

    def decode_direct(self, w):

        # Convert w to a tensor
        w = torch.tensor(w, dtype=self.dtype, device=self.device)
        D = w.shape[0]

        # Reseed the RNG
        self.generator.manual_seed(self.key)

        # Compute correlation directly to reduce memory usage
        seq = []
        S = torch.zeros((1, self.L + 2*D), dtype=w.dtype, device=w.device)
        N = torch.zeros((self.L + D + 1,), dtype=w.dtype, device=w.device)

        basis = []
        for i in range(self.M):  # 96-bit max
            carrier = self.basis[i].to(device=self.device, dtype=self.dtype)
            invnorms = self.invnorms[i].to(device=self.device, dtype=self.dtype)

            L = carrier.shape[0]

            # make filter from w
            weight = w.reshape(1,1,-1)
            weight_f = w.flip(dims=(0,)).reshape(1,1,-1) # for correlation

            # circular padding, D-1 should be enough but keep aligned for speed
            S[0,:D] = carrier[-D:]
            S[0,D:L+D] = carrier
            S[0,L+D:] = carrier[:D]
            N[:D] = invnorms[-D:]
            N[D:L+D] = invnorms
            N[L+D:] = invnorms[0]

            # Correlate
            CZ = torch.nn.functional.conv1d(S, weight)[0]
            CZ *= N


            T = 2**self.BITS
            if self.use_flip:
                c0, v0 = self.lowmem_absmax(CZ)
                CY = torch.nn.functional.conv1d(S, weight_f)[0]
                CY *= N.roll(D)

                c1, v1 = self.lowmem_absmax(CY)
                if abs(c0) > abs(c1):
                    value = (v0 + L - D) % T # v0 - D % L
                    c = c0
                else:
                    value = T + (v1 + L - 1) % T # v1 - 1 % L
                    c = c1
            else:
                C = CZ[D:L+D]
                # Get best correlation
                c, value = self.lowmem_absmax(C)[:T]

            # restore sign
            if self.use_sign and c < 0:
                value = -value

            chunk = int.to_bytes(int(value), self.N, byteorder="big", signed=self.use_sign)
            seq.append(chunk)

        data = b''.join(seq)
        # print("decode_plain : ",data)
        return data

    def decode(self, w):
        if self.direct:
            return self.decode_direct(w)

        # Convert w to a tensor
        w = torch.tensor(w, dtype=self.dtype, device=self.device)
        D = w.shape[0]

        # Reseed the RNG
        self.generator.manual_seed(self.key)

        # Compute correlation in parallel via FFT
        seq = []
        basis = []
        for i in range(self.M):  # 96-bit max
            carrier = self.basis[i].to(device=self.device, dtype=self.dtype)
            invnorms = self.invnorms[i].to(device=self.device, dtype=self.dtype)

            L = carrier.shape[0]

            fB = torch.fft.rfft(carrier)

            # allocate for zero-padding
            Z = torch.zeros(self.L, dtype=self.dtype, device=self.device)

            # Correlate via FFT
            # Zero-pad signal
            Z[:w.shape[0]] = w
            fY = torch.fft.rfft(Z)

            fCZ = fB * fY.conj()
            CZ = torch.fft.irfft(fCZ)

            caib = fB * fY.real
            CAIB = torch.fft.irfft(caib)
            fY.real = 0
            fB *= fY
            dbia = fB
            torch.fft.irfft(dbia, out=Z)
            DBIA = Z
            CZ = CAIB - DBIA

            if self.use_flip:

                CY = CAIB + DBIA

                CZ *= invnorms
                CY *= invnorms.roll(D)

                # cat to restore MSB: if flipped second part will match
                C = torch.cat((CZ[:2**self.BITS], CY[:2**self.BITS]), dim=0)
            else:

                CZ *= invnorms
                C = CZ[:2**self.BITS]

            # Get best correlation
            value = torch.argmax(torch.abs(C) if self.use_sign else C)

            c = C[value]
            # restore sign
            if self.use_sign and c < 0:
                value = -value

            chunk = int.to_bytes(int(value), self.N, byteorder="big", signed=self.use_sign)
            seq.append(chunk)

        data = b''.join(seq)

        return data


class ZnLatticeModulator(Modulator):

    def __init__(self, dim, r2):
        # import here to avoid dependency if unused
        from lattices import Zn_lattice
        #dim = 256
        #r2=7 # 50.58 bits
        #r2=8 # 56.54 bits
        #r2=9 # 62.33 bits KO
        #r2=10 # 67.95 bits

        #dim = 128
        #r2=4 # 27.35 bits OK 
        #r2=5 # 32.98 bits OK
        #r2=6 # 38.34 KO

        #dim = 64
        #r2=2 # 12.98 bits
        #r2=3 # 18.35 bits KO

        self.dim = dim
        self.r2 = r2
        self.C = 256 // dim
        self.codec = Zn_lattice.ZnCodec(dim, r2)
        print("nv=%d %.2f bits" % (self.codec.nv, np.log2(float(self.codec.nv))))

    def encode(self, data, bitlen=None):
        #print("data = ", data)
        x = int.from_bytes(data, byteorder="little")
        codes = []
        while x:
            y = x % self.codec.nv
            x //= self.codec.nv
            codes.append(y)
        codes = np.array(codes, dtype=np.uint64)
        assert len(codes) <= self.C, "%d > %d (%d codes = 2**%f)" % (len(codes), self.C, self.codec.nv, np.log2(self.codec.nv))
            
        C = self.C
        
        # encode, flatten and normalize
        encoded_vector = self.codec.decode(codes).reshape(-1) / np.sqrt(self.r2 * C)
        return encoded_vector

    def decode(self, w):
        input_vector = w.copy()
        #print("input_vector = ", input_vector, input_vector.dot(input_vector), input_vector.mean())
        v = input_vector.reshape(-1, self.dim)
        #print("v = ", v)
        decoded = self.codec.encode(v)
        x = int(0)
        for c in reversed(decoded):
            x = int(x * self.codec.nv + c)
        #print("decoded = ", x)
        #print("size = ", int(np.ceil(np.log2(float(self.codec.nv)) * self.C / 8)))
        data = int.to_bytes(int(x), int(np.ceil(np.log2(float(self.codec.nv)) * self.C / 8)), byteorder="little") # TODO: fix
        #print("data = ", data)
        return data

# register modulators and demodulators
MODULATIONS = {
    'cyclic':      lambda k: CyclicModulator(k),
    'Zn256,3':     lambda k: ZnLatticeModulator(256, 3),
}

if __name__ == '__main__':
    import numpy as np
    import tqdm

    USE_FLIP=True
    M=2
    N=4

    np.random.seed(0)   

    # test cyclic modulator
    print("mit GPU")
    modulator = CyclicModulator(42, use_flip=USE_FLIP, N=N, M=M, use_mod1=False)
    errors = 0
    for i in tqdm.tqdm(range(1000)):
        datae = np.random.bytes(N*M)
        v = modulator.encode(datae)

        dataf = modulator.decode_direct(v)
        if dataf != datae:
            print("dataf = ", dataf)
            print("datae = ", datae)
            errors += 1
            print(errors)
        
    print("errors: ", errors)

    modulator = CyclicModulator(42, use_fp16=True, use_flip=USE_FLIP, M=M)
    errors = 0
    for i in tqdm.tqdm(range(1000)):
        datae = np.random.bytes(3*M)
        v = modulator.encode(datae)
        datad = modulator.decode(v)

        if datad != datae:

            errors += 1
    print("errors: ", errors)

    print("no GPU")
    modulator = CyclicModulator(42, use_gpu=False, use_flip=USE_FLIP, M=M)
    errors = 0
    for i in tqdm.tqdm(range(100)):
        datae = np.random.bytes(3*M)
        v = modulator.encode(datae)
        datad = modulator.decode(v)
        if datad != datae:

            errors += 1

    print("errors: ", errors)
