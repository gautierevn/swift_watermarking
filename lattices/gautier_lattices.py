import numpy as np
from concurrent.futures import ThreadPoolExecutor
import math

from collections import Counter
def comb(n, k):
    # Combination calculation, equivalent to nCk
    from math import comb
    return comb(n, k)

def decode_comb_1(n, k1, r):
    while comb(r, k1) > n:
        r -= 1
    n -= comb(r, k1)
    return r

        
class Repeat:
    def __init__(self, value, count):
        self.val = value
        self.n = count

class Repeats:
    def __init__(self, dim, c=None):
        self.dim = dim
        print("test",type(dim))
        self.repeats = []
        if c is None:
            c = np.zeros(self.dim, dtype='float32')

            print("youhou")
        repeat_counts = Counter(c)
        for value, count in repeat_counts.items():
            if any(r.val == value for r in self.repeats):
                for r in self.repeats:
                    if r.val == value:
                        r.n += 1
            else:
                self.repeats.append(Repeat(value, count))

    def count(self):
        accu = 1
        remain = self.dim
        for r in self.repeats:
            accu *= self.comb(remain, r.n)
            remain -= r.n
        return accu

    def encode(self, c):
        if self.dim < 64:
            return self.repeats_encode_64(c)
        else:
            return self.repeats_encode_large(c)

    def decode(self, code,c):
        # c = np.zeros(self.dim, dtype=float)
        # print(self)
        print("ADDDDDDDDDDDD")
        if self.dim < 64:
            self.repeats_decode_64(code, c)
        else :
            decoded = np.zeros(self.dim, dtype=bool)
            nfree = self.dim
            print(self.repeats)
            for r in self.repeats:
                max_comb = comb(nfree, r.n)
                code_comb = code % max_comb
                code //= max_comb

                occ = 0
                rank = nfree
                next_rank = decode_comb_1(code_comb, r.n, rank)
                for i in range(self.dim - 1, -1):
                    exit()
                    print(decoded)
                    if not decoded[i]:
                        print("bbbbbbbbbbbbbbbbbbb")
                        rank -= 1
                        if rank == next_rank:
                            decoded[i] = True
                            c[i] = r.val

                            occ += 1
                            if occ == r.n:
                                break
                            next_rank = decode_comb_1(code_comb, r.n - occ, next_rank)
                nfree -= r.n

    def repeats_encode_64(self, c):
        coded = 0
        nfree = self.dim
        code = 0
        shift = 1
        for r in self.repeats:
            print(r.n)
            rank = 0
            occ = 0
            code_comb = 0
            tosee = ~coded
            while True:
                if not tosee:
                    break
                i = (tosee & -tosee).bit_length() - 1
                print(i, occ)
                tosee &= ~(1 << i)
                print(c[i], r.val)
                if c[i] == r.val:
                    code_comb += self.comb(rank, occ + 1)
                    occ += 1
                    coded |= 1 << i
                    if occ == r.n:
                        break
                rank += 1
            max_comb = self.comb(nfree, r.n)
            code += shift * code_comb
            shift *= max_comb
            nfree -= r.n
        return code

    def repeats_decode_64(self, code, c):
        decoded = 0
        nfree = self.dim
        for r in self.repeats:
            # print("val",r.val)
            max_comb = self.comb(nfree, r.n)
            code_comb = code % max_comb
            code //= max_comb

            occ = 0
            rank = nfree
            next_rank = self.decode_comb_1(code_comb, r.n, rank)
            tosee = ((1 << self.dim) - 1) ^ decoded
            while True:
                if not tosee:
                    break
                leading_zeros = 64 - tosee.bit_length()

                # Calculate the desired value
                i = 63 - leading_zeros

                tosee &= ~(1 << i)
                rank -= 1
                if rank == next_rank:
                    # print(r.val)
                    decoded |= 1 << i
                    c[i] = r.val
                    occ += 1
                    if occ == r.n:
                        break
                    next_rank = self.decode_comb_1(code_comb, r.n - occ, next_rank)

    @staticmethod
    def comb(n, k):
        from math import comb
        return comb(n, k)

    @staticmethod
    def decode_comb_1(n, k1, r):
        while Repeats.comb(r, k1) > n:
            r -= 1
        n -= Repeats.comb(r, k1)
        return r

# class CodeSegment(Repeats):
#     def __init__(self, r, c0=0, signbits=0):
#         super().__init__(r.dim, [repeat.val for repeat in r.repeats for _ in range(repeat.n)])  # Initialize the base class (Repeats)
#         self.c0 = c0  # First code assigned to segment
#         self.signbits = signbits  # Initialize signbits
#         self.dim = r.dim
#         print("DIMENSION",self.dim)
#         self.repeats = r.repeats  # Copy the repeats list from r

#     def initialize_segment(self, c0, signbits):
#         self.c0 = c0
#         self.signbits = signbits

class VectorCodec:
    def __init__(self, dim):
        self.dim = dim

    def encode_multi(self, n, c):
        codes = np.zeros(n, dtype=np.uint64)
        with ThreadPoolExecutor(max_workers=None) if n > 1000 else nullcontext() as executor:
            futures = {executor.submit(self.encode, c[i * self.dim:(i + 1) * self.dim]): i for i in range(n)}
            for future in futures:
                codes[futures[future]] = future.result()
        return codes

    def decode_multi(self, n, codes):
        c_out = np.zeros(n * self.dim)
        with ThreadPoolExecutor(max_workers=None) if n > 1000 else nullcontext() as executor:
            futures = {executor.submit(self.decode, codes[i], c_out[i * self.dim:(i + 1) * self.dim]): i for i in range(n)}
            for future in futures:
                future.result()
        return c_out

    def find_nn(self, nc, codes, nq, xq):
        labels = np.full(nq, -1, dtype=int)
        distances = np.full(nq, -np.inf)
        c = np.zeros(self.dim)

        for i in range(nc):
            code = codes[i]
            self.decode(code, c)
            for j in range(nq):
                x = xq[j * self.dim:(j + 1) * self.dim]
                dis = np.dot(x, c)
                if dis > distances[j]:
                    distances[j] = dis
                    labels[j] = i

        return labels, distances

def sum_of_sq(total, v, n, add=0):
    def sqr(x):
        return x**2
    
    if total < 0:
        return np.array([])
    elif n == 1:
        while sqr(v + add) > total:
            v -= 1
        if sqr(v + add) == total:
            return np.array([v + add])
        else:
            return np.array([])
    else:
        res = []
        while v >= 0:
            sub_points = sum_of_sq(total - sqr(v + add), v, n - 1, add)
            if sub_points.size > 0:
                for i in range(0, sub_points.size, n - 1):
                    res.append(v + add)
                    res.extend(sub_points[i:i + n - 1])
            v -= 1
        return np.array(res)
        
class ZnSphereSearch:
    def __init__(self, dim, r2):
        self.dimS = dim
        self.r2 = r2
        self.voc = sum_of_sq(r2, int(np.ceil(np.sqrt(r2)) + 1), dim)
        self.natom = len(self.voc) // dim

    def search(self, x, c, tmp=None, tmp_int=None, ibest_out=None):
        if tmp is None:
            tmp = np.zeros(self.dimS * 2)
        if tmp_int is None:
            tmp_int = np.zeros(self.dimS, dtype=int)

        dim = self.dimS
        assert self.natom > 0
        xabs = np.abs(x)
        o = np.argsort(-xabs)  # negative for descending sort
        xperm = xabs[o]

        dpbest = -np.inf
        ibest = -1
        for i in range(self.natom):
            dp = np.dot(self.voc[i * dim:(i + 1) * dim], xperm)
            if dp > dpbest:
                dpbest = dp
                ibest = i
        
        cin = self.voc[ibest * dim:(ibest + 1) * dim]
        for i in range(dim):
            c[o[i]] = np.copysign(cin[i], x[o[i]])
        
        if ibest_out is not None:
            ibest_out[0] = ibest

        return dpbest

    def search_multi(self, n, x, c_out, dp_out):
        def task(i):
            dp_out[i] = self.search(x[i * self.dimS:(i + 1) * self.dimS],
                                    c_out[i * self.dimS:(i + 1) * self.dimS])

        with ThreadPoolExecutor() as executor:
            executor.map(task, range(n))
            
class ZnSphereCodecRecDEUX:
    def __init__(self, dim, r2):
        self.dim = dim
        self.r2 = r2
        self.log2_dim = 0
        while dim > (1 << self.log2_dim):
            self.log2_dim += 1
        assert dim == (1 << self.log2_dim), "dimension must be a power of 2"

        self.all_nv = np.zeros((self.log2_dim + 1) * (r2 + 1), dtype=np.uint64)
        self.all_nv_cum = np.zeros(((self.log2_dim + 1) * (r2 + 1)) ** 2, dtype=np.uint64)

        for r2a in range(r2 + 1):
            r = int(np.sqrt(r2a))
            if r * r == r2a:
                self.all_nv[r2a] = 1 if r == 0 else 2
            else:
                self.all_nv[r2a] = 0

        for ld in range(1, self.log2_dim + 1):
            for r2sub in range(r2 + 1):
                nv = 0
                for r2a in range(r2sub + 1):
                    r2b = r2sub - r2a
                    self.set_nv_cum(ld, r2sub, r2a, nv)
                    nv += self.get_nv(ld - 1, r2a) * self.get_nv(ld - 1, r2b)
                self.all_nv[ld * (r2 + 1) + r2sub] = nv

        self.nv = self.get_nv(self.log2_dim, r2)
        nvx = int(self.nv)
        self.code_size = 0
        while nvx > 0:
            nvx >>= 8
            self.code_size += 1

        self.cache_level = min(3, self.log2_dim - 1)
        self.decode_cache_ld = 0
        assert self.cache_level <= self.log2_dim
        # print([x for x in self.all_nv[:r2 + 1]]) 
        self.decode_cache = [np.zeros(int(1 << self.cache_level) * int(nvi),dtype=np.float32) for nvi in self.all_nv[:r2 + 1]]

        for r2sub in range(r2 + 1):
            ld = self.cache_level
            nvi = self.get_nv(ld, r2sub)
            self.decode_cache[r2sub] = np.zeros(int(nvi) * (1 << self.cache_level),dtype=np.float32)
            c = np.zeros(dim, dtype=np.float32)
            code0 = self.get_nv_cum(self.cache_level + 1, r2, r2 - r2sub)
            for i in range(nvi):
                print("before decode", c)
                self.decode(i + code0, c)
                print("after decode", c)
                self.decode_cache[r2sub][i * (1 << self.cache_level): (i + 1) * (1 << self.cache_level)] = c[self.dim - (1 << self.cache_level):]

    def get_nv(self, ld, r2a):
        return self.all_nv[ld * (self.r2 + 1) + r2a]

    def get_nv_cum(self, ld, r2t, r2a):
        return self.all_nv_cum[(ld * (self.r2 + 1) + r2t) * (self.r2 + 1) + r2a]

    def set_nv_cum(self, ld, r2t, r2a, cum):
        self.all_nv_cum[(ld * (self.r2 + 1) + r2t) * (self.r2 + 1) + r2a] = cum

    def decode(self, code, c):
        codes = np.zeros(self.dim, dtype=np.uint64)
        norm2s = np.zeros(self.dim, dtype=int)
        codes[0] = code
        norm2s[0] = self.r2
        dim2 = 1
        for ld in range(self.log2_dim, self.decode_cache_ld, -1):
            for i in range(dim2 - 1, -1, -1):
                r2sub = norm2s[i]
                i0 = 0
                i1 = r2sub + 1
                codei = codes[i]
                cum = self.all_nv_cum[(ld * (self.r2 + 1) + r2sub) * (self.r2 + 1):(ld * (self.r2 + 1) + r2sub + 1) * (self.r2 + 1)]
                while i1 > i0 + 1:
                    imed = (i0 + i1) // 2
                    
                    if cum[imed] <= codei:
                        i0 = imed
                    else:
                        i1 = imed
                r2a = i0
                r2b = r2sub - i0
                codei -= cum[r2a]
                # print("codei :",codei)
                
                norm2s[2 * i] = r2a
                norm2s[2 * i + 1] = r2b
                code_a = codei / self.get_nv(ld - 1, r2b)
                # print("code a :",code_a)
                code_b = codei % self.get_nv(ld - 1, r2b)
                # print("code b :",code_b)
                codes[2 * i] = code_a
                codes[2 * i + 1] = code_b
            # print(f"{i}_codes", codes)
            dim2 *= 2
        # print(codes)
        # exit()
        if self.decode_cache_ld == 0:
            for i in range(self.dim):
                if norm2s[i] == 0:
                    c[i] = 0
                else:
                    r = np.sqrt(norm2s[i])
                    assert r * r == norm2s[i], "Square root check failed"
                    c[i] = r if codes[i] == 0 else -r
        else:
            subdim = 1 << self.decode_cache_ld
            assert (dim2 * subdim) == self.dim, "Dimension mismatch in decode"
            for i in range(dim2):
                cache = self.decode_cache[norm2s[i]]
                assert codes[i] < len(cache), "Code index out of bounds"
                c[i * subdim:(i + 1) * subdim] = cache[codes[i] * subdim:(codes[i] + 1) * subdim]
        return c

    def encode_centroid(self, c):
        codes = np.zeros(self.dim, dtype=np.uint64)
        norm2s = np.zeros(self.dim, dtype=int)
        for i in range(self.dim):
            if c[i] == 0:
                codes[i] = 0
                norm2s[i] = 0
            else:
                r2i = int(c[i] * c[i])
                norm2s[i] = r2i
                codes[i] = 0 if c[i] >= 0 else 1
        dim2 = self.dim // 2
        for ld in range(1, self.log2_dim + 1):
            for i in range(dim2):
                r2a = norm2s[2 * i]
                r2b = norm2s[2 * i + 1]
                code_a = codes[2 * i]
                code_b = codes[2 * i + 1]
                codes[i] = self.get_nv_cum(ld, r2a + r2b, r2a) + code_a * self.get_nv(ld - 1, r2b) + code_b
                norm2s[i] = r2a + r2b
            dim2 //= 2
        # print("encoded",codes)
        return codes[0]

    def encode(self, c):
        return self.encode_centroid(c)
    
class ZnSphereCodecRec:
    def __init__(self, dim, r2):
        self.dim = dim
        self.r2 = r2
        self.log2_dim = 0
        while dim > (1 << self.log2_dim):
            self.log2_dim += 1
        assert dim == (1 << self.log2_dim), "dimension must be a power of 2"

        self.all_nv = np.zeros((self.log2_dim + 1) * (r2 + 1), dtype=np.uint64)
        self.all_nv_cum = np.zeros(((self.log2_dim + 1) * (r2 + 1)) ** 2, dtype=np.uint64)

        for r2a in range(r2 + 1):
            r = int(np.sqrt(r2a))
            if r * r == r2a:
                self.all_nv[r2a] = 1 if r == 0 else 2
            else:
                self.all_nv[r2a] = 0

        for ld in range(1, self.log2_dim + 1):
            for r2sub in range(r2 + 1):
                nv = 0
                for r2a in range(r2sub + 1):
                    r2b = r2sub - r2a
                    self.set_nv_cum(ld, r2sub, r2a, nv)
                    nv += self.get_nv(ld - 1, r2a) * self.get_nv(ld - 1, r2b)
                self.all_nv[ld * (r2 + 1) + r2sub] = nv

        self.nv = self.get_nv(self.log2_dim, r2)
        nvx = self.nv
        self.code_size = 0
        while nvx > 0:
            nvx >>= 8
            self.code_size += 1

        self.cache_level = min(3, self.log2_dim - 1)
        self.decode_cache_ld = 0
        assert self.cache_level <= self.log2_dim
        self.decode_cache = [np.zeros((1 << self.cache_level) * int(nvi)) for nvi in self.all_nv[:r2 + 1]]

        for r2sub in range(r2 + 1):
            ld = self.cache_level
            nvi = self.get_nv(ld, r2sub)
            self.decode_cache[r2sub] = np.zeros(nvi * (1 << self.cache_level))
            c = np.zeros(dim)
            code0 = self.get_nv_cum(self.cache_level + 1, r2, r2 - r2sub)
            for i in range(nvi):
                self.decode(i + code0, c)
                self.decode_cache[r2sub][i * (1 << self.cache_level): (i + 1) * (1 << self.cache_level)] = c[self.dim - (1 << self.cache_level):]

    def get_nv(self, ld, r2a):
        return int(self.all_nv[ld * (self.r2 + 1) + r2a])

    def get_nv_cum(self, ld, r2t, r2a):
        return int(self.all_nv_cum[(ld * (self.r2 + 1) + r2t) * (self.r2 + 1) + r2a])

    def set_nv_cum(self, ld, r2t, r2a, cum):
        self.all_nv_cum[(ld * (self.r2 + 1) + r2t) * (self.r2 + 1) + r2a] = cum

    def decode(self, code,c):
        codes = np.zeros(self.dim, dtype=np.uint64)
        norm2s = np.zeros(self.dim, dtype=int)

        codes[0] = code
        norm2s[0] = self.r2
        dim2 = 1
        for ld in range(self.log2_dim, self.decode_cache_ld, -1):
            for i in range(dim2 - 1, -1, -1):
                r2sub = norm2s[i]
                i0 = 0
                i1 = r2sub + 1
                codei = codes[i]
                cum = self.all_nv_cum[(ld * (self.r2 + 1) + r2sub) * (self.r2 + 1):(ld * (self.r2 + 1) + r2sub + 1) * (self.r2 + 1)]
                while i1 > i0 + 1:
                    imed = (i0 + i1) // 2
                    if cum[imed] <= codei:
                        i0 = imed
                    else:
                        i1 = imed
                r2a = i0
                r2b = r2sub - i0
                codei -= cum[r2a]
                norm2s[2 * i] = r2a
                norm2s[2 * i + 1] = r2b
                code_a = codei // self.get_nv(ld - 1, r2b)
                code_b = codei % self.get_nv(ld - 1, r2b)
                codes[2 * i] = code_a
                codes[2 * i + 1] = code_b
            dim2 *= 2

        if self.decode_cache_ld == 0:
            for i in range(self.dim):
                if norm2s[i] == 0:
                    c[i] = 0
                else:
                    r = np.sqrt(norm2s[i])
                    assert r * r == norm2s[i], "Square root check failed"
                    c[i] = r if codes[i] == 0 else -r
        else:
            subdim = 1 << self.decode_cache_ld
            assert (dim2 * subdim) == self.dim, "Dimension mismatch in decode"
            for i in range(dim2):
                cache = self.decode_cache[norm2s[i]]
                assert codes[i] < len(cache), "Code index out of bounds"
                c[i * subdim:(i + 1) * subdim] = cache[codes[i] * subdim:(codes[i] + 1) * subdim]

    def encode_centroid(self, c):
        dim = self.dim
        log2_dim = self.log2_dim
        codes = [0] * dim
        norm2s = [0] * dim
        # print(c)
        for i in range(dim):
            if c[i] == 0:
                codes[i] = 0
                norm2s[i] = 0
            else:
                r2i = int(c[i] * c[i])
                norm2s[i] = r2i
                codes[i] = 0 if c[i] >= 0 else 1

        dim2 = dim // 2
        for ld in range(1, log2_dim + 1):
            for i in range(dim2):
                r2a = norm2s[2 * i]
                r2b = norm2s[2 * i + 1]

                code_a = codes[2 * i]
                code_b = codes[2 * i + 1]

                codes[i] = (
                    self.get_nv_cum(ld, r2a + r2b, r2a) +
                    code_a * self.get_nv(ld - 1, r2b) +
                    code_b
                )
                norm2s[i] = r2a + r2b

            dim2 //= 2

        return codes[0]

    def encode(self,c):
        return self.encode_centroid(c)

class ZnSphereSearch:
    def __init__(self, dim, r2):
        self.dimS = dim
        self.r2 = r2
        self.voc = sum_of_sq(r2, int(np.ceil(np.sqrt(r2)) + 1), dim);
        self.natom = int(len(self.voc) / dim)
    def search(self, x,c,ibest_out = None):
        # c = np.zeros_like(x)
        tmp = np.zeros(2 * self.dimS, dtype=float)
        tmp_int = np.zeros(self.dimS, dtype=int)
        return self.search_full(x, c, tmp, tmp_int,ibest_out)

    def search_full(self, x, c, tmp, tmp_int, ibest_out=None):
        dim = self.dimS
        assert self.natom > 0

        o = tmp_int
        xabs = tmp[:dim]
        xperm = tmp[dim:2*dim]

        # argsort
        for i in range(dim):
            o[i] = i
            xabs[i] = abs(x[i])
        o = sorted(o, key=lambda a: xabs[a], reverse=True)
        for i in range(dim):
            xperm[i] = xabs[o[i]]

        # find best
        ibest = -1
        dpbest = -100
        for i in range(self.natom):
            dp = np.dot(self.voc[i * dim:(i + 1) * dim], xperm)
            if dp > dpbest:
                dpbest = dp
                ibest = i

        # revert sort
        cin = self.voc[ibest * dim:(ibest + 1) * dim]
        for i in range(dim):
            c[o[i]] = math.copysign(cin[i], x[o[i]])

        if ibest_out is not None:
            ibest_out[0] = ibest
        return dpbest

    def search_multi(self, n, x, c_out, dp_out):
        # Assuming x is a flattened array with n vectors of dimension dimS
        for i in range(n):
            xi = x[i * self.dimS:(i + 1) * self.dimS]
            ci = c_out[i * self.dimS:(i + 1) * self.dimS]
            dp_out[i] = self.search(xi, ci)

class CodeSegment(Repeats):
    def __init__(self, r):
        super().__init__(r.dim)
        print("r.dim : ",r.dim)
        print("using gautier latticse")
        self.dim = r.dim
        self.c0 = 0
        self.signbits = 0

class ZnSphereCodec(ZnSphereSearch, VectorCodec):


    def __init__(self, dim, r2):
        ZnSphereSearch.__init__(self, dim, r2)
        VectorCodec.__init__(self, dim)
        self.code_segments = []
        self.nv = 0
        for i in range(int(self.natom)):
            print("-"*20)
            repeats = Repeats(dim, self.voc[i * dim:(i + 1) * dim])
            print(repeats.repeats[0].val)
            cs = CodeSegment(repeats)
            cs.c0 = self.nv
            br = repeats.repeats[-1]
            cs.signbits = dim if br.val != 0 else (dim - br.n)
            self.code_segments.append(cs)
            self.nv += repeats.count() << cs.signbits

        nvx = self.nv
        self.code_size = 0
        while nvx > 0:
            nvx >>= 8
            self.code_size += 1

    def search_and_encode(self, x):

        ano = np.zeros(1, dtype=int)
        c = np.zeros(self.dim)
        self.search(x,c,ibest_out=ano)
        signs = 0
        cabs = np.abs(c)
        nnz = 0
        for i in range(self.dim):
            if c[i] != 0:
                if c[i] < 0:
                    signs |= 1 << nnz
                nnz += 1
        cs = self.code_segments[ano[0]]

        assert nnz == cs.signbits
        code = cs.c0 + signs
        code += cs.encode(cabs) << cs.signbits

        return code

    def encode(self, x):
        return self.search_and_encode(x)

    def decode(self, code, c):
        i0, i1 = 0, self.natom
        while i0 + 1 < i1:
            imed = (i0 + i1) // 2
            if self.code_segments[imed].c0 <= code:
                i0 = imed
            else:
                i1 = imed
        cs = self.code_segments[i0]
        code -= cs.c0
        signs = code
        code >>= cs.signbits
        print("before decoding ",c)
        cs.decode(code, c)

        nnz = 0
        for i in range(self.dim):
            if c[i] != 0:
                if signs & (1 << nnz):
                    c[i] = -c[i]
                nnz += 1

# Additional classes like Repeats, CodeSegment would need to be defined to match the functionality you're expecting.

