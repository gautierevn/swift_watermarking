import numpy as np

def sqr(x):
    return x * x

def popcount64(x):
    return bin(x).count("1")

def sum_of_sq(total, v, n, add=0):
    if total < 0:
        return []
    elif n == 1:
        while (v + add) ** 2 > total:
            v -= 1
        if (v + add) ** 2 == total:
            return [v + add]
        else:
            return []
    else:
        res = []
        while v >= 0:
            sub_points = sum_of_sq(total - (v + add) ** 2, v, n - 1, add)
            for i in range(0, len(sub_points), n - 1):
                res.append(v + add)
                res.extend(sub_points[i:i + n - 1])
            v -= 1
        return res

class Comb:
    def __init__(self, nmax):
        self.nmax = nmax
        self.tab = [[0] * nmax for _ in range(nmax)]
        self.tab[0][0] = 1
        for i in range(1, nmax):
            self.tab[i][0] = 1
            for j in range(1, i + 1):
                self.tab[i][j] = self.tab[i - 1][j] + self.tab[i - 1][j - 1]

    def __call__(self, n, p):
        if p > n:
            return 0
        return self.tab[n][p]

comb = Comb(300)

class Repeats:
    def __init__(self, dim=0, c=None):
        self.dim = dim
        self.repeats = []
        if c is not None:
            for i in range(dim):
                found = False
                for j in range(len(self.repeats)):
                    if self.repeats[j][0] == c[i]:
                        self.repeats[j][1] += 1
                        found = True
                        break
                if not found:
                    self.repeats.append([c[i], 1])

    def count(self):
        accu = 1
        remain = self.dim
        for i in range(len(self.repeats)):
            accu *= comb(remain, self.repeats[i][1])
            remain -= self.repeats[i][1]
        return accu

    def encode(self, c):
        if self.dim < 64:
            return self.encode_64(c)
        else:
            return self.encode_bool(c)

    def encode_64(self, c):
        coded = 0
        nfree = self.dim
        code = 0
        shift = 1
        for r in self.repeats:
            rank = 0
            occ = 0
            code_comb = 0
            tosee = ~coded
            while True:
                i = tosee.bit_length() - 1
                tosee &= ~(1 << i)
                if c[i] == r[0]:
                    code_comb += comb(rank, occ + 1)
                    occ += 1
                    coded |= 1 << i
                    if occ == r[1]:
                        break
                rank += 1
            max_comb = comb(nfree, r[1])
            code += shift * code_comb
            shift *= max_comb
            nfree -= r[1]
        return code

    def encode_bool(self, c):
        coded = [False] * self.dim
        nfree = self.dim
        code = 0
        shift = 1
        for r in self.repeats:
            rank = 0
            occ = 0
            code_comb = 0
            while True:
                i = next(j for j in range(self.dim) if not coded[j])
                if c[i] == r[0]:
                    code_comb += comb(rank, occ + 1)
                    occ += 1
                    coded[i] = True
                    if occ == r[1]:
                        break
                rank += 1
            max_comb = comb(nfree, r[1])
            code += shift * code_comb
            shift *= max_comb
            nfree -= r[1]
        return code

    def decode(self, code):
        if self.dim < 64:
            return self.decode_64(code)
        else:
            return self.decode_bool(code)

    def decode_64(self, code):
        decoded = 0
        nfree = self.dim
        c = [0] * self.dim
        for r in self.repeats:
            max_comb = comb(nfree, r[1])
            code_comb = code % max_comb
            code //= max_comb
            occ = 0
            rank = nfree
            next_rank = self.decode_comb_1(code_comb, r[1], rank)
            tosee = ((1 << self.dim) - 1) ^ decoded
            while True:
                i = tosee.bit_length() - 1
                tosee &= ~(1 << i)
                rank -= 1
                if rank == next_rank:
                    decoded |= 1 << i
                    c[i] = r[0]
                    occ += 1
                    if occ == r[1]:
                        break
                    next_rank = self.decode_comb_1(code_comb, r[1] - occ, next_rank)
            nfree -= r[1]
        return c

    def decode_bool(self, code):
        decoded = [False] * self.dim
        nfree = self.dim
        c = [0] * self.dim
        for r in self.repeats:
            max_comb = comb(nfree, r[1])
            code_comb = code % max_comb
            code //= max_comb
            occ = 0
            rank = nfree
            next_rank = self.decode_comb_1(code_comb, r[1], rank)
            for i in range(self.dim - 1, -1, -1):
                if not decoded[i]:
                    rank -= 1
                    if rank == next_rank:
                        decoded[i] = True
                        c[i] = r[0]
                        occ += 1
                        if occ == r[1]:
                            break
                        next_rank = self.decode_comb_1(code_comb, r[1] - occ, next_rank)
            nfree -= r[1]
        return c

    def decode_comb_1(self, n, k1, r):
        while comb(r, k1) > n:
            r -= 1
        n -= comb(r, k1)
        return r

def fvec_inner_product(x, y, d):
    return np.dot(x[:d], y[:d])

class ZnSphereSearch:
    def __init__(self, dim, r2):
        self.dimS = dim
        self.r2 = r2
        self.voc = sum_of_sq(r2, int(np.ceil(np.sqrt(r2)) + 1), dim)
        self.natom = len(self.voc) // dim

    def search(self, x, c=None):
        tmp = np.zeros(2 * self.dimS)
        tmp_int = np.zeros(self.dimS, dtype=np.int32)
        return self.search_impl(x, c, tmp, tmp_int)

    def search_impl(self, x, c, tmp, tmp_int, ibest_out=None):
        dim = self.dimS
        assert self.natom > 0
        o = tmp_int
        xabs = tmp[:dim]
        xperm = tmp[dim:]

        # argsort
        o[:] = np.argsort(np.abs(x))[::-1]
        xabs[:] = np.abs(x)[o]
        xperm[:] = xabs

        # find best
        ibest = -1
        dpbest = -100
        for i in range(self.natom):
            dp = fvec_inner_product(self.voc[i * dim:(i + 1) * dim], xperm, dim)
            if dp > dpbest:
                dpbest = dp
                ibest = i

        # revert sort
        cin = self.voc[ibest * dim:(ibest + 1) * dim]
        if c is None:
            c = np.zeros(dim)
        c[o] = np.where(x[o] >= 0, cin, -cin)
        if ibest_out is not None:
            ibest_out[0] = ibest
        return dpbest

    def search_multi(self, x, c_out, dp_out):
        n = x.shape[0]
        for i in range(n):
            dp_out[i] = self.search(x[i], c_out[i])

class VectorCodec:
    def __init__(self, dim):
        self.nv = 0
        self.dim = dim

    def encode(self, x):
        raise NotImplementedError

    def decode(self, code):
        raise NotImplementedError

    def encode_multi(self, c, codes):
        n = c.shape[0]
        for i in range(n):
            codes[i] = self.encode(c[i])

    def decode_multi(self, codes, c):
        n = codes.shape[0]
        for i in range(n):
            c[i] = self.decode(codes[i])

    def find_nn(self, codes, xq, labels, distances):
        nq = xq.shape[0]
        distances[:] = -1e20
        labels[:] = -1
        c = np.zeros(self.dim)
        for i in range(len(codes)):
            code = codes[i]
            self.decode(code, c)
            for j in range(nq):
                dis = fvec_inner_product(xq[j], c, self.dim)
                if dis > distances[j]:
                    distances[j] = dis
                    labels[j] = i
                    
class CodeSegment(Repeats):
    def __init__(self, r):
        super().__init__(r.dim)
        print("r.dim : ",r.dim)
        self.dim = r.dim
        self.c0 = 0
        self.signbits = 0
        
class ZnSphereCodec(ZnSphereSearch, VectorCodec):
    def __init__(self, dim, r2):
        super().__init__(dim, r2)
        self.code_segments = []
        self.nv = 0
        for i in range(self.natom):
            repeats = Repeats(dim, self.voc[i * dim:(i + 1) * dim])
            cs = CodeSegment(repeats)
            # cs = repeats.repeats
            cs.c0 = self.nv
            br = repeats.repeats[-1]
            cs.signbits = dim if br[0] != 0 else dim - br[1]
            self.code_segments.append(cs)
            self.nv += repeats.count() << cs.signbits

        nvx = self.nv
        self.code_size = 0
        while nvx > 0:
            nvx >>= 8
            self.code_size += 1

    def search_and_encode(self, x):
        tmp = np.zeros(2 * self.dim)
        tmp_int = np.zeros(self.dim, dtype=np.int32)
        ano = np.zeros(1, dtype=np.int32)
        c = np.zeros(self.dim)
        self.search_impl(x, c, tmp, tmp_int, ano)
        signs = 0
        cabs = np.abs(c)
        nnz = np.count_nonzero(c)
        cs = self.code_segments[ano[0]]
        assert nnz == cs.signbits
        code = cs.c0 + (c < 0).astype(np.uint64).dot(1 << np.arange(nnz, dtype=np.uint64))
        code += cs.encode(cabs) << cs.signbits
        return code

    def encode(self, x):
        return self.search_and_encode(x)

    def decode(self, code):
        i0 = 0
        i1 = self.natom
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
        c = cs.decode(code)
        nnz = 0
        for i in range(self.dim):
            if c[i] != 0:
                if signs & (1 << nnz):
                    c[i] = -c[i]
                nnz += 1
        return c

class ZnSphereCodecRec(VectorCodec):
    def __init__(self, dim, r2):
        super().__init__(dim)
        self.r2 = r2
        self.log2_dim = int(np.log2(dim))
        assert dim == (1 << self.log2_dim), "dimension must be a power of 2"
        self.all_nv = np.zeros((self.log2_dim + 1, r2 + 1), dtype=np.uint64)
        self.all_nv_cum = np.zeros((self.log2_dim + 1, r2 + 1, r2 + 1), dtype=np.uint64)
        for r2a in range(r2 + 1):
            r = int(np.sqrt(r2a))
            if r * r == r2a:
                self.all_nv[0, r2a] = 1 if r == 0 else 2
            else:
                self.all_nv[0, r2a] = 0
        for ld in range(1, self.log2_dim + 1):
            for r2sub in range(r2 + 1):
                nv = 0
                for r2a in range(r2sub + 1):
                    r2b = r2sub - r2a
                    self.all_nv_cum[ld, r2sub, r2a] = nv
                    nv += self.all_nv[ld - 1, r2a] * self.all_nv[ld - 1, r2b]
                self.all_nv[ld, r2sub] = nv
        self.nv = self.all_nv[self.log2_dim, r2]
        nvx = self.nv
        self.code_size = 0
        while nvx > 0:
            nvx = np.int64(nvx)
            print(type(nvx))
            nvx >>= 8
            self.code_size += 1
        cache_level = min(3, self.log2_dim - 1)
        self.decode_cache_ld = 0
        assert cache_level <= self.log2_dim
        self.decode_cache = [None] * (r2 + 1)
        for r2sub in range(r2 + 1):
            ld = cache_level
            nvi = self.all_nv[ld, r2sub]
            dimsub = (1 << cache_level)
            cache = np.zeros((nvi, dimsub), dtype=np.float32)
            code0 = self.all_nv_cum[cache_level + 1, r2, r2 - r2sub]
            for i in range(nvi):
                c = self.decode(i + code0)
                cache[i] = c[-dimsub:]
            self.decode_cache[r2sub] = cache
        self.decode_cache_ld = cache_level

    def encode(self, c):
        return self.encode_centroid(c)

    def encode_centroid(self, c):
        codes = np.zeros(self.dim, dtype=np.uint64)
        norm2s = np.zeros(self.dim, dtype=np.int32)
        norm2s[:] = np.where(c != 0, np.round(c ** 2).astype(np.int32), 0)
        codes[:] = np.where(c != 0, np.where(c >= 0, 0, 1), 0)
        dim2 = self.dim // 2
        for ld in range(1, self.log2_dim + 1):
            for i in range(dim2):
                r2a = norm2s[2 * i]
                r2b = norm2s[2 * i + 1]
                code_a = codes[2 * i]
                code_b = codes[2 * i + 1]
                codes[i] = self.all_nv_cum[ld, r2a + r2b, r2a] + code_a * self.all_nv[ld - 1, r2b] + code_b
                norm2s[i] = r2a + r2b
            dim2 //= 2
        return codes[0]

    def decode(self, code):
        codes = np.zeros(self.dim, dtype=np.uint64)
        norm2s = np.zeros(self.dim, dtype=np.int32)
        codes[0] = code
        norm2s[0] = self.r2
        dim2 = 1
        for ld in range(self.log2_dim, self.decode_cache_ld - 1, -1):
            for i in range(dim2 - 1, -1, -1):
                r2sub = norm2s[i]
                i0 = 0
                i1 = r2sub + 1
                codei = codes[i]
                cum = self.all_nv_cum[ld, r2sub]
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
                code_a = codei // self.all_nv[ld - 1, r2b]
                code_b = codei % self.all_nv[ld - 1, r2b]
                codes[2 * i] = code_a
                codes[2 * i + 1] = code_b
            dim2 *= 2
        if self.decode_cache_ld == 0:
            c = np.zeros(self.dim, dtype=np.float32)
            for i in range(self.dim):
                if norm2s[i] == 0:
                    c[i] = 0
                else:
                    r = np.sqrt(norm2s[i])
                    assert r * r == norm2s[i]
                    c[i] = r if codes[i] == 0 else -r
        else:
            subdim = 1 << self.decode_cache_ld
            assert dim2 * subdim == self.dim
            c = np.zeros(self.dim, dtype=np.float32)
            for i in range(dim2):
                cache = self.decode_cache[norm2s[i]]
                assert codes[i] < len(cache)
                c[i * subdim:(i + 1) * subdim] = cache[codes[i]]
        return c
