import unittest
import numpy as np
import lechat_utils as ZnSphere
from tqdm import tqdm

class TestZnCodec(unittest.TestCase):
    def test_codec(self):
        self.do_test(32, 14)

    def test_codec_rec(self):
        self.do_test(24, 79)

    def do_test(self, dim, r2):
        codec = ZnSphere.ZnSphereCodec(dim, r2)
        rs = np.random.RandomState(123)
        n = 2000
        x = rs.randn(n, dim).astype('float32')
        x /= np.sqrt((x ** 2).sum(1)).reshape(-1, 1)
        quant = codec.search(x)
        codes = codec.encode(x)
        x_decoded = codec.decode(codes)
        assert np.all(x_decoded == quant)
        codec2 = ZnSphere.ZnSphereCodecRec(dim, r2)
        quant2 = codec2.search(x)
        assert np.all(quant == quant2)

class BasicTest(unittest.TestCase):
    def test_comb(self):
        assert ZnSphere.comb(2, 1) == 2

    def test_repeats(self):
        rs = np.random.RandomState(123)
        dim = 2
        for i in tqdm(range(10)):
            vec = np.floor((rs.rand(dim) ** 7) * 3).astype('float32')
            vecs = vec.copy(); vecs.sort()
            repeats = ZnSphere.Repeats(dim, vecs)
            code = repeats.encode(vec)
            vec2 = repeats.decode(code)
            assert np.all(vec == vec2)

class TestZnSphereCodecRec(unittest.TestCase):
    def test_encode_centroid(self):
        dim = 8
        r2 = 5
        ref_codec = ZnSphere.ZnSphereCodec(dim, r2)
        codec = ZnSphere.ZnSphereCodecRec(dim, r2)
        assert ref_codec.nv == codec.nv
        s = set()
        for i in tqdm(range(ref_codec.nv)):
            c = ref_codec.decode(i)
            code = codec.encode_centroid(c)
            assert 0 <= code < codec.nv
            s.add(code)
        assert len(s) == codec.nv

    def test_codec(self):
        dim = 16
        r2 = 6
        codec = ZnSphere.ZnSphereCodecRec(dim, r2)
        for i in tqdm(range(codec.nv)):
            c = codec.decode(i)
            code = codec.encode_centroid(c)
            assert code == i

if __name__ == '__main__':
    unittest.main()
