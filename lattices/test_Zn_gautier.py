import gautier_lattices
from gautier_lattices import ZnSphereCodecRec,ZnSphereCodecRecDEUX ,ZnSphereCodec
# from lechat_utils import ZnSphereCodecRec, ZnSphereCodec
import unittest
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
# class TestZnCodec(unittest.TestCase):
#     def test_codec(self):
#         self.do_test(32, 14)

#     def test_codec_rec(self):
#         self.do_test(24, 79)

#     def do_test(self, dim, r2):
#         codec = ZnCodec(dim, r2)
#         rs = np.random.RandomState(123)

#         n = 2000
#         x = rs.randn(n, dim).astype('float32')
#         x /= np.sqrt((x ** 2).sum(1)).reshape(-1, 1)
#         quant = codec.quantize(x)

#         codes = codec.encode(x)
#         x_decoded = codec.decode(codes.reshape(-1, 1))

#         np.testing.assert_array_equal(x_decoded, quant, "Decoded values do not match quantized values.")

#         codec2 = ZnCodecPy(dim, r2)

#         quant2 = codec2.quantize(x)
#         np.testing.assert_array_equal(quant, quant2, "Quantization between ZnCodec and ZnCodecPy does not match.")

# class TestZnSphereCodec(unittest.TestCase):

#     def setUp(self):
#         self.dim = 8
#         self.r2 = 5
#         self.codec = ZnSphereCodec(self.dim, self.r2)

#     def test_initialization(self):
#         codec = self.codec
#         self.assertEqual(codec.dim, self.dim)
#         self.assertGreater(len(codec.voc), 0)
#         self.assertGreater(codec.natom, 0)
#         self.assertGreater(len(codec.code_segments), 0)
#         self.assertGreater(codec.nv, 0)
#         self.assertGreater(codec.code_size, 0)

#     def test_search(self):
#         x = np.random.rand(self.dim)
#         c = np.zeros(self.dim)
#         tmp = np.zeros(self.dim * 2)
#         tmp_int = np.zeros(self.dim, dtype=int)
#         ano = [0]
#         self.codec.search(x, c)
#         self.assertEqual(len(c), self.dim)
#         self.assertEqual(len(tmp), self.dim * 2)
#         self.assertEqual(len(tmp_int), self.dim)
#         self.assertIsInstance(ano[0], int)

#     def test_search_and_encode(self):
#         x = np.random.rand(self.dim)
#         code = self.codec.search_and_encode(x)
#         self.assertIsInstance(code, int)
#         self.assertGreaterEqual(code, 0)

#     def test_encode(self):
#         x = np.random.rand(self.dim)
#         code = self.codec.encode(x)
#         self.assertIsInstance(code, int)
#         self.assertGreaterEqual(code, 0)

#     def test_decode(self):
#         x = np.random.rand(self.dim)
#         encoded_code = self.codec.encode(x)
#         c = np.zeros(self.dim)
#         self.codec.decode(encoded_code, c)
#         self.assertEqual(len(c), self.dim)
#         self.assertTrue(np.all(c <= np.abs(x)))

#     def test_round_trip_encoding(self):
#         x = np.random.randn(self.dim)
#         encoded_code = self.codec.encode(x)
#         print(encoded_code)
#         c = np.zeros(self.dim)
#         self.codec.decode(encoded_code, c)
#         print(c)
#         np.testing.assert_array_almost_equal(x, c, decimal=5)

class TestZnSphereCodec(unittest.TestCase):

    def test_codec(self):

        dim = 32
        r2 = 14
        codec = ZnSphereCodec(dim, r2)
        # print("nb atoms", codec.natom)
        rs = np.random.RandomState(123)
        for i in range(2):
            x = rs.randn(dim).astype('float32')
            ref_c = np.zeros(dim, dtype='float32')
            codec.search(x, ref_c)
            code = codec.search_and_encode(x)
            print("x :", x," code :", code)
            c = np.zeros(dim, dtype='float32')
            codec.decode(code, c)
            print(ref_c, c)
# class TestZnSphereCodecRec(unittest.TestCase):

#     def test_encode_centroid(self):
#         dim = 8
#         r2 = 5
#         ref_codec = ZnSphereCodec(dim, r2)
#         codec = ZnSphereCodecRec(dim, r2)
#         # print(ref_codec.nv, codec.nv)
#         assert ref_codec.nv == codec.nv
#         s = set()
#         l = []
#         for i in range(ref_codec.nv):
#             # print(i)
#             c = np.zeros(dim, dtype='float32')
#             codec.decode(i, c)
#             # print("c",c)
#             code = codec.encode_centroid(c)
#             # print(code)
#             # print(code)
#             assert 0 <= code < codec.nv
#             s.add(code)
#             # print(code)
#             l.append(code)
#         # print(s,len(l),codec.nv)
#         assert len(s) == codec.nv

    # def test_codec(self):
    #     dim = 16
    #     r2 = 6
    #     codec = ZnSphereCodecRecDEUX(dim, r2)
    #     print("nv=", codec.nv)
    #     for i in range(codec.nv):
    #         if i%10000==0:print(i)
    #         c = np.zeros(dim, dtype='float32')
    #         codec.decode(i, c)
    #         code = codec.encode_centroid(c)
    #         assert code == i
# class TestZnSphereCodec(unittest.TestCase):
#     def test_codec(self):
#         dim = 32
#         r2 = 14
#         codec = ZnSphereCodecRecDEUX(dim, r2)
#         rs = np.random.RandomState(42)
#         for i in range(2):
#             x = rs.randn(dim).astype(np.float32)
#             ref_c = np.zeros(dim, dtype=np.float32)
#             code = codec.encode(x)
#             c = np.zeros(dim, dtype=np.float32)
#             c = codec.decode(code, c)
#             np.testing.assert_array_almost_equal(ref_c, c, decimal=5, err_msg="Decode results differ from expected.")

# Add additional test classes as necessary

if __name__ == '__main__':
    unittest.main()