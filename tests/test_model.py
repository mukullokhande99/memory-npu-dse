import unittest

from memory_npu.model import MemoryTechnology, NPUConfig, Workload, evaluate


class ModelInvariantTests(unittest.TestCase):
    def setUp(self):
        self.npu = NPUConfig(256, 500, 10, 15.0, 0.8, 8)
        self.workload = Workload("unit", "cnn", 1_000_000, 500_000, 100_000, 100_000, 4.0, 10)

    def test_small_memory_requires_more_weight_tiles(self):
        small = MemoryTechnology("small", "sram", 256, 64, .1, .1, .5, None, None, 0, 1, None)
        large = MemoryTechnology("large", "sram", 2048, 64, .1, .1, .5, None, None, 0, 1, None)
        self.assertGreater(evaluate(self.npu, small, self.workload)["weight_tiles"], evaluate(self.npu, large, self.workload)["weight_tiles"])

    def test_edram_refresh_cost_is_nonzero(self):
        edram = MemoryTechnology("eDRAM", "edram", 1024, 64, .1, .1, .3, 1, 1, 1.0, 1, None)
        self.assertGreater(evaluate(self.npu, edram, self.workload)["refresh_energy_nj"], 0)

    def test_rram_reports_finite_wear_budget(self):
        rram = MemoryTechnology("RRAM", "rram", 1024, 64, .1, 5.0, .2, 1, None, 0, 1, 10_000_000)
        self.assertGreater(evaluate(self.npu, rram, self.workload)["wear_limited_inferences"], 0)


if __name__ == "__main__":
    unittest.main()
