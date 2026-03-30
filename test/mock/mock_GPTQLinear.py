class MockGPTQLinear():
    def __init__(self, q, z, s):
        self.qweight = q
        self.qzeros = z
        self.scales = s
        self.bias = None
