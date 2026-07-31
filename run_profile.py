import torch
import triton
import triton.testing
from triton_impl.matmul_gptq import matmul_gptq
# ---------------------------------------------------------
# 1. 数据准备函数 (支持等比例放大)
# ---------------------------------------------------------
def prepare_data(scale_factor=1, device="cuda"):
    base_weight_r, base_weight_c = 448, 2560
    base_zeros_r, base_zeros_c = 28, 320
    base_scales_r, base_scales_c = 28, 2560

    weight = torch.randint(
        0, 0xFFFFFFFF + 1,
        size=(base_weight_r * scale_factor, base_weight_c * scale_factor),
        dtype=torch.int64, device=device
    ).to(torch.int32)

    zeros = torch.randint(
        0, 0xFFFFFFFF + 1,
        size=(base_zeros_r * scale_factor, base_zeros_c * scale_factor),
        dtype=torch.int64, device=device
    ).to(torch.int32)

    scales = torch.empty((base_scales_r * scale_factor, base_scales_c * scale_factor), 
                         dtype=torch.float16, device=device)
    scales.uniform_(0.005, 0.02)

    return weight, zeros, scales

def prepare_hidden_size(M, N_base=448, scale_factor=1, device="cuda"):
    N_scaled = N_base * scale_factor
    return torch.randn((M, N_scaled), dtype=torch.float16, device=device).clamp(-4.0, 4.0)


# ---------------------------------------------------------
# 2. Mock 包装类 (适配你 matmul_gptq 接口中的 down_proj)
# ---------------------------------------------------------
class MockDownProj:
    def __init__(self, qweight, qzeros, scales, bias=None):
        self.qweight = qweight
        self.qzeros = qzeros
        self.scales = scales
        self.bias = bias

# ---------------------------------------------------------
# 4. Triton Benchmark 吞吐量测试 (测试不同 M 的性能)
# ---------------------------------------------------------
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M'],  # 横坐标：动态变化的维度 M (通常为 BatchSize * SeqLen)
        x_vals=[1, 16, 32, 64, 128, 256, 512, 1024, 2048], 
        x_log=True,     # X 轴使用对数比例
        line_arg='provider',
        line_vals=['custom_gptq'],
        line_names=['Triton GPTQ Kernel'],
        styles=[('blue', '-')],
        ylabel='Time (ms)',
        plot_name='gptq-matmul-performance',
        args={'scale_factor': 2}  # 设定矩阵整体放大的倍数
    )
)
def benchmark_triton(M, scale_factor, provider):
    weight, zeros, scales = prepare_data(scale_factor=scale_factor)
    x = prepare_hidden_size(M=M, N_base=448, scale_factor=scale_factor)
    down_proj = MockDownProj(weight, zeros, scales)

    quant_fn = lambda: matmul_gptq(x, down_proj)

    # do_bench 自动处理 Warmup 和多次采样，返回 中位数, 最大值, 最小值 (ms)
    ms, min_ms, max_ms = triton.testing.do_bench(quant_fn, warmup=25, rep=100)
    return ms, max_ms, min_ms


# ---------------------------------------------------------
# 5. NVTX & NCU 精准剖析模式 (用于底层性能调优)
# ---------------------------------------------------------
def profile_for_ncu(M=1024, scale_factor=2):
    print(f"\n[NCU Profile Mode] Matrix Scale: {scale_factor}x, M: {M}")
    weight, zeros, scales = prepare_data(scale_factor=scale_factor)
    x = prepare_hidden_size(M=M, N_base=448, scale_factor=scale_factor)
    down_proj = MockDownProj(weight, zeros, scales)

    # 1. Warmup: 触发 JIT 编译并让 GPU 预热
    for _ in range(10):
        _ = matmul_gptq(x, down_proj)
    torch.cuda.synchronize()

    # 2. Profile 区间: 仅记录此范围，排除无关开销
    torch.cuda.nvtx.range_push("Triton_GPTQ_Kernel")
    torch.cuda.cudart().cudaProfilerStart()

    _ = matmul_gptq(x, down_proj)

    torch.cuda.cudart().cudaProfilerStop()
    torch.cuda.nvtx.range_pop()
    print("NCU Profile completed. (Please run this script with `ncu` command to capture trace)")


import torch

def profile_with_torch(M=1024, scale_factor=2):
    print(f"[Torch Profile Mode] Matrix Scale: {scale_factor}x, M: {M}")
    weight, zeros, scales = prepare_data(scale_factor=scale_factor)
    x = prepare_hidden_size(M=M, N_base=448, scale_factor=scale_factor)
    down_proj = MockDownProj(weight, zeros, scales)

    # 1. Warmup
    for _ in range(10):
        _ = matmul_gptq(x, down_proj)
    torch.cuda.synchronize()

    # 2. 导出 Profile
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        _ = matmul_gptq(x, down_proj)
        torch.cuda.synchronize()

    prof.export_chrome_trace("gptq_trace.json")
    print("\n✅ Trace 导出成功！保存为 gptq_trace.json")
    print("👉 下载该文件后，在本地 Chrome 浏览器打开 chrome://tracing 或 https://www.speedscope.app 拖入查看。")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['benchmark', 'ncu'], default='benchmark',
                        help="选择运行模式：'benchmark' 绘制性能曲线，'ncu' 用于 Nsight Compute 采集")
    args = parser.parse_args()

    if args.mode == 'benchmark':
        print("Running Triton Benchmark. This will measure execution time across different M values.")
        # 运行 Benchmark 并将结果打印至终端 (如果是在 Jupyter 中可以 show_plots=True 显示图表)
        benchmark_triton.run(print_data=True, show_plots=False)
    else:
        # 用于命令行执行: ncu --profile-from-start 0 -o gptq_profile python profiler.py --mode ncu
        profile_with_torch(scale_factor=2)