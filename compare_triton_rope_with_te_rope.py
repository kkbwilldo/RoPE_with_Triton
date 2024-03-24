import torch
import triton
import triton.language as tl
import triton.testing as testing
import torch.autograd as autograd
from transformer_engine.pytorch.attention import apply_rotary_pos_emb

"""
Triton으로 작성되는 RoPE 코드는 Unsloth의 코드를 참고하여 작성하였습니다.
ref: https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/rope_embedding.py
"""

ROPE_GROUP_SIZE = 4
MAX_FUSED_SIZE = 65536
next_power_of_2 = triton.next_power_of_2


def calculate_settings(n):
    """
    블록 사이즈를 설정하고, 블록 사이즈에 따른 워프 수를 계산합니다.
    """
    BLOCK_SIZE = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch Triton kernel since n = {n} exceeds "\
                           f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
    num_warps = 4
    if   BLOCK_SIZE >= 32768: num_warps = 32
    elif BLOCK_SIZE >=  8192: num_warps = 16
    elif BLOCK_SIZE >=  2048: num_warps = 8
    return BLOCK_SIZE, num_warps


class RoPEKernel:

    """
    RoPEKernel은 RoPE의 forward와 backward 연산을 수행하는 Triton 커널입니다.
    """

    @staticmethod
    @triton.jit
    def forward(
        tokens_ptr, tokens_row_stride,
        freqs_ptr,  freqs_row_stride,
        seqlen,
        head_dim      : tl.constexpr,
        n_heads       : tl.constexpr,
        BLOCK_SIZE    : tl.constexpr,
    ):
        # 블록의 오프셋을 계산합니다.
        row_position  = tl.program_id(0)
        group_head_position = tl.program_id(1)
        col_offsets  = tl.arange(0, BLOCK_SIZE)
        half_head_dim = head_dim // 2
        mask = col_offsets < half_head_dim

        # freqs를 로드하고 sin, cos를 계산합니다.
        freqs = tl.load(freqs_ptr + (row_position % seqlen)*freqs_row_stride + \
                       half_head_dim*0 + col_offsets, mask = mask, other = 0)

        sin = tl.sin(freqs)
        cos = tl.cos(freqs)

        # RoPE_GROUP_SIZE만큼의 헤드를 처리합니다.
        head_start = group_head_position * ROPE_GROUP_SIZE
        head_end = min((head_start + ROPE_GROUP_SIZE), n_heads)

        # tokens*cos + rotate_half(tokens)*sin 연산을 수행합니다.
        for k in range(head_start, head_end):
            offs_t1 = row_position * tokens_row_stride + k * head_dim + col_offsets
            offs_t2 = row_position * tokens_row_stride + k * head_dim + col_offsets + half_head_dim

            t1 = tl.load(tokens_ptr + offs_t1, mask = mask, other = 0).to(sin.dtype)
            t2 = tl.load(tokens_ptr + offs_t2, mask = mask, other = 0).to(sin.dtype)

            tl.store(tokens_ptr + offs_t1, t1*cos - t2*sin, mask = mask)
            tl.store(tokens_ptr + offs_t2, t2*cos + t1*sin, mask = mask)

    @staticmethod
    @triton.jit
    def backward(
        tokens_ptr, tokens_row_stride,
        freqs_ptr,  freqs_row_stride,
        seqlen,
        head_dim      : tl.constexpr,
        n_heads       : tl.constexpr,
        BLOCK_SIZE    : tl.constexpr,
    ):
        
        row_position  = tl.program_id(0)
        group_head_position = tl.program_id(1)
        col_offsets  = tl.arange(0, BLOCK_SIZE)
        half_head_dim = head_dim // 2
        mask = col_offsets < half_head_dim

        freqs = tl.load(freqs_ptr + (row_position % seqlen)*freqs_row_stride + \
                       half_head_dim*0 + col_offsets, mask = mask, other = 0)

        sin = tl.sin(freqs)
        cos = tl.cos(freqs)

        sin = -sin

        head_start = group_head_position * ROPE_GROUP_SIZE
        head_end = min((head_start + ROPE_GROUP_SIZE), n_heads)

        for k in range(head_start, head_end):
            offs_t1 = row_position * tokens_row_stride + k * head_dim + col_offsets
            offs_t2 = row_position * tokens_row_stride + k * head_dim + col_offsets + half_head_dim

            t1 = tl.load(tokens_ptr + offs_t1, mask = mask, other = 0).to(sin.dtype)
            t2 = tl.load(tokens_ptr + offs_t2, mask = mask, other = 0).to(sin.dtype)

            tl.store(tokens_ptr + offs_t1, t1*cos - t2*sin, mask = mask)
            tl.store(tokens_ptr + offs_t2, t2*cos + t1*sin, mask = mask)


class RoPEFunction(autograd.Function):

    """
    RoPEFunction은 RoPE의 forward와 backward 연산을 수행하는 PyTorch autograd 함수입니다.
    """

    @staticmethod
    def forward(ctx, tokens, freqs):

        batch, seq_len, n_heads, head_dim = tokens.shape
        tokens = tokens.reshape(batch*seq_len, n_heads*head_dim)
        n_rows, n_cols = tokens.shape

        # [TODO] Changing blocksize to head_dim//2 seems to have
        # some concurrency / un-deterministic issues.
        BLOCK_SIZE, num_warps = calculate_settings(head_dim//2)
        
        # group_size = 4 # 4 or 8, too large group_size can hurt performance.
        div, mod = divmod(n_heads, ROPE_GROUP_SIZE)
        n_groups = div + (mod != 0)

        # RoPE 커널을 호출합니다.
        RoPEKernel.forward[(n_rows, n_groups, )](
            tokens,   tokens.stride(0),
            freqs,    freqs.stride(0),
            seq_len,
            head_dim, n_heads,
            BLOCK_SIZE = BLOCK_SIZE,
            num_warps  = num_warps,
        )

        # backward 연산을 위한 컨텍스트(작업 정보)를 저장합니다.
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps  = num_warps
        ctx.n_groups = n_groups
        ctx.freqs = freqs

        return tokens.view(batch, seq_len, n_heads, head_dim)

    @staticmethod
    def backward(ctx, grad_output):
        batch, seq_len, n_heads, head_dim = grad_output.shape
        grad_output = grad_output.reshape(batch*seq_len, n_heads*head_dim)
        n_rows, n_cols = grad_output.shape

        freqs = ctx.freqs

        # RoPE 커널을 호출합니다.
        RoPEKernel.backward[(n_rows, ctx.n_groups, )](
            grad_output,  grad_output.stride(0),
            freqs, freqs.stride(0),
            seq_len, head_dim, n_heads,
            BLOCK_SIZE = ctx.BLOCK_SIZE,
            num_warps  = ctx.num_warps,
        )
        grad_output = grad_output.view(batch, seq_len, n_heads, head_dim)

        # gradient를 반환합니다.
        return grad_output, None, None,
    

def apply_rotary_pos_emb_triton(tokens, freqs, tensor_format="sbhd"):
    """
    2가지 텐서 형식을 지원하는 RoPE 연산을 수행합니다.
    """

    freqs = freqs.squeeze()

    if tensor_format == "bshd":
        # apply api를 사용해 PyTorch가 연산을 추적할 수 있도록 합니다.
        tokens = RoPEFunction.apply(tokens.transpose(1, 2), freqs).transpose(1, 2)
    elif tensor_format == "sbhd":
        tokens = tokens.transpose(0, 1)
        tokens = RoPEFunction.apply(tokens.transpose(1, 2), freqs).transpose(1, 2)
        tokens = tokens.transpose(0, 1)
    else:
        raise NotImplementedError(f"{tensor_format} 텐서 형식은 지원하지 않습니다.")
    return tokens


@testing.perf_report(
    [
        testing.Benchmark(
            x_names=["seq_len"],
            x_vals=[2**i for i in range(5, 10)],
            x_log=True,
            line_arg="backend",
            line_vals=["triton", "transforemer_engine"],
            line_names=["Triton", "Transformer Engine"],
            ylabel="milliseconds",
            plot_name="rope-performance",
            args={"num_batches": 4},
        ),
    ]
)
def benchmark(num_batches, seq_len, backend):

    """
    RoPE 연산의 성능을 측정합니다.
    성능은 연산의 실행 시간을 말하며, sequence length가 커짐에 따라 성능이 어떻게 변하는지 확인합니다.
    """

    DIM = 1024
    NUM_HEADS = 32
    HEAD_DIM = DIM // NUM_HEADS

    tokens = torch.randn(num_batches, seq_len, NUM_HEADS, HEAD_DIM, device="cuda").transpose(1, 2)
    freqs = torch.randn(seq_len, HEAD_DIM, device="cuda").unsqueeze(0).unsqueeze(0)

    # triton 커널
    if backend == "triton":
        return testing.do_bench(lambda: apply_rotary_pos_emb_triton(tokens, freqs, "bshd"))
    # transformer engine 커널
    else:
        return testing.do_bench(lambda: apply_rotary_pos_emb(tokens, freqs.transpose(0,2), "bshd", fused=True))


def main():

    # 벤치마크를 실행합니다.
    benchmark.run(show_plots=False, print_data=True)
    
    SEQ_LEN = 1024
    DIM = 1024
    BATCH_SIZE = 4
    NUM_HEADS = 32
    HEAD_DIM = DIM // NUM_HEADS
    
    tokens = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM, device="cuda")
    freqs = torch.randn(SEQ_LEN, HEAD_DIM, device="cuda").unsqueeze(0).unsqueeze(0)

    tokens_triton = apply_rotary_pos_emb_triton(tokens, freqs, "bshd") 
    tokens_te = apply_rotary_pos_emb(tokens, freqs.transpose(0,2), "bshd", fused=True)

    # Triton과 TE의 RoPE 연산 결과가 동일한지 확인합니다.
    print(f"tokens_triton: {tokens_triton[0][0][0][:10]}")
    print(f"tokens_te: {tokens_te[0][0][0][:10]}")

    assert torch.allclose(tokens_te, tokens_triton)


if __name__ == "__main__":

    main()
