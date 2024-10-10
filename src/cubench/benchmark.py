import torch
from typing import Callable, Optional, List, Literal, Union

ReturnMode = Literal["min", "max", "mean", "median"]


def _summarize_statistics(
    times: torch.Tensor, quantiles: Optional[List[float]], return_mode: ReturnMode
):
    if quantiles is not None:
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    if return_mode == "all":
        return times.tolist()
    return getattr(torch, return_mode)(times).item()


# copied from https://github.com/triton-lang/triton/blob/cc0cf2d04c39c7571fe0194a8172af37fcd69a7e/python/triton/testing.py#L95-L162
def do_bench(
    fn: Callable,
    n_warmup: int = 3,
    n_repeat: int = 10,
    grad_to_none: Optional[torch.Tensor] = None,
    quantiles: Optional[List[float]] = None,
    fast_flush: bool = True,
    return_mode: ReturnMode = "mean",
) -> Union[float, List[float]]:
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    Args:
        fn (Callable): Function to benchmark.
        warmup (int, optional): Warmup time in milliseconds. Defaults to 3.
        rep (int, optional): Repetition time in milliseconds. Defaults to 10.
        grad_to_none (torch.tensor, optional): Reset the gradient of the provided tensor to None. Defaults to None.
        quantiles (list[float], optional): Performance percentile to return in addition to the median. Defaults to None.
        fast_flush (bool, optional): Use faster kernel to flush L2 between measurements. Defaults to True.
        return_mode (str, optional): The statistical measure to return. Options are "min", "max", "mean", "median", or "all". Defaults to "mean".
    """

    assert return_mode in ["min", "max", "mean", "median"]

    fn()
    torch.cuda.synchronize()

    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2
    # doesn't contain any input data before the run
    if fast_flush:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    else:
        cache = torch.empty(int(256e6), dtype=torch.int8, device="cuda")

    start_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]

    # Warm-up
    for _ in range(n_warmup):
        fn()

    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        cache.zero_()

        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()

    # Record clocks
    torch.cuda.synchronize()
    times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float
    )
    return _summarize_statistics(times, quantiles, return_mode)
