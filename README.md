# Deep Dive: CUDA — Euclidean Norm (C vs x86-64 vs SIMD vs CUDA Unified Memory)

## 0. Group Members, Specification and AI Usage

**Group 3:** Matthew Fabregas, Kane Ibanez, Bowen Li, John Teoxon

**Google Colab notebook:** _https://colab.research.google.com/drive/1-fHUjgFFIr_MJmbnM2q4u1HPkSkK4_-A?authuser=1#scrollTo=LsC4P1TIiTsN_ (the notebook is also in this repository as `Deep_Dive_CUDA.ipynb`)

### Project specification (from our SIMD project)

Compute the Euclidean norm of a double-precision vector `A` of length `n`:

$$Z = \sqrt{\sum_{i=0}^{n-1} A_i^2}$$

| Item | Value |
|---|---|
| Data type | `double` (8 bytes) |
| Vector size | n = 2²⁴ = 16,777,216 elements (128 MiB) |
| Initialization | `A[i] = sin(i * 0.001)` (same as our SIMD project) |
| Expected answer | Z = 2896.2749920235 |
| Runs per kernel | 30 (avg, min and max reported) |
| CUDA block sizes | 256, 512, 1024 |
| CUDA grid size | `numBlocks = (n + blockSize - 1) / blockSize` |

### Versions

| # | Version | Platform |
|---|---|---|
| 1 | C program (baseline and answer key) | Colab CPU (`gcc -O2`) |
| 2 | CUDA grid-stride loop, Unified Memory, no prefetch, no page creation, no mem advise | Colab Tesla T4 |
| 3 | CUDA grid-stride loop + prefetch (no page creation, no mem advise) | Colab Tesla T4 |
| 4 | CUDA grid-stride loop + prefetch + page creation (no mem advise) | Colab Tesla T4 |
| 5 | CUDA grid-stride loop + prefetch + page creation + mem advise | Colab Tesla T4 |
| — | C, x86-64, SIMD XMM, SIMD YMM (from our SIMD project) | Laptop CPU, Visual Studio, x64 Release |

### AI usage declaration

We used Claude (Anthropic) as an assistant for this project. It helped us understand the requirements, write the CUDA versions and the Colab notebook, compile-check the code, add the 2²⁴ size and min/max timing to our SIMD program, and draft this README from our measured results. All programs were run by our group on Google Colab and on our own laptop, and every number and screenshot in this report comes from those runs. We reviewed the code and the analysis and are responsible for the submission.

---

## How the CUDA programs work

All four CUDA versions run the **same kernel**. The only difference between them is how the Unified Memory is managed, so any time difference comes from memory handling.

```cuda
__global__ void norm_kernel(const double* A, long long n, double* sum) {
    __shared__ double s[1024];
    long long index  = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    double local = 0.0;
    for (long long i = index; i < n; i += stride)   // grid-stride loop
        local += A[i] * A[i];

    s[threadIdx.x] = local;                          // shared-memory block reduction
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k)
            s[threadIdx.x] += s[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(sum, s[0]);                        // one atomic per block
}
```

The Euclidean norm is a **reduction**: every element adds into one sum. If every thread did `*sum += A[i] * A[i]`, threads would overwrite each other's updates (a race condition). Each block therefore first adds its own values in shared memory, and only thread 0 of each block does one `atomicAdd` into the global `sum`. The host then computes `sqrt(sum)`.

| Version | What happens to `A` before the kernel |
|---|---|
| 2 Unified | `cudaMallocManaged`, filled on the **CPU**. The kernel page-faults and the pages migrate to the GPU on demand. |
| 3 Prefetch | Filled on the CPU, then `cudaMemPrefetchAsync(A, bytes, device)` moves it to the GPU before the kernel. |
| 4 Prefetch + page creation | The empty buffer is prefetched to the GPU, then filled by a GPU kernel (`init_kernel`), so its pages are **created directly in GPU memory** and nothing is copied from the CPU. |
| 5 Prefetch + page creation + mem advise | Same as 4, plus `cudaMemAdviseSetPreferredLocation` = GPU (before the prefetch) and `cudaMemAdviseSetReadMostly` (after `A` is filled, since the kernel only reads it). |

**Our definition of page creation:** `cudaMallocManaged` only reserves addresses; physical pages are created when memory is first touched. In Versions 4 and 5 we make the first touch happen on the GPU.

**Timing:** each of the 30 runs resets `sum` with `cudaMemset` (not timed), then only the kernel is timed with CUDA events. Run 1 is also reported separately because it includes any page migration.

**Correctness:** each CUDA result is compared to the C answer key. Because the GPU adds the numbers in a different order, the last digits can differ, so a result is `CORRECT` if the relative difference is at most 1e-9 (the same tolerance as our SIMD project).

---

## i. Program Output (correctness check and execution time)

### C program (Version 1)

<!-- SCREENSHOT: Colab output of the Version 1 cell (!./euclid_c 24) -->
![C output](screenshots/v1_c.png)

### CUDA Unified (Version 2)

<!-- SCREENSHOT: Colab output of the Version 2 run cell (all 3 block sizes) -->
![CUDA Unified output](screenshots/v2_unified.png)

### CUDA Prefetch (Version 3)

<!-- SCREENSHOT: Colab output of the Version 3 run cell -->
![CUDA Prefetch output](screenshots/v3_prefetch.png)

### CUDA Prefetch + page creation (Version 4)

<!-- SCREENSHOT: Colab output of the Version 4 run cell -->
![CUDA Prefetch + page creation output](screenshots/v4_pages.png)

### CUDA Prefetch + page creation + mem advise (Version 5)

<!-- SCREENSHOT: Colab output of the Version 5 run cell -->
![CUDA Prefetch + page creation + mem advise output](screenshots/v5_advise.png)

### Results summary (Colab)

<!-- SCREENSHOT: Colab output of the "Results summary" pandas cell -->
![Results summary](screenshots/summary_table.png)

### C, x86-64, XMM and YMM (SIMD project, Visual Studio x64 Release)

![SIMD output](screenshots/simd_output.png)

Every version printed `correct` / `CORRECT`. All CUDA versions gave Z = 2896.2749920236 against the C answer key 2896.2749920235, a relative difference of about 4e-14.

---

## ii. Nsight Systems (nsys) — all CUDA versions (block size 256)

For each version, the screenshot shows the kernel summary (`cuda_gpu_kern_sum`), the memory transfer summary (`cuda_gpu_mem_time_sum` / `cuda_gpu_mem_size_sum`) and the Unified Memory summary.

### Version 2 — CUDA Unified

<!-- SCREENSHOT: nsys output for cuda_unified: cuda_gpu_kern_sum, cuda_gpu_mem_time_sum, cuda_gpu_mem_size_sum, um_total_sum -->
![nsys Unified](screenshots/nsys_v2_unified.png)

### Version 3 — CUDA Prefetch

<!-- SCREENSHOT: nsys output for cuda_prefetch: cuda_gpu_kern_sum, cuda_gpu_mem_time_sum, cuda_gpu_mem_size_sum -->
![nsys Prefetch](screenshots/nsys_v3_prefetch.png)

### Version 4 — CUDA Prefetch + page creation

<!-- SCREENSHOT: nsys output for cuda_pages: cuda_gpu_kern_sum (shows init_kernel), cuda_gpu_mem_size_sum -->
![nsys Prefetch + page creation](screenshots/nsys_v4_pages.png)

### Version 5 — CUDA Prefetch + page creation + mem advise

<!-- SCREENSHOT: nsys output for cuda_advise: cuda_gpu_kern_sum, cuda_gpu_mem_size_sum -->
![nsys Prefetch + page creation + mem advise](screenshots/nsys_v5_advise.png)

### What nsys shows

| | Unified | Prefetch | Prefetch + page creation | + mem advise |
|---|---|---|---|---|
| Host-to-Device transfers | 2,378 | 64 | 0 | 0 |
| Data moved Host-to-Device | 134.2 MB | 134.2 MB | 0 MB | 0 MB |
| Transfer size | 4 KB – 1 MB (avg 0.056 MB) | 2.097 MB each | — | — |
| Total transfer time | 16.90 ms | 11.18 ms | — | — |
| Transfer speed | 7.9 GB/s | 12.0 GB/s | — | — |
| GPU page faults | 2,642 | 0 | 0 | 0 |
| CPU page faults | 385 | 385 | 1 | 1 |
| `init_kernel` on GPU | — | — | 5.48 ms (once) | 5.11 ms (once) |

### Nsight Compute (ncu)

<!-- SCREENSHOT (optional): ncu output for one version: GPU Speed Of Light + Occupancy sections -->
![ncu](screenshots/ncu_v2_unified.png)

ncu profiled one `norm_kernel` launch (block size 256) for each version. All four showed nearly identical results: duration **3.86 ms**, Compute (SM) throughput **83.5%**, DRAM throughput **13.8%**, achieved occupancy **95.3%**. (ncu locks the GPU clock at 585 MHz, so its duration is longer than our benchmark times.)

---

## iii. Comparative Table of Execution Time (2²⁴ elements)

Times are in milliseconds over 30 runs. The CUDA times are **kernel only**.

### Main table

| 2²⁴ elements | Block size | Time avg \| min \| max (ms) | Speedup vs C (same machine) |
|---|---|---|---|
| C (Colab CPU, `gcc -O2`) | — | 43.2398 \| 24.6698 \| 73.2367 | 1.00x |
| C (laptop, VS Release) | — | 18.3172 \| 17.7056 \| 21.3269 | 1.00x |
| x86-64 (laptop) | — | 23.1282 \| 18.5724 \| 66.6737 | 0.79x |
| x86-64 SIMD XMM (laptop) | — | 10.8573 \| 9.5171 \| 12.8959 | 1.69x |
| x86-64 SIMD YMM (laptop) | — | 9.9456 \| 7.6551 \| 27.6573 | 1.84x |
| CUDA Unified | 256 | 3.4607 \| 2.2836 \| 37.4334 | 12.49x |
| | 512 | 3.2900 \| 2.1829 \| 32.7905 | 13.14x |
| | 1024 | 3.9207 \| 2.8385 \| 35.2356 | 11.03x |
| CUDA Prefetch | 256 | 2.2967 \| 2.2838 \| 2.6215 | 18.83x |
| | 512 | 2.8120 \| 2.7873 \| 3.1533 | 15.38x |
| | 1024 | 3.0398 \| 3.0250 \| 3.3524 | 14.22x |
| CUDA Prefetch + page creation | 256 | 2.4697 \| 2.4639 \| 2.5518 | 17.51x |
| | 512 | 2.7943 \| 2.5963 \| 2.8260 | 15.47x |
| | 1024 | 3.0410 \| 3.0356 \| 3.0604 | 14.22x |
| CUDA Prefetch + page creation + mem advise | 256 | **1.8216 \| 1.8178 \| 1.8479** | **23.74x** |
| | 512 | 2.1324 \| 2.1141 \| 2.1746 | 20.28x |
| | 1024 | 2.7801 \| 2.7771 \| 2.8062 | 15.55x |

**Which C is the baseline?** The CUDA speedups are against the C program on the **same Colab machine** (43.24 ms). The x86-64, XMM and YMM speedups are against the C program on the **same laptop** (18.32 ms). Comparing each version with C on its own machine keeps the CPU hardware out of the comparison. A cross-platform comparison is in the next table.

### Run 1 vs the other 29 runs

| Version (block size 256) | Run 1 (ms) | Avg of runs 2–30 (ms) |
|---|---|---|
| CUDA Unified | 37.4334 | 2.2892 |
| CUDA Prefetch | 2.6215 | 2.2855 |
| CUDA Prefetch + page creation | 2.5518 | 2.4669 |
| CUDA Prefetch + page creation + mem advise | 1.8479 | 1.8206 |

### Cross-platform comparison (everything against the laptop C, 18.32 ms)

| Version (best block size) | Avg time (ms) | Speedup vs laptop C | Effective bandwidth |
|---|---|---|---|
| C (laptop) | 18.3172 | 1.00x | 7.3 GB/s |
| x86-64 SIMD YMM (laptop) | 9.9456 | 1.84x | 13.5 GB/s |
| CUDA Unified (512) | 3.2900 | 5.57x | — |
| CUDA Prefetch (256) | 2.2967 | 7.98x | 58.4 GB/s |
| CUDA Prefetch + page creation (256) | 2.4697 | 7.42x | 54.3 GB/s |
| CUDA Prefetch + page creation + mem advise (256) | 1.8216 | 10.06x | 73.7 GB/s |
| **CUDA Unified, run 1 only (256)** | **37.4334** | **0.49x** | — |

Effective bandwidth = bytes read / time = (2²⁴ × 8 bytes) / time, using the formula from the lecture.

---

## iv. Analysis of Results

### Kernel execution time

**1. The first run of the Unified version is about 16x slower than the rest.** Run 1 of CUDA Unified took 32.8–37.4 ms, but runs 2–30 took only about 2.3 ms. In the Unified version `A` is filled on the CPU, so its pages live in CPU memory. When the kernel touches a page that is not on the GPU, the GPU stalls, the page migrates, and then the threads continue. nsys shows this clearly: 2,642 GPU page faults and 2,378 small Host-to-Device transfers moving 134.2 MB, all inside the first kernel. After run 1, the data stays on the GPU, so runs 2–30 have no faults. This is why the **max** column for CUDA Unified is about 10x the average, and why its 30-run average (3.29–3.92 ms) is the worst of the four CUDA versions.

**2. Prefetching removes the slow first run.** In CUDA Prefetch, run 1 (2.62 ms) is almost the same as the other runs (2.29 ms), and the max is only 2.62 ms. The 128 MB are moved before the kernel starts. nsys shows the prefetch moved the same 134.2 MB in only **64 transfers of 2 MB**, at 12.0 GB/s, compared with 7.9 GB/s for the on-demand migration. Large transfers use the PCIe bus better than thousands of small faults.

**3. Page creation removes the transfer completely.** In Versions 4 and 5, `A` is created and filled on the GPU, so nsys shows **0 bytes** copied from host to device (only the 4 KB `sum` result is copied back). The CPU page faults also drop from 385 to 1. Filling the vector on the GPU took about 5.1–5.5 ms (`init_kernel`), which is not part of the kernel time.

**4. Once the data is on the GPU, the kernel time is nearly the same for Versions 3, 4 and 5.** In our benchmark run, Version 5 (mem advise) was the fastest at 1.82 ms, and Versions 3 and 4 were 2.30 and 2.47 ms. However, these small differences changed between runs. In the separate nsys run, the order was different (Prefetch 3.53 ms, page creation 2.60 ms, mem advise 2.43 ms). ncu gives the strongest evidence: it locks the GPU clock, and with a fixed clock **all four versions took exactly 3.86 ms**. The kernel does the same work on the same data in GPU memory, so the memory handling only changes how and when the data gets to the GPU, not the steady-state speed of the kernel. We believe most of the run-to-run differences come from the Colab GPU's changing clock speed. Version 5 was the fastest in both of our runs, but we do not think a 0.5 ms gap alone proves that mem advise speeds up the kernel itself.

**5. The kernel is limited by the GPU's compute side, not by memory bandwidth.** The best version reached 73.7 GB/s, only about 23% of the T4's theoretical 320 GB/s (lecture 2). ncu shows why: Compute (SM) throughput is 83.5%, but DRAM throughput is only 13.8%. With `numBlocks = (n + blockSize - 1) / blockSize`, each thread handles just **one** element, then takes part in an 8-step shared-memory reduction with a `__syncthreads()` at each step, then exits. Most of the work is this per-thread overhead (64-bit index math, shared-memory adds, barriers, block start-up for 65,536 blocks) rather than the one multiply-add per element. The T4 also has very few double-precision units (2 FP64 lanes per SM for CC 7.5, from the lecture's compute capability table, compared with 64 FP32 lanes), and both the squaring and the reduction use doubles. This is the opposite of the CPU, where our SIMD project found the kernel was limited by memory speed.

### a) How does block size affect execution time? Which block size do we recommend?

**256 was the fastest block size for every prefetch version, and 1024 was always the slowest.**

| Version | 256 | 512 | 1024 |
|---|---|---|---|
| CUDA Prefetch | 2.30 | 2.81 | 3.04 |
| CUDA Prefetch + page creation | 2.47 | 2.79 | 3.04 |
| CUDA Prefetch + page creation + mem advise | 1.82 | 2.13 | 2.78 |

(CUDA Unified is the exception: 512 had the best average, because its average is dominated by the slow first run, which varies by a few ms per run.)

Why smaller blocks are faster here:

- **More blocks fit on an SM at once.** The T4 (CC 7.5) allows at most 1,024 threads per SM (lecture 3). With 256 threads per block, 4 blocks run on an SM together; with 1024, only 1 block fits. ncu confirms 4 blocks per SM and 95% achieved occupancy at block size 256.
- **Barriers hurt big blocks more.** At each `__syncthreads()` step of the reduction, fewer and fewer threads do useful work while the rest wait. With 4 independent blocks on an SM, while one block waits at a barrier the others can keep loading and computing. With one 1024-thread block, the whole SM waits, and the reduction needs 10 steps instead of 8.
- **More atomics did not matter.** Block size 256 needs 65,536 `atomicAdd`s (one per block), four times more than 1024 needs (16,384), yet 256 was still fastest. One atomic per block is cheap compared with the reduction work.

**Recommendation: block size 256** for this kernel. It was the fastest in every prefetch version and gives the scheduler the most flexibility.

On the number of blocks: we used the maximum needed for one element per thread, `(n + blockSize - 1) / blockSize` (65,536 blocks at 256). Because each thread only does one element, the grid-stride loop runs once per thread. From the ncu result above, we expect that launching fewer blocks (for example a multiple of the 40 SMs), so each thread adds many elements in its grid-stride loop before the reduction, would reduce the overhead further. We did not measure this, so we list it as a possible improvement.

### b) Is prefetching always recommended, or should CUDA manage memory?

For our use case, **prefetching (or creating the data on the GPU) is clearly better.** We read the whole vector in order, and we know in advance that the GPU needs all of it. Prefetching moved the data in 2 MB chunks at 12.0 GB/s, instead of 2,378 page faults at 7.9 GB/s, and removed the 35 ms first-run penalty.

Prefetching is **not always** the best choice. Letting CUDA manage memory on demand can be better when:

- **The GPU only touches a small or unpredictable part of a large array** (for example, a sparse lookup, a graph traversal, or a search that stops early). Prefetching everything would move data that is never used, while on-demand faults move only the pages that are actually touched.
- **The data is bigger than GPU memory.** Unified Memory can oversubscribe GPU memory and page in what is needed; prefetching all of it is not possible.
- **The CPU and GPU take turns working on the same data in small pieces.** Prefetching back and forth every time can cost more than letting only the touched pages move.
- **Simplicity matters more than speed,** such as in a prototype or a program that runs once on a small array.

Prefetching is better when the kernel reads (almost) all of the data, the access pattern is known in advance, and the data fits in GPU memory, which is exactly the case for vector operations like ours. Best of all, as Versions 4 and 5 show, is to **create the data on the GPU** if possible, so it never has to cross the PCIe bus.

Note: lecture 2 mentions that Windows does not support on-demand page migration (data is automatically prefetched). Colab runs Linux, so our Version 2 really shows the cost of on-demand page faults.

### c) Between SIMD and SIMT, which one is faster?

**For the kernel alone, SIMT on the GPU is much faster.** The best CUDA version (1.82 ms) was **5.5x faster than YMM** (9.95 ms) and **10x faster than C** (18.32 ms) on our laptop. Even the slowest CUDA average (Unified at 1024, 3.92 ms) beat YMM by 2.5x.

**But it depends on where the data starts.** If the data is created on the CPU and only used once, the GPU must first receive 128 MB over PCIe. In our Unified version, the first run (with the migration) took 37.4 ms, which is **slower than plain C on the laptop (18.32 ms)** and almost 4x slower than YMM. Even with the faster prefetch, moving the data took 11.2 ms, which is already longer than YMM needed to compute the whole answer (9.95 ms). For a simple kernel that does very little math per element, the cost of moving the data can be larger than the speed gained.

SIMD (CPU) is better when:
- the data is already in CPU memory and is used once, as in a single norm of a freshly computed vector;
- the vector is small (at 2²⁰ elements, YMM took only 0.35 ms, less than one GPU kernel launch plus transfer);
- there is no NVIDIA GPU, or the computation is part of a mostly sequential CPU program.

SIMT (GPU) is better when:
- the data is already on the GPU or can be created there (Versions 4 and 5);
- the same data is used many times, so the transfer is paid once and spread over many kernel runs (our 30-run benchmark is an example);
- the data is large and there is a lot of independent work per element.

### Speed across all platforms

From slowest to fastest kernel time at 2²⁴: Colab C (43.2 ms) → x86-64 (23.1 ms) → laptop C (18.3 ms) → XMM (10.9 ms) → YMM (9.9 ms) → CUDA Unified (3.3–3.9 ms average, but 33–37 ms in run 1) → CUDA Prefetch / page creation (2.3–3.0 ms) → CUDA mem advise (1.8–2.8 ms).

- **C on two machines:** the same C code took 43.2 ms on Colab's CPU but 18.3 ms on our laptop, and the Colab times were very noisy (24.7–73.2 ms) because the Colab CPU is shared. This is why we compare each version with C on its own machine.
- **x86-64 vs C:** In Release, the compiler keeps `i` and `sum` in registers just like our hand-written assembly, so x86-64 and C are about tied. At 2²⁰ and 2²⁶ they were within 2%, and x86-64's best run at 2²⁴ (18.57 ms) matched C. Its 2²⁴ average (23.1 ms) was pulled up by a few runs that were interrupted by other programs on the laptop (max 66.7 ms).
- **XMM and YMM** were 1.69x and 1.84x faster than C, well below the theoretical 2x and 4x, because the CPU kernel is limited by memory speed (YMM read at only 13.5 GB/s).
- **CUDA** was 5–10x faster than laptop C for the kernel itself, because 2,560 CUDA cores and the T4's 320 GB/s memory work on the vector in parallel, while the CPU uses one core.

---

## v. Problems Encountered, Solutions and AHA Moments

- **The norm is a reduction, which is harder on a GPU.** A simple `*sum += A[i] * A[i]` in every thread is a race condition (lecture 4), and an `atomicAdd` for every element would serialize 16.7 million updates. **Solution:** each thread adds its elements in a register, each block adds its threads' values in shared memory, and only one `atomicAdd` per block goes to global memory.
- **The lecture's prefetch and mem advise calls did not compile on Colab.** Colab uses CUDA 13.0, which changed `cudaMemPrefetchAsync` and `cudaMemAdvise` to take a `cudaMemLocation` instead of a device number. **Solution:** small wrapper functions that check `CUDART_VERSION` and use the right form, so the notebook works on CUDA 12 and 13.
- **"Page creation" is not a CUDA function.** We had to decide what it means. We defined it as creating the pages directly in GPU memory by prefetching the empty buffer and then filling it with a GPU kernel. nsys confirmed it worked: 0 bytes copied from host to device.
- **The GPU result does not exactly match C.** The GPU adds in a tree order, so the last digits differ (2896.2749920236 vs 2896.2749920235). **Solution:** compare with a relative tolerance of 1e-9, as in our SIMD project. The actual difference was about 4e-14.
- **Our SIMD program did not have 2²⁴ or min/max.** Our earlier project measured 2²⁰, 2²⁶ and 2³⁰ and only reported averages. We added 2²⁴ and min/max timing. The first re-run still showed the old sizes because Visual Studio had the old file open and saved it over our change; reloading the file fixed it.
- **The profilers change the timings.** Under ncu, run 1 took 2.2–2.5 seconds because ncu replays the kernel 10 times to collect counters, and nsys also added overhead. **Solution:** all benchmark numbers come from the normal (unprofiled) runs; the profilers are used only to explain them.
- **Colab timings vary between runs.** The same versions gave different times in the benchmark run and in the nsys run. We report all three statistics (avg, min, max) and use ncu's fixed-clock result to check which differences are real.

**AHA moments**

- **Faults vs prefetch is about transfer size.** Both moved the same 134 MB, but on-demand faults used 2,378 small transfers while prefetch used 64 transfers of 2 MB, which was about 50% faster.
- **The fastest transfer is no transfer.** Creating the data on the GPU removed all 134 MB of Host-to-Device traffic.
- **Moving the data can cost more than computing it.** The 128 MB transfer to the GPU (11–17 ms) took longer than YMM needed to compute the whole norm on the CPU (9.9 ms). The GPU only wins when the data is already there or is reused.
- **The GPU kernel was not memory-bound, the CPU kernel was.** On the CPU, wider SIMD barely helped because RAM was the bottleneck. On the GPU, ncu showed only 14% DRAM use but 83% compute use, because each thread did very little real work compared with its reduction overhead.
- **Smaller blocks were faster.** We expected the maximum block size (1024) to be best, but 256 won every time because 4 blocks can share an SM and hide each other's `__syncthreads()` waits.

---

## vi. SIMD vs SIMT

### How each one handles parallelism in our project

**SIMD (XMM / YMM on the CPU):** one CPU thread runs one instruction stream, and each instruction works on a wider register: 2 doubles in XMM, 4 in YMM. We had to write the parallelism ourselves in assembly: load 4 doubles with `vmovupd`, square them with `vmulpd`, add them to 4 partial sums with `vaddpd`, combine the 4 lanes at the end, and handle the 1–3 leftover elements with a scalar loop. The parallelism is limited to the register width (at most 4 doubles per instruction with YMM).

**SIMT (CUDA on the GPU):** we write the code for **one thread** (square my elements and add them), and the GPU runs it on 16.7 million threads. Threads are grouped into warps of 32 that execute the same instruction together, which is SIMD inside the hardware, but each thread has its own index, registers and control flow. The hardware hides memory latency by switching between the many warps on each SM. Leftover elements are handled automatically by the `i < n` check in the grid-stride loop, so we did not need a separate tail loop. The cost is that threads cannot simply share a sum: we needed shared memory, `__syncthreads()` and `atomicAdd`.

### Pros and cons for our use case (Euclidean norm)

| | SIMD (XMM / YMM) | SIMT (CUDA) |
|---|---|---|
| **Speed (kernel only, 2²⁴)** | YMM 9.95 ms, 1.84x faster than C | 1.82 ms at best, 10x faster than laptop C |
| **Data transfer** | None, data is already in RAM | 128 MB must reach GPU memory (11–17 ms), unless created there |
| **Reduction (the sum)** | Easy: keep 2 or 4 partial sums in one register, combine at the end | Harder: shared memory, barriers and atomics needed to avoid race conditions |
| **Leftover elements** | Need a separate scalar tail loop | Handled by the `i < n` check |
| **Programming effort** | Low-level assembly, register-by-register | High-level C++, but memory management and launch configuration must be tuned |
| **Scaling** | Limited by register width (4 doubles) and one core | Thousands of threads; scales with GPU size |
| **Bottleneck in our results** | Memory bandwidth (13.5 GB/s) | Per-thread overhead and FP64 units (83% compute, 14% DRAM) |
| **Double precision** | Full-speed on the CPU | Slow on the T4 (2 FP64 lanes per SM) |
| **Hardware needed** | Any modern x86-64 CPU | An NVIDIA GPU |

**Summary:** SIMD is simple, has no transfer cost, and is the better choice for a one-off norm of data that is already in CPU memory, especially for small vectors. SIMT gives a much faster kernel and scales far better, so it is the better choice when the data is already on the GPU, can be created there, or is reused many times, so the transfer cost is paid only once.
