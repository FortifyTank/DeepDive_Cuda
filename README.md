# Deep Dive: CUDA — Euclidean Norm - C vs Assembly vs SIMD vs CUDA

## Group Members, Specification and AI Usage

**Group 3:** Matthew Fabregas, Kane Ibanez, Bowen Li, John Teoxon

**Google Colab notebook:** https://colab.research.google.com/drive/1aB99_AnJZpkkZSQYR4ayNUfPiIx2kcsR?usp=sharing

### Project specification (SIMD project)

Compute the Euclidean norm of a double-precision vector `A` of length `n`:

$$Z = \sqrt{\sum_{i=0}^{n-1} A_i^2}$$

| Item | Value |
|---|---|
| Data type | `double` (8 bytes) |
| Main vector size | n = 2²⁴ = 16,777,216 elements (128 MiB) |
| Extra sizes | 2²⁰ (8 MiB) and 2²⁶ (512 MiB), SIMD project sizes|
| Initialization | `A[i] = sin(i * 0.001)` (SIMD project) |
| Expected answer (2²⁴) | Z = 2896.2749920235 |
| Runs per kernel | 30 (avg, min and max reported; run 1 also reported separately "cold start") |
| CUDA block sizes | 256, 512, 1024 |
| CUDA grid size (main runs) | number of SMs × resident blocks per SM = 160 / 80 / 40 blocks on a T4 |
| CUDA grid size (comparison) | `(n + blockSize - 1) / blockSize` = maximum blocks, one element per thread |
| Correctness tolerance | relative difference ≤ 1e-9 against the C answer key |

### Versions

| # | Version | Platform |
|---|---|---|
| 1 | C program (baseline and answer key) | Colab CPU (`gcc -O2`) |
| 2 | CUDA grid-stride loop, Unified Memory, no prefetch, no page creation, no mem advise | Colab T4 |
| 3 | CUDA grid-stride loop + prefetch (no page creation, no mem advise) | Colab T4 |
| 4 | CUDA grid-stride loop + prefetch + page creation (no mem advise) | Colab T4 |
| 5 | CUDA grid-stride loop + prefetch + page creation + mem advise | Colab T4 |
| — | C, x86-64, SIMD XMM, SIMD YMM (from our SIMD project) | Laptop CPU, Visual Studio, x64 Release |

### AI usage declaration

We used Claude to assist us in our project. We noticed in some parts of the code that there are some values that don't seem to make sense, such as the grid-stride, which should vary but wasn't being implemented well. We were also confused on how page creation would be implemented and how to use NCU, so we did research via Google Search and asked Claude for help with syntax. Additionally, we asked it for assistance on what to expect and to theorize what the results would be, which is why we had to do multiple iterations to get a decent comparison of the different versions. All our results were run using a T4 GPU in Google Colab.

---

## How the CUDA programs work

All four CUDA versions run the **same kernel**. The only difference between them is how the Unified Memory is managed, so any time difference comes from memory handling.

```cuda
__global__ void norm_kernel(const double* A, long long n, double* sum) {
    extern __shared__ double s[];                    // blockSize doubles, size given at launch
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

The Euclidean norm is a reduction wherein every element adds into one sum. If every thread did `*sum += A[i] * A[i]`, threads would overwrite each other's updates (a race condition). So each thread first adds its own elements in a register, each block adds its threads' values in shared memory, and only thread 0 of each block does one `atomicAdd` into the global `sum`. The host then computes `sqrt(sum)`.

### Grid size: making the grid-stride loop actually loop

| Launch | Number of blocks | Elements per thread | Used for |
|---|---|---|---|
| **Blocks that fit on the GPU** (`gridMode 1`) | SMs × resident blocks per SM (from `cudaOccupancyMaxActiveBlocksPerMultiprocessor`): **160 / 80 / 40** for block size 256 / 512 / 1024 | about 410 | All main runs |
| **Maximum blocks** (`gridMode 0`) | `(n + blockSize - 1) / blockSize`: **65,536 / 32,768 / 16,384** | 1 | Comparison for guide question (a) |

With maximum blocks there is one thread per element, so `stride ≥ n` and the grid-stride loop body runs only once. With blocks that fit on the GPU, each thread walks through about 410 elements, which is the point of a grid-stride loop. (we messed up in our first version since we kept getting same times)

### Memory handling per version

| Version | What happens to `A` before the kernel |
|---|---|
| 2 Unified | `cudaMallocManaged`, filled on the CPU. The kernel page-faults and the pages migrate to the GPU on demand. |
| 3 Prefetch | Filled on the CPU, then `cudaMemPrefetchAsync(A, bytes, device)` moves it to the GPU before the kernel. |
| 4 Prefetch + page creation | The empty buffer is prefetched to the GPU, then filled by a GPU kernel (`init_kernel`, also a grid-stride loop), so its pages are created directly in GPU memory and nothing is copied from the CPU. |
| 5 Prefetch + page creation + mem advise | Same as 4, plus `cudaMemAdviseSetPreferredLocation` = GPU (before the prefetch) and `cudaMemAdviseSetReadMostly` (after `A` is filled, since the kernel only reads it). |

**Page Creation:** `cudaMallocManaged` only reserves addresses; physical pages are created when memory is first touched. In Versions 4 and 5 we make the first touch happen on the GPU.

**Timing:** each of the 30 runs resets `sum` with `cudaMemset` , then only the kernel is timed with CUDA events. Run 1 is also reported separately because it includes any page migration, and so is the average of runs 2–30.

**Correctness:** each CUDA result is compared to the C answer key. Because the GPU adds the numbers in a different order, the last digits can differ, so a result is `CORRECT` if the relative difference is at most 1e-9 (tolerance in our SIMD project). A boundary test (n = 1, 3, 1000, 1001, 1002, 1003, the same sizes as our SIMD project) checks that sizes that do not divide evenly are handled.

---

## I. Program Output (correctness check and execution time)

### C program (Version 1)

<!-- SCREENSHOT: output of the Version 1 cell (!./euclid_c 24) -->
![C output](screenshots/v1_c.png)

### CUDA Unified (Version 2)

<!-- SCREENSHOT: output of the Version 2 run cell (all 3 block sizes) -->
![CUDA Unified output](screenshots/v2_unified.png)

### CUDA Prefetch (Version 3)

<!-- SCREENSHOT: output of the Version 3 run cell -->
![CUDA Prefetch output](screenshots/v3_prefetch.png)

### CUDA Prefetch + page creation (Version 4)

<!-- SCREENSHOT: output of the Version 4 run cell -->
![CUDA Prefetch + page creation output](screenshots/v4_pages.png)

### CUDA Prefetch + page creation + mem advise (Version 5)

<!-- SCREENSHOT: output of the Version 5 run cell -->
![CUDA Prefetch + page creation + mem advise output](screenshots/v5_advise.png)

### Maximum blocks (one element per thread), all versions

<!-- SCREENSHOT(S): output of the "Block size and number of blocks" cell -->
![Maximum blocks output 1](screenshots/grid_compare_1.png)
![Maximum blocks output 2](screenshots/grid_compare_2.png)

### Boundary test

<!-- SCREENSHOT: output of the boundary test cell -->
![Boundary test](screenshots/boundary.png)

### Results summary (Colab)

<!-- SCREENSHOT: first two tables of the "Results summary" cell (main runs + gridMode 0 vs 1) -->
![Results summary](screenshots/summary_table.png)

<!-- SCREENSHOT: the "All sizes" tables at the end of the "Results summary" cell -->
![Results for all sizes](screenshots/summary_sizes.png)

### C, x86-64, XMM and YMM (SIMD project, Visual Studio x64 Release)

![SIMD output](screenshots/simd_output.png)

Every version printed `correct` at every size, block size and launch. All CUDA versions gave Z = 2896.2749920236 at 2²⁴ against the C answer key 2896.2749920235, a very small and acceptable difference. The boundary test matched C for all six sizes.

---

## II. Nsight Systems (nsys) — all CUDA versions (2²⁴, block size 256, blocks that fit on the GPU)

For each version, the screenshot shows the kernel summary (`cuda_gpu_kern_sum`) and the memory transfer summaries (`cuda_gpu_mem_time_sum` / `cuda_gpu_mem_size_sum`). Version 2 also shows the Unified Memory totals (`um_total_sum`).

### Version 2 — CUDA Unified

<!-- SCREENSHOT: nsys for cuda_unified: cuda_gpu_kern_sum, cuda_gpu_mem_time_sum, cuda_gpu_mem_size_sum, um_total_sum -->
![nsys Unified](screenshots/nsys_v2_unified.png)

### Version 3 — CUDA Prefetch

<!-- SCREENSHOT: nsys for cuda_prefetch: cuda_gpu_kern_sum, cuda_gpu_mem_time_sum, cuda_gpu_mem_size_sum -->
![nsys Prefetch](screenshots/nsys_v3_prefetch.png)

### Version 4 — CUDA Prefetch + page creation

<!-- SCREENSHOT: nsys for cuda_pages: cuda_gpu_kern_sum (shows init_kernel), cuda_gpu_mem_time_sum, cuda_gpu_mem_size_sum -->
![nsys Prefetch + page creation](screenshots/nsys_v4_pages.png)

### Version 5 — CUDA Prefetch + page creation + mem advise

<!-- SCREENSHOT: nsys for cuda_advise: cuda_gpu_kern_sum, cuda_gpu_mem_time_sum, cuda_gpu_mem_size_sum -->
![nsys Prefetch + page creation + mem advise](screenshots/nsys_v5_advise.png)

### What nsys shows

| | Unified | Prefetch | Prefetch + page creation | + mem advise |
|---|---|---|---|---|
| Host-to-Device transfers | 7,569 | 64 | 0 | 0 |
| Data moved Host-to-Device | 132.1 MB | 134.2 MB | 0 MB | 0 MB |
| Transfer size | 4 KB – 0.84 MB (avg 0.017 MB) | 2.097 MB each | — | — |
| Total transfer time | 28.71 ms | 11.14 ms | — | — |
| Transfer speed | 4.6 GB/s | 12.0 GB/s | — | — |
| GPU page faults | 365 | 0 | 0 | 0 |
| CPU page faults | 385 | 385 | 1 | 1 |
| `init_kernel` on GPU | — | — | 6.11 ms (once) | 5.89 ms (once) |
| `norm_kernel` median (30 runs) | 0.536 ms | 0.533 ms | 0.528 ms | 0.528 ms |

There was a popup from nsys that says "SKIPPED … does not contain GPU page faults data" for Versions 3–5 because there were no GPU page faults to record.

### Nsight Compute (ncu)

<!-- SCREENSHOT: ncu for cuda_unified: "GPU Speed Of Light Throughput" and "Occupancy" sections -->
![ncu](screenshots/ncu_v2_unified.png)

ncu profiled one `norm_kernel` launch (2²⁴, block size 256, 160 blocks) for each version. All four looked the same:

| Metric | Value (all four versions) |
|---|---|
| Duration | 0.544 – 0.553 ms |
| DRAM throughput | **93.2 – 95.2 %** |
| Compute (SM) throughput | 67.5 – 67.9 % |
| Achieved occupancy | about 100 % |

**NOTE: **The kernel keeps the GPU memory almost fully busy, so it is memory-bound: it runs about as fast as the T4 can read 128 MB. (ncu fixes the SM clock at 585 MHz, so its durations are a little longer than our benchmark times.)

---

## III. Comparative Table of Execution Time (2²⁴ elements)

Times are in milliseconds over 30 runs. The CUDA times are kernel only.

### Main table

| 2²⁴ elements | Block size | Time avg \| min \| max (ms) | Speedup vs C |
|---|---|---|---|
| C (Colab CPU, `gcc -O2`) | — | 31.2619 \| 24.6707 \| 46.3645 | 1.00x |
| C (laptop, VS Release) | — | 18.3172 \| 17.7056 \| 21.3269 | 1.00x |
| x86-64 (laptop) | — | 23.1282 \| 18.5724 \| 66.6737 | 0.79x |
| x86-64 SIMD XMM (laptop) | — | 10.8573 \| 9.5171 \| 12.8959 | 1.69x |
| x86-64 SIMD YMM (laptop) | — | 9.9456 \| 7.6551 \| 27.6573 | 1.84x |
| CUDA Unified | 256 | 1.5049 \| 0.5243 \| 29.8330 | 20.77x |
| | 512 | 1.4769 \| 0.5223 \| 29.0670 | 21.17x |
| | 1024 | 1.4333 \| 0.5223 \| 27.7745 | 21.81x |
| CUDA Prefetch | 256 | 0.5310 \| 0.5278 \| 0.5479 | 58.87x |
| | 512 | 0.5266 \| 0.5222 \| 0.5714 | 59.36x |
| | 1024 | 0.5290 \| 0.5254 \| 0.5386 | 59.10x |
| CUDA Prefetch + page creation | 256 | 0.5293 \| 0.5246 \| 0.5456 | 59.07x |
| | 512 | **0.5248 \| 0.5222 \| 0.5366** | **59.57x** |
| | 1024 | 0.5336 \| 0.5302 \| 0.5467 | 58.59x |
| CUDA Prefetch + page creation + mem advise | 256 | 0.5318 \| 0.5260 \| 0.5509 | 58.78x |
| | 512 | 0.5316 \| 0.5253 \| 0.5673 | 58.81x |
| | 1024 | 0.5275 \| 0.5243 \| 0.5407 | 59.26x |

CUDA rows use the main launch (blocks that fit on the GPU: 160 / 80 / 40 blocks).

**Baseline C program** The CUDA speedups are against the C program on the same Colab machine (31.26 ms). The x86-64, XMM and YMM speedups are against the C program on the same laptop (18.32 ms). Comparing each version with C on its own machine keeps the CPU hardware out of the comparison.

### Run 1 vs the other 29 runs (block size 256)

| Version | Run 1 (ms) | Avg of runs 2–30 (ms) |
|---|---|---|
| CUDA Unified | 29.8330 | 0.5281 |
| CUDA Prefetch | 0.5374 | 0.5308 |
| CUDA Prefetch + page creation | 0.5407 | 0.5289 |
| CUDA Prefetch + page creation + mem advise | 0.5509 | 0.5312 |

### Blocks that fit on the GPU vs maximum blocks (2²⁴, average kernel time in ms)

| Version | Max blocks 256 | Max blocks 512 | Max blocks 1024 | Fit 256 | Fit 512 | Fit 1024 |
|---|---|---|---|---|---|---|
| Blocks launched | 65,536 | 32,768 | 16,384 | 160 | 80 | 40 |
| CUDA Unified | 4.2808 | 3.0069 | 3.4801 | 1.5049 | 1.4769 | 1.4333 |
| CUDA Prefetch | 1.4945 | 2.5558 | 2.8908 | 0.5310 | 0.5266 | 0.5290 |
| CUDA Prefetch + page creation | 1.4911 | 1.8635 | 2.4159 | 0.5293 | 0.5248 | 0.5336 |
| CUDA Prefetch + page creation + mem advise | 1.5036 | 1.8720 | 3.0487 | 0.5318 | 0.5316 | 0.5275 |
| Bandwidth (page creation, runs 2–30) | 90 GB/s | 72 GB/s | 56 GB/s | 254 GB/s | 256 GB/s | 252 GB/s |

### Other sizes: 2²⁰ and 2²⁶ (average kernel time in ms; speedup vs Colab C of the same size)

| Version | 2²⁰ max blocks (256 / 512 / 1024) | 2²⁰ fit (256 / 512 / 1024) | 2²⁶ max blocks (256 / 512 / 1024) | 2²⁶ fit (256 / 512 / 1024) |
|---|---|---|---|---|
| C (Colab) | 1.5324 | | 98.1717 | |
| CUDA Unified | 0.3192 / 0.3711 / 0.2667 | 0.1630 / 0.1649 / 0.2081 | 11.5043 / 12.1039 / 19.0830 | 9.7535 / 6.1413 / 6.0885 |
| CUDA Prefetch | 0.1059 / 0.1311 / 0.1673 | 0.0392 / 0.0407 / 0.0409 | 8.2521 / 10.2887 / 9.8538 | 2.1278 / 2.1133 / 2.1351 |
| CUDA Prefetch + page creation | 0.0976 / 0.1230 / 0.1595 | **0.0391** / 0.0413 / 0.0415 | 6.7569 / 8.0207 / 10.4316 | 2.1098 / 2.1134 / 2.1310 |
| CUDA Prefetch + page creation + mem advise | 0.0980 / 0.1228 / 0.1608 | 0.0404 / 0.0406 / 0.0414 | 6.8088 / 8.0556 / 9.9455 | **2.1094** / 2.1139 / 2.1289 |
| Best speedup vs Colab C | 15.7x | **39.2x** | 14.5x | **46.5x** |

### Cross-platform comparison (everything against Matt's laptop C)

| 2²⁴, best block size | Avg time (ms) | Speedup vs laptop C (18.32 ms) | Effective bandwidth |
|---|---|---|---|
| C (laptop) | 18.3172 | 1.00x | 7.3 GB/s |
| x86-64 SIMD YMM (laptop) | 9.9456 | 1.84x | 13.5 GB/s |
| CUDA Unified (1024) | 1.4333 | 12.8x | — |
| CUDA Prefetch (512) | 0.5266 | 34.8x | 255 GB/s |
| CUDA Prefetch + page creation (512) | 0.5248 | **34.9x** | **256 GB/s** |
| CUDA Prefetch + page creation + mem advise (1024) | 0.5275 | 34.7x | 254 GB/s |
| **CUDA Unified, run 1 only (256)** | **29.8330** | **0.61x** | — |

| Other sizes | Laptop C | Laptop YMM | Best CUDA | CUDA vs YMM |
|---|---|---|---|---|
| 2²⁰ | 1.1758 ms | 0.3528 ms | 0.0391 ms | 9.0x faster |
| 2²⁶ | 72.5739 ms | 30.9901 ms | 2.1094 ms | 14.7x faster |

Formula used: Effective bandwidth = bytes read / time = (n × 8 bytes) / time.

---

## IV. Analysis of Results

### Kernel execution time

1. The first run of the Unified version is way slower than the rest (over 50x slower). For CUDA Unified, run 1 took anywhere from 27.7 to 29.8 ms, while runs 2 through 30 only took about 0.53 ms. Since we filled A on the CPU for this version, all its pages start out living in the CPU's memory. The second the kernel tries to touch a page that isn't on the GPU yet, the whole GPU stalls out, it migrates that page over, and then the threads finally keep going. You can see this super clearly in nsys. It logs 365 GPU page faults and 7,569 Host-to-Device transfers moving around 132 MB, and all of that happens right inside the first kernel call. Once run 1 finishes, the data just hangs out on the GPU, so runs 2 to 30 don't hit any page faults at all. That's why the max column for the Unified version is over 50x higher than the minimum, and why its overall 30-run average (1.43–1.50 ms) ends up being almost 3 times higher than all the other versions.

2. Prefetching gets rid of that slow first run. For the CUDA Prefetch version, run 1 took 0.537 ms, which is the exact same as the other runs (0.531 ms). This is because all 128 MB get moved over before the kernel even starts running. When we checked nsys, the prefetch moved 134.2 MB in just 64 big 2 MB transfers running at 12.0 GB/s, compared to the super slow 4.6 GB/s we got from on-demand migration in Version 2. Those huge batch transfers use the PCIe bus way better than thousands of tiny page faults.

3. Page creation skips the transfer completely. For Versions 4 and 5, we created and filled A directly on the GPU, so nsys literally shows 0 bytes copied from the host to the device (the only thing it copies back is the tiny 4 KB sum result). The CPU page faults also dropped from 385 down to just 1. Filling the vector on the GPU using init_kernel only took about 6 ms, which doesn't even count towards the actual kernel time. For comparison, Versions 2 and 3 do all this on the CPU, which means one single CPU core has to sit there and compute 16.7 million sin values.

4. Once the data is actually on the GPU, Versions 3, 4, and 5 all run at pretty much the exact same speed. Their averages all hovered around 0.525–0.534 ms across every block size, and their ranges overlap. Both nsys and ncu backed this up too. Since the kernel is doing the exact same math on the exact same GPU memory, actions such as prefetch, page creation, and mem advise really only affect how and when the data gets to the GPU, not how fast the kernel actually runs once it's there. The mem advise hints didn't really give us any measurable boost here since the GPU only reads our data once per run and doesn't share it back with the CPU anyway.

5. The kernel is memory-bound and maxing out the hardware. Our best runs hit around 256 GB/s, which is 80% of the T4 GPU's theoretical max of 320 GB/s. ncu also showed the DRAM throughput at around at 93–95%, while the actual SMs (compute cores) were only about 68% busy. Since each element needs 8 bytes of memory but only takes one simple multiply-add operation, the GPU spends almost all its time just waiting for data instead of actually computing. Because it's already maxing out the memory bandwidth, throwing more threads at it wouldn't really speed things up much.

6. The run-to-run timings were a bit confusing. The times shift between runs because the Colab GPU's clock speed ramps up and down based on temperature and load, plus the Colab CPU is shared. For example, CUDA Prefetch with a 512 block size (max blocks) took 2.56 ms, but Versions 4 and 5 only took 1.86–1.87 ms for the exact same setup—even though the kernel code is 100% identical. However, when we checked a single run, the min and avg are quite close, so it's definitely just a clock speed issue, not a code difference. Because Colab can be noisy randomly, we made sure to compare the average, minimum, and maximum times instead of just relying on one lucky run.

### A) How does block size affect execution time (various sizes, maximum blocks)? Which block size do we recommend?

With maximum blocks (one element per thread), the block size matters a ton, and 256 is definitely the best. When we tested at 2²⁴, Versions 3 through 5 took around 1.49 ms with a 256 block size, but slowed down to 1.86–2.56 ms at 512, and 2.42–3.05 ms at 1024. We saw the exact same pattern at 2²⁰ (going from 0.098 ms to 0.123 ms to 0.160 ms) and at 2²⁶ (where Version 4 went from 6.76 ms to 8.02 ms to 10.43 ms). We believed that this occurs because of the ff:
- Each thread is doing a single multiply-add, and then it immediately joins the shared-memory reduction (which takes 8 steps at 256, and 10 steps at 1024). Each of those steps needs a __syncthreads(), and then one thread per block does an atomicAdd. Because of that, the per-block overhead is where most of the work actually happens.
- The T4 GPU maxes out at 1024 threads per SM. If we use 256 threads per block, we can fit 4 blocks on a single SM. That means while one block is stuck waiting at a barrier, the other blocks can keep loading data. But if we use 1024 threads per block, only 1 block fits, so the entire SM just sits there waiting at every single barrier.
- In our very first version of the notebook, ncu actually showed this launch setup hitting only 14% DRAM throughput but 83% SM throughput. This means that the GPU was busy dealing with overhead instead of actually reading our data.

With blocks that fit on the GPU, block size barely matters. At 2²⁴, all three block sizes finished in about 0.525–0.534 ms (under a 2% difference). The exact same thing happened at 2²⁰ (0.039–0.042 ms) and 2²⁶ (2.11–2.14 ms). Since each thread is looping through about 410 elements in the grid-stride loop before it even hits the reduction, that reduction and atomic overhead is only paid once per thread instead of once per element. At that point, the kernel just becomes completely limited by memory speed. Fixing the launch setup made way more of a difference than the block size did as it made things around 2.8x to 4.5x faster than the max blocks setup at 2²⁴, and 3.2x to 4.9x faster at 2²⁶.

The vector size matters as well. The bandwidth for the "blocks that fit" setup was around 215 GB/s at 2²⁰, but it jumped up to about 255 GB/s at 2²⁴ and 2²⁶. Because the whole kernel only takes around 39 µs at 2²⁰, the fixed cost of just launching the kernel and doing the final reduction takes up a much bigger chunk of the total execution time.   

**Recommendation:** We recommend launching just the blocks that fit on the GPU and sticking with a block size of 256. With that specific launch setup, 256, 512, and 1024 are all just as fast, but 256 is the clear winner if a program ever ends up falling back to launching one thread per element. Furthermore, it gives the scheduler 4 blocks per SM to easily swap between, which keeps things running smoothly.

### B) Is prefetching always recommended, or should CUDA manage memory?

For our case, prefetching is good since we have to read the entire vector in order anyway, so we already know the GPU is gonna need all of it. Prefetching moved the data in 64 massive 2 MB chunks running at 12.0 GB/s, which beat the 7,569 on-demand transfers running at 4.6 GB/s. Additionally, it wiped out the 28–30 ms penalty we saw on the first run.

HOWEVER, prefetching is not always the best move. Just letting CUDA handle the memory on-demand is actually better on the following conditions:

- The GPU only needs a small or random part of a huge array. If we prefetch everything, we're just wasting time moving data we'll never even use. On-demand faults are way better here because they only move the pages the kernel actually touches.
- The data is bigger than our GPU memory. Unified Memory can oversubscribe the GPU memory and just page in exactly what it needs as it goes. If we try to prefetch all of it at once, it just won't fit.
- The CPU and GPU are passing the same data back and forth in small chunks. If they're taking turns working on the same stuff, constantly prefetching everything back and forth is gonna cost way more overhead than just letting the touched pages migrate on their own.
- CUDA manage memory is much more simpler to use so its better if you don't care too much of "perfect" optimizations.

In summary, you want to use prefetching when the kernel is going to read all the data, you already know the access pattern, and the data actually fits on the GPU. Looking at our results, Versions 4 and 5 were efficient because they completely avoided transferring the vector from the CPU to the GPU by just creating the data directly on the GPU itself. This killed the data transfer overhead, though actually generating that vector on the GPU using init_kernel still took about 6 ms.

### C) Between SIMD and SIMT, which one is faster?

If we are just looking at the kernel itself, SIMT on the GPU is way faster. At the 2²⁴ size, our best CUDA run took around 0.525 ms, which is 19x faster than YMM (9.95 ms) and 35x faster than plain C (18.32 ms). At 2²⁰, CUDA was 9x faster than YMM, and at 2²⁶ it was 14.7x faster. The GPU is using memory at roughly 255 GB/s, while the laptop CPU running YMM maxed out at only 13.5 GB/s.

However, it depends on where your data starts out. If the data gets created on the CPU and you only need to use it once, the GPU has to receive that entire 128 MB over the PCIe bus first. In our Unified version, that first run (has migration) took 29.8 ms, which is slower than just running plain C on the laptop (18.32 ms) and 3x slower than YMM. Even when we used prefetching, moving the data still took about 11.1 ms, which is longer than YMM needed to just compute the whole answer right there on the CPU (9.95 ms). Since our Euclidean norm project just squares each element and adds it to a sum, transferring a massive vector over to the GPU can easily take more time than just crunching the numbers with CPU SIMD. So CUDA isn't automatically the faster choice, especially for "one-and-done" calculations.

**Summary**

SIMD (CPU) is better when:
- the data is already in CPU memory and is used once, as in a single norm of a freshly computed vector;
- the vector is small (at 2²⁰ elements, YMM took only 0.35 ms, much less than a GPU transfer plus kernel launch);
- there is no NVIDIA GPU, or the computation is part of a mostly sequential CPU program.

SIMT (GPU) is better when:
- the data is already on the GPU or can be created there (Versions 4 and 5);
- the same data is used many times, so the transfer is paid once and spread over many kernel runs (our 30-run benchmark is an example);
- the data is large and there is a lot of independent work per element.

### Speed across all platforms

From slowest to fastest kernel time at 2²⁴: Colab C (31.3 ms) → x86-64 (23.1 ms) → laptop C (18.3 ms) → XMM (10.9 ms) → YMM (9.9 ms) → CUDA Unified (1.43–1.50 ms average, but 28–30 ms in run 1) → CUDA Prefetch / page creation / mem advise (0.525–0.534 ms).

- C on two machines: The exact same C code took 31.3 ms on Colab's CPU and only 18.3 ms on Matt's laptop. The Colab times were also noisy just because we're sharing that CPU with other people. That is why we made sure to only compare each version against the C baseline on its own specific machine.
- We also compared assembly and C. Since we ran them in Release mode, the compiler is smart enough to keep i and sum in registers just like we did in our assembly, so x86-64 and C tied. At the 2²⁰ and 2²⁶ sizes, they were within 2% of each other, and x86-64's best run at 2²⁴ (18.57 ms) matched C. Its 2²⁴ average (23.1 ms) only looked worse because a few runs got slowed down by background processes running on the laptop.
-  XMM and YMM: These ended up being 1.69x and 1.84x faster than plain C. We didn't hit those theoretical 2x and 4x speedups mostly because the CPU kernel just gets bottlenecked by memory speed (YMM was only reading at about 13.5 GB/s).
- CUDA is about 35x faster in the C program test than Matt's laptop. Both the CPU SIMD versions and the GPU are limited by memory speed, but the T4 GPU's memory is just 19x faster than what a single CPU core can read.

---

## V. Problems Encountered, Solutions and AHA Moments

- Our first grid-stride loop didn't actually loop. For our first version, we launched (n + blockSize - 1) / blockSize blocks (only one thread per element). This meant our stride = n, so every single thread just ran the loop body exactly once. It technically worked, but it didn't really take advantage of the grid-stride loop. Additionally we had 65,536 blocks all trying to do their own reduction and atomicAdd. Kane caught this while looking over the notebook. Our solution is that we changed it to only launch the exact number of blocks that can fit on the GPU at once (using SMs × resident blocks per SM, which we got from cudaOccupancyMaxActiveBlocksPerMultiprocessor). Doing it this way means each thread actually loops over about 410 elements. This made the kernel 2.8x faster (dropping from 1.49 ms to 0.53 ms at a 256 block size) and our bandwidth shot up from 90 GB/s to 255 GB/s. We kept the old launch setup around to use as a comparison for guide question (a).
- The norm is a reduction, which is a lot harder to do on a GPU. If we just did a simple *sum += A[i] * A[i] in every thread, we'd get a massive race condition. And if we tried using an atomicAdd for every single element, it would serialize 16.7 million updates which would completely tank performance. The solution for this is that we set it up so each thread adds its own elements into a local register first. Then, each block adds up all its threads' values inside shared memory, and we only send one atomicAdd per block out to global memory.   
- The prefetch and mem advise calls wouldn't compile on Colab for some reason. Colab apparently uses CUDA 13.0, which completely changed cudaMemPrefetchAsync and cudaMemAdvise so they require a cudaMemLocation instead of just a device number. So we wrote some quick wrapper functions that check the CUDART_VERSION and automatically use the right format, that way the notebook works perfectly on both CUDA 12 and 13.
- We got confused on the Page creation as it isn't actually a CUDA function. We kinda had to figure out what it meant for this assignment so we just created the pages directly inside the GPU memory. To do this, we just prefetch the empty buffer and then fill it up using a GPU kernel. We checked nsys and it confirmed this worked perfectly and exactly 0 bytes were copied from the host to the device.
- The GPU result didn't match the C version exactly. Because the GPU adds everything up in a totally different order, the last few digits are a little off (2896.2749920236 vs 2896.2749920235). So we just compared them using a relative tolerance from our SIMD Project.
- Our SIMD program didn't have the 2²⁴ size or min/max timings. For our older project, we only measured 2²⁰, 2²⁶, and 2³⁰, and we only reported the averages so we went back and added the 2²⁴ size and the min/max timings.
- The profilers messrf with the timings. When we ran ncu, run 1 took over 2 full seconds because it replays the kernel like 10 times to collect all its counters, and nsys adds a bunch of overhead. Which is why we pulled all our actual benchmark numbers from the normal unprofiled runs, and just used the profilers to help explain what was going on while the code was running.
- Colab timings are inconsistent between runs. It didn't help that we kept getting an error that Their is no T4 GPU available at night time (a lot of people probably are using it at that time) so we made sure to report the avg, min, and max, and kept the "cold start run" seperate and onyl trusted the next sequential runs.

**AHA moments**

- Fewer blocks actually made the kernel faster. We expected that launching one thread per element would be the fastest way to do it. But it turned out that only launching the 160 blocks that fit perfectly on the GPU was 2.8x faster. This is because every thread's reduction and atomic overhead gets shared across 410 elements instead of just doing it for one.
- The block size stopped mattering when we got the launch setup right. We used the maximum blocks approach, a block size of 1024 was almost 2x slower than 256. However, once we swapped to only launching the blocks that fit on the GPU, the 256, 512, and 1024 block sizes all performed within 2% of each other.
- The difference between faults and prefetching comes down to just transfer size. Both setups moved roughly the same 132 to 134 MB of data, but doing it with on-demand faults caused 7,569 tiny transfers running at 4.6 GB/s, while prefetching did it in 64 huge 2 MB transfers running at 12.0 GB/s.
- The absolute fastest transfer is no transfer. We realized that creating the data straight on the GPU was better because it completely skipped having to transfer the vector over from the CPU. The GPU only took about 6 ms to just initialize the data itself.
- Moving the data around can cost more than actually computing it as transferring that 128 MB over to the GPU took around 11 to 29 ms, which is actually longer than it took YMM to compute the entire norm over on the CPU (9.9 ms). So the GPU only wins when the data is already sitting there or if it gets reused a lot.
- Both the CPU and GPU are limited by memory at different speeds. On the CPU, using wider SIMD barely did anything because the RAM was bottlenecking it at 13.5 GB/s. But on the GPU, ncu showed we were hitting 95% DRAM utilization. So our kernel was maxed out running at around 255 GB/s, and throwing more threads at it wouldn't make it any faster.

---

## VI. SIMD vs SIMT

### How each one handles parallelism in our project

**SIMD (XMM / YMM on the CPU):** a single CPU thread runs one instruction stream, and each instruction works on a wide register (2 doubles at a time for XMM, or 4 for YMM). We had to hardcode all the parallelism ourselves in assembly. We loaded 4 doubles using vmovupd, square them with vmulpd, add them to our 4 partial sums with vaddpd, and then combine all 4 lanes at the very end. We also had to write a whole separate scalar tail loop just to handle the 1 to 3 leftover elements. The main downside is that the parallelism is hard-capped by the register width and it's all running on a single core.

**SIMT (CUDA on the GPU):** We wrote the code from the perspective of one single thread, and then the GPU blasts it out to run on 40,960 threads at the exact same time (160 blocks × 256 threads). The hardware groups these threads into warps of 32 that all run the same instruction together (like SIMD), but every thread still gets its own index, registers, and control flow. The GPU is smart about hiding memory latency, because it just rapidly switches between all the different warps on each SM. Additionally, the leftover elements get handled automatically because of i < n check in our grid-stride loop. The only real problem is that threads can't just easily share a single sum variable. We had to do all the extra work setting up shared memory, using __syncthreads(), and doing an atomicAdd. Furthermore, we had to be careful about how we set up the launch configuration.   

### Pros and cons for our use case (Euclidean norm)

| | SIMD (XMM / YMM) | SIMT (CUDA) |
|---|---|---|
| **Speed (kernel only, 2²⁴)** | YMM 9.95 ms, 1.84x faster than C | 0.525 ms, 35x faster than laptop C, 19x faster than YMM |
| **Data transfer** | None, data is already in RAM | 128 MB must reach GPU memory (11–29 ms), unless created there |
| **Reduction (the sum)** | Easy: keep 2 or 4 partial sums in one register, combine at the end | Harder: shared memory, barriers and atomics needed to avoid race conditions |
| **Leftover elements** | Need a separate scalar tail loop | Handled by the `i < n` check |
| **Tuning** | Mainly register width | Launch configuration matters a lot (up to 4.5x between our two launches) |
| **Programming effort** | Low-level assembly, register by register | High-level C++, but memory management and launch configuration must be tuned |
| **Scaling** | Limited by register width (4 doubles) and one core | Tens of thousands of threads; scales with GPU size |
| **Bottleneck in our results** | Memory bandwidth (13.5 GB/s) | Memory bandwidth (about 255 GB/s, 95% DRAM use) |
| **Hardware needed** | Any modern x86-64 CPU | An NVIDIA GPU |

**Summary:** Based on our tests, SIMD is a good choice for smaller data because it does not need to transfer data to the GPU. CUDA is faster when the data is already on the GPU, especially for larger vectors, so it is the better choice when the data is already on the GPU, the data can be created there, or is reused many times, so the transfer cost is paid only once.
