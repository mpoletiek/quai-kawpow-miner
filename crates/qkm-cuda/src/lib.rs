//! CUDA backend: compiles kernels with NVRTC straight to native code
//! (CUBIN for the device's `sm_XY`, so a toolkit newer than the driver
//! still loads), generates the DAG on the GPU and runs the KawPoW search.

use std::collections::HashMap;
use std::ffi::{CString, c_char};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceRepr, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::{Ptx, result as nvrtc, sys as nvrtc_sys};
use qkm_core::kernel::{
    DAG_FNS, DAG_SOURCE, HASH_FN, MAX_RESULTS, RESULT_WORDS, SEARCH_FN, SearchParams, search_source,
};
use qkm_core::{Epoch, Program, period};

/// Backend errors (driver, NVRTC, or setup).
#[derive(Debug, Clone)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}

impl From<cudarc::driver::DriverError> for Error {
    fn from(e: cudarc::driver::DriverError) -> Self {
        Error(format!("CUDA driver: {e}"))
    }
}
impl From<cudarc::nvrtc::result::NvrtcError> for Error {
    fn from(e: cudarc::nvrtc::result::NvrtcError) -> Self {
        Error(format!("NVRTC: {e:?}"))
    }
}

/// Result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// The header hash as the kernel takes it (8 little-endian words).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Header(pub [u32; 8]);

// SAFETY: a plain `#[repr(C)]` array of u32, matching `header_t` in the kernel.
unsafe impl DeviceRepr for Header {}

impl Header {
    /// From the 32 header bytes (words read little-endian).
    pub fn from_bytes(b: &[u8; 32]) -> Header {
        let mut w = [0u32; 8];
        for (i, x) in w.iter_mut().enumerate() {
            *x = u32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
        }
        Header(w)
    }
}

/// Compiles CUDA C to a CUBIN for `sm_{major}{minor}`.
pub fn compile_cubin(src: &str, arch: (i32, i32)) -> Result<Vec<u8>> {
    let csrc = CString::new(src).map_err(|e| Error(e.to_string()))?;
    let prog = nvrtc::create_program(&csrc, None)?;
    let opts = [format!("--gpu-architecture=sm_{}{}", arch.0, arch.1)];
    // SAFETY: `prog` is a live program; it is destroyed on every path below.
    unsafe {
        if let Err(e) = nvrtc::compile_program(prog, &opts) {
            let log = nvrtc::get_program_log(prog)
                .map(|l| {
                    let bytes: Vec<u8> = l
                        .iter()
                        .take_while(|c| **c != 0)
                        .map(|c| *c as u8)
                        .collect();
                    String::from_utf8_lossy(&bytes).into_owned()
                })
                .unwrap_or_default();
            let _ = nvrtc::destroy_program(prog);
            return Err(Error(format!("NVRTC compile failed: {e:?}\n{log}")));
        }
        let mut size = 0usize;
        if let Err(e) = nvrtc_sys::nvrtcGetCUBINSize(prog, &mut size).result() {
            let _ = nvrtc::destroy_program(prog);
            return Err(e.into());
        }
        let mut cubin = vec![0u8; size];
        let r = nvrtc_sys::nvrtcGetCUBIN(prog, cubin.as_mut_ptr() as *mut c_char).result();
        let _ = nvrtc::destroy_program(prog);
        r?;
        Ok(cubin)
    }
}

/// One GPU.
pub struct Gpu {
    /// The device context.
    pub ctx: Arc<CudaContext>,
    /// Compute capability.
    pub arch: (i32, i32),
    /// Device name.
    pub name: String,
}

impl Gpu {
    /// Opens device `ordinal`.
    pub fn new(ordinal: usize) -> Result<Gpu> {
        let ctx = CudaContext::new(ordinal)?;
        // Each stream owns its buffers and the DAG is finished before any
        // search, so cudarc's cross-stream event tracking is only overhead.
        // SAFETY: no buffer is shared between streams without a synchronize.
        unsafe { ctx.disable_event_tracking() };
        let arch = ctx.compute_capability()?;
        let name = ctx.name()?;
        Ok(Gpu { ctx, arch, name })
    }

    /// Free and total device memory in bytes.
    pub fn memory(&self) -> Result<(usize, usize)> {
        Ok(self.ctx.mem_get_info()?)
    }

    /// Returns freed device memory to the driver: frees are stream-ordered
    /// (`cuMemFreeAsync`) and the pool keeps them until a synchronize, so
    /// call this after dropping a DAG and before allocating the next one.
    pub fn release_memory(&self) -> Result<()> {
        self.ctx.synchronize()?;
        self.ctx.bind_to_thread()?;
        let mut pool = std::ptr::null_mut();
        // SAFETY: a valid device handle and an out-pointer for its default pool.
        unsafe {
            cudarc::driver::sys::cuDeviceGetDefaultMemPool(&mut pool, self.ctx.cu_device())
                .result()?;
            cudarc::driver::sys::cuMemPoolTrimTo(pool, 0).result()?;
        }
        Ok(())
    }

    /// Loads a CUBIN.
    pub fn load(&self, cubin: Vec<u8>) -> Result<Arc<CudaModule>> {
        Ok(self.ctx.load_module(Ptx::from_binary(cubin))?)
    }

    /// Builds the DAG of `epoch` from its light cache (u32 words) on the GPU.
    pub fn generate_dag(&self, epoch: Epoch, light: &[u32]) -> Result<Dag> {
        let stream = self.ctx.default_stream();
        let module = self.load(compile_cubin(DAG_SOURCE, self.arch)?)?;
        let fs = DAG_FNS
            .iter()
            .map(|n| module.load_function(n))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let light_dev = stream.clone_htod(light)?;
        let light_rows = (light.len() / 16) as u32;
        let items = epoch.dag_items();
        let dag = DeviceBuffer::new(&self.ctx, items as usize * 64)?;
        const CHUNK: u64 = 1 << 21;
        const BLOCK: u32 = 256;
        let mut start = 0u64;
        while start < items {
            let end = (start + CHUNK).min(items);
            let (s, e) = (start as u32, end as u32);
            let cfg = LaunchConfig {
                grid_dim: ((end - start).div_ceil(u64::from(BLOCK)) as u32, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };
            for (n, f) in fs.iter().enumerate() {
                // dag_mix runs four threads per item.
                let threads = if n == 1 { 4 } else { 1 };
                let cfg = LaunchConfig {
                    grid_dim: (
                        ((end - start) * threads).div_ceil(u64::from(BLOCK)) as u32,
                        1,
                        1,
                    ),
                    ..cfg
                };
                let mut b = stream.launch_builder(f);
                b.arg(&s).arg(&e);
                if n < 2 {
                    b.arg(&light_dev).arg(&light_rows);
                }
                b.arg(&dag.ptr);
                // SAFETY: argument types match the `dag_*` kernels; `dag`
                // holds `items` 64-byte items.
                unsafe { b.launch(cfg) }?;
            }
            start = end;
        }
        stream.synchronize()?;
        Ok(Dag { epoch, buf: dag })
    }
}

/// Device memory from `cuMemAlloc` (not the stream-ordered pool), freed on
/// drop. The DAG lives here, like kawpowminer's `cudaMalloc`'d DAG.
pub struct DeviceBuffer {
    /// The device address, passed to kernels as a pointer argument.
    pub ptr: u64,
    /// Size in bytes.
    pub bytes: usize,
    ctx: Arc<CudaContext>,
}

impl DeviceBuffer {
    /// Allocates `bytes` on the context's device.
    pub fn new(ctx: &Arc<CudaContext>, bytes: usize) -> Result<DeviceBuffer> {
        ctx.bind_to_thread()?;
        let mut ptr = 0u64;
        // SAFETY: an out-pointer for the allocation; the context is current.
        unsafe { cudarc::driver::sys::cuMemAlloc_v2(&mut ptr, bytes).result()? };
        Ok(DeviceBuffer {
            ptr,
            bytes,
            ctx: ctx.clone(),
        })
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.ctx.bind_to_thread().is_ok() {
            // SAFETY: `ptr` came from cuMemAlloc on this context and is freed once.
            unsafe {
                let _ = cudarc::driver::sys::cuMemFree_v2(self.ptr);
            }
        }
    }
}

/// A DAG resident on the GPU.
pub struct Dag {
    /// Its epoch.
    pub epoch: Epoch,
    /// The dataset.
    pub buf: DeviceBuffer,
}

/// A compiled search module for one period.
pub struct SearchKernel {
    /// The period.
    pub period: u64,
    /// The tunables it was built with.
    pub params: SearchParams,
    _module: Arc<CudaModule>,
    search: CudaFunction,
    hash: CudaFunction,
}

type Pending = JoinHandle<Result<Vec<u8>>>;

/// Search kernels by period, with the next period compiled in the
/// background (a new program every 3 heights).
pub struct Kernels {
    arch: (i32, i32),
    params: SearchParams,
    ready: HashMap<(u64, u32), Arc<SearchKernel>>,
    pending: Mutex<HashMap<(u64, u32), Pending>>,
}

fn source_for(per: u64, dag_elements: u32, params: SearchParams) -> String {
    search_source(&Program::new(per), dag_elements, params)
}

impl Kernels {
    /// For a device and the tunables baked into each module.
    pub fn new(gpu: &Gpu, params: SearchParams) -> Kernels {
        Kernels {
            arch: gpu.arch,
            params,
            ready: HashMap::new(),
            pending: Mutex::new(HashMap::new()),
        }
    }

    /// Starts compiling the kernel for `per` in the background.
    pub fn prefetch(&self, per: u64, dag_elements: u32) {
        let key = (per, dag_elements);
        if self.ready.contains_key(&key) {
            return;
        }
        let mut p = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        if p.contains_key(&key) {
            return;
        }
        let (arch, params) = (self.arch, self.params);
        p.insert(
            key,
            std::thread::spawn(move || compile_cubin(&source_for(per, dag_elements, params), arch)),
        );
    }

    /// The kernel for `height` with `epoch`'s DAG; compiles it if needed and
    /// prefetches the next period's.
    pub fn get(&mut self, gpu: &Gpu, height: u64, epoch: Epoch) -> Result<Arc<SearchKernel>> {
        let per = period(height);
        let de = epoch.dag_elements();
        let key = (per, de);
        if !self.ready.contains_key(&key) {
            let job = self
                .pending
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&key);
            let cubin = match job {
                Some(j) => j
                    .join()
                    .map_err(|_| Error("compile thread panicked".into()))??,
                None => compile_cubin(&source_for(per, de, self.params), self.arch)?,
            };
            let module = gpu.load(cubin)?;
            let search = module.load_function(SEARCH_FN)?;
            let hash = module.load_function(HASH_FN)?;
            // Keep the current and the next period only.
            self.ready.retain(|k, _| k.0 + 1 >= per && k.0 <= per + 1);
            self.ready.insert(
                key,
                Arc::new(SearchKernel {
                    period: per,
                    params: self.params,
                    _module: module,
                    search,
                    hash,
                }),
            );
        }
        self.prefetch(per + 1, de);
        Ok(self.ready[&key].clone())
    }
}

/// A solution reported by the GPU (to be checked on the CPU).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Found {
    /// The nonce.
    pub nonce: u64,
    /// The mix digest words.
    pub mix: [u32; 8],
}

/// Launch geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    /// Blocks per launch.
    pub grid: u32,
    /// Streams kept in flight.
    pub streams: usize,
}

impl Geometry {
    /// Nonces per launch with `block` threads per block.
    pub fn batch(&self, block: u32) -> u64 {
        u64::from(self.grid) * u64::from(block)
    }
}

/// Runs search launches on several streams so the GPU never idles while
/// the host reads results.
pub struct Searcher {
    streams: Vec<Arc<CudaStream>>,
    outs: Vec<CudaSlice<u32>>,
    geom: Geometry,
}

impl Searcher {
    /// Streams and result buffers for `geom`.
    pub fn new(gpu: &Gpu, geom: Geometry) -> Result<Searcher> {
        let mut streams = Vec::new();
        let mut outs = Vec::new();
        for _ in 0..geom.streams.max(1) {
            let s = gpu.ctx.new_stream()?;
            outs.push(s.alloc_zeros::<u32>(RESULT_WORDS)?);
            streams.push(s);
        }
        Ok(Searcher {
            streams,
            outs,
            geom,
        })
    }

    fn launch(
        &mut self,
        i: usize,
        k: &SearchKernel,
        dag: &Dag,
        header: Header,
        target: u64,
        start: u64,
    ) -> Result<()> {
        let stream = self.streams[i].clone();
        stream.memset_zeros(&mut self.outs[i])?;
        let cfg = LaunchConfig {
            grid_dim: (self.geom.grid, 1, 1),
            block_dim: (k.params.block, 1, 1),
            shared_mem_bytes: 0,
        };
        let hack = 0u32;
        let mut b = stream.launch_builder(&k.search);
        b.arg(&start)
            .arg(&header)
            .arg(&target)
            .arg(&dag.buf.ptr)
            .arg(&mut self.outs[i])
            .arg(&hack);
        // SAFETY: arguments match `kawpow_search`; the output buffer holds RESULT_WORDS.
        unsafe { b.launch(cfg) }?;
        Ok(())
    }

    fn collect(&self, i: usize, start: u64, found: &mut Vec<Found>) -> Result<()> {
        let out = self.streams[i].clone_dtoh(&self.outs[i])?;
        let n = out[0].min(MAX_RESULTS) as usize;
        for r in 0..n {
            let o = 1 + r * 9;
            let mut mix = [0u32; 8];
            mix.copy_from_slice(&out[o + 1..o + 9]);
            found.push(Found {
                nonce: start.wrapping_add(u64::from(out[o])),
                mix,
            });
        }
        Ok(())
    }

    /// Searches from `start` until `keep_going(nonces_done, &found)` returns
    /// false, re-launching each stream as soon as its results are read.
    /// Returns the nonces searched and the solutions found.
    pub fn run(
        &mut self,
        k: &SearchKernel,
        dag: &Dag,
        header: Header,
        target: u64,
        start: u64,
        keep_going: &mut dyn FnMut(u64, &[Found]) -> bool,
    ) -> Result<(u64, Vec<Found>)> {
        let batch = self.geom.batch(k.params.block);
        let n = self.streams.len();
        let mut starts = vec![0u64; n];
        let mut next = start;
        for (i, s) in starts.iter_mut().enumerate() {
            *s = next;
            self.launch(i, k, dag, header, target, next)?;
            next = next.wrapping_add(batch);
        }
        let mut found = Vec::new();
        let mut done = 0u64;
        let mut i = 0;
        loop {
            self.collect(i, starts[i], &mut found)?;
            done += batch;
            if !keep_going(done, &found) {
                for (j, s) in self.streams.iter().enumerate() {
                    if j != i {
                        s.synchronize()?;
                    }
                }
                return Ok((done, found));
            }
            starts[i] = next;
            self.launch(i, k, dag, header, target, next)?;
            next = next.wrapping_add(batch);
            i = (i + 1) % n;
        }
    }

    /// Every nonce's (mix digest, final hash words) for `count` nonces from
    /// `start`, with the debug kernel (rounded up to whole blocks).
    pub fn hash_all(
        &self,
        k: &SearchKernel,
        dag: &Dag,
        header: Header,
        start: u64,
        count: u32,
    ) -> Result<Vec<([u32; 8], [u32; 8])>> {
        let block = k.params.block;
        let count = count.div_ceil(block) * block;
        let stream = self.streams[0].clone();
        let mut out = stream.alloc_zeros::<u32>(count as usize * 16)?;
        let cfg = LaunchConfig {
            grid_dim: (count / block, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(&k.hash);
        b.arg(&start).arg(&header).arg(&dag.buf.ptr).arg(&mut out);
        // SAFETY: arguments match `kawpow_hash`; `out` holds 16 words per nonce.
        unsafe { b.launch(cfg) }?;
        let v = stream.clone_dtoh(&out)?;
        Ok(v.chunks_exact(16)
            .map(|c| {
                let mut m = [0u32; 8];
                let mut f = [0u32; 8];
                m.copy_from_slice(&c[..8]);
                f.copy_from_slice(&c[8..]);
                (m, f)
            })
            .collect())
    }
}
