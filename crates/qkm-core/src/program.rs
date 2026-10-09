//! The per-period ProgPoW program that KawPoW runs in each of its 64 DAG
//! loops, taken from `qkm_kawpow::common::init(period)` and the KISS99
//! sequence exactly as `qkm_kawpow::common::prog_loop` consumes it.
//!
//! The GPU kernel is generated from this program (see `kernel`); the tests
//! interpret it on the CPU and require the result to equal
//! `qkm_kawpow::common::prog_loop`, the code rs-quai and go-quai agree on.

use qkm_kawpow::common::{
    CACHE_WORDS, CNT_CACHE, CNT_MATH, DAG_LOADS, LANES, REGS, init, math, merge,
};

/// One step of the inner loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// `mix[dst] = merge(mix[dst], c_dag[mix[src] % CACHE_WORDS], merge)`.
    Cache {
        /// Register whose value indexes the cached DAG.
        src: u8,
        /// Register merged into.
        dst: u8,
        /// Merge selector.
        merge: u32,
    },
    /// `mix[dst] = merge(mix[dst], math(mix[src1], mix[src2], math), merge)`.
    Math {
        /// First operand register.
        src1: u8,
        /// Second operand register (never `src1`).
        src2: u8,
        /// Math selector.
        math: u32,
        /// Register merged into.
        dst: u8,
        /// Merge selector.
        merge: u32,
    },
}

/// The program for one period (`height / 3` for KawPoW).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    /// The period this program belongs to.
    pub period: u64,
    /// Cache and math steps, in execution order.
    pub ops: Vec<Op>,
    /// The four DAG-word merges at the end of the loop: `(dst, merge)`;
    /// the first always targets register 0.
    pub dag: [(u8, u32); DAG_LOADS],
}

impl Program {
    /// Derives the program for `period` (the same KISS99 draws as
    /// `prog_loop`, in the same order).
    pub fn new(period: u64) -> Program {
        let (mut rs, dst_seq, src_seq) = init(period);
        let regs = REGS as u32;
        let mut src_c = 0usize;
        let mut dst_c = 0usize;
        let mut ops = Vec::with_capacity((CNT_CACHE + CNT_MATH) as usize);
        for i in 0..CNT_MATH {
            if i < CNT_CACHE {
                let src = src_seq[src_c % REGS] as u8;
                src_c += 1;
                let dst = dst_seq[dst_c % REGS] as u8;
                dst_c += 1;
                let merge = rs.next();
                ops.push(Op::Cache { src, dst, merge });
            }
            let src_rnd = rs.next() % (regs * (regs - 1));
            let src1 = src_rnd % regs;
            let mut src2 = src_rnd / regs;
            if src2 >= src1 {
                src2 += 1;
            }
            let math = rs.next();
            let dst = dst_seq[dst_c % REGS] as u8;
            dst_c += 1;
            let merge = rs.next();
            ops.push(Op::Math {
                src1: src1 as u8,
                src2: src2 as u8,
                math,
                dst,
                merge,
            });
        }
        let mut dag = [(0u8, 0u32); DAG_LOADS];
        dag[0] = (0, rs.next());
        for d in dag.iter_mut().skip(1) {
            let dst = dst_seq[dst_c % REGS] as u8;
            dst_c += 1;
            *d = (dst, rs.next());
        }
        Program { period, ops, dag }
    }

    /// One DAG loop on the CPU, with the same arguments and effect as
    /// `qkm_kawpow::common::prog_loop(self.period, loop_, ...)`.
    pub fn run_loop(
        &self,
        loop_: u32,
        mix: &mut [[u32; REGS]; LANES],
        lookup: &mut dyn FnMut(u32) -> [u8; 256],
        cdag: &[u32],
        dataset_size: u32,
    ) {
        let lanes = LANES as u32;
        let loads = DAG_LOADS as u32;
        let g_offset =
            mix[(loop_ % lanes) as usize][0] % (64u32.wrapping_mul(dataset_size) / (lanes * loads));
        let entry = lookup(g_offset.wrapping_mul(lanes).wrapping_mul(loads));
        for l in 0..lanes {
            let lane = &mut mix[l as usize];
            for op in &self.ops {
                match *op {
                    Op::Cache { src, dst, merge: r } => {
                        let data = cdag[(lane[src as usize] % CACHE_WORDS as u32) as usize];
                        merge(&mut lane[dst as usize], data, r);
                    }
                    Op::Math {
                        src1,
                        src2,
                        math: m,
                        dst,
                        merge: r,
                    } => {
                        let data = math(lane[src1 as usize], lane[src2 as usize], m);
                        merge(&mut lane[dst as usize], data, r);
                    }
                }
            }
            let index = (((l ^ loop_) % lanes) * loads) as usize;
            for (k, &(dst, r)) in self.dag.iter().enumerate() {
                let o = 4 * (index + k);
                let w = u32::from_le_bytes([entry[o], entry[o + 1], entry[o + 2], entry[o + 3]]);
                merge(&mut lane[dst as usize], w, r);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qkm_kawpow::common::prog_loop;

    /// xorshift64* for test data.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 32) as u32
        }
    }

    #[test]
    fn program_matches_reference_prog_loop() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let cdag: Vec<u32> = (0..CACHE_WORDS).map(|_| rng.next()).collect();
        // A deterministic fake DAG: entry words are a hash of the base.
        let mut lookup = |base: u32| {
            let mut out = [0u8; 256];
            let mut s = Rng(u64::from(base) * 0x9e37_79b9 + 1);
            for c in out.chunks_mut(4) {
                c.copy_from_slice(&s.next().to_le_bytes());
            }
            out
        };
        let mut lookup2 = lookup;
        for period in [
            0u64,
            1,
            2,
            1_521_745,
            1_521_902,
            4_666_666,
            u64::from(u32::MAX) + 7,
        ] {
            let prog = Program::new(period);
            assert_eq!(prog.ops.len(), (CNT_CACHE + CNT_MATH) as usize);
            for loop_ in [0u32, 1, 15, 16, 63] {
                let mut a = [[0u32; REGS]; LANES];
                for lane in a.iter_mut() {
                    for w in lane.iter_mut() {
                        *w = rng.next();
                    }
                }
                let mut b = a;
                let ds = 24_117_247; // dataset_size / 256 at epoch 608
                prog_loop(period, loop_, &mut a, &mut lookup, &cdag, ds);
                prog.run_loop(loop_, &mut b, &mut lookup2, &cdag, ds);
                assert_eq!(a, b, "period {period} loop {loop_}");
            }
        }
    }
}
