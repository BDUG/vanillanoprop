use libc::size_t;

extern "C" {
    pub fn kai_matmul(a: *const f32, b: *const f32, c: *mut f32, m: size_t, n: size_t, k: size_t);
}
