pub mod dbm {
    /*
     * This is an implementation of a DBM.
     * The matrix is declared as a single array, as this guarantees a single, coherent block of memory, rather than having all the rows spread around in the heap.
     * As such, it should be indexed with an offset.
     *
     */
    use crate::bitvector::Bitvector;

struct DBM<T, N> {
        matrix: Vec<T>,
        clock_names: Vec<N>,
        bitvec: Bitvector,
    }

    enum ConstraintOp{
        LessThan,
        LessThanEqual,
    }

    impl<T, N> DBM<T, N> {
        fn new(clocks: Vec<N>) { //Intentionally doesn't take a reference, as we would like the names to be owned by the data structure

        }

        fn up() {

        }

        fn down() {

        }

        fn free(_clock_to_free: N) {

        }

        fn reset(_clock_to_reset: N, _reset_val: T) {

        }

        fn copy(_clock_target: N, _clock_src: N) {

        }

        fn and(_clock: N, _constraint: (T, ConstraintOp)) { //Constraint is a tuple over T and ConstraintOP, as this mimicks the notation in Timed Automata: Semantics, Algorithms and Tools by Bengtsson and Yi

        }

        fn normalize() {

        }

        fn shift(_clock: N, _d_val: T) {

        }

        fn get_index_of_clock(_clock: N) {

        }
    }
}

pub mod interface {
    type raw_t = u64;
    type cindex_t = i64;

    #[no_mangle]
    extern "C" fn dbm_init(_dbm: *mut raw_t, _dim: cindex_t) {}

    #[no_mangle]
    extern "C" fn dbm_zero(_dbm: *mut raw_t, _dim: cindex_t) {}

    #[no_mangle]
    extern "C" fn dbm_isEqualToInit(_dbm: *const raw_t, _dim: cindex_t) {}
}
