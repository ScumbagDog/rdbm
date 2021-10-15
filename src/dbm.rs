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

    enum ConstraintOp {
        LessThan,
        LessThanEqual,
    }

    impl<T: std::default::Default, //For all T's that implement Default
         N: std::cmp::Eq + std::default::Default, //all N's that implement Eq and Default
         > DBM<T, N> {

        fn new(mut clocks: Vec<N>) -> DBM<T, N> { //Intentionally doesn't take a reference, as we would like the names to be owned by the data structure
            let bitvector = Bitvector::init_with_length(clocks.len() as u32);
            clocks.insert(0, Default::default());
            let matrix_size = clocks.len() * clocks.len();
            let mut matrix: Vec<T>  = Vec::new();
            matrix.resize_with(matrix_size, Default::default);
            Self{
                matrix: matrix,
                clock_names: clocks,
                bitvec: bitvector,
            }
        }

        fn get_dimsize(&self) -> usize { //only need one function, as the matrices are always quadratic
           self.clock_names.len()
        }

        fn get_element(&self, row: usize, col: usize) -> Option<&T>{
            match self.get_dimsize() {
                0 => None, //If dimsize is 0 (ie. DBM is empty), return none
                dimsize | dimsize if row > dimsize-1 => None, //dimsize is indexed by 1, whereas rows and columns are 0-indexed
                dimsize | dimsize if col > dimsize-1 => None, //and we don't want users to break our dimensions
                dimsize => self.matrix.get((row * dimsize + col) as usize) //offset by row, then index by col
            }
        }

        fn get_index_of_clock(&self, clock: N) -> Option<usize> {
            self.clock_names.iter().position(|e| e == &clock)
        }

        fn consistent() {

        }

        fn relation() {

        }

        fn satisfied() {

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
    }

    #[test]
    fn dbm_index_test1() {
        let clocks = vec!["c1", "c2", "c3", "c4"];
        let dbm = DBM::<u32, &str>::new(clocks);
        let elem = *dbm.get_element(0, 0).unwrap();
        assert_eq!(elem, 0);
    }

    #[test]
    fn dbm_index_test2() {
        let clocks = vec!["c1", "c2", "c3", "c4"];
        let dbm = DBM::<u32, &str>::new(clocks);
        let elem = *dbm.get_element(2, 3).unwrap();
        assert_eq!(elem, 0);
    }

    #[test]
    fn dbm_index_test3() {
        let clocks = vec!["c1", "c2", "c3", "c4"];
        let dbm = DBM::<u32, &str>::new(clocks);
        let elem = *dbm.get_element(3, 1).unwrap();
        assert_eq!(elem, 0);
    }

    #[test]
    fn dbm_index_test_bad_index1() {
        let clocks = vec!["c1", "c2", "c3", "c4"];
        let dbm = DBM::<u32, &str>::new(clocks);
        let elem = dbm.get_element(4, 0);
        assert_eq!(elem, None);
    }

    #[test]
    fn dbm_index_test_bad_index2() {
        let clocks = vec!["c1", "c2", "c3", "c4"];
        let dbm = DBM::<u32, &str>::new(clocks);
        let elem = dbm.get_element(0, 4);
        assert_eq!(elem, None);
    }

    #[test]
    fn dbm_index_test_bad_index3() {
        let clocks = vec!["c1", "c2", "c3", "c4"];
        let dbm = DBM::<u32, &str>::new(clocks);
        let elem = dbm.get_element(4, 4);
        assert_eq!(elem, None);
    }

    #[test]
    fn dbm_index_test_empty_dbm() {
        let clocks = vec![];
        let dbm = DBM::<u32, &str>::new(clocks);
        let elem = dbm.get_element(0, 0);
        assert_eq!(elem, None);
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
