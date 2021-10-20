pub mod dbm {
    /*
     * This is an implementation of a DBM.
     * The matrix is declared as a single array, as this guarantees a single, coherent block of memory, rather than having all the rows spread around in the heap.
     * As such, it should be indexed with an offset.
     *
     */
    use crate::dbm::dbm::ConstraintOp::{LessThanEqual, LessThan};
    use crate::bitvector::Bitvector;
    use std::fmt;
    struct DBM<T, N> {
        matrix: Vec<T>,
        clock_names: Vec<N>,
        bitvec: Bitvector,
    }

    pub enum ConstraintOp {
        LessThanEqual,
        LessThan,
    }

    impl From<bool> for ConstraintOp {

        fn from(val: bool) -> Self {
            match val {
                true => LessThan,
                false => LessThanEqual
            }
        }
    }

    impl fmt::Display for ConstraintOp {

        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
            match self {
                LessThanEqual =>    write!(f, "≤"),
                LessThan =>         write!(f, "<"),
            }
        }
    }

    impl<
            T: std::default::Default + std::cmp::PartialOrd, //For all T's that implement Default trait
            N: std::default::Default + std::cmp::Ord, //all N's that implement Ord and Default traits
        > DBM<T, N>
    {
        fn new(mut clocks: Vec<N>) -> DBM<T, N> {
            //Intentionally doesn't take a reference, as we would like the names to be owned by the data structure
            clocks.insert(0, Default::default());
            let bitvector = Bitvector::init_with_length((clocks.len() * clocks.len()) as u32);
            let matrix_size = clocks.len() * clocks.len();
            let mut matrix: Vec<T> = Vec::new();
            matrix.resize_with(matrix_size, Default::default);
            Self {
                matrix: matrix,
                clock_names: clocks,
                bitvec: bitvector,
            }
        }

        fn get_dimsize(&self) -> usize {
            //only need one function, as the matrices are always quadratic
            self.clock_names.len()
        }

        fn get_element(&self, row: usize, col: usize) -> Option<&T> {
            match self.get_dimsize() {
                0 => None, //If dimsize is 0 (ie. DBM is empty), return none
                dimsize | dimsize if row > dimsize - 1 => None, //dimsize is indexed by 1, whereas rows and columns are 0-indexed
                dimsize | dimsize if col > dimsize - 1 => None, //and we don't want users to break our dimensions
                dimsize => self.matrix.get((row * dimsize + col) as usize), //offset by row, then index by col
            }
        }

        fn get_bitval(&self, row: usize, col: usize) -> bool {
            self.bitvec
                .get_bit((row * self.get_dimsize() + col) as u32)
                .unwrap()
        }

        fn set_bitval(&mut self, row: usize, col: usize, val: bool) {
            self.bitvec
                .set_bit((row * self.get_dimsize() + col) as u32, val)
                .unwrap()
        }

        fn get_index_of_clock(&self, clock: N) -> Option<usize> {
            self.clock_names.iter().position(|e| e == &clock)
        }

        fn consistent(&self) -> bool {
            let mut upper_bounds: Vec<(usize, usize)> = vec![]; // List of pairs to check.
            for r in 0..self.get_dimsize() {
                //cycle through all columns and rows
                for c in 0..self.get_dimsize() {
                    //maybe not the prettiest or fastest, but it will work.
                    if c > r {
                        //only act on cells that are above the diagonal, that is, where column index is strictly greater than row index
                        upper_bounds.push((r, c));
                    }
                }
            }
            //now we have all the indexes for upper bounds. We then have to compare them with the lower bounds.
            // Map over the values in upper bound and compare them with their respective lower bounds. Check values first, and operators second
            let bound_consistency: bool = upper_bounds
                .iter()
                .map(|(r, c)| //map every element
                     self.get_element(*r, *c).unwrap() > self.get_element(*c, *r).unwrap() || //and check if HB value is greater than LB value, short circuit if it is.
                     self.get_bitval(*r, *c) >= self.get_bitval(*c, *r)) //if not, then check if HB constraint is as great as LB constrait and return its value.
                .fold(true, |acc, e| acc && e); //finally check if all values are true. Fold will return false if not
            bound_consistency //return bound_consistency as resulting value
        }

        fn relation() {}

        fn satisfied() {}

        fn up() {}

        fn down() {}

        fn free(_clock_to_free: N) {}

        fn reset(_clock_to_reset: N, _reset_val: T) {}

        fn copy(_clock_target: N, _clock_src: N) {}

        fn and(_clock: N, _constraint: (T, ConstraintOp)) { //Constraint is a tuple over T and ConstraintOP, as this mimicks the notation in Timed Automata: Semantics, Algorithms and Tools by Bengtsson and Yi
        }

        fn normalize() {}

        fn shift(_clock: N, _d_val: T) {}
    }

    impl<
            T: std::default::Default + std::cmp::PartialOrd + std::fmt::Display,
            N: std::default::Default + std::cmp::Ord,
        > fmt::Display for DBM<T, N>
    {

        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
            let mut bitstring = self.bitvec.get_bits_as_vec();
            let mut bitstring_chunks = bitstring.chunks(self.get_dimsize());
            for r in self.matrix.chunks(self.get_dimsize()) {
                let mut bit_r = bitstring_chunks.next().unwrap().iter();
                write!(f, "|");
                let mut r = r.into_iter().peekable();
                while let Some(e) = r.next() {
                    let e_bitval = bit_r.next().unwrap();
                    write!(f, "({}, {})", e, ConstraintOp::from(*e_bitval));
                    match r.peek() {
                        Some(a) => write!(f, ", "),
                        None => write!(f, "|\n"),
                    }.unwrap();
                };
            }
        Ok(())
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
    fn dbm_bit_consistency_test1() {
        let clocks = vec!["c1", "c2", "c3", "c4"];
        let dbm = DBM::<u32, &str>::new(clocks);
        assert_eq!(dbm.get_dimsize() * dbm.get_dimsize(), dbm.bitvec.get_length() as usize);
    }

    #[test]
    fn dbm_index_test_bad_index1() {
        let clocks = vec!["c1", "c2", "c3", "c4"];
        let dbm = DBM::<u32, &str>::new(clocks);
        let elem = dbm.get_element(5, 0);
        assert_eq!(elem, None);
    }

    #[test]
    fn dbm_index_test_bad_index2() {
        let clocks = vec!["c1", "c2", "c3", "c4"];
        let dbm = DBM::<u32, &str>::new(clocks);
        let elem = dbm.get_element(0, 5);
        assert_eq!(elem, None);
    }

    #[test]
    fn dbm_index_test_bad_index3() {
        let clocks = vec!["c1", "c2", "c3", "c4"];
        let dbm = DBM::<u32, &str>::new(clocks);
        let elem = dbm.get_element(5, 5);
        assert_eq!(elem, None);
    }

    #[test]
    fn dbm_index_test_empty_dbm() {
        let clocks = vec![];
        let dbm = DBM::<u32, &str>::new(clocks);
        let elem = *dbm.get_element(0, 0).unwrap();
        assert_eq!(elem, 0);
    }

    #[test]
    fn dbm_print_test() {
        let clocks = vec!["c1", "c2", "c3", "c4"];
        //lines declared seperately, as it makes it much nicer to look at
        let line1 = "|(0, ≤), (0, ≤), (0, ≤), (0, ≤), (0, ≤)|\n"; //these strings are borrowed (&str)
        let line2 = "|(0, ≤), (0, ≤), (0, ≤), (0, ≤), (0, ≤)|\n";
        let line3 = "|(0, ≤), (0, ≤), (0, ≤), (0, ≤), (0, ≤)|\n";
        let line4 = "|(0, ≤), (0, ≤), (0, ≤), (0, ≤), (0, ≤)|\n";
        let line5 = "|(0, ≤), (0, ≤), (0, ≤), (0, ≤), (0, ≤)|\n";
        let printed_vec = String::new() + line1 + line2 + line3 + line4 + line5; //so to concatenate, we need an owned string (String) to concat into.
    let dbm = DBM::<u32, &str>::new(clocks);
    assert_eq!(format!("{}", dbm), printed_vec); //check if format runs
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
