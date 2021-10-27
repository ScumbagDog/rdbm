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
                     {
                         let higher_bound = self.get_element(*r, *c).unwrap(); //get the higher and lower bounds for readability
                         let lower_bound = self.get_element(*c, *r).unwrap();
                         higher_bound > lower_bound || { //first check if HB is greater than LB. If yes, short circuit, if no, evaluate the next code block
                             higher_bound == lower_bound && { //check if HB is equal to LB. If no, short circuit, if yes, evaluate next code block
                                 let higher_constraint = self.get_bitval(*r, *c); //get the bitvals. We get them in here to avoid the call in the outer scope
                                 let lower_constraint = self.get_bitval(*c, *r);
                                 higher_constraint >= lower_constraint //check if HB constraint isn't smaller than LB constraint, and let this be the value of the entire block.
                             }
                         }
                     })
                     //self.get_element(*r, *c).unwrap() > self.get_element(*c, *r).unwrap() ||//and check if HB value is greater than LB value, short circuit if it is.
                     //self.get_bitval(*r, *c) >= self.get_bitval(*c, *r)) //if not, then check if HB constraint is as great as LB constrait and return its value.
                .fold(true, |acc, e| acc && e); //finally check if all values are true. Fold will return false if not
            bound_consistency //return bound_consistency as resulting value
        }

        fn relation(first: &DBM<T, N>, second: &DBM<T, N>) -> bool { //I considered doing this as a method, but I ultimately decided to do it as a normal function, as it looks somewhat nicer
            first.get_dimsize() == second.get_dimsize() && { //dimension check for short circuit and iterator size later.
                let first_iter = first.matrix.iter();
                let second_iter = second.matrix.iter();
                first_iter.le(second_iter) //check if every element in first_iter is less than or equal to its corresponding element in second_iter.
                //Will return true even if first_iter has fewer elements than second_iter (so long as they are less than or equal to second_iter), so dimension check is still needed
            }
        }

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
    fn dbm_consistency_test1() {
        let clocks = vec!["c1", "c2", "c3", "c4"];
        let dbm = DBM::<u32, &str>::new(clocks);
        assert_eq!(dbm.consistent(), true); //a dbm filled with (0, lte) should be consistent
    }

    #[test]
    fn dbm_consistency_test2() {
        let clocks = vec!["c1", "c2", "c3", "c4"];
        let mut dbm = DBM::<u32, &str>::new(clocks);
        let mut val = dbm.matrix.get_mut(1).unwrap(); //get mutable reference to the value in (0, 1)
        *val = 1; //set (0, 1) to be 1
        assert_eq!(dbm.consistent(), true); // as the upper bound in (0, 1) isn't smaller than the lower bound in (1, 0), dbm should be consistent
    }

    #[test]
    fn dbm_consistency_test3() {
        let clocks = vec!["c1", "c2", "c3", "c4"];
        let mut dbm = DBM::<u32, &str>::new(clocks);
        let dimsize = dbm.get_dimsize();
        let mut val = dbm.matrix.get_mut(dimsize).unwrap(); //get mutable reference to the value in (1, 0). (we use dimsize to skip the first row)
        *val = 1; //set (1, 0) to be 1
        assert_eq!(dbm.consistent(), false); // as the upper bound in (0, 1) IS smaller than the lower bound in (1, 0), dbm shouldn't be consistent
    }
    #[test]
    fn dbm_relation_test1() {
        let clocks = vec!["c1", "c2", "c3", "c4"];
        let mut dbm_le = DBM::<u32, &str>::new(clocks.to_owned());
        let mut dbm_gt = DBM::<u32, &str>::new(clocks);
        let mut val = dbm_gt.matrix.get_mut(1).unwrap(); //get mutable reference to the value in (0, 1)
        *val = 1; //set (0, 1) to be 1. dbm_gt should now be greater than dbm_le
        assert_eq!(DBM::relation(&dbm_le, &dbm_gt), true);
        assert_eq!(DBM::relation(&dbm_gt, &dbm_le), false);
    }

    #[test]
    fn dbm_relation_test2() {
        let clocks = vec!["c1", "c2", "c3", "c4"];
        let mut dbm_le = DBM::<u32, &str>::new(clocks.to_owned());
        let mut dbm_gte = DBM::<u32, &str>::new(clocks);
        assert_eq!(DBM::relation(&dbm_le, &dbm_gte), true); //now they are equal
        assert_eq!(DBM::relation(&dbm_gte, &dbm_le), true); //both checks should run
    }

    #[test]
    fn dbm_relation_test3() {
        let clocks = vec!["c1", "c2", "c3", "c4"];
        let mut dbm_le = DBM::<u32, &str>::new(clocks.to_owned());
        let mut dbm_gte = DBM::<u32, &str>::new(clocks);
        let mut val_gte = dbm_gte.matrix.get_mut(1).unwrap();
        let mut val_lte = dbm_le.matrix.get_mut(1).unwrap();
        *val_gte = 1;
        *val_lte = 1;
        assert_eq!(DBM::relation(&dbm_le, &dbm_gte), true); //they are still equal
        assert_eq!(DBM::relation(&dbm_gte, &dbm_le), true); //So both should return true
    }

    #[test]
    fn dbm_relation_test4() {
        let clocks = vec!["c1", "c2", "c3", "c4"];
        let smol_clocks = vec!["c1", "c2", "c3"];
        let mut dbm_le = DBM::<u32, &str>::new(smol_clocks);
        let mut dbm_gte = DBM::<u32, &str>::new(clocks);
        let mut val_gte = dbm_gte.matrix.get_mut(1).unwrap();
        let mut val_lte = dbm_le.matrix.get_mut(1).unwrap();
        *val_gte = 1;
        *val_lte = 1;
        assert_eq!(DBM::relation(&dbm_le, &dbm_gte), false); //values are equal
        assert_eq!(DBM::relation(&dbm_gte, &dbm_le), false); //but dimensions are different, so return false
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
        assert_eq!(format!("{}", dbm), printed_vec);
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
