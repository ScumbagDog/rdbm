pub mod dbm {
    /*
     * This is an implementation of a DBM.
     * The matrix is declared as a single array, as this guarantees a single, coherent block of memory, rather than having all the rows spread around in the heap.
     * As such, it should be indexed with an offset.
     *
     */
    
    use std::ops::Add;
use crate::dbm::dbm::ConstraintOp::{LessThanEqual, LessThan};
    use crate::bitvector::Bitvector;
    use std::fmt;
    struct DBM<T, N> {
        matrix: Vec<T>,
        clock_names: Vec<N>,
        bitvec: Bitvector,
    }

    #[derive(PartialEq, PartialOrd, Debug, Default)]
    struct Bound<T> {
        boundval: T,
        constraint_op: ConstraintOp,
    }

    #[derive(PartialEq, PartialOrd, Debug)]
    pub enum ConstraintOp {
        LessThanEqual,
        LessThan,
    }

     impl<T: std::fmt::Display> fmt::Display for Bound<T> {

        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
            write!(f, "({}, {})", self.boundval, self.constraint_op)
        }
     }

    impl From<bool> for ConstraintOp {

        fn from(val: bool) -> Self {
            match val {
                true => LessThan,
                false => LessThanEqual
            }
        }
    }

    impl From<&bool> for ConstraintOp {

        fn from(val: &bool) -> Self {
            match *val {
                true => LessThan,
                false => LessThanEqual
            }
        }
    }

    impl From<ConstraintOp> for bool {

        fn from(val: ConstraintOp) -> Self {
            match val {
                LessThan => true,
                LessThanEqual => false
            }
        }
    }

    impl Default for ConstraintOp {

        fn default() -> Self {
            LessThanEqual
        }
    }

    impl Add for ConstraintOp {

        fn add(self, rhs: Self) -> <Self as std::ops::Add<Self>>::Output {
            (self.into() && rhs.into()).into() //cast to bool to perform and operator, then cast back once we are done
        }
        type Output = Self;
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
            T: std::default::Default + std::cmp::PartialOrd + Clone + std::ops::Add<Output = T>, //For all T's that implement Default trait
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

        fn set_element(&mut self, row: usize, col: usize, elem: T) -> Result<(), ()>{
            match self.get_dimsize() {
                dimsize | dimsize if row > dimsize - 1 => Err(()),
                dimsize | dimsize if col > dimsize - 1 => Err(()),
                dimsize => {
                    let old_elem = self.matrix.get_mut(row * dimsize + col).unwrap();
                    *old_elem = elem;
                    Ok(())
                }
            }
        }

        fn get_bitval(&self, row: usize, col: usize) -> Option<bool> {
            self.bitvec
                .get_bit((row * self.get_dimsize() + col) as u32)
        }

        fn set_bitval(&mut self, row: usize, col: usize, val: bool) -> Result<(), ()> {
            self.bitvec.set_bit((row * self.get_dimsize() + col) as u32, val)
        }

        fn get_bound(&self, row: usize, col: usize) -> Option<Bound<T>> {
            let boundval_option = self.get_element(row, col);
            if let Some(val) = boundval_option {
                let constraint_op_option = self.get_bitval(row, col);
                if let Some(boolval) = constraint_op_option {
                    return Some(Bound {
                        boundval: val.clone(),
                        constraint_op: ConstraintOp::from(boolval),
                    });
                }
            }
            return None;

        }

        fn set_bound(&mut self, row: usize, col: usize, bound: Bound<T>) -> Result<(), ()> {
            let set_element_status = self.set_element(row, col, bound.boundval);
            let set_bit_status = self.set_bitval(row, col, bound.constraint_op.into());
            match set_element_status {
                Ok(_) => {
                    match set_bit_status {
                        Ok(_) => Ok(()),
                        Err(_) => Err(()),
                    }
                }
                Err(_) => Err(()),
            }
        }

        fn get_bound_iter(&self) -> impl Iterator<Item = Bound<&T>> + '_ { //This function looks deceptively simple. It is an absolute mess behind the scenes.
            let matrix_iter = self.matrix.iter();                          //If you ever decide to refactor it, prepare yourself for generic
            let bitvec_iter = self.bitvec.get_iterator();
            matrix_iter.zip(bitvec_iter)
                       .map(|(val, bitval)| Bound{boundval: val, constraint_op: ConstraintOp::from(bitval),})
        }

        fn get_index_of_clock(&self, clock: N) -> Option<usize> {
            self.clock_names.iter().position(|e| e == &clock)
        }

        fn consistent(&self) -> bool { //should be run after the tighten function, as index (i, i) will be annotated there
                                       //alternatively after another transformation, as it preserves the consistent status
            (0..self.get_dimsize())
                .map(|c| self.get_bound(c, c).unwrap() == Default::default())
                .fold(true, |acc, e| acc && e)
        }

        fn relation(first: &DBM<T, N>, second: &DBM<T, N>) -> bool { //I considered doing this as a method, but I ultimately decided to do it as a normal function, as it looks somewhat nicer
            first.get_dimsize() == second.get_dimsize() && { //dimension check for short circuit and iterator size later.
                let first_iter = first.get_bound_iter();
                let second_iter = second.get_bound_iter();
                first_iter.le(second_iter) //check if every element in first_iter is less than or equal to its corresponding element in second_iter.
                //Will return true even if first_iter has fewer elements than second_iter (so long as they are less than or equal to second_iter), so dimension check is still needed
            }
        }

        fn satisfied(row_clock: N, col_clock: N, op: ConstraintOp, constant: T) {

        }

        fn tighten(&mut self){ //Essentially the Floyd-Warshall algorithm
            let dimsize = self.get_dimsize();
            for k in 0..dimsize {
                for i in 0..dimsize {
                    for j in 0..dimsize {
                        let ij_bound = self.get_bound(i, j).unwrap();
                        let ik_bound = self.get_bound(i, k).unwrap();
                        let kj_bound = self.get_bound(k, j).unwrap();
                        let ikj_bound = Bound
                        {boundval: ik_bound.boundval + kj_bound.boundval,
                         constraint_op: ik_bound.constraint_op + kj_bound.constraint_op,
                        };
                        if ij_bound > ikj_bound {
                            self.set_bound(i, j, ikj_bound).unwrap();
                        }
                    }
                }
            }
        }

        fn up() {}

        fn down() {}

        fn free(_clock_to_free: N) {}

        fn reset(_clock_to_reset: N, _reset_val: T) {}

        fn copy(_clock_target: N, _clock_src: N) {}

        fn and(row_clock: N, col_clock: N, op: ConstraintOp, constant: T) { //Constraint is a tuple over T and ConstraintOP, as this mimicks the notation in Timed Automata: Semantics, Algorithms and Tools by Bengtsson and Yi
        }

        fn shift(_clock: N, _d_val: T) {}
    }

    impl<
            T: std::default::Default + std::cmp::PartialOrd + std::fmt::Display + Clone + std::ops::Add<Output = T>,
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
                    let bound = Bound {boundval: e, constraint_op: ConstraintOp::from(*e_bitval)};
                    write!(f, "{}", bound);
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
    fn bound_lte_test1() {
        let smaller_bound = Bound { boundval: 0, constraint_op: ConstraintOp::LessThanEqual };
        let greater_bound = Bound { boundval: 0, constraint_op: ConstraintOp::LessThanEqual };
        assert_eq!(smaller_bound <= greater_bound, true);
    }

    #[test]
    fn bound_lte_test2() {
        let smaller_bound = Bound { boundval: 0, constraint_op: ConstraintOp::LessThanEqual };
        let greater_bound = Bound { boundval: 10, constraint_op: ConstraintOp::LessThanEqual };
        assert_eq!(smaller_bound <= greater_bound, true);
        assert_eq!(greater_bound <= smaller_bound, false);
    }

    #[test]
    fn bound_lte_test3() {
        let smaller_bound = Bound { boundval: 0, constraint_op: ConstraintOp::LessThanEqual };
        let greater_bound = Bound { boundval: 0, constraint_op: ConstraintOp::LessThan };
        assert_eq!(smaller_bound <= greater_bound, true);
        assert_eq!(greater_bound <= smaller_bound, false);
    }

    #[test]
    fn bound_le_test1() {
        let smaller_bound = Bound { boundval: 0, constraint_op: ConstraintOp::LessThanEqual };
        let greater_bound = Bound { boundval: 0, constraint_op: ConstraintOp::LessThan };
        assert_eq!(smaller_bound < greater_bound, true);
        assert_eq!(greater_bound < smaller_bound, false);
    }

    #[test]
    fn bound_le_test2() {
        let smaller_bound = Bound { boundval: 0, constraint_op: ConstraintOp::LessThanEqual };
        let greater_bound = Bound { boundval: 0, constraint_op: ConstraintOp::LessThanEqual };
        assert_eq!(smaller_bound < greater_bound, false);
        assert_eq!(greater_bound < smaller_bound, false);
    }

    #[test]
    fn bound_le_test3() {
        let smaller_bound = Bound { boundval: 0, constraint_op: ConstraintOp::LessThanEqual };
        let greater_bound = Bound { boundval: 10, constraint_op: ConstraintOp::LessThanEqual };
        assert_eq!(smaller_bound < greater_bound, true);
        assert_eq!(greater_bound < smaller_bound, false);
    }


    #[test]
    fn dbm_index_test1() {
        let clocks = vec!["c1", "c2", "c3", "c4"];
        let dbm = DBM::<i32, &str>::new(clocks);
        let elem = *dbm.get_element(0, 0).unwrap();
        assert_eq!(elem, 0);
    }

    #[test]
    fn dbm_index_test2() {
        let clocks = vec!["c1", "c2", "c3", "c4"];
        let dbm = DBM::<i32, &str>::new(clocks);
        let elem = *dbm.get_element(2, 3).unwrap();
        assert_eq!(elem, 0);
    }

    #[test]
    fn dbm_index_test3() {
        let clocks = vec!["c1", "c2", "c3", "c4"];
        let dbm = DBM::<i32, &str>::new(clocks);
        let elem = *dbm.get_element(3, 1).unwrap();
        assert_eq!(elem, 0);
    }

    #[test]
    fn dbm_bit_consistency_test1() {
        let clocks = vec!["c1", "c2", "c3", "c4"];
        let dbm = DBM::<i32, &str>::new(clocks);
        assert_eq!(dbm.get_dimsize() * dbm.get_dimsize(), dbm.bitvec.get_length() as usize);
    }

    #[test]
    fn dbm_index_test_bad_index1() {
        let clocks = vec!["c1", "c2", "c3", "c4"];
        let dbm = DBM::<i32, &str>::new(clocks);
        let elem = dbm.get_element(5, 0);
        assert_eq!(elem, None);
    }

    #[test]
    fn dbm_index_test_bad_index2() {
        let clocks = vec!["c1", "c2", "c3", "c4"];
        let dbm = DBM::<i32, &str>::new(clocks);
        let elem = dbm.get_element(0, 5);
        assert_eq!(elem, None);
    }

    #[test]
    fn dbm_index_test_bad_index3() {
        let clocks = vec!["c1", "c2", "c3", "c4"];
        let dbm = DBM::<i32, &str>::new(clocks);
        let elem = dbm.get_element(5, 5);
        assert_eq!(elem, None);
    }

    #[test]
    fn dbm_index_test_empty_dbm() {
        let clocks = vec![];
        let dbm = DBM::<i32, &str>::new(clocks);
        let elem = *dbm.get_element(0, 0).unwrap();
        assert_eq!(elem, 0);
    }

    #[test]
    fn dbm_bound_test1() {
        let clocks = vec!["c1", "c2", "c3", "c4"];
        let mut dbm = DBM::<i32, &str>::new(clocks);
        dbm.set_bitval(0, 1, true);
        let mut elem = dbm.matrix.get_mut(1).unwrap();
        *elem = 4;
        let bound = dbm.get_bound(0, 1);
        assert_eq!(bound.unwrap(), Bound{boundval: 4, constraint_op: ConstraintOp::LessThan});
    }


    #[test]
    fn dbm_consistency_test1() {
        let clocks = vec!["c1", "c2", "c3", "c4"];
        let mut dbm = DBM::<i32, &str>::new(clocks);
        dbm.tighten();
        assert_eq!(dbm.consistent(), true); //a dbm filled with (0, lte) should be consistent
    }

    #[test]
    fn dbm_consistency_test2() {
        let clocks = vec!["c1", "c2", "c3", "c4"];
        let mut dbm = DBM::<i32, &str>::new(clocks);
        let mut val = dbm.matrix.get_mut(1).unwrap(); //get mutable reference to the value in (0, 1)
        *val = 1; //set (0, 1) to be 1
        dbm.tighten();
        assert_eq!(dbm.consistent(), true); // as the upper bound in (0, 1) isn't smaller than the lower bound in (1, 0), dbm should be consistent
    }

    #[test]
    fn dbm_consistency_test3() {
        let clocks = vec!["c1", "c2", "c3", "c4"];
        let mut dbm = DBM::<i32, &str>::new(clocks);
        let dimsize = dbm.get_dimsize();
        let mut val = dbm.matrix.get_mut(dimsize).unwrap(); //get mutable reference to the value in (1, 0). (we use dimsize to skip the first row)
        *val = 1; //set (1, 0) to be 1, meaning that 0-x <= 1 ~ -x <= 1
        dbm.tighten();
        assert_eq!(dbm.consistent(), true); // While this means that the zone is behind the x-axis which violates our implicit constraint that clocks are non-negative, it doesn't make it inconsistent, as we can tighten the bound.
    }

    #[test]
    fn dbm_consistency_test4() {
        let clocks = vec!["c1", "c2", "c3", "c4"];
        let mut dbm = DBM::<i32, &str>::new(clocks);
        let dimsize = dbm.get_dimsize();
        let mut val = dbm.matrix.get_mut(dimsize).unwrap(); //get mutable reference to the value in (1, 0). (we use dimsize to skip the first row)
        *val = -1; //set (1, 0) to be -1, meaning that 0-x <= -1 ~ -x <= -1
        dbm.tighten();
        assert_eq!(dbm.consistent(), false); //As the corresponding bound in (0, 1) is (0 <=), this would mean that the intersection between the bounds is Ø (empty), as no x is both smaller than 0 and greater than 1/ !!Inconsistent
    }

    #[test]
    fn dbm_consistency_test5() {
        let clocks = vec!["c1", "c2", "c3", "c4"];
        let mut dbm = DBM::<i32, &str>::new(clocks);
        let mut val = dbm.matrix.get_mut(1).unwrap(); //get mutable reference to the value in (0, 1)
        *val = -1;
        dbm.tighten();
        assert_eq!(dbm.consistent(), false); // Same deal as in consistency test 4, the bounds are flipped. No x that is smaller than -1 and greater than 0 => inconsistent DBM
    }

    #[test]
    fn dbm_relation_test1() {
        let clocks = vec!["c1", "c2", "c3", "c4"];
        let mut dbm_le = DBM::<i32, &str>::new(clocks.to_owned());
        let mut dbm_gt = DBM::<i32, &str>::new(clocks);
        let mut val = dbm_gt.matrix.get_mut(1).unwrap(); //get mutable reference to the value in (0, 1)
        *val = 1; //set (0, 1) to be 1. dbm_gt should now be greater than dbm_le
        assert_eq!(DBM::relation(&dbm_le, &dbm_gt), true);
        assert_eq!(DBM::relation(&dbm_gt, &dbm_le), false);
    }

    #[test]
    fn dbm_relation_test2() {
        let clocks = vec!["c1", "c2", "c3", "c4"];
        let mut dbm_le = DBM::<i32, &str>::new(clocks.to_owned());
        let mut dbm_gte = DBM::<i32, &str>::new(clocks);
        assert_eq!(DBM::relation(&dbm_le, &dbm_gte), true); //now they are equal
        assert_eq!(DBM::relation(&dbm_gte, &dbm_le), true); //both checks should run
    }

    #[test]
    fn dbm_relation_test3() {
        let clocks = vec!["c1", "c2", "c3", "c4"];
        let mut dbm_le = DBM::<i32, &str>::new(clocks.to_owned());
        let mut dbm_gte = DBM::<i32, &str>::new(clocks);
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
        let mut dbm_le = DBM::<i32, &str>::new(smol_clocks);
        let mut dbm_gte = DBM::<i32, &str>::new(clocks);
        let mut val_gte = dbm_gte.matrix.get_mut(1).unwrap();
        let mut val_lte = dbm_le.matrix.get_mut(1).unwrap();
        *val_gte = 1;
        *val_lte = 1;
        assert_eq!(DBM::relation(&dbm_le, &dbm_gte), false); //values are equal
        assert_eq!(DBM::relation(&dbm_gte, &dbm_le), false); //but dimensions are different, so return false
    }

     #[test]
    fn dbm_relation_test5() {
        let clocks = vec!["c1", "c2", "c3", "c4"];
        let mut dbm_le = DBM::<i32, &str>::new(clocks.to_owned());
        let mut dbm_gte = DBM::<i32, &str>::new(clocks);
        dbm_gte.set_bitval(0, 1, true);
        assert_eq!(DBM::relation(&dbm_le, &dbm_gte), true); //bound on gte(0, 1) is greater
        assert_eq!(DBM::relation(&dbm_gte, &dbm_le), false); //so le is related to gte, but gte isn't related to le
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
        let dbm = DBM::<i32, &str>::new(clocks);
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
