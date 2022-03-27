/*
 * This is an implementation of a DBM.
 * The matrix is declared as a single array, as this guarantees a single, coherent block of memory, rather than having all the rows spread around in the heap.
 * As such, it should be indexed with an offset.
 *
 */

use crate::bitvector::Bitvector;
use crate::rdbm::ConstraintOp::{LessThan, LessThanEqual};
use num::Bounded;
use num::Zero;
use std::fmt;
use std::ops::Add;

#[derive(Debug, PartialEq, Clone)]
pub struct DBM<T> {
    matrix: Vec<T>,
    clock_names: Vec<u8>,
    ops: Bitvector,
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Default, Clone)]
struct Bound<T> {
    boundval: T,
    constraint_op: ConstraintOp,
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Copy, Clone)]
pub enum ConstraintOp {
    LessThanEqual,
    LessThan,
}

impl From<bool> for ConstraintOp {
    fn from(val: bool) -> Self {
        match val {
            true => LessThan,
            false => LessThanEqual,
        }
    }
}

impl From<&bool> for ConstraintOp {
    fn from(val: &bool) -> Self {
        match *val {
            true => LessThan,
            false => LessThanEqual,
        }
    }
}

impl From<ConstraintOp> for bool {
    fn from(val: ConstraintOp) -> Self {
        match val {
            LessThan => true,
            LessThanEqual => false,
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
            LessThanEqual => write!(f, "≤"),
            LessThan => write!(f, "<"),
        }
    }
}

impl<T: std::fmt::Display> fmt::Display for Bound<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        write!(f, "({}, {})", self.boundval, self.constraint_op)
    }
}

impl<T: num::Saturating + std::ops::Add<Output = T>> Add for Bound<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> <Self as std::ops::Add<Self>>::Output {
        Bound {
            boundval: self.boundval.saturating_add(rhs.boundval),
            constraint_op: self.constraint_op + rhs.constraint_op,
        }
    }
}

impl<'a, 'b, T: num::Saturating + std::ops::Add<Output = T> + Clone> Add<&'b Bound<T>>
    for &'a Bound<T>
{
    type Output = Bound<T>;
    fn add(self, rhs: &'b Bound<T>) -> Bound<T> {
        Bound {
            boundval: self.boundval.clone().saturating_add(rhs.boundval.clone()),
            constraint_op: self.constraint_op + rhs.constraint_op,
        }
    }
}

impl<T: Bounded> Bounded for Bound<T> {
    fn max_value() -> Self {
        Bound {
            boundval: T::max_value(),
            constraint_op: LessThanEqual,
        }
    }
    fn min_value() -> Self {
        Bound {
            boundval: T::min_value(),
            constraint_op: LessThanEqual,
        }
    }
}

impl<T: Zero + PartialEq + num::Saturating> Zero for Bound<T> {
    fn zero() -> Self {
        Bound {
            boundval: num::zero(),
            constraint_op: LessThanEqual,
        }
    }

    fn is_zero(&self) -> bool {
        self.boundval == num::zero() && self.constraint_op == LessThanEqual
    }
}

impl<T: std::cmp::PartialEq + num::Bounded> Bound<T> {
    fn get_infinite_bound() -> Bound<T> {
        Bound {
            boundval: Bounded::max_value(),
            constraint_op: LessThanEqual,
        }
    }
}

impl<T> DBM<T> {
    //This impl is without traits, which allows us to print without satisfying all the other traits we don't need anyway (because the dimsize function now has a traitless-impl)

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

    fn set_element(&mut self, row: usize, col: usize, elem: T) -> Result<(), ()> {
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
        self.ops.get_bit((row * self.get_dimsize() + col) as u32)
    }

    fn set_bitval(&mut self, row: usize, col: usize, val: bool) -> Result<(), ()> {
        self.ops
            .set_bit((row * self.get_dimsize() + col) as u32, val)
    }

    fn get_clock_index(&self, clock: u8) -> Option<usize> {
        self.clock_names.iter().position(|c| c == &clock)
    }
}

impl<
        T: std::cmp::Ord
            + Clone
            + std::ops::Add<Output = T>
            + num::Bounded
            + num::Zero
            + num::Saturating
            + std::ops::Neg<Output = T>, //For all T's that implement the following traits
    > DBM<T>
{
    pub fn zero(mut clocks: Vec<u8>) -> DBM<T> {
        //Intentionally doesn't take a reference, as we would like the names to be owned by the data structure
        clocks.insert(0, Zero::zero());
        let bitvector = Bitvector::init_with_length((clocks.len() * clocks.len()) as u32);
        let matrix_size = clocks.len() * clocks.len();
        let mut matrix: Vec<T> = Vec::new();
        matrix.resize_with(matrix_size, Zero::zero);
        Self {
            matrix: matrix,
            clock_names: clocks,
            ops: bitvector,
        }
    }

    pub fn new(mut clocks: Vec<u8>) -> DBM<T> {
        clocks.insert(0, Zero::zero());
        let dim = clocks.len();
        let matrix_size = dim * dim;
        let bitvector = Bitvector::init_with_length(matrix_size as u32);
        let mut matrix: Vec<T> = Vec::new();
        matrix.resize_with(matrix_size, Bounded::max_value);
        let mut dbm = Self {
            matrix: matrix,
            clock_names: clocks,
            ops: bitvector,
        };
        let zero_val = T::zero();
        for i in 0..dim {
            let _opt_val = dbm.set_element(0, i, zero_val.clone());
            let _opt_val_2 = dbm.set_element(i, i, zero_val.clone());
        }
        dbm
    }

    fn get_bound(&self, row: usize, col: usize) -> Option<Bound<T>> {
        let boundval_option = self.get_element(row, col);
        if let Some(val) = boundval_option {
            //if we actually get a boundval, cast it to val
            let constraint_op_option = self.get_bitval(row, col);
            if let Some(boolval) = constraint_op_option {
                // If we get a boolval, cast it to boolval
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
            Ok(_) => match set_bit_status {
                Ok(_) => Ok(()),
                Err(_) => Err(()),
            },
            Err(_) => Err(()),
        }
    }

    fn get_bound_iter(&self) -> impl Iterator<Item = Bound<&T>> + '_ {
        //This function looks deceptively simple. It is an absolute mess behind the scenes.
        let matrix_iter = self.matrix.iter(); //If you ever decide to refactor it, prepare yourself for generic
        let bitvec_iter = self.ops.get_iterator();
        matrix_iter.zip(bitvec_iter).map(|(val, bitval)| Bound {
            boundval: val,
            constraint_op: ConstraintOp::from(bitval),
        })
    }

    pub fn consistent(dbm: &DBM<T>) -> bool {
        //should be run after the close function, as index (i, i) will be annotated there (or not, depending on whether there are negative cycles or not)
        //alternatively after another transformation, as it preserves the consistent status
        (0..dbm.get_dimsize())
            .map(|c| dbm.get_bound(c, c).unwrap() == Zero::zero()) //could be optimized to short circuit early, but this is easier to implement and looks nicer, so ¯\_(ツ)_/¯
            .fold(true, |acc, e| acc && e)
    }

    pub fn is_included_in(first: &DBM<T>, second: &DBM<T>) -> bool {
        //I considered doing this as a method, but I ultimately decided to do it as a normal function, as it looks somewhat nicer
        first.get_dimsize() == second.get_dimsize() && {
            //dimension check for short circuit and iterator size later.
            let first_iter = first.get_bound_iter();
            let second_iter = second.get_bound_iter();
            first_iter
                .zip(second_iter)
                .map(|(x, y)| -> bool { x <= y })
                .fold(true, |acc, x| -> bool { acc && x })
        }
    }

    pub fn satisfied(
        dbm: &DBM<T>,
        row_name: u8,
        col_name: u8,
        op: ConstraintOp,
        val: T,
    ) -> Result<bool, ()> {
        //Should only be called on DBMs that are in their canonical form
        // Also note that this function just checks if the DBM is still consistent after conjunction of the bound.
        let row_opt = dbm.get_clock_index(row_name);
        let col_opt = dbm.get_clock_index(col_name);
        if row_opt == None || col_opt == None {
            Err(())
        } else {
            let row = row_opt.unwrap();
            let col = col_opt.unwrap();
            let local_bound = dbm.get_bound(col, row).unwrap(); //into usize type, as get_bound doesn't take u8s
            let new_bound = Bound {
                boundval: val,
                constraint_op: op,
            };
            let zero_bound: Bound<T> = num::Zero::zero();
            Ok((local_bound + new_bound) > zero_bound)
        }
    }

    pub fn close(dbm: &mut DBM<T>) {
        //Essentially the Floyd-Warshall algorithm
        let dimsize = dbm.get_dimsize();
        for k in 0..dimsize {
            for i in 0..dimsize {
                for j in 0..dimsize {
                    let ij_bound = dbm.get_bound(i, j).unwrap();
                    let ik_bound = dbm.get_bound(i, k).unwrap();
                    let kj_bound = dbm.get_bound(k, j).unwrap();
                    let ikj_bound = ik_bound + kj_bound;
                    if ij_bound > ikj_bound {
                        dbm.set_bound(i, j, ikj_bound).unwrap();
                    }
                }
            }
        }
    }

    pub fn up(dbm: &mut DBM<T>) -> Result<(), ()> {
        let dimsize = dbm.get_dimsize();
        let infinite_bound: Bound<T> = Bound::get_infinite_bound(); //we just pretend that the max value is the infinite :). As such, it is the responsibility of whoever uses a DBM to ensure, that we never exceed the max value that T can capture.
        for i in 1..dimsize {
            if let Err(()) = dbm.set_bound(i, 0, infinite_bound.clone()) {
                //If we get an error while setting the bound, return the error. Realistically though, this should not happen.
                return Err(());
            }
        }
        Ok(())
    }

    pub fn down(dbm: &mut DBM<T>) -> Result<(), ()> {
        let dimsize = dbm.get_dimsize();
        let zero_bound: Bound<T> = num::zero();
        for i in 1..dimsize {
            if let Err(()) = dbm.set_bound(0, i, zero_bound.clone()) {
                return Err(());
            }
            for j in 1..dimsize {
                let ij_bound_option = dbm.get_bound(i, j);
                if let Some(ij_bound) = ij_bound_option {
                    if ij_bound < zero_bound {
                        dbm.set_bound(0, i, ij_bound).unwrap(); //This can be optimized by only making one check on ij, and then setting (0,i) to either the value of D_ij or zero, instead of setting (0,i) and then potentially resetting it.
                    } //But this follows the pseudocode more closely, so implemented like this for now.
                }
            }
        }
        Ok(())
    }

    pub fn free(dbm: &mut DBM<T>, clock_to_free: u8) -> Result<(), ()> {
        let clock_opt = dbm.get_clock_index(clock_to_free);
        if clock_opt == None {
            Err(())
        } else {
            let clock_index = clock_opt.unwrap();
            let inf_bound: Bound<T> = Bound::get_infinite_bound();
            for i in 1..dbm.get_dimsize() {
                if i != clock_index {
                    let i0_bound = dbm.get_bound(i, 0).unwrap();
                    dbm.set_bound(clock_index, i, inf_bound.clone())?;
                    dbm.set_bound(i, clock_index, i0_bound)?;
                }
            }
            Ok(())
        }
    }

    pub fn reset(dbm: &mut DBM<T>, clock_to_reset: u8, reset_val: T) -> Result<(), ()> {
        let clock_opt = dbm.get_clock_index(clock_to_reset);
        if clock_opt == None {
            Err(())
        } else {
            let clock_index = clock_opt.unwrap();
            for i in 0..dbm.get_dimsize() {
                let zero_i_bound = dbm.get_bound(0, i).unwrap();
                let i_zero_bound = dbm.get_bound(i, 0).unwrap();
                let positive_bound = Bound {
                    boundval: reset_val.clone(),
                    constraint_op: LessThanEqual,
                };
                let negative_bound = Bound {
                    boundval: -reset_val.clone(),
                    constraint_op: LessThanEqual,
                };
                dbm.set_bound(clock_index, i, positive_bound + zero_i_bound)?;
                dbm.set_bound(i, clock_index, i_zero_bound + negative_bound)?;
            }
            Ok(())
        }
    }

    pub fn copy(dbm: &mut DBM<T>, clock_target: u8, clock_src: u8) -> Result<(), ()> {
        let target_opt = dbm.get_clock_index(clock_target);
        let src_opt = dbm.get_clock_index(clock_src);
        if target_opt == None || src_opt == None {
            Err(())
        } else {
            let target_index = target_opt.unwrap();
            let src_index = src_opt.unwrap();
            for i in 0..dbm.get_dimsize() {
                if i != target_index {
                    dbm.set_bound(target_index, i, dbm.get_bound(src_index, i).unwrap())?;
                    dbm.set_bound(i, target_index, dbm.get_bound(i, src_index).unwrap())?;
                }
            }
            let zero_bound: Bound<T> = Bound::zero();
            dbm.set_bound(src_index, target_index, zero_bound.clone())?;
            dbm.set_bound(target_index, src_index, zero_bound)?;
            Ok(())
        }
    }

    pub fn and(
        dbm: &mut DBM<T>,
        row_clock: u8,
        col_clock: u8,
        op: ConstraintOp,
        constant: T,
    ) -> Result<(), ()> {
        //Constraint is a tuple over T and ConstraintOP, as this mimicks the notation in Timed Automata: Semantics, Algorithms and Tools by Bengtsson and Yi
        let row_opt = dbm.get_clock_index(row_clock);
        let col_opt = dbm.get_clock_index(col_clock);
        if row_opt == None || col_opt == None {
            Err(())
        } else {
            let row_index = row_opt.unwrap();
            let col_index = col_opt.unwrap();
            let local_bound = dbm.get_bound(row_index, col_index).unwrap(); //the bound in (y,x)
            let and_bound = Bound {
                boundval: constant,
                constraint_op: op,
            };
            if (&local_bound + &and_bound) < num::zero() {
                //num zero means a zero-bound
                dbm.set_bound(
                    0,
                    0,
                    Bound {
                        boundval: num::Bounded::min_value(),
                        constraint_op: LessThanEqual,
                    },
                )?; //We use min_value in place of a negative number, and then it is the user's responsibility to call it with a signed type (or a type where min<zero)
            } else if and_bound < local_bound {
                dbm.set_bound(row_index, col_index, and_bound)?;
                for i in 0..dbm.get_dimsize() {
                    for j in 0..dbm.get_dimsize() {
                        let ix_bound = dbm.get_bound(i, row_index).unwrap();
                        let xj_bound = dbm.get_bound(row_index, j).unwrap();
                        let ij_bound = dbm.get_bound(i, j).unwrap();
                        if &ix_bound + &xj_bound < ij_bound.clone() {
                            dbm.set_bound(i, j, ix_bound + xj_bound).unwrap();
                        }
                        let iy_bound = dbm.get_bound(i, col_index).unwrap();
                        let yj_bound = dbm.get_bound(col_index, j).unwrap();
                        if &iy_bound + &yj_bound < ij_bound {
                            dbm.set_bound(i, j, iy_bound + yj_bound).unwrap();
                        }
                    }
                }
            }

            Ok(())
        }
    }

    pub fn shift(dbm: &mut DBM<T>, clock: u8, delta_val: T) -> Result<(), ()> {
        let clock_opt = dbm.get_clock_index(clock);
        if clock_opt == None {
            Err(())
        } else {
            let clock_index = clock_opt.unwrap();
            let local_bound = Bound {
                boundval: delta_val,
                constraint_op: LessThanEqual,
            };
            for i in 0..dbm.get_dimsize() {
                if i != clock_index {
                    dbm.set_bound(
                        clock_index,
                        i,
                        dbm.get_bound(clock_index, i).unwrap() + local_bound.clone(),
                    )
                    .unwrap();
                    dbm.set_bound(
                        i,
                        clock_index,
                        dbm.get_bound(i, clock_index).unwrap() + local_bound.clone(),
                    )
                    .unwrap();
                }
            }
            dbm.set_bound(
                clock_index,
                0,
                std::cmp::min(num::zero(), dbm.get_bound(clock_index, 0).unwrap()),
            )
            .unwrap();
            dbm.set_bound(
                0,
                clock_index,
                std::cmp::min(num::zero(), dbm.get_bound(0, clock_index).unwrap()),
            )
            .unwrap();
            Ok(())
        }
    }

    pub fn normalise(dbm: &mut DBM<T>, upper: T, lower: T) -> Result<(), ()> {
        let upper_bound = Bound {
            boundval: upper,
            constraint_op: LessThanEqual,
        };
        let lower_bound = Bound {
            boundval: -lower,
            constraint_op: LessThanEqual,
        };
        for i in 0..dbm.get_dimsize() {
            for j in 0..dbm.get_dimsize() {
                let local_bound = dbm.get_bound(i, j).unwrap();
                if local_bound < Bound::max_value() && local_bound > upper_bound {
                    dbm.set_bound(i, j, Bound::max_value())?;
                } else if local_bound < Bound::max_value() && local_bound < lower_bound {
                    dbm.set_bound(i, j, lower_bound.clone())?;
                }
            }
        }
        Ok(())
    }
}

impl<T: fmt::Display> fmt::Display for DBM<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        let bitstring = self.ops.get_bits_as_vec();
        let mut bitstring_chunks = bitstring.chunks(self.get_dimsize());
        for r in self.matrix.chunks(self.get_dimsize()) {
            let mut bit_r = bitstring_chunks.next().unwrap().iter();
            write!(f, "|").unwrap();
            let mut r = r.into_iter().peekable();
            while let Some(e) = r.next() {
                let e_bitval = bit_r.next().unwrap();
                let bound = Bound {
                    boundval: e,
                    constraint_op: ConstraintOp::from(*e_bitval),
                };
                write!(f, "{}", bound).unwrap();
                match r.peek() {
                    Some(_a) => write!(f, ", "),
                    None => write!(f, "|\n"),
                }
                .unwrap();
            }
        }
        Ok(())
    }
}

#[test]
fn bound_lte_test1() {
    let smaller_bound = Bound {
        boundval: 0,
        constraint_op: ConstraintOp::LessThanEqual,
    };
    let greater_bound = Bound {
        boundval: 0,
        constraint_op: ConstraintOp::LessThanEqual,
    };
    assert_eq!(smaller_bound <= greater_bound, true);
}

#[test]
fn bound_lte_test2() {
    let smaller_bound = Bound {
        boundval: 0,
        constraint_op: ConstraintOp::LessThanEqual,
    };
    let greater_bound = Bound {
        boundval: 10,
        constraint_op: ConstraintOp::LessThanEqual,
    };
    assert_eq!(smaller_bound <= greater_bound, true);
    assert_eq!(greater_bound <= smaller_bound, false);
}

#[test]
fn bound_lte_test3() {
    let smaller_bound = Bound {
        boundval: 0,
        constraint_op: ConstraintOp::LessThanEqual,
    };
    let greater_bound = Bound {
        boundval: 0,
        constraint_op: ConstraintOp::LessThan,
    };
    assert_eq!(smaller_bound <= greater_bound, true);
    assert_eq!(greater_bound <= smaller_bound, false);
}

#[test]
fn bound_le_test1() {
    let smaller_bound = Bound {
        boundval: 0,
        constraint_op: ConstraintOp::LessThanEqual,
    };
    let greater_bound = Bound {
        boundval: 0,
        constraint_op: ConstraintOp::LessThan,
    };
    assert_eq!(smaller_bound < greater_bound, true);
    assert_eq!(greater_bound < smaller_bound, false);
}

#[test]
fn bound_le_test2() {
    let smaller_bound = Bound {
        boundval: 0,
        constraint_op: ConstraintOp::LessThanEqual,
    };
    let greater_bound = Bound {
        boundval: 0,
        constraint_op: ConstraintOp::LessThanEqual,
    };
    assert_eq!(smaller_bound < greater_bound, false);
    assert_eq!(greater_bound < smaller_bound, false);
}

#[test]
fn bound_le_test3() {
    let smaller_bound = Bound {
        boundval: 0,
        constraint_op: ConstraintOp::LessThanEqual,
    };
    let greater_bound = Bound {
        boundval: 10,
        constraint_op: ConstraintOp::LessThanEqual,
    };
    assert_eq!(smaller_bound < greater_bound, true);
    assert_eq!(greater_bound < smaller_bound, false);
}

#[test]
fn bound_add_overflowing() {
    let max_bound = Bound::<i8>::get_infinite_bound();
    let other_max_bound = max_bound.clone();
    assert_eq!(
        max_bound + other_max_bound,
        Bound::<i8>::get_infinite_bound()
    );
}

#[test]
fn bound_reference_add_overflowing() {
    let max_bound = Bound::<i8>::get_infinite_bound();
    let other_max_bound = max_bound.clone();
    assert_eq!(
        &max_bound + &other_max_bound,
        Bound::<i8>::get_infinite_bound()
    );
}

#[test]
fn dbm_index_test1() {
    let clocks = vec![1, 2, 3, 4];
    let dbm = DBM::<i32>::zero(clocks);
    let elem = *dbm.get_element(0, 0).unwrap();
    assert_eq!(elem, 0);
}

#[test]
fn dbm_index_test2() {
    let clocks = vec![1, 2, 3, 4];
    let dbm = DBM::<i32>::zero(clocks);
    let elem = *dbm.get_element(2, 3).unwrap();
    assert_eq!(elem, 0);
}

#[test]
fn dbm_index_test3() {
    let clocks = vec![1, 2, 3, 4];
    let dbm = DBM::<i32>::zero(clocks);
    let elem = *dbm.get_element(3, 1).unwrap();
    assert_eq!(elem, 0);
}

#[test]
fn dbm_bit_consistency_test1() {
    let clocks = vec![1, 2, 3, 4];
    let dbm = DBM::<i32>::zero(clocks);
    assert_eq!(
        dbm.get_dimsize() * dbm.get_dimsize(),
        dbm.ops.length as usize
    );
}

#[test]
fn dbm_index_test_bad_index1() {
    let clocks = vec![1, 2, 3, 4];
    let dbm = DBM::<i32>::zero(clocks);
    let elem = dbm.get_element(5, 0);
    assert_eq!(elem, None);
}

#[test]
fn dbm_index_test_bad_index2() {
    let clocks = vec![1, 2, 3, 4];
    let dbm = DBM::<i32>::zero(clocks);
    let elem = dbm.get_element(0, 5);
    assert_eq!(elem, None);
}

#[test]
fn dbm_index_test_bad_index3() {
    let clocks = vec![1, 2, 3, 4];
    let dbm = DBM::<i32>::zero(clocks);
    let elem = dbm.get_element(5, 5);
    assert_eq!(elem, None);
}

#[test]
fn dbm_index_test_empty_dbm() {
    let clocks = vec![];
    let dbm = DBM::<i32>::zero(clocks);
    let elem = *dbm.get_element(0, 0).unwrap();
    assert_eq!(elem, 0);
}

#[test]
fn dbm_bound_test1() {
    let clocks = vec![1, 2, 3, 4];
    let mut dbm = DBM::<i32>::zero(clocks);
    dbm.set_bitval(0, 1, true).unwrap();
    let elem = dbm.matrix.get_mut(1).unwrap();
    *elem = 4;
    let bound = dbm.get_bound(0, 1);
    assert_eq!(
        bound.unwrap(),
        Bound {
            boundval: 4,
            constraint_op: ConstraintOp::LessThan
        }
    );
}

#[test]
fn dbm_consistency_test1() {
    let clocks = vec![1, 2, 3, 4];
    let mut dbm = DBM::<i32>::zero(clocks);
    DBM::close(&mut dbm);
    assert_eq!(DBM::consistent(&dbm), true); //a dbm filled with (0, lte) should be consistent
}

#[test]
fn dbm_consistency_test2() {
    let clocks = vec![1, 2, 3, 4];
    let mut dbm = DBM::<i32>::zero(clocks);
    let val = dbm.matrix.get_mut(1).unwrap(); //get mutable reference to the value in (0, 1)
    *val = 1; //set (0, 1) to be 1
    DBM::close(&mut dbm);
    assert_eq!(DBM::consistent(&dbm), true); // as the upper bound in (0, 1) isn't smaller than the lower bound in (1, 0), dbm should be consistent
}

#[test]
fn dbm_consistency_test3() {
    let clocks = vec![1, 2, 3, 4];
    let mut dbm = DBM::<i32>::zero(clocks);
    let dimsize = dbm.get_dimsize();
    let val = dbm.matrix.get_mut(dimsize).unwrap(); //get mutable reference to the value in (1, 0). (we use dimsize to skip the first row)
    *val = 1; //set (1, 0) to be 1, meaning that 0-x <= 1 ~ -x <= 1
    DBM::close(&mut dbm);
    assert_eq!(DBM::consistent(&dbm), true); // While this means that the zone is behind the x-axis which violates our implicit constraint that clocks are non-negative, it doesn't make it inconsistent, as we can tighten the bound.
}

#[test]
fn dbm_consistency_test4() {
    let clocks = vec![1, 2, 3, 4];
    let mut dbm = DBM::<i32>::zero(clocks);
    let dimsize = dbm.get_dimsize();
    let val = dbm.matrix.get_mut(dimsize).unwrap(); //get mutable reference to the value in (1, 0). (we use dimsize to skip the first row)
    *val = -1; //set (1, 0) to be -1, meaning that 0-x <= -1 ~ -x <= -1
    DBM::close(&mut dbm);
    assert_eq!(DBM::consistent(&dbm), false); //As the corresponding bound in (0, 1) is (0 <=), this would mean that the intersection between the bounds is Ø (empty), as no x is both smaller than 0 and greater than 1/ !!Inconsistent
}

#[test]
fn dbm_consistency_test5() {
    let clocks = vec![1, 2, 3, 4];
    let mut dbm = DBM::<i32>::zero(clocks);
    let val = dbm.matrix.get_mut(1).unwrap(); //get mutable reference to the value in (0, 1)
    *val = -1;
    DBM::close(&mut dbm);
    assert_eq!(DBM::consistent(&dbm), false); // Same deal as in consistency test 4, the bounds are flipped. No x that is smaller than -1 and greater than 0 => inconsistent DBM
}

#[test]
fn dbm_inclusion_test1() {
    let clocks = vec![1, 2, 3, 4];
    let dbm_le = DBM::<i32>::zero(clocks.to_owned());
    let mut dbm_gt = DBM::<i32>::zero(clocks);
    let val = dbm_gt.matrix.get_mut(1).unwrap(); //get mutable reference to the value in (0, 1)
    *val = 1; //set (0, 1) to be 1. dbm_gt should now be greater than dbm_le
    assert_eq!(DBM::is_included_in(&dbm_le, &dbm_gt), true);
    assert_eq!(DBM::is_included_in(&dbm_gt, &dbm_le), false);
}

#[test]
fn dbm_inclusion_test2() {
    let clocks = vec![1, 2, 3, 4];
    let dbm_le = DBM::<i32>::zero(clocks.to_owned());
    let dbm_gte = DBM::<i32>::zero(clocks);
    assert_eq!(DBM::is_included_in(&dbm_le, &dbm_gte), true); //now they are equal
    assert_eq!(DBM::is_included_in(&dbm_gte, &dbm_le), true); //both checks should run
}

#[test]
fn dbm_inclusion_test3() {
    let clocks = vec![1, 2, 3, 4];
    let mut dbm_le = DBM::<i32>::zero(clocks.to_owned());
    let mut dbm_gte = DBM::<i32>::zero(clocks);
    let val_gte = dbm_gte.matrix.get_mut(1).unwrap();
    let val_lte = dbm_le.matrix.get_mut(1).unwrap();
    *val_gte = 1;
    *val_lte = 1;
    assert_eq!(DBM::is_included_in(&dbm_le, &dbm_gte), true); //they are still equal
    assert_eq!(DBM::is_included_in(&dbm_gte, &dbm_le), true); //So both should return true
}

#[test]
fn dbm_inclusion_test4() {
    let clocks = vec![1, 2, 3, 4];
    let smol_clocks = vec![1, 2, 3];
    let mut dbm_le = DBM::<i32>::zero(smol_clocks);
    let mut dbm_gte = DBM::<i32>::zero(clocks);
    let val_gte = dbm_gte.matrix.get_mut(1).unwrap();
    let val_lte = dbm_le.matrix.get_mut(1).unwrap();
    *val_gte = 1;
    *val_lte = 1;
    assert_eq!(DBM::is_included_in(&dbm_le, &dbm_gte), false); //values are equal
    assert_eq!(DBM::is_included_in(&dbm_gte, &dbm_le), false); //but dimensions are different, so return false
}

#[test]
fn dbm_inclusion_test5() {
    let clocks = vec![1, 2, 3, 4];
    let dbm_le = DBM::<i32>::zero(clocks.to_owned());
    let mut dbm_gte = DBM::<i32>::zero(clocks);
    dbm_gte.set_bitval(0, 1, true).unwrap();
    assert_eq!(DBM::is_included_in(&dbm_le, &dbm_gte), true); //bound on gte(0, 1) is greater
    assert_eq!(DBM::is_included_in(&dbm_gte, &dbm_le), false); //so le is related to gte, but gte isn't related to le
}

#[test]
fn test_reset_zero() {
    let clocks = vec![1, 2, 3, 4];
    let mut dbm: DBM<i8> = DBM::zero(clocks.to_owned());
    let dbm2: DBM<i8> = DBM::zero(clocks);
    DBM::reset(&mut dbm, 1, 10).unwrap(); //set clock 1 to a value of 10
    assert_eq!(DBM::is_included_in(&dbm2, &dbm), false); //as dbm has clock 1 set to 10, it will not include the zero dbm
    assert_eq!(DBM::is_included_in(&dbm, &dbm2), false); //likewise, the zero dbm does not include dbm
}

#[test]
fn test_restrict() {
    let dim: usize = 3;
    let mut dbm: DBM<i8> = DBM::new((1..dim as u8).collect());
    let dbm2 = dbm.clone();
    DBM::and(&mut dbm, 1, 0, LessThanEqual, 10).unwrap();
    assert_eq!(DBM::is_included_in(&dbm, &dbm2), true); //since dbm has been restricted, dbm2 should now include it, but not the other way around.
    assert_eq!(DBM::is_included_in(&dbm2, &dbm), false);
}

#[test]
fn test_restrict_different_order() {
    let dim: usize = 10;
    let mut dbm: DBM<i8> = DBM::new((1..dim as u8).collect());
    let mut dbm_reordered = dbm.clone();

    DBM::and(&mut dbm_reordered, 1, 2, LessThanEqual, 10).unwrap();
    DBM::and(&mut dbm_reordered, 1, 0, LessThanEqual, 15).unwrap();
    DBM::and(&mut dbm_reordered, 2, 3, LessThanEqual, 20).unwrap();

    DBM::and(&mut dbm, 2, 3, LessThanEqual, 20).unwrap();
    DBM::and(&mut dbm, 1, 2, LessThanEqual, 10).unwrap();
    DBM::and(&mut dbm, 1, 0, LessThanEqual, 15).unwrap();

    assert_eq!(DBM::is_included_in(&dbm, &dbm_reordered), true); //order of restricts shouldn't matter for equality, dbms should be equal
    assert_eq!(DBM::is_included_in(&dbm_reordered, &dbm), true);
}

#[test]
fn test_restrict_with_satisfies() {
    let dim: usize = 10;
    let mut dbm: DBM<i8> = DBM::new((1..dim as u8).collect());
    DBM::and(&mut dbm, 1, 0, LessThanEqual, 10).unwrap();
    //panic!(format!("\n{}", dbm));
    assert_eq!(DBM::satisfied(&dbm, 1, 0, LessThanEqual, 15).unwrap(), true);
    assert_eq!(DBM::satisfied(&dbm, 1, 0, LessThanEqual, 5).unwrap(), true);
}

#[test]
fn dbm_print_test() {
    let clocks = vec![1, 2, 3, 4];
    //lines declared seperately, as it makes it much nicer to look at
    let line1 = "|(0, ≤), (0, ≤), (0, ≤), (0, ≤), (0, ≤)|\n"; //these strings are borrowed (&str)
    let line2 = "|(0, ≤), (0, ≤), (0, ≤), (0, ≤), (0, ≤)|\n";
    let line3 = "|(0, ≤), (0, ≤), (0, ≤), (0, ≤), (0, ≤)|\n";
    let line4 = "|(0, ≤), (0, ≤), (0, ≤), (0, ≤), (0, ≤)|\n";
    let line5 = "|(0, ≤), (0, ≤), (0, ≤), (0, ≤), (0, ≤)|\n";
    let printed_vec = String::new() + line1 + line2 + line3 + line4 + line5; //so to concatenate, we need an owned string (String) to concat into.
    let dbm = DBM::<i32>::zero(clocks);
    assert_eq!(format!("{}", dbm), printed_vec);
}
