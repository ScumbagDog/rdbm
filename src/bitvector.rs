pub struct Bitvector {
    pub bits: Vec<u8>,
    pub length: u32,
}

pub struct BitvectorIterator<'a> {
    bitvec: &'a Bitvector,
    current_index: u32,
}

impl Bitvector {
    pub fn init() -> Bitvector {
        let vector: Vec<u8> = Vec::new();
        Bitvector {
            bits: vector,
            length: 0,
        }
    }

    pub fn init_with_length(length: u32) -> Bitvector {
        //Normally, we would use usize here, but as length refers to amount of bits and not bytes, the benefit is diminished.
        let mut bvec = Bitvector::init();
        bvec.set_length(length);
        bvec
    }

    pub fn get_bit(&self, index: u32) -> Option<bool> {
        let element_offset = index / 8;
        if let Some(element) = self.bits.get(element_offset as usize) {
            let bit_offset = index % 8;
            let val_bitmask = (1 as u8) << (7 - bit_offset); //logic is the same as in set_bit
            Some((val_bitmask & element) != 0) //return val of val_bitmask AND element. There's probably a better way to cast u8 to bool, but this also works.
        } else {
            None
        }
    }

    pub fn set_bit(&mut self, index: u32, val: bool) -> Result<(), ()> {
        if index > self.length {
            return Err(());
        } //Error if we try to get an index out of bounds
        let element_offset = index / 8;
        //Otherwise, the order of the bitvec is left->right while the elements inside the bytes are ordered left<-right
        if let Some(element) = self.bits.get_mut(element_offset as usize) {
            //If we index properly (ie. not out of bounds), continue
            let bit_offset = index % 8;
            let val_bitmask = (1 as u8) << (7 - bit_offset); //lshift 7 - bit_offset so we're in the bit_offset'th place from the left (0 included)
            match val {
                true => {
                    *element |= val_bitmask; //This is easy enough, just or element with bitmask, and our bit will be set to true
                }
                false => {
                    *element = !(val_bitmask | !*element); //Bit more complex. Negate element, then or with bitmask, then negate once more for the result.
                }
            }
            Ok(()) //If all's well, return ok
        } else {
            Err(()) //If we somehow index improperly, just error. Should be handled above, but just in case.
        }
    }

    pub fn set_length(&mut self, new_length: u32) {
        self.length = new_length;
        self.bits.resize_with(
            ((new_length / 8) + (new_length % 8 != 0) as u32) as usize,
            Default::default,
        );
    }

    pub fn get_bits_as_vec(&self) -> Vec<bool> {
        let mut res: Vec<bool> = vec![];
        for i in 0..self.length {
            res.push(self.get_bit(i).unwrap());
        }
        res
    }

    pub fn get_iterator(&self) -> impl Iterator<Item = bool> + '_ {
        BitvectorIterator {
            bitvec: self,
            current_index: 0,
        }
    }
}

impl Iterator for BitvectorIterator<'_> {
    type Item = bool;
    fn next(&mut self) -> std::option::Option<<Self as std::iter::Iterator>::Item> {
        let next_val = self.bitvec.get_bit(self.current_index);
        self.current_index += 1;
        next_val
    }
}

#[cfg(test)]
mod tests {
    use crate::bitvector::Bitvector;

    #[test]
    fn test_setlength1() {
        //check allocation for exact bitmatches
        let mut vector = Bitvector::init();
        vector.set_length(16);
        assert_eq!(vector.length, 16);
        assert_eq!(vector.bits.len(), 2);
    }

    #[test]
    fn test_setlength2() {
        //check allocation for match +1 bit
        let mut vector = Bitvector::init();
        vector.set_length(17);
        assert_eq!(vector.length, 17);
        assert_eq!(vector.bits.len(), 3); //as we need one bit of the next element, we should have three elements allocated
    }

    #[test]
    fn test_setlength3() {
        //check for allocation for match -1 bit
        let mut vector = Bitvector::init();
        vector.set_length(15);
        assert_eq!(vector.length, 15);
        assert_eq!(vector.bits.len(), 2); //should be the same as for exact match
    }

    #[test]
    fn test_setbit1() {
        //basic test for setting the very first bit
        let mut vector = Bitvector::init();
        vector.set_length(1);
        match vector.set_bit(0, true) {
            Ok(_) => (),
            Err(_) => panic!("couldn't set bit"),
        }
        assert_eq!(vector.bits.get(0).unwrap().to_owned(), 0b10000000u8); //deliberately not using get_bit for these tests in order to avoid a circular testing dependency.
    }

    #[test]
    fn test_setbit2() {
        //test for checking if byte indexing works on the next element
        let mut vector = Bitvector::init();
        vector.set_length(9);
        match vector.set_bit(8, true) {
            Ok(_) => (),
            Err(_) => panic!("could't set bit"),
        }
        assert_eq!(vector.bits.get(1).unwrap().to_owned(), 0b10000000u8);
    }

    #[test]
    #[should_panic]
    fn test_setbit3() {
        //test for checking if byte indexing errors properly
        let mut vector = Bitvector::init();
        vector.set_length(8); //should panic as length of vector is 8
        match vector.set_bit(8, true) {
            //but we are trying to set the 9th bit at index 8
            Ok(_) => (),
            Err(_) => panic!("could't set bit"),
        }
        assert_eq!(vector.bits.get(1).unwrap().to_owned(), 0b10000000u8);
    }

    #[test]
    fn test_setbit4() {
        //test for checking if byte indexing works, now several elements down the vec
        let mut vector = Bitvector::init();
        vector.set_length(129);
        match vector.set_bit(128, true) {
            Ok(_) => (),
            Err(_) => panic!("could't set bit"),
        }

        for n in 0..=15 {
            assert_eq!(vector.bits.get(n).unwrap().to_owned(), 0b00000000u8); //check that all other elements are zero
        }
        assert_eq!(vector.bits.get(16).unwrap().to_owned(), 0b10000000u8);
    }

    #[test]
    fn test_setbit5() {
        //test for checking if bitindexing works
        let mut vector = Bitvector::init();
        vector.set_length(8);
        match vector.set_bit(1, true) {
            Ok(_) => (),
            Err(_) => panic!("could't set bit"),
        }
        match vector.set_bit(5, true) {
            Ok(_) => (),
            Err(_) => panic!("could't set bit"),
        }

        assert_eq!(vector.bits.get(0).unwrap().to_owned(), 0b01000100u8);
    }

    #[test]
    fn test_setbit6() {
        //test for checking if bitindexing works on another element
        let mut vector = Bitvector::init();
        vector.set_length(16);
        match vector.set_bit(9, true) {
            Ok(_) => (),
            Err(_) => panic!("could't set bit"),
        }
        match vector.set_bit(13, true) {
            Ok(_) => (),
            Err(_) => panic!("could't set bit"),
        }

        assert_eq!(vector.bits.get(1).unwrap().to_owned(), 0b01000100u8);
    }

    #[test]
    fn test_getbit1() {
        //basic test for getting the very first bit. We just reuse everything from before, but now get in the assert instead.
        let mut vector = Bitvector::init();
        vector.set_length(1);
        match vector.set_bit(0, true) {
            Ok(_) => (),
            Err(_) => panic!("couldn't set bit"),
        }
        assert_eq!(vector.get_bit(0).unwrap(), true);
    }

    #[test]
    fn test_getbit2() {
        //test for checking if byte indexing works on the next element
        let mut vector = Bitvector::init();
        vector.set_length(9);
        match vector.set_bit(8, true) {
            Ok(_) => (),
            Err(_) => panic!("could't set bit"),
        }
        assert_eq!(vector.get_bit(8).unwrap(), true);
    }

    #[test]
    #[should_panic]
    fn test_getbit3() {
        //test for checking if byte indexing errors properly
        let mut vector = Bitvector::init();
        vector.set_length(9);
        match vector.set_bit(8, true) {
            Ok(_) => (),
            Err(_) => panic!("could't set bit"),
        }
        assert_eq!(vector.get_bit(9).unwrap(), true);
    }

    #[test]
    fn test_getbit4() {
        //test for checking if byte indexing works, now several elements down the vec
        let mut vector = Bitvector::init();
        vector.set_length(129);
        match vector.set_bit(128, true) {
            Ok(_) => (),
            Err(_) => panic!("could't set bit"),
        }

        for n in 0..128 {
            //range isn't inclusive
            assert_eq!(vector.get_bit(n).unwrap(), false); //check that all other elements are zero
        }
        assert_eq!(vector.get_bit(128).unwrap(), true);
    }

    #[test]
    fn test_getbit5() {
        //test for checking if bitindexing works
        let mut vector = Bitvector::init();
        vector.set_length(8);
        match vector.set_bit(1, true) {
            Ok(_) => (),
            Err(_) => panic!("could't set bit"),
        }
        match vector.set_bit(5, true) {
            Ok(_) => (),
            Err(_) => panic!("could't set bit"),
        }

        assert_eq!(vector.get_bit(1).unwrap(), true);
        assert_eq!(vector.get_bit(5).unwrap(), true);
    }

    #[test]
    fn test_getbit6() {
        //test for checking if bitindexing works on another element
        let mut vector = Bitvector::init();
        vector.set_length(16);
        match vector.set_bit(9, true) {
            Ok(_) => (),
            Err(_) => panic!("could't set bit"),
        }
        match vector.set_bit(13, true) {
            Ok(_) => (),
            Err(_) => panic!("could't set bit"),
        }

        assert_eq!(vector.get_bit(9).unwrap(), true);
        assert_eq!(vector.get_bit(13).unwrap(), true);
    }

    #[test]
    fn test_init_with_length() {
        let bvec = Bitvector::init_with_length(32);
        assert_eq!(bvec.length, 32);
    }
}
