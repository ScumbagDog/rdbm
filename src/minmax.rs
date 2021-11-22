pub trait MinMax: Min + Max {} //Used for setting infinity and negative infinity. If you want another datatype to represent a bound, implement min and max for it. Take care to not let either be the same as default.

pub trait Min {
    fn min() -> Self;
}

pub trait Max {
    fn max() -> Self;
}

impl Min for i8 {
    fn min() -> Self {
        std::i8::MIN
    }
}

impl Min for i16 {

    fn min() -> Self {
        std::i16::MIN
    }
}

impl Min for i32 {

    fn min() -> Self {
        std::i32::MIN
    }
}

impl Min for i64 {

    fn min() -> Self {
        std::i64::MIN
    }
}

impl Min for i128 {

    fn min() -> Self {
        std::i128::MIN
    }
}

impl Max for i8 {

    fn max() -> Self {
        std::i8::MAX
    }
}

impl Max for i16 {

    fn max() -> Self {
        std::i16::MAX
    }
}

impl Max for i32 {

    fn max() -> Self {
        std::i32::MAX
    }
}

impl Max for i64 {

    fn max() -> Self {
        std::i64::MAX
    }
}

impl Max for i128 {

    fn max() -> Self {
        std::i128::MAX
    }
}
