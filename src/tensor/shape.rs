use std::fmt;
use std::ops::Index;

/// Represents the shape of a tensor
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    dims: [usize; 8],
    len: usize,
}

impl Shape {
    /// Creates a new shape from a vector of dimensions
    pub fn new(slice: impl AsRef<[usize]>) -> Self {
        let slice = slice.as_ref();
        let mut dims = [0; 8];
        let len = slice.len().min(8);
        dims[..len].copy_from_slice(&slice[..len]);
        Self { dims, len }
    }

    /// Returns the number of dimensions
    pub fn ndim(&self) -> usize {
        self.len
    }

    /// Returns the size of a specific dimension
    pub fn dim(&self, index: usize) -> Option<usize> {
        if index < self.len { Some(self.dims[index]) } else { None }
    }

    /// Returns an iterator over all dimensions
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.dims[..self.len].iter().copied()
    }

    /// Returns the product of all dimensions (total number of elements)
    pub fn product(&self) -> usize {
        self.dims[..self.len].iter().product()
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, &dim) in self.dims[..self.len].iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", dim)?;
        }
        write!(f, "]")
    }
}

impl From<&[usize]> for Shape {
    fn from(slice: &[usize]) -> Self {
        Self::new(slice)
    }
}

impl From<Vec<usize>> for Shape {
    fn from(vec: Vec<usize>) -> Self {
        Self::new(&vec)
    }
}

impl Index<usize> for Shape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(
            index < self.len,
            "Shape index out of bounds: index {} >= len {}",
            index,
            self.len
        );
        &self.dims[index]
    }
}

impl AsRef<[usize]> for Shape {
    fn as_ref(&self) -> &[usize] {
        &self.dims[..self.len]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_empty() {
        let s = Shape::new(&[] as &[usize]);
        assert_eq!(s.ndim(), 0);
    }

    #[test]
    fn new_one_dim() {
        let s = Shape::new([5]);
        assert_eq!(s.ndim(), 1);
        assert_eq!(s.dim(0), Some(5));
        assert_eq!(s.product(), 5);
    }

    #[test]
    fn new_two_dims() {
        let s = Shape::new(&[2, 3]);
        assert_eq!(s.ndim(), 2);
        assert_eq!(s.dim(0), Some(2));
        assert_eq!(s.dim(1), Some(3));
        assert_eq!(s.product(), 6);
    }

    #[test]
    fn new_max_dims() {
        let s = Shape::new([1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(s.ndim(), 8);
    }

    #[test]
    fn new_truncates_over_eight() {
        let s = Shape::new(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(s.ndim(), 8);
    }

    #[test]
    fn dim_out_of_bounds() {
        let s = Shape::new([2, 3]);
        assert_eq!(s.dim(2), None);
        assert_eq!(s.dim(100), None);
    }

    #[test]
    fn iter_single() {
        let s = Shape::new([4]);
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![4]);
    }

    #[test]
    fn iter_multiple() {
        let s = Shape::new(&[2, 3]);
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![2, 3]);
    }

    #[test]
    fn iter_empty() {
        let s = Shape::new(&[] as &[usize]);
        assert!(s.iter().next().is_none());
    }

    #[test]
    fn product_empty_is_one() {
        let s = Shape::new(&[] as &[usize]);
        assert_eq!(s.product(), 1);
    }

    #[test]
    fn product_with_zeros() {
        let s = Shape::new(&[2, 0, 3]);
        assert_eq!(s.product(), 0);
    }

    #[test]
    fn display_empty() {
        let s = Shape::new(&[] as &[usize]);
        assert_eq!(format!("{}", s), "[]");
    }

    #[test]
    fn display_one() {
        let s = Shape::new([5]);
        assert_eq!(format!("{}", s), "[5]");
    }

    #[test]
    fn display_two() {
        let s = Shape::new(&[2, 3]);
        assert_eq!(format!("{}", s), "[2, 3]");
    }

    #[test]
    fn from_slice() {
        let s: Shape = (&[10, 20][..]).into();
        assert_eq!(s.ndim(), 2);
        assert_eq!(s.dim(0), Some(10));
        assert_eq!(s.dim(1), Some(20));
    }

    #[test]
    fn from_vec() {
        let s: Shape = vec![3, 4].into();
        assert_eq!(s.ndim(), 2);
        assert_eq!(s.dim(0), Some(3));
    }

    #[test]
    fn index_valid() {
        let s = Shape::new(&[7, 8]);
        assert_eq!(s[0], 7);
        assert_eq!(s[1], 8);
    }

    #[test]
    #[should_panic(expected = "Shape index out of bounds")]
    fn index_out_of_bounds() {
        let s = Shape::new([2, 3]);
        let _ = s[2];
    }

    #[test]
    #[should_panic(expected = "Shape index out of bounds")]
    fn index_empty_shape() {
        let s = Shape::new(&[] as &[usize]);
        let _ = s[0];
    }

    #[test]
    fn as_ref_basic() {
        let s = Shape::new(&[5, 6]);
        assert_eq!(s.as_ref(), &[5, 6]);
    }

    #[test]
    fn as_ref_empty() {
        let s = Shape::new(&[] as &[usize]);
        assert_eq!(s.as_ref(), &[] as &[usize]);
    }

    #[test]
    fn clone_and_copy() {
        let s = Shape::new(&[2, 3]);
        let s2 = s;
        let _s3 = s;
        assert_eq!(s.ndim(), s2.ndim());
        assert_eq!(s.dim(0), s2.dim(0));
    }

    #[test]
    fn partial_eq() {
        assert_eq!(Shape::new([2, 3]), Shape::new(&[2, 3]));
        assert_ne!(Shape::new([2, 3]), Shape::new([2, 4]));
        assert_ne!(Shape::new([2, 3]), Shape::new([1, 6]));
    }

    #[test]
    fn debug() {
        let s = Shape::new([2, 3]);
        let out = format!("{:?}", s);
        assert!(out.starts_with("Shape"));
        assert!(out.contains("dims"));
    }
}
