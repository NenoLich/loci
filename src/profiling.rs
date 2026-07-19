#[cfg(feature = "profiling")]
macro_rules! range_push {
    ($name:literal) => {
        nvtx::range_push!($name);
    };
}

#[cfg(feature = "profiling")]
macro_rules! range_pop {
    () => {
        nvtx::range_pop!();
    };
}

#[cfg(not(feature = "profiling"))]
macro_rules! range_push {
    ($name:literal) => {};
}

#[cfg(not(feature = "profiling"))]
macro_rules! range_pop {
    () => {};
}

pub(crate) use {range_pop, range_push};
