#[cfg(feature = "path_oram")]
pub use path_oram as oram;

#[cfg(feature = "circuit_oram")]
pub use circuit_oram as oram;

#[cfg(not(any(feature = "path_oram", feature = "circuit_oram")))]
compile_error!("Exactly one of these features must be enabled: path_oram, circuit_oram");
