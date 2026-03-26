use std::time::Duration;

/// Results of a benchmark
pub struct BenchmarkResult {
    /// Time taken in ON mode
    pub access_on_duration: Duration,
    /// Time taken in OFF mode (online performance - excluding `turn_on`)
    pub access_off_duration: Duration,
    /// Time taken to turn ORAM back on
    pub turn_on_duration: Duration,
}

/// Statistics of a benchmark
pub struct BenchmarkStats {
    /// Time taken in ON mode
    pub access_on_duration: Stats,
    /// Time taken in OFF mode (overall performance - including `turn_on`)
    pub off_total_duration: Stats,
    /// Time taken in OFF mode (online performance - excluding `turn_on`)
    pub access_off_duration: Stats,
    /// Time taken to turn ORAM back on
    pub turn_on_duration: Stats,
}

/// Statistics (mean and standard deviation) for a single parameter
pub struct Stats {
    /// Mean value
    pub mean: Duration,
    /// Standard deviation
    pub stddev: Duration,
}

/// Calculate statistics for measurements of a single parameter
pub fn mean_and_standard_deviation(data: &[Duration]) -> Stats {
    let mean =
        data.iter().copied().reduce(|acc, x| acc + x).unwrap() / u32::try_from(data.len()).unwrap();
    let variance = data
        .iter()
        .map(|x| {
            let diff = (x.abs_diff(mean)).as_nanos();
            diff * diff
        })
        .reduce(|acc, x| acc + x)
        .unwrap()
        / u128::try_from(data.len() - 1).unwrap();
    let stddev = Duration::from_nanos(variance.isqrt().try_into().unwrap());
    Stats { mean, stddev }
}

/// Calculate statistics for all parameters in a benchmark
pub fn benchmark_stats(results: &[BenchmarkResult]) -> BenchmarkStats {
    let access_on_durations = results
        .iter()
        .map(|x| x.access_on_duration)
        .collect::<Vec<_>>();
    let off_total_durations = results
        .iter()
        .map(|result| result.access_off_duration + result.turn_on_duration)
        .collect::<Vec<_>>();
    let access_off_durations = results
        .iter()
        .map(|x| x.access_off_duration)
        .collect::<Vec<_>>();
    let turn_on_durations = results
        .iter()
        .map(|x| x.turn_on_duration)
        .collect::<Vec<_>>();
    BenchmarkStats {
        access_on_duration: mean_and_standard_deviation(&access_on_durations),
        off_total_duration: mean_and_standard_deviation(&off_total_durations),
        access_off_duration: mean_and_standard_deviation(&access_off_durations),
        turn_on_duration: mean_and_standard_deviation(&turn_on_durations),
    }
}
