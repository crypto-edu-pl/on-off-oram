use std::time::Instant;

use rand::{distributions::Standard, rngs::OsRng, Rng};
use static_assertions::const_assert;

use oram::{path_oram::LINEAR_TIME_ORAM_CUTOFF, Address, DefaultOram, Oram};

const ARRAY_SIZE: u64 = 4096;

const_assert!(ARRAY_SIZE >= LINEAR_TIME_ORAM_CUTOFF);

fn main() {
    let mut rng = OsRng;

    let mut oram_array = DefaultOram::<u64>::new(ARRAY_SIZE, &mut rng).unwrap();

    let mut values = (&mut rng).sample_iter(Standard).take(ARRAY_SIZE as usize).collect::<Vec<u64>>();
    values.sort();

    for (i, value) in values.iter().enumerate() {
        oram_array.write(i as Address, *value, &mut rng).unwrap();
    }

    // Run for a fixed number of iterations to prevent leakage from the number of iterations and keep execution times consistent
    let n_iterations = ARRAY_SIZE.next_power_of_two().ilog2();
    
    for _ in 0..20 {
        let search_val = rng.gen::<u64>();

        let mut l = 0;
        let mut r = ARRAY_SIZE - 1;
        let mut found = None;

        let start = Instant::now();
        for _ in 0..n_iterations {
            let mid = (l + r) / 2;
            let val = oram_array.read(mid, &mut rng).unwrap();

            if l <= r {
                if val < search_val {
                    l = mid + 1;
                } else if val > search_val {
                    r = mid - 1;
                } else {
                    found = Some(mid);
                }
            }
        }
        let duration = start.elapsed();

        println!("Found {:?} in {:?}", found, duration);
    }
}
