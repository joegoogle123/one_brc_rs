use std::fmt::{Display, Formatter, Write};
use std::fs::File;
use std::hash::Hash;
use std::hash::BuildHasherDefault;
use std::io::{Seek, SeekFrom};
use std::ops::{Deref, DerefMut, Div, Mul};
use fxhash::{FxBuildHasher, FxHasher};
use hashbrown::{Equivalent, HashMap};
use memchr::arch::all::is_equal;
use std::io::Read;
use std::os::unix::fs::FileExt;
use std::thread;

use clap::Parser;

use memchr::{memchr, memchr_iter, memrchr};


#[derive(Debug, Clone)]
pub struct FileOffset {
    pub offset: usize,
    pub length: usize,
}
impl FileOffset {
    pub fn new(offset: usize, length: usize) -> Self {
        FileOffset { offset, length }
    }

    pub fn offset_file(&self, mut file: File) -> Result<File, String> {
        file.seek(SeekFrom::Start(self.offset as u64))
            .map_err(|e| e.to_string())?;
        Ok(file)
    }
}

#[derive(PartialEq, Debug)]
pub struct ResolvedTemperatureStatistics {
    min: f32,
    max: f32,
    mean: f32,
}

impl ResolvedTemperatureStatistics {
    #[must_use]
    pub fn new(min: f32, max: f32, mean: f32) -> ResolvedTemperatureStatistics {
        ResolvedTemperatureStatistics { min, max, mean }
    }
}

impl Equivalent<Key> for [u8] {
    #[inline]
    fn equivalent(&self, key: &Key) -> bool {
        let x = self;
        let y = key.0.as_slice();
        is_equal(x, y)
    }
}

#[derive(PartialEq, Debug)]
pub struct ResolvedMeasurement(pub String, pub ResolvedTemperatureStatistics);

impl Display for ResolvedMeasurement {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}={}", self.0, self.1)
    }
}

#[derive(Debug)]
pub struct UnresolvedTemperatureStatistics {
    count: usize,
    sum: i32,
    min: i16,
    max: i16,
}

impl UnresolvedTemperatureStatistics {
    fn merge(&mut self, that: &UnresolvedTemperatureStatistics) {
        self.count += that.count;
        self.sum += that.sum;
        self.min = self.min.min(that.min);
        self.max = self.max.max(that.max);
    }

    pub fn add(&mut self, temperature: i16) {
        let this = self;
        this.count += 1;
        this.sum += i32::from(temperature);
        this.min = this.min.min(temperature);
        this.max = this.max.max(temperature);
    }
}


impl Default for UnresolvedTemperatureStatistics {
    fn default() -> Self {
        UnresolvedTemperatureStatistics {
            count: 0,
            sum: 0,
            min: i16::MAX,
            max: i16::MIN,
        }
    }
}

impl Display for ResolvedTemperatureStatistics {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        //:.1$
        let s = format!("{:.1}/{:.1}/{:.1}", self.min, self.mean, self.max);
        f.write_str(s.as_str())
    }
}

#[derive(Debug, PartialEq)]
pub struct AggregatedMeasurements(pub Vec<ResolvedMeasurement>);

impl Display for AggregatedMeasurements {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_char('{')?;
        for (index, m) in self.0.iter().enumerate() {
            if index == self.0.len() - 1 {
                write!(f, "{m}")?;
            } else {
                write!(f, "{m}, ")?;
            }
        }

        f.write_char('}')
    }
}

impl From<UnresolvedTemperatureStatistics> for ResolvedTemperatureStatistics {
    fn from(value: UnresolvedTemperatureStatistics) -> Self {
        let min = (value.min as f32).div(10f32);
        let max = (value.max as f32).div(10f32);
        let mean: f32 = (value.sum as f32).div(value.count.mul(10) as f32);
        let mean = round(mean);
        ResolvedTemperatureStatistics { min, max, mean }
    }
}

fn round(f: f32) -> f32 {
    (f * 10f32).round() / 10f32
}


impl Default for Measurements {
    fn default() -> Self {
        let map: HashMap<Key, UnresolvedTemperatureStatistics, BuildHasherDefault<FxHasher>> =
            HashMap::with_capacity_and_hasher(10000, FxBuildHasher::default());
        Measurements(map)
    }
}

#[derive(Hash)]
pub struct Key(pub Vec<u8>);

pub struct Measurements(
    pub HashMap<Key, UnresolvedTemperatureStatistics, BuildHasherDefault<FxHasher>>,
);


impl Measurements {
    #[inline]
    pub fn add_measurement(&mut self, city_as_bytes: &[u8], temp: i16) {
        let (_, value) = self
            .0
            .raw_entry_mut()
            .from_key(city_as_bytes)
            .or_insert_with(|| {
                (
                    Key(city_as_bytes.to_vec()),
                    UnresolvedTemperatureStatistics::default(),
                )
            });
        value.add(temp);
    }

    #[must_use]
    pub fn merge(self, that: Measurements) -> Measurements {
        let mut this = self;
        let this_map = &mut this.0;

        let Measurements(that_map) = that;

        for (key, value) in that_map {
            this_map
                .raw_entry_mut()
                .from_key(key.0.as_slice())
                .and_modify(|_, old_v| old_v.merge(&value))
                .or_insert_with(|| (key, value));
        }

        this
    }
}

impl Deref for Measurements {
    type Target = HashMap<Key, UnresolvedTemperatureStatistics, BuildHasherDefault<FxHasher>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Measurements {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Measurements {
    pub fn calculate_measurements(self) -> AggregatedMeasurements {
        let mut actual: Vec<ResolvedMeasurement> = self
            .0
            .into_iter()
            .map(|(k, v)| {
                let city = String::from_utf8(k.0).unwrap();
                let temp = v.into();
                ResolvedMeasurement(city, temp)
            })
            .collect();

        actual.sort_unstable_by(|m1, m2| m1.0.as_str().cmp(m2.0.as_str()));

        AggregatedMeasurements(actual)
    }
}

fn aggregate_measurements(
    mut file: File,
    file_offset: &FileOffset,
    buffer_len: usize,
) -> Measurements {
    let mut measurements = Measurements::default();

    file = file_offset.offset_file(file).unwrap();

    let mut buffer = vec![0; buffer_len];
    let mut remainder_buffer = Vec::with_capacity(128);
    let mut total_read = 0;

    while total_read < file_offset.length {
        let mut read = {
            let spillover_len = remainder_buffer.len();
            let (left, right) = buffer.split_at_mut(spillover_len);
            left.copy_from_slice(remainder_buffer.as_slice());
            remainder_buffer.clear();
            let num_bytes_read = read_greedy(&mut file, right);
            num_bytes_read + spillover_len
        };

        read = read.min(file_offset.length - total_read);

        let read_slice = &buffer[..read];

        let last_newline_index = memrchr(b'\n', read_slice).unwrap();
        let actual_bytes_read = last_newline_index + 1;
        total_read += actual_bytes_read;

        let (aligned_buffer, remainder) = read_slice.split_at(actual_bytes_read); // include the new line in this line
        remainder_buffer.extend_from_slice(remainder);

        let mut city_start = 0;

        for delimiter_index in memchr_iter(b';', aligned_buffer) {
            let city_end = delimiter_index;

            let city_name_bytes = &aligned_buffer[city_start..city_end];

            let temperature_start = delimiter_index + 1;
            let temperature_bytes = &aligned_buffer[temperature_start..];
            let (temperature, temperature_len) = parse_temperature_and_len(temperature_bytes);
            city_start = temperature_start + temperature_len + 1;
            measurements.add_measurement(city_name_bytes, temperature);
        }
    }

    measurements
}
#[inline]
#[allow(clippy::cast_possible_truncation)]
fn parse_temperature_and_len(temperature_bytes: &[u8]) -> (i16, usize) {
    // Cast next 8 bytes as i64 for optimized temperature parsing
    let next_word = unsafe { temperature_bytes.as_ptr().cast::<i64>().read_unaligned() };
    // offset to . in bits + 4 e.g. 99.9 -> 8 + 8 + 4 = 20
    let dot_offset = (!next_word & 0x1010_1000).trailing_zeros();
    let temperature_len = ((dot_offset >> 3) + 2) as usize;
    (parse_temp_fast_v2(next_word, dot_offset) as i16, temperature_len)
}

#[inline]
fn parse_temp_fast_v2(word: i64, dot_offset: u32) -> i64 {
    let shift = 28 - dot_offset;
    let signed = (!word << 59) >> 63;
    let dsmask = !(signed & 0xFF);
    let digits = ((word & dsmask) << shift) & 0x000F_000F_0F00;
    let abs_val = ((digits.wrapping_mul(0x640a_0001)) >> 32) & 0x3FF;
    (abs_val ^ signed) - signed
}

fn read_greedy(file: &mut File, mut buf: &mut [u8]) -> usize {
    let mut read = 0;
    while !buf.is_empty() {
        match file.read(buf) {
            Ok(0) => break,
            Ok(n) => {
                read += n;
                buf = &mut buf[n..];
            }
            Err(e) => panic!("{}", e.to_string()),
        }
    }

    read
}

fn aggregate_temperatures_single_threaded(file_path: &str, buffer_size: usize) -> AggregatedMeasurements {
    let file = File::open(file_path).expect("Unable to open file");
    let file_offset = FileOffset::new(0, usize::try_from(file.metadata().unwrap().len()).unwrap());
    let measurements = aggregate_measurements(file, &file_offset, buffer_size);
    measurements.calculate_measurements()
}

fn aggregate_temperatures_multi_threaded(file_path: &str, num_workers: usize, buffer_size: usize) -> AggregatedMeasurements {
    let mut file: File = File::open(file_path).unwrap();
    let mut threads = Vec::with_capacity(num_workers);
    let segments = calculate_file_segments(&mut file, num_workers);

    for segment in segments {
        let file: File = File::open(file_path).unwrap();
        let thread = thread::spawn(move || aggregate_measurements(file, &segment, buffer_size));
        threads.push(thread);
    }

    let merged_measurements = threads
        .into_iter()
        .map(|t| t.join().unwrap())
        .reduce(Measurements::merge)
        .unwrap();

    merged_measurements.calculate_measurements()
}

fn calculate_file_segments(file: &mut File, num_segments: usize) -> Vec<FileOffset> {
    let file_len = file.metadata().unwrap().len();

    let mut file_offsets: Vec<FileOffset> = Vec::with_capacity(num_segments);

    let segment_len = file_len / num_segments as u64;
    let mut probe_buf = vec![0; 128];

    let mut current_offset = 0;

    while current_offset < file_len {
        let probe_start = (current_offset + segment_len).min(file_len - 1);
        let start = current_offset as usize;
        let end = match file.read_at(&mut probe_buf, probe_start) {
            Ok(read) => {
                probe_start as usize + memchr(b'\n', &probe_buf[..read]).unwrap_or(read - 1)
            }
            Err(e) => {
                panic!("{}", e.to_string())
            }
        };

        let file_offset = FileOffset::new(start, end - start + 1);
        current_offset = (end + 1) as u64;
        file_offsets.push(file_offset);
    }

    file_offsets
}

const DEFAULT_BUFFER_LEN: usize = 1024 * 512;
const NUM_THREADS: usize = 16;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Command {
    path: String,
    #[arg(short, long)]
    threads: Option<usize>,
    #[arg(short, long)]
    buffer_size: Option<usize>,
}


fn main() {
    let Command{path, threads, buffer_size} = Command::parse();

    let num_workers = threads.unwrap_or(NUM_THREADS);
    let buffer_size = buffer_size.unwrap_or(DEFAULT_BUFFER_LEN);

    let measurements = match num_workers {
        1 =>  aggregate_temperatures_single_threaded(path.as_str(), buffer_size),
        num_workers => aggregate_temperatures_multi_threaded(path.as_str(), num_workers, buffer_size)
    };

    println!("{measurements}");
}


#[cfg(test)]
mod tests {
    use std::fs;
    use std::fs::File;
    use std::os::unix::fs::FileExt;
    use crate::{calculate_file_segments, parse_temperature_and_len, ResolvedMeasurement, ResolvedTemperatureStatistics, aggregate_temperatures_multi_threaded, aggregate_temperatures_single_threaded, AggregatedMeasurements};

    #[test]
    fn temperature_test() {
        assert_eq!((999, 4), parse_temperature_and_len("99.9".as_bytes()));
        assert_eq!((-22, 4), parse_temperature_and_len("-2.2".as_bytes()));
        assert_eq!((11, 3), parse_temperature_and_len("1.1".as_bytes()));
        assert_eq!((111, 4), parse_temperature_and_len("11.1".as_bytes()));
        assert_eq!((-111, 5), parse_temperature_and_len("-11.1".as_bytes()));
    }

    #[test]
    fn test_single_threaded_temperature_aggregations_from_file() {
        let path = "resources/measurements_small.txt";
        let actual: AggregatedMeasurements = aggregate_temperatures_single_threaded(path, 32);

        let expected = AggregatedMeasurements(vec![
            ResolvedMeasurement(
                "Christchurch".to_string(),
                ResolvedTemperatureStatistics::new(23.5, 23.5, 23.5),
            ),
            ResolvedMeasurement(
                "Virginia Beach".to_string(),
                ResolvedTemperatureStatistics::new(21.7, 35.0, 28.4),
            ),
            ResolvedMeasurement(
                "Yakutsk".to_string(),
                ResolvedTemperatureStatistics::new(-27.6, -0.8, -14.2),
            ),
        ]);

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_multi_threaded_temperature_aggregations_from_file() {
        let path = "resources/measurements_small.txt";
        let single_threaded_measurements = aggregate_temperatures_single_threaded(path, 32);
        let multi_threaded_measurements = aggregate_temperatures_multi_threaded(path, 2, 32);
        assert_eq!(single_threaded_measurements, multi_threaded_measurements);
    }

    #[test]
    fn test_calculate_file_segments() {
        let path = "resources/measurements_small.txt";
        let mut file = File::open(path).expect("Unable to open file");
        let len = usize::try_from(file.metadata().unwrap().len()).unwrap();
        let segments = calculate_file_segments(&mut file, 5);
        let actual_len: usize = segments.iter().map(|s| s.length).sum();
        assert_eq!(actual_len, len);

        let all_bytes_expected = fs::read(path).unwrap();


        let mut all_bytes_actual = vec![];

        for offset in segments {
            let mut buffer = vec![0; offset.length];
            file.read_exact_at(&mut buffer[..], offset.offset as u64)
                .unwrap();
            all_bytes_actual.extend_from_slice(buffer.as_slice());
        }

        assert_eq!(all_bytes_expected, all_bytes_actual);
    }

}
