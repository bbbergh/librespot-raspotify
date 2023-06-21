use crate::SAMPLE_RATE as SOURCE_SAMPLE_RATE;

const INPUT_SIZE: usize = 147;

// Reciprocals allow us to multiply instead of divide during interpolation.
const HZ48000_RESAMPLE_FACTOR_RECIPROCAL: f64 = SOURCE_SAMPLE_RATE as f64 / 48_000.0;
const HZ88200_RESAMPLE_FACTOR_RECIPROCAL: f64 = SOURCE_SAMPLE_RATE as f64 / 88_200.0;
const HZ96000_RESAMPLE_FACTOR_RECIPROCAL: f64 = SOURCE_SAMPLE_RATE as f64 / 96_000.0;

// sample rate * channels
const HZ44100_SAMPLES_PER_SECOND: f64 = 44_100.0 * 2.0;
const HZ48000_SAMPLES_PER_SECOND: f64 = 48_000.0 * 2.0;
const HZ88200_SAMPLES_PER_SECOND: f64 = 88_200.0 * 2.0;
const HZ96000_SAMPLES_PER_SECOND: f64 = 96_000.0 * 2.0;

const HZ48000_INTERPOLATION_OUTPUT_SIZE: usize =
    (INPUT_SIZE as f64 * (1.0 / HZ48000_RESAMPLE_FACTOR_RECIPROCAL)) as usize;
const HZ88200_INTERPOLATION_OUTPUT_SIZE: usize =
    (INPUT_SIZE as f64 * (1.0 / HZ88200_RESAMPLE_FACTOR_RECIPROCAL)) as usize;
const HZ96000_INTERPOLATION_OUTPUT_SIZE: usize =
    (INPUT_SIZE as f64 * (1.0 / HZ96000_RESAMPLE_FACTOR_RECIPROCAL)) as usize;

// Blackman Window coefficients
const BLACKMAN_A0: f64 = 0.42;
const BLACKMAN_A1: f64 = 0.5;
const BLACKMAN_A2: f64 = 0.08;

// Constants for calculations
const TWO_TIMES_PI: f64 = 2.0 * std::f64::consts::PI;
const FOUR_TIMES_PI: f64 = 4.0 * std::f64::consts::PI;

#[derive(Clone, Copy, Debug, Default)]
pub enum InterpolationQuality {
    #[default]
    Low,
    Medium,
    High,
}

impl std::str::FromStr for InterpolationQuality {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use InterpolationQuality::*;

        let lowercase_input = s.to_lowercase();

        match lowercase_input.as_str() {
            "low" => Ok(Low),
            "medium" => Ok(Medium),
            "high" => Ok(High),
            _ => Err(()),
        }
    }
}

impl std::fmt::Display for InterpolationQuality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use InterpolationQuality::*;

        match self {
            Low => write!(f, "Low"),
            Medium => write!(f, "Medium"),
            High => write!(f, "High"),
        }
    }
}

impl InterpolationQuality {
    fn get_interpolation_coefficients(&self, resample_factor_reciprocal: f64) -> Vec<f64> {
        let interpolation_coefficients_length = self.get_interpolation_coefficients_length();

        let mut coefficients = Vec::with_capacity(interpolation_coefficients_length);

        let last_index = interpolation_coefficients_length as f64 - 1.0;

        let sinc_center = last_index * 0.5;

        let mut coefficient_sum = 0.0;

        coefficients.extend((0..interpolation_coefficients_length).map(
            |interpolation_coefficient_index| {
                let index_float = interpolation_coefficient_index as f64;
                let sample_index_fractional = (index_float * resample_factor_reciprocal).fract();
                let sinc_center_offset = index_float - sinc_center;

                let sample_index_fractional_sinc_weight = Self::sinc(sample_index_fractional);

                let sinc_value = Self::sinc(sinc_center_offset);
                // Calculate the Blackman window function for the given center offset
                // w(n) = A0 - A1*cos(2πn / (N-1)) + A2*cos(4πn / (N-1)),
                // where n is the center offset, N is the window size,
                // and A0, A1, A2 are precalculated coefficients

                let two_pi_n = TWO_TIMES_PI * index_float;
                let four_pi_n = FOUR_TIMES_PI * index_float;

                let blackman_window_value = BLACKMAN_A0
                    - BLACKMAN_A1 * (two_pi_n / last_index).cos()
                    + BLACKMAN_A2 * (four_pi_n / last_index).cos();

                let sinc_window = sinc_value * blackman_window_value;

                let coefficient = sinc_window * sample_index_fractional_sinc_weight;

                coefficient_sum += coefficient;

                coefficient
            },
        ));

        coefficients
            .iter_mut()
            .for_each(|coefficient| *coefficient /= coefficient_sum);

        coefficients
    }

    fn get_interpolation_coefficients_length(&self) -> usize {
        use InterpolationQuality::*;
        match self {
            Low => 0,
            Medium => 129,
            High => 257,
        }
    }

    fn sinc(x: f64) -> f64 {
        if x.abs() < f64::EPSILON {
            1.0
        } else {
            let pi_x = std::f64::consts::PI * x;
            pi_x.sin() / pi_x
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub enum SampleRate {
    #[default]
    Hz44100,
    Hz48000,
    Hz88200,
    Hz96000,
}

impl std::str::FromStr for SampleRate {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use SampleRate::*;

        let lowercase_input = s.to_lowercase();

        // Match against both the actual
        // stringified value and how most
        // humans would write a sample rate.
        match lowercase_input.as_str() {
            "hz44100" | "44100hz" | "44100" | "44.1khz" => Ok(Hz44100),
            "hz48000" | "48000hz" | "48000" | "48khz" => Ok(Hz48000),
            "hz88200" | "88200hz" | "88200" | "88.2khz" => Ok(Hz88200),
            "hz96000" | "96000hz" | "96000" | "96khz" => Ok(Hz96000),
            _ => Err(()),
        }
    }
}

impl std::fmt::Display for SampleRate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use SampleRate::*;

        match self {
            // Let's make these more human readable.
            // "Hz44100" is just awkward.
            Hz44100 => write!(f, "44.1kHz"),
            Hz48000 => write!(f, "48kHz"),
            Hz88200 => write!(f, "88.2kHz"),
            Hz96000 => write!(f, "96kHz"),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ResampleSpec {
    resample_factor_reciprocal: f64,
    interpolation_output_size: usize,
}

impl SampleRate {
    pub fn as_u32(&self) -> u32 {
        use SampleRate::*;

        match self {
            Hz44100 => 44100,
            Hz48000 => 48000,
            Hz88200 => 88200,
            Hz96000 => 96000,
        }
    }

    pub fn duration_to_normalisation_coefficient(&self, duration: std::time::Duration) -> f64 {
        (-1.0 / (duration.as_secs_f64() * self.samples_per_second())).exp()
    }

    pub fn normalisation_coefficient_to_duration(&self, coefficient: f64) -> std::time::Duration {
        std::time::Duration::from_secs_f64(-1.0 / coefficient.ln() / self.samples_per_second())
    }

    fn samples_per_second(&self) -> f64 {
        use SampleRate::*;

        match self {
            Hz44100 => HZ44100_SAMPLES_PER_SECOND,
            Hz48000 => HZ48000_SAMPLES_PER_SECOND,
            Hz88200 => HZ88200_SAMPLES_PER_SECOND,
            Hz96000 => HZ96000_SAMPLES_PER_SECOND,
        }
    }

    fn get_resample_spec(&self) -> ResampleSpec {
        use SampleRate::*;

        match self {
            // Dummy values to satisfy
            // the match statement.
            // 44.1kHz will be bypassed.
            Hz44100 => ResampleSpec {
                resample_factor_reciprocal: 1.0,
                interpolation_output_size: INPUT_SIZE,
            },
            Hz48000 => ResampleSpec {
                resample_factor_reciprocal: HZ48000_RESAMPLE_FACTOR_RECIPROCAL,
                interpolation_output_size: HZ48000_INTERPOLATION_OUTPUT_SIZE,
            },
            Hz88200 => ResampleSpec {
                resample_factor_reciprocal: HZ88200_RESAMPLE_FACTOR_RECIPROCAL,
                interpolation_output_size: HZ88200_INTERPOLATION_OUTPUT_SIZE,
            },
            Hz96000 => ResampleSpec {
                resample_factor_reciprocal: HZ96000_RESAMPLE_FACTOR_RECIPROCAL,
                interpolation_output_size: HZ96000_INTERPOLATION_OUTPUT_SIZE,
            },
        }
    }
}

struct DelayLine {
    buffer: std::collections::VecDeque<f64>,
    interpolation_coefficients_length: usize,
}

impl DelayLine {
    fn new(interpolation_coefficients_length: usize) -> DelayLine {
        Self {
            buffer: std::collections::VecDeque::with_capacity(interpolation_coefficients_length),
            interpolation_coefficients_length,
        }
    }

    fn push(&mut self, sample: f64) {
        self.buffer.push_back(sample);

        while self.buffer.len() > self.interpolation_coefficients_length {
            self.buffer.pop_front();
        }
    }

    fn clear(&mut self) {
        self.buffer.clear();
    }
}

impl<'a> IntoIterator for &'a DelayLine {
    type Item = &'a f64;
    type IntoIter = std::collections::vec_deque::Iter<'a, f64>;

    fn into_iter(self) -> Self::IntoIter {
        self.buffer.iter()
    }
}

struct WindowedSincInterpolator {
    interpolation_coefficients: Vec<f64>,
    interpolation_coefficients_sum: f64,
    delay_line: DelayLine,
}

impl WindowedSincInterpolator {
    fn new(interpolation_quality: InterpolationQuality, resample_factor_reciprocal: f64) -> Self {
        let interpolation_coefficients =
            interpolation_quality.get_interpolation_coefficients(resample_factor_reciprocal);

        let interpolation_coefficients_sum = interpolation_coefficients.iter().sum();

        let delay_line = DelayLine::new(interpolation_coefficients.len());

        Self {
            interpolation_coefficients,
            interpolation_coefficients_sum,
            delay_line,
        }
    }

    fn interpolate(&mut self, sample: f64) -> f64 {
        // Since our interpolation coefficients are pre-calculated
        // we can basically pretend like the Interpolator is a FIR filter.
        self.delay_line.push(sample);

        // Temporal convolution
        let mut output_sample = self
            .interpolation_coefficients
            .iter()
            .zip(&self.delay_line)
            .fold(0.0, |acc, (coefficient, delay_line_sample)| {
                acc + coefficient * delay_line_sample
            });

        if output_sample.is_normal() {
            // Make sure that interpolation does not add any gain.
            output_sample /= self.interpolation_coefficients_sum;
        }

        output_sample
    }

    fn clear(&mut self) {
        self.delay_line.clear();
    }
}

trait MonoResampler {
    fn new(sample_rate: SampleRate, interpolation_quality: InterpolationQuality) -> Self
    where
        Self: Sized;

    fn stop(&mut self);
    fn get_latency_pcm(&mut self) -> u64;
    fn resample(&mut self, samples: &[f64]) -> Option<Vec<f64>>;
}

struct MonoSincResampler {
    interpolator: WindowedSincInterpolator,
    input_buffer: Vec<f64>,
    resample_factor_reciprocal: f64,
    delay_line_latency: u64,
    interpolation_output_size: usize,
}

impl MonoResampler for MonoSincResampler {
    fn new(sample_rate: SampleRate, interpolation_quality: InterpolationQuality) -> Self {
        let spec = sample_rate.get_resample_spec();

        let delay_line_latency = (interpolation_quality.get_interpolation_coefficients_length()
            as f64
            * spec.resample_factor_reciprocal)
            .round() as u64;

        Self {
            interpolator: WindowedSincInterpolator::new(
                interpolation_quality,
                spec.resample_factor_reciprocal,
            ),

            input_buffer: Vec::with_capacity(SOURCE_SAMPLE_RATE as usize),
            resample_factor_reciprocal: spec.resample_factor_reciprocal,
            delay_line_latency,
            interpolation_output_size: spec.interpolation_output_size,
        }
    }

    fn get_latency_pcm(&mut self) -> u64 {
        self.input_buffer.len() as u64 + self.delay_line_latency
    }

    fn stop(&mut self) {
        self.interpolator.clear();
        self.input_buffer.clear();
    }

    fn resample(&mut self, samples: &[f64]) -> Option<Vec<f64>> {
        self.input_buffer.extend_from_slice(samples);

        let num_buffer_chunks = self.input_buffer.len().saturating_div(INPUT_SIZE);

        if num_buffer_chunks == 0 {
            return None;
        }

        let input_size = num_buffer_chunks * INPUT_SIZE;
        // The size of the output after interpolation.
        let output_size = num_buffer_chunks * self.interpolation_output_size;

        let mut output = Vec::with_capacity(output_size);

        output.extend((0..output_size).map(|ouput_index| {
            // The factional weights are already calculated and factored
            // into our interpolation coefficients so all we have to
            // do is pretend we're doing nearest-neighbor interpolation
            // and push samples though the Interpolator and what comes
            // out the other side is Sinc Windowed Interpolated samples.
            let sample_index = (ouput_index as f64 * self.resample_factor_reciprocal) as usize;
            let sample = self.input_buffer[sample_index];
            self.interpolator.interpolate(sample)
        }));

        self.input_buffer.drain(..input_size);

        Some(output)
    }
}

struct MonoLinearResampler {
    input_buffer: Vec<f64>,
    resample_factor_reciprocal: f64,
    interpolation_output_size: usize,
}

impl MonoResampler for MonoLinearResampler {
    fn new(sample_rate: SampleRate, _: InterpolationQuality) -> Self {
        let spec = sample_rate.get_resample_spec();

        Self {
            input_buffer: Vec::with_capacity(SOURCE_SAMPLE_RATE as usize),
            resample_factor_reciprocal: spec.resample_factor_reciprocal,
            interpolation_output_size: spec.interpolation_output_size,
        }
    }

    fn get_latency_pcm(&mut self) -> u64 {
        self.input_buffer.len() as u64
    }

    fn stop(&mut self) {
        self.input_buffer.clear();
    }

    fn resample(&mut self, samples: &[f64]) -> Option<Vec<f64>> {
        self.input_buffer.extend_from_slice(samples);

        let num_buffer_chunks = self.input_buffer.len().saturating_div(INPUT_SIZE);

        if num_buffer_chunks == 0 {
            return None;
        }

        let input_size = num_buffer_chunks * INPUT_SIZE;
        // The size of the output after interpolation.
        // We have to account for the fact that to do effective linear
        // interpolation we need an extra sample to be able to throw away later.
        let output_size = num_buffer_chunks * self.interpolation_output_size + 1;

        let mut output = Vec::with_capacity(output_size);

        output.extend((0..output_size).map(|output_index| {
            let sample_index = output_index as f64 * self.resample_factor_reciprocal;
            let sample_index_fractional = sample_index.fract();
            let sample_index = sample_index as usize;
            let sample = *self.input_buffer.get(sample_index).unwrap_or(&0.0);
            let next_sample = *self.input_buffer.get(sample_index + 1).unwrap_or(&0.0);
            let sample_index_fractional_complementary = 1.0 - sample_index_fractional;
            sample * sample_index_fractional_complementary + next_sample * sample_index_fractional
        }));

        // Remove the last garbage sample.
        output.pop();

        self.input_buffer.drain(..input_size);

        Some(output)
    }
}

enum ResampleTask {
    Stop,
    Terminate,
    GetLatency,
    ProcessSamples(Vec<f64>),
}

enum ResampleResult {
    Latency(u64),
    ProcessedSamples(Option<Vec<f64>>),
}

struct ResampleWorker {
    task_sender: Option<std::sync::mpsc::Sender<ResampleTask>>,
    result_receiver: Option<std::sync::mpsc::Receiver<ResampleResult>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl ResampleWorker {
    fn new(mut resampler: impl MonoResampler + std::marker::Send + 'static, name: String) -> Self {
        let (task_sender, task_receiver) = std::sync::mpsc::channel();
        let (result_sender, result_receiver) = std::sync::mpsc::channel();

        let builder = std::thread::Builder::new().name(name.clone());

        let handle = match builder.spawn(move || loop {
            match task_receiver.recv() {
                Err(e) => {
                    match std::thread::current().name() {
                        Some(name) => error!("Error in <ResampleWorker> [{name}] thread: {e}"),
                        None => error!("Error in <ResampleWorker> thread: {e}"),
                    }

                    std::process::exit(1);
                }
                Ok(task) => match task {
                    ResampleTask::Stop => resampler.stop(),
                    ResampleTask::GetLatency => {
                        let latency = resampler.get_latency_pcm();

                        result_sender.send(ResampleResult::Latency(latency)).ok();
                    }
                    ResampleTask::ProcessSamples(samples) => {
                        let samples = resampler.resample(&samples);

                        result_sender
                            .send(ResampleResult::ProcessedSamples(samples))
                            .ok();
                    }
                    ResampleTask::Terminate => {
                        match std::thread::current().name() {
                            Some(name) => debug!("drop <ResampleWorker> [{name}] thread"),
                            None => debug!("drop <ResampleWorker> thread"),
                        }

                        break;
                    }
                },
            }
        }) {
            Ok(handle) => {
                debug!("Created <ResampleWorker> [{name}] thread");
                handle
            }
            Err(e) => {
                error!("Error creating <ResampleWorker> [{name}] thread: {e}");
                std::process::exit(1);
            }
        };

        Self {
            task_sender: Some(task_sender),
            result_receiver: Some(result_receiver),
            handle: Some(handle),
        }
    }

    fn get_latency_pcm(&mut self) -> u64 {
        self.task_sender
            .as_mut()
            .and_then(|sender| sender.send(ResampleTask::GetLatency).ok());

        self.result_receiver
            .as_mut()
            .and_then(|result_receiver| result_receiver.recv().ok())
            .and_then(|result| match result {
                ResampleResult::Latency(latency) => Some(latency),
                _ => None,
            })
            .unwrap_or_default()
    }

    fn stop(&mut self) {
        self.task_sender
            .as_mut()
            .and_then(|sender| sender.send(ResampleTask::Stop).ok());
    }

    fn process(&mut self, samples: Vec<f64>) {
        self.task_sender
            .as_mut()
            .and_then(|sender| sender.send(ResampleTask::ProcessSamples(samples)).ok());
    }

    fn receive_result(&mut self) -> Option<Vec<f64>> {
        self.result_receiver
            .as_mut()
            .and_then(|result_receiver| result_receiver.recv().ok())
            .and_then(|result| match result {
                ResampleResult::ProcessedSamples(samples) => samples,
                _ => None,
            })
    }
}

impl Drop for ResampleWorker {
    fn drop(&mut self) {
        self.task_sender
            .take()
            .and_then(|sender| sender.send(ResampleTask::Terminate).ok());

        self.result_receiver
            .take()
            .and_then(|result_receiver| loop {
                let drained = result_receiver.recv().ok();

                if drained.is_none() {
                    break drained;
                }
            });

        self.handle.take().and_then(|handle| handle.join().ok());
    }
}

#[derive(Default)]
enum Resampler {
    #[default]
    Bypass,
    Worker {
        left_resampler: ResampleWorker,
        right_resampler: ResampleWorker,
    },
}

#[derive(Default)]
pub struct StereoInterleavedResampler {
    resampler: Resampler,
    latency_flag: bool,
    process_flag: bool,
}

impl StereoInterleavedResampler {
    pub fn new(sample_rate: SampleRate, interpolation_quality: InterpolationQuality) -> Self {
        debug!("Sample Rate: {sample_rate}");

        let resampler = match sample_rate {
            SampleRate::Hz44100 => {
                debug!("Interpolation Type: Bypass");
                debug!("No <ResampleWorker> threads required");

                Resampler::Bypass
            }
            _ => {
                debug!("Interpolation Quality: {interpolation_quality}");

                let left_thread_name = "resampler:left".to_string();
                let right_thread_name = "resampler:right".to_string();

                match interpolation_quality {
                    InterpolationQuality::Low => {
                        debug!("Interpolation Type: Linear");

                        let left = MonoLinearResampler::new(sample_rate, interpolation_quality);
                        let right = MonoLinearResampler::new(sample_rate, interpolation_quality);

                        Resampler::Worker {
                            left_resampler: ResampleWorker::new(left, left_thread_name),
                            right_resampler: ResampleWorker::new(right, right_thread_name),
                        }
                    }
                    _ => {
                        debug!("Interpolation Type: Windowed Sinc");

                        let left = MonoSincResampler::new(sample_rate, interpolation_quality);
                        let right = MonoSincResampler::new(sample_rate, interpolation_quality);

                        Resampler::Worker {
                            left_resampler: ResampleWorker::new(left, left_thread_name),
                            right_resampler: ResampleWorker::new(right, right_thread_name),
                        }
                    }
                }
            }
        };

        Self {
            resampler,
            latency_flag: true,
            process_flag: false,
        }
    }

    pub fn get_latency_pcm(&mut self) -> u64 {
        let alternate_latency_flag = self.alternate_latency_flag();

        match &mut self.resampler {
            Resampler::Bypass => 0,
            Resampler::Worker {
                left_resampler,
                right_resampler,
            } => {
                if alternate_latency_flag {
                    left_resampler.get_latency_pcm()
                } else {
                    right_resampler.get_latency_pcm()
                }
            }
        }
    }

    fn alternate_latency_flag(&mut self) -> bool {
        // We only actually need the latency
        // from one channel for PCM frame latency
        // to balance the load we alternate.
        let current_flag = self.latency_flag;
        self.latency_flag = !self.latency_flag;
        current_flag
    }

    fn alternate_process_flag(&mut self) -> bool {
        // This along with the latency_flag makes
        // sure that all worker calls alternate
        // for load balancing.
        let current_flag = self.process_flag;
        self.process_flag = !self.process_flag;
        current_flag
    }

    pub fn process(&mut self, input_samples: &[f64]) -> Option<Vec<f64>> {
        let alternate_process_flag = self.alternate_process_flag();

        match &mut self.resampler {
            // Bypass is basically a no-op.
            Resampler::Bypass => Some(input_samples.to_vec()),
            Resampler::Worker {
                left_resampler,
                right_resampler,
            } => {
                let (left_samples, right_samples) = Self::deinterleave_samples(input_samples);

                let (processed_left_samples, processed_right_samples) = if alternate_process_flag {
                    left_resampler.process(left_samples);
                    right_resampler.process(right_samples);

                    let processed_left_samples = left_resampler.receive_result();
                    let processed_right_samples = right_resampler.receive_result();

                    (processed_left_samples, processed_right_samples)
                } else {
                    right_resampler.process(right_samples);
                    left_resampler.process(left_samples);

                    let processed_right_samples = right_resampler.receive_result();
                    let processed_left_samples = left_resampler.receive_result();

                    (processed_left_samples, processed_right_samples)
                };

                processed_left_samples.and_then(|left_samples| {
                    processed_right_samples.map(|right_samples| {
                        Self::interleave_samples(&left_samples, &right_samples)
                    })
                })
            }
        }
    }

    pub fn stop(&mut self) {
        match &mut self.resampler {
            // Stop does nothing
            // if we're bypassed.
            Resampler::Bypass => (),
            Resampler::Worker {
                left_resampler,
                right_resampler,
            } => {
                left_resampler.stop();
                right_resampler.stop();
            }
        }
    }

    fn interleave_samples(left_samples: &[f64], right_samples: &[f64]) -> Vec<f64> {
        // Re-interleave the resampled channels.
        left_samples
            .iter()
            .zip(right_samples.iter())
            .flat_map(|(&x, &y)| vec![x, y])
            .collect()
    }

    fn deinterleave_samples(samples: &[f64]) -> (Vec<f64>, Vec<f64>) {
        // Split the stereo interleaved samples into left and right channels.
        let (left_samples, right_samples): (Vec<f64>, Vec<f64>) = samples
            .chunks(2)
            .map(|chunk| {
                let [left_sample, right_sample] = [chunk[0], chunk[1]];
                (left_sample, right_sample)
            })
            .unzip();

        (left_samples, right_samples)
    }
}
