use crate::{MS_PER_PAGE, NUM_CHANNELS, PAGES_PER_MS};
use std::time::{Duration, Instant};

const PRELOAD_NEXT_TRACK_BEFORE_END_DURATION_MS: u32 = 30000;

pub struct PlayerTime;

impl PlayerTime {
    pub fn samples_to_pcm(samples: &[f64]) -> u64 {
        (samples.len() / NUM_CHANNELS as usize) as u64
    }

    fn position_ms_to_pcm(position_ms: u32) -> u64 {
        (position_ms as f64 * PAGES_PER_MS) as u64
    }

    pub fn get_relative_position_pcm(position_ms: u32, duration_ms: u32) -> u64 {
        let position_pcm = (position_ms as f64 * PAGES_PER_MS).round() as u64;
        let duration_pcm = (duration_ms as f64 * PAGES_PER_MS) as u64;

        position_pcm.min(duration_pcm)
    }

    pub fn get_maybe_relative_position_pcm(
        position_ms: u32,
        maybe_duration_ms: Option<u32>,
    ) -> u64 {
        match maybe_duration_ms {
            Some(duration_ms) => Self::get_relative_position_pcm(position_ms, duration_ms),
            None => Self::position_ms_to_pcm(position_ms),
        }
    }

    pub fn get_relative_position_ms(position_pcm: u64, duration_ms: u32) -> u32 {
        let position_ms = (position_pcm as f64 * MS_PER_PAGE).round() as u32;

        position_ms.min(duration_ms)
    }

    pub fn should_preload(position_pcm: u64, duration_ms: u32) -> bool {
        duration_ms.saturating_sub(Self::get_relative_position_ms(position_pcm, duration_ms))
            < PRELOAD_NEXT_TRACK_BEFORE_END_DURATION_MS
    }

    pub fn get_nominal_start_time(position_ms: u32, duration_ms: u32) -> Option<Instant> {
        let position_ms = position_ms.min(duration_ms);

        Instant::now().checked_sub(Duration::from_millis(position_ms as u64))
    }

    pub fn should_notify(
        position_pcm: u64,
        latency_pcm: u64,
        duration_ms: u32,
        start_time: Option<Instant>,
    ) -> Option<(Option<Instant>, u32)> {
        let adjusted_position_pcm = position_pcm.saturating_sub(latency_pcm);

        let position_ms = Self::get_relative_position_ms(adjusted_position_pcm, duration_ms);

        let track_position = Duration::from_millis(position_ms as u64);

        let new_start_time = Self::get_nominal_start_time(position_ms, duration_ms);

        match start_time {
            None => Some((new_start_time, position_ms)),
            Some(start_time) => {
                // Only notify if we're behind,
                // more than likely due to sample pipeline latency.
                if Instant::now()
                    .saturating_duration_since(start_time)
                    .saturating_sub(track_position)
                    .as_secs()
                    == 0
                {
                    None
                } else {
                    Some((new_start_time, position_ms))
                }
            }
        }
    }
}
