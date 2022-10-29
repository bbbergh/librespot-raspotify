use super::{Open, Sink, SinkAsBytes, SinkError, SinkResult};
use crate::config::AudioFormat;
use crate::convert::Converter;
use crate::decoder::AudioPacket;
use crate::{NUM_CHANNELS, SAMPLE_RATE};
use alsa::device_name::HintIter;
use alsa::pcm::{Access, Format, Frames, HwParams, PCM};
use alsa::{Direction, ValueOr};
use std::process::exit;
use thiserror::Error;

const MAX_BUFFER: Frames = SAMPLE_RATE as Frames;
const OPTIMAL_BUFFER: Frames = MAX_BUFFER / 2;
const OPTIMAL_PERIODS: Frames = 5;

#[derive(Debug, Error)]
enum AlsaError {
    #[error("<AlsaSink> Device {device} Unsupported Format {alsa_format} ({format:?}), {e}")]
    UnsupportedFormat {
        device: String,
        alsa_format: Format,
        format: AudioFormat,
        e: alsa::Error,
    },

    #[error("<AlsaSink> Device {device} Unsupported Channel Count {channel_count}, {e}")]
    UnsupportedChannelCount {
        device: String,
        channel_count: u8,
        e: alsa::Error,
    },

    #[error("<AlsaSink> Device {device} Unsupported Sample Rate {samplerate}, {e}")]
    UnsupportedSampleRate {
        device: String,
        samplerate: u32,
        e: alsa::Error,
    },

    #[error("<AlsaSink> Device {device} Unsupported Access Type RWInterleaved, {e}")]
    UnsupportedAccessType { device: String, e: alsa::Error },

    #[error("<AlsaSink> Device {device} May be Invalid, Busy, or Already in Use, {e}")]
    PcmSetUp { device: String, e: alsa::Error },

    #[error("<AlsaSink> Failed to Drain PCM Buffer, {0}")]
    DrainFailure(alsa::Error),

    #[error("<AlsaSink> {0}")]
    OnWrite(alsa::Error),

    #[error("<AlsaSink> Hardware, {0}")]
    HwParams(alsa::Error),

    #[error("<AlsaSink> Software, {0}")]
    SwParams(alsa::Error),

    #[error("<AlsaSink> PCM, {0}")]
    Pcm(alsa::Error),

    #[error("<AlsaSink> Could Not Parse Output Name(s) and/or Description(s), {0}")]
    Parsing(alsa::Error),

    #[error("<AlsaSink>")]
    NotConnected,
}

impl From<AlsaError> for SinkError {
    fn from(e: AlsaError) -> SinkError {
        use AlsaError::*;
        let es = e.to_string();
        match e {
            DrainFailure(_) | OnWrite(_) => SinkError::OnWrite(es),
            PcmSetUp { .. } => SinkError::ConnectionRefused(es),
            NotConnected => SinkError::NotConnected(es),
            _ => SinkError::InvalidParams(es),
        }
    }
}

impl From<AudioFormat> for Format {
    fn from(f: AudioFormat) -> Format {
        use AudioFormat::*;
        match f {
            F64 => Format::float64(),
            F32 => Format::float(),
            S32 => Format::s32(),
            S24 => Format::s24(),
            S24_3 => Format::s24_3(),
            S16 => Format::s16(),
        }
    }
}

pub struct AlsaSink {
    pcm: Option<PCM>,
    format: AudioFormat,
    device: String,
    buffer_size: Frames,
    period_size: Frames,
    period_buffer: Vec<u8>,
}

fn list_compatible_devices() -> SinkResult<()> {
    let i = HintIter::new_str(None, "pcm").map_err(AlsaError::Parsing)?;

    println!("\n\n\tCompatible alsa device(s):\n");
    println!("\t------------------------------------------------------\n");

    for a in i {
        if let Some(Direction::Playback) = a.direction {
            if let Some(name) = a.name {
                if let Ok(pcm) = PCM::new(&name, Direction::Playback, false) {
                    if let Ok(hwp) = HwParams::any(&pcm) {
                        // Only show devices that support
                        // 2 ch 44.1 Interleaved.

                        if hwp.set_access(Access::RWInterleaved).is_ok()
                            && hwp.set_rate(SAMPLE_RATE, ValueOr::Nearest).is_ok()
                            && hwp.set_channels(NUM_CHANNELS as u32).is_ok()
                        {
                            let mut supported_formats = vec![];

                            for f in &[
                                AudioFormat::S16,
                                AudioFormat::S24,
                                AudioFormat::S24_3,
                                AudioFormat::S32,
                                AudioFormat::F32,
                                AudioFormat::F64,
                            ] {
                                if hwp.test_format(Format::from(*f)).is_ok() {
                                    supported_formats.push(format!("{f:?}"));
                                }
                            }

                            if !supported_formats.is_empty() {
                                println!("\tDevice:\n\n\t\t{name}\n");

                                println!(
                                    "\tDescription:\n\n\t\t{}\n",
                                    a.desc.unwrap_or_default().replace('\n', "\n\t\t")
                                );

                                println!(
                                    "\tSupported Format(s):\n\n\t\t{}\n",
                                    supported_formats.join(" ")
                                );

                                println!(
                                    "\t------------------------------------------------------\n"
                                );
                            }
                        }
                    };
                }
            }
        }
    }

    Ok(())
}

impl Open for AlsaSink {
    fn open(device: Option<String>, format: AudioFormat) -> Self {
        let name = match device.as_deref() {
            Some("?") => match list_compatible_devices() {
                Ok(_) => {
                    exit(0);
                }
                Err(e) => {
                    error!("{e}");
                    exit(1);
                }
            },
            Some(device) => device,
            None => "default",
        }
        .to_string();

        info!("Using AlsaSink with format: {format:?}");

        Self {
            pcm: None,
            format,
            device: name,
            buffer_size: 0,
            period_size: 0,
            period_buffer: vec![],
        }
    }
}

impl Sink for AlsaSink {
    fn start(&mut self) -> SinkResult<()> {
        if self.pcm.is_none() {
            self.open_device()?;
        }

        Ok(())
    }

    fn stop(&mut self) -> SinkResult<()> {
        if self.pcm.is_some() {
            // Zero fill the remainder of the period buffer and
            // write any leftover data before draining the actual PCM buffer.
            self.period_buffer.resize(self.period_buffer.capacity(), 0);
            self.write_buf()?;

            let pcm = self.pcm.take().ok_or(AlsaError::NotConnected)?;

            pcm.drain().map_err(AlsaError::DrainFailure)?;
        }

        Ok(())
    }

    sink_as_bytes!();
}

impl SinkAsBytes for AlsaSink {
    fn write_bytes(&mut self, data: &[u8]) -> SinkResult<()> {
        let mut start_index = 0;
        let data_len = data.len();
        let capacity = self.period_buffer.capacity();

        loop {
            let data_left = data_len - start_index;
            let space_left = capacity - self.period_buffer.len();
            let data_to_buffer = data_left.min(space_left);
            let end_index = start_index + data_to_buffer;

            self.period_buffer
                .extend_from_slice(&data[start_index..end_index]);

            if self.period_buffer.len() == capacity {
                self.write_buf()?;
            }

            if end_index == data_len {
                break Ok(());
            }

            start_index = end_index;
        }
    }
}

impl AlsaSink {
    pub const NAME: &'static str = "alsa";

    fn get_buffer_size(hwp: &HwParams) -> Frames {
        let mut buffer_size = 1;

        let min_buffer = hwp.clone().get_buffer_size_min().unwrap_or(1);

        let max_buffer = hwp
            .clone()
            .get_buffer_size_max()
            .unwrap_or(MAX_BUFFER)
            .min(MAX_BUFFER);

        let supported_buffer_range = min_buffer..=max_buffer;

        trace!("Supported Buffer Range in Frames: {supported_buffer_range:?}");

        if supported_buffer_range.contains(&OPTIMAL_BUFFER)
            && hwp
                .clone()
                .set_buffer_size_near(OPTIMAL_BUFFER)
                .unwrap_or(0)
                == OPTIMAL_BUFFER
        {
            buffer_size = OPTIMAL_BUFFER;

            trace!("The Optimal Buffer Size ({OPTIMAL_BUFFER}) is in Range and Supported");
        } else {
            let supported_buffer_sizes: Vec<Frames> = supported_buffer_range
                .into_iter()
                .filter(|buffer_size| {
                    hwp.clone().set_buffer_size_near(*buffer_size).unwrap_or(0) == *buffer_size
                })
                .collect();

            trace!("Supported Buffer Sizes: {supported_buffer_sizes:#?}");

            let closest_buffer_size = supported_buffer_sizes
                .iter()
                .min_by_key(|x| x.abs_diff(OPTIMAL_BUFFER))
                .unwrap_or(&OPTIMAL_BUFFER);

            trace!("Closest Buffer Size to Optimal ({OPTIMAL_BUFFER}): {closest_buffer_size}");

            if hwp
                .clone()
                .set_buffer_size_near(*closest_buffer_size)
                .unwrap_or(0)
                == *closest_buffer_size
            {
                buffer_size = *closest_buffer_size;

                trace!("Buffer Size in Frames: {buffer_size}");
            } else {
                trace!("Error setting Buffer Size, falling back to the device's defaults");
            }
        }

        buffer_size
    }

    fn get_period_size(hwp: &HwParams, buffer_size: Frames) -> Frames {
        let optimal_period_size = buffer_size / OPTIMAL_PERIODS;
        // There must always be at least 2 periods per buffer.
        let max_period_size = buffer_size / 2;
        let mut period_size = 1;

        let min_period = hwp.clone().get_period_size_min().unwrap_or(1);

        let max_period = hwp
            .clone()
            .get_period_size_max()
            .unwrap_or(max_period_size)
            .min(max_period_size);

        let supported_period_range = min_period..=max_period;

        trace!("Supported Period Range in Frames: {supported_period_range:?}");

        if supported_period_range.contains(&optimal_period_size)
            && hwp
                .clone()
                .set_period_size_near(optimal_period_size, ValueOr::Nearest)
                .unwrap_or(0)
                == optimal_period_size
        {
            period_size = optimal_period_size;

            trace!("The Optimal Period Size ({optimal_period_size}) is in Range and Supported");
        } else {
            let supported_period_sizes: Vec<Frames> = supported_period_range
                .into_iter()
                .filter(|period_size| {
                    hwp.clone()
                        .set_period_size_near(*period_size, ValueOr::Nearest)
                        .unwrap_or(0)
                        == *period_size
                })
                .collect();

            trace!("Supported Period Sizes: {supported_period_sizes:#?}");

            let closest_period_size = supported_period_sizes
                .iter()
                .min_by_key(|x| x.abs_diff(optimal_period_size))
                .unwrap_or(&optimal_period_size);

            trace!("Closest Period Size to Optimal ({optimal_period_size}): {closest_period_size}");

            if hwp
                .clone()
                .set_period_size_near(*closest_period_size, ValueOr::Nearest)
                .unwrap_or(0)
                == *closest_period_size
            {
                period_size = *closest_period_size;

                trace!("Period Size in Frames: {period_size}");
            } else {
                trace!("Error setting Period Size, falling back to the device's defaults");
            }
        }

        period_size
    }

    fn open_device(&mut self) -> SinkResult<()> {
        let pcm = PCM::new(&self.device, Direction::Playback, false).map_err(|e| {
            AlsaError::PcmSetUp {
                device: self.device.clone(),
                e,
            }
        })?;

        let bytes_per_period = {
            let hwp = HwParams::any(&pcm).map_err(AlsaError::HwParams)?;

            hwp.set_access(Access::RWInterleaved).map_err(|e| {
                AlsaError::UnsupportedAccessType {
                    device: self.device.clone(),
                    e,
                }
            })?;

            let alsa_format = Format::from(self.format);

            hwp.set_format(alsa_format)
                .map_err(|e| AlsaError::UnsupportedFormat {
                    device: self.device.clone(),
                    alsa_format,
                    format: self.format,
                    e,
                })?;

            hwp.set_rate(SAMPLE_RATE, ValueOr::Nearest).map_err(|e| {
                AlsaError::UnsupportedSampleRate {
                    device: self.device.clone(),
                    samplerate: SAMPLE_RATE,
                    e,
                }
            })?;

            hwp.set_channels(NUM_CHANNELS as u32).map_err(|e| {
                AlsaError::UnsupportedChannelCount {
                    device: self.device.clone(),
                    channel_count: NUM_CHANNELS,
                    e,
                }
            })?;

            if self.buffer_size != 0 && self.period_size != 0 {
                // Use the cached buffer and period sizes to avoid
                // recalculating them.
                hwp.set_buffer_size_near(self.buffer_size)
                    .map_err(AlsaError::HwParams)?;

                hwp.set_period_size_near(self.period_size, ValueOr::Nearest)
                    .map_err(AlsaError::HwParams)?;

                pcm.hw_params(&hwp).map_err(AlsaError::Pcm)?;
            } else {
                // The initial opening of the card.
                // Calculate a buffer and period size as close
                // to optimal as possible.

                // hwp continuity is very important.
                let hwp_clone = hwp.clone();

                let buffer_size = Self::get_buffer_size(&hwp_clone);
                let mut period_size = 1;

                if buffer_size > 1 {
                    hwp_clone
                        .set_buffer_size_near(buffer_size)
                        .map_err(AlsaError::HwParams)?;

                    period_size = Self::get_period_size(&hwp_clone, buffer_size);

                    if period_size > 1 {
                        hwp_clone
                            .set_period_size_near(period_size, ValueOr::Nearest)
                            .map_err(AlsaError::HwParams)?;
                    }
                }

                if buffer_size > 1 && period_size > 1 {
                    pcm.hw_params(&hwp_clone).map_err(AlsaError::Pcm)?;
                } else {
                    pcm.hw_params(&hwp).map_err(AlsaError::Pcm)?;
                }
            }

            let hwp = pcm.hw_params_current().map_err(AlsaError::Pcm)?;

            // Don't assume we got what we wanted. Ask to make sure.
            self.buffer_size = hwp.get_buffer_size().map_err(AlsaError::HwParams)?;

            self.period_size = hwp.get_period_size().map_err(AlsaError::HwParams)?;

            let swp = pcm.sw_params_current().map_err(AlsaError::Pcm)?;

            swp.set_start_threshold(self.buffer_size - self.period_size)
                .map_err(AlsaError::SwParams)?;

            pcm.sw_params(&swp).map_err(AlsaError::Pcm)?;

            // Let ALSA do the math for us.
            pcm.frames_to_bytes(self.period_size) as usize
        };

        self.pcm = Some(pcm);

        if self.period_buffer.capacity() != bytes_per_period {
            trace!("Period Buffer size in bytes: {bytes_per_period}");

            self.period_buffer = Vec::with_capacity(bytes_per_period);
        }

        Ok(())
    }

    fn write_buf(&mut self) -> SinkResult<()> {
        if self.pcm.is_some() {
            let pcm = self.pcm.as_mut().ok_or(AlsaError::NotConnected)?;

            if let Err(e) = pcm.io_bytes().writei(&self.period_buffer) {
                // Capture and log the original error as a warning, and then try to recover.
                // If recovery fails then forward that error back to player.
                warn!("Error writing from AlsaSink buffer to PCM, trying to recover, {e}");

                pcm.try_recover(e, false).map_err(AlsaError::OnWrite)?;
            }
        }

        self.period_buffer.clear();

        Ok(())
    }
}
