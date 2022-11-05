use super::{Open, Sink, SinkAsBytes, SinkError, SinkResult};
use crate::config::AudioFormat;
use crate::convert::Converter;
use crate::decoder::AudioPacket;
use crate::{NUM_CHANNELS, SAMPLE_RATE};
use libpulse_binding::{self as pulse, def::BufferAttr, error::PAErr, stream::Direction};
use libpulse_simple_binding::Simple;
use std::env;
use thiserror::Error;

#[derive(Debug, Error)]
enum PulseError {
    #[error("<PulseAudioSink> Unsupported Pulseaudio Sample Spec, Format {pulse_format:?} ({format:?}), Channels {channels}, Rate {rate}")]
    InvalidSampleSpec {
        pulse_format: pulse::sample::Format,
        format: AudioFormat,
        channels: u8,
        rate: u32,
    },

    #[error("<PulseAudioSink> {0}")]
    ConnectionRefused(PAErr),

    #[error("<PulseAudioSink> Failed to Drain Pulseaudio Buffer, {0}")]
    DrainFailure(PAErr),

    #[error("<PulseAudioSink>")]
    NotConnected,

    #[error("<PulseAudioSink> {0}")]
    OnWrite(PAErr),
}

impl From<PulseError> for SinkError {
    fn from(e: PulseError) -> SinkError {
        use PulseError::*;
        let es = e.to_string();
        match e {
            DrainFailure(_) | OnWrite(_) => SinkError::OnWrite(es),
            ConnectionRefused(_) => SinkError::ConnectionRefused(es),
            NotConnected => SinkError::NotConnected(es),
            InvalidSampleSpec { .. } => SinkError::InvalidParams(es),
        }
    }
}

pub struct PulseAudioSink {
    sink: Option<Simple>,
    device: Option<String>,
    app_name: String,
    stream_desc: String,
    format: AudioFormat,
    sample_spec: pulse::sample::Spec,
    buffer_attr: BufferAttr,
    buffer: Vec<u8>,
}

impl Open for PulseAudioSink {
    fn open(device: Option<String>, format: AudioFormat) -> Self {
        let app_name = env::var("PULSE_PROP_application.name").unwrap_or_default();
        let stream_desc = env::var("PULSE_PROP_stream.description").unwrap_or_default();

        let mut actual_format = format;

        if actual_format == AudioFormat::F64 {
            warn!("PulseAudio currently does not support F64 output");
            actual_format = AudioFormat::F32;
        }

        info!("Using PulseAudioSink with format: {actual_format:?}");

        let pulse_format = match actual_format {
            AudioFormat::F32 => pulse::sample::Format::FLOAT32NE,
            AudioFormat::S32 => pulse::sample::Format::S32NE,
            AudioFormat::S24 => pulse::sample::Format::S24_32NE,
            AudioFormat::S24_3 => pulse::sample::Format::S24NE,
            AudioFormat::S16 => pulse::sample::Format::S16NE,
            _ => unreachable!(),
        };

        let sample_spec = pulse::sample::Spec {
            format: pulse_format,
            channels: NUM_CHANNELS,
            rate: SAMPLE_RATE,
        };

        // 1000ms / 1 sec
        let maxlength = SAMPLE_RATE * NUM_CHANNELS as u32 * actual_format.size() as u32;
        // 100ms
        let tlength = maxlength / 10;
        // 500ms
        let prebuf = tlength / 2;

        let buffer_attr = BufferAttr {
            maxlength,
            tlength,
            prebuf,
            minreq: u32::MAX,
            fragsize: u32::MAX,
        };

        let buffer = Vec::with_capacity(tlength as usize);

        Self {
            sink: None,
            device,
            app_name,
            stream_desc,
            format: actual_format,
            sample_spec,
            buffer_attr,
            buffer,
        }
    }
}

impl Sink for PulseAudioSink {
    fn start(&mut self) -> SinkResult<()> {
        if self.sink.is_none() {
            if !self.sample_spec.is_valid() {
                let pulse_error = PulseError::InvalidSampleSpec {
                    pulse_format: self.sample_spec.format,
                    format: self.format,
                    channels: NUM_CHANNELS,
                    rate: SAMPLE_RATE,
                };

                return Err(SinkError::from(pulse_error));
            }

            let sink = Simple::new(
                None,                    // Use the default server.
                &self.app_name,          // Our application's name.
                Direction::Playback,     // Direction.
                self.device.as_deref(),  // Our device (sink) name.
                &self.stream_desc,       // Description of our stream.
                &self.sample_spec,       // Our sample format.
                None,                    // Use default channel map.
                Some(&self.buffer_attr), // Use our buffering attributes.
            )
            .map_err(PulseError::ConnectionRefused)?;

            self.sink = Some(sink);
        }

        Ok(())
    }

    fn stop(&mut self) -> SinkResult<()> {
        if self.sink.is_some() {
            // Zero fill the remainder of the buffer and
            // write any leftover data before draining the actual PulseAudio buffer.
            self.buffer.resize(self.buffer.capacity(), 0);
            self.write_buf()?;

            let sink = self.sink.take().ok_or(PulseError::NotConnected)?;

            sink.drain().map_err(PulseError::DrainFailure)?;
        }

        Ok(())
    }

    sink_as_bytes!();
}

impl SinkAsBytes for PulseAudioSink {
    fn write_bytes(&mut self, data: &[u8]) -> SinkResult<()> {
        let mut start_index = 0;
        let data_len = data.len();
        let capacity = self.buffer.capacity();

        loop {
            let data_left = data_len - start_index;
            let space_left = capacity - self.buffer.len();
            let data_to_buffer = data_left.min(space_left);
            let end_index = start_index + data_to_buffer;

            self.buffer.extend_from_slice(&data[start_index..end_index]);

            if self.buffer.len() == capacity {
                self.write_buf()?;
            }

            if end_index == data_len {
                break Ok(());
            }

            start_index = end_index;
        }
    }
}

impl PulseAudioSink {
    pub const NAME: &'static str = "pulseaudio";

    fn write_buf(&mut self) -> SinkResult<()> {
        if self.sink.is_some() {
            let sink = self.sink.as_mut().ok_or(PulseError::NotConnected)?;

            sink.write(&self.buffer).map_err(PulseError::OnWrite)?;
        }

        self.buffer.clear();

        Ok(())
    }
}
