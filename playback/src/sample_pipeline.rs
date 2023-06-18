use crate::{
    config::PlayerConfig, mixer::VolumeGetter, normaliser::Normaliser, player::NormalisationData,
    resampler::StereoInterleavedResampler,
};

pub struct SamplePipeline {
    resampler: StereoInterleavedResampler,
    normaliser: Normaliser,
}

impl SamplePipeline {
    pub fn new(config: &PlayerConfig, volume_getter: Box<dyn VolumeGetter>) -> Self {
        let resampler =
            StereoInterleavedResampler::new(config.sample_rate, config.interpolation_quality);

        let normaliser = Normaliser::new(config, volume_getter);

        Self {
            resampler,
            normaliser,
        }
    }

    pub fn stop(&mut self) {
        self.resampler.stop();
        self.normaliser.stop();
    }

    pub fn set_normalisation_factor(
        &mut self,
        auto_normalise_as_album: bool,
        data: NormalisationData,
    ) {
        self.normaliser.set_factor(auto_normalise_as_album, data);
    }

    pub fn process(&mut self, samples: &[f64]) -> Option<Vec<f64>> {
        self.resampler
            .process(samples)
            .map(|processed_samples| self.normaliser.normalise(&processed_samples))
    }
}
