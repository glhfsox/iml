### Observations: training_dataset_v2

- **639 clips**, **259 recording IDs**, and **45 inferred/metadata speaker IDs**. Identifiers use documented filename conventions.
- Allowed-class clips cover **52 recording IDs**, including **6** in validation. Correlated excerpts mean clip count overstates independent evidence.
- **0 recording IDs cross splits**. There are **0 exact waveform duplicate groups**, with **0** crossing splits. These checks do not establish absence of acoustic near duplicates or shared sessions.
- **33 clips** contain int16 PCM-limit samples. This does not prove analog clipping or rule out other distortion.
- Median RMS spans **10.2 dB** across displayed groups (M: -26.5 dBFS; T: -16.3 dBFS). Voice and recording conditions may both contribute.
- Features for all **639 clips** use the shared loader and normalization, with shape **80 × 301**. Normalization changes feature scale; it does not denoise waveforms.
- Dataset fingerprint unchanged: `10ac1fb033e3b1845bc52ea2e2229c2982ad11f0`. No waveform, class label, or split was modified.
