//! Activation profiler for neuron hotness analysis
//!
//! Runs the model forward pass on diverse inputs and tracks which FFN neurons
//! consistently activate. Produces a "hot neuron index" used to decide which
//! neurons stay on GPU vs CPU.

use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Activation statistics for a single neuron
#[derive(Debug, Clone)]
pub struct NeuronStats {
    /// How many times this neuron was evaluated
    pub samples: usize,
    /// Sum of activation magnitudes (for computing mean)
    pub magnitude_sum: f64,
    /// Number of times activation exceeded the hotness threshold
    pub hot_count: usize,
    /// Maximum activation magnitude seen
    pub max_magnitude: f32,
}

impl NeuronStats {
    fn new() -> Self {
        Self {
            samples: 0,
            magnitude_sum: 0.0,
            hot_count: 0,
            max_magnitude: 0.0,
        }
    }

    /// Mean activation magnitude
    pub fn mean_magnitude(&self) -> f64 {
        if self.samples == 0 {
            0.0
        } else {
            self.magnitude_sum / self.samples as f64
        }
    }

    /// Fraction of inputs where this neuron was "hot" (above threshold)
    pub fn hotness_ratio(&self) -> f64 {
        if self.samples == 0 {
            0.0
        } else {
            self.hot_count as f64 / self.samples as f64
        }
    }
}

/// Per-layer activation profile
#[derive(Debug, Clone)]
pub struct LayerProfile {
    /// Layer index
    pub layer_idx: usize,
    /// Per-neuron stats [n_ff]
    pub neurons: Vec<NeuronStats>,
}

/// Complete activation profile across all layers
#[derive(Debug, Clone)]
pub struct ActivationProfile {
    /// Per-layer profiles
    pub layers: Vec<LayerProfile>,
    /// Number of input samples processed
    pub total_samples: usize,
    /// Hotness threshold used
    pub threshold: f32,
}

impl ActivationProfile {
    /// Create a new empty profile
    pub fn new(n_layers: usize, n_ff: usize, threshold: f32) -> Self {
        Self::new_with_dims(&vec![n_ff; n_layers], threshold)
    }

    /// Create a new empty profile with per-layer activation widths.
    pub fn new_with_dims(layer_dims: &[usize], threshold: f32) -> Self {
        let layers = layer_dims
            .iter()
            .enumerate()
            .map(|(layer_idx, &width)| LayerProfile {
                layer_idx,
                neurons: (0..width).map(|_| NeuronStats::new()).collect(),
            })
            .collect();

        Self {
            layers,
            total_samples: 0,
            threshold,
        }
    }

    /// Record activations for one layer
    pub fn record_layer(&mut self, layer_idx: usize, gate_activations: &[f32]) {
        let Some(layer) = self.layers.get_mut(layer_idx) else {
            return;
        };

        for (stats, &act) in layer.neurons.iter_mut().zip(gate_activations.iter()) {
            let mag = act.abs();
            stats.samples += 1;
            stats.magnitude_sum += mag as f64;
            stats.max_magnitude = stats.max_magnitude.max(mag);
            if mag > self.threshold {
                stats.hot_count += 1;
            }
        }
    }

    /// Increment sample counter
    pub fn finish_sample(&mut self) {
        self.total_samples += 1;
    }

    /// Get hot neuron indices per layer (sorted by hotness ratio, descending)
    pub fn hot_neurons(&self, min_hotness: f64) -> Vec<Vec<usize>> {
        self.layers
            .iter()
            .map(|layer| {
                let mut hot: Vec<usize> = (0..layer.neurons.len())
                    .filter(|&i| layer.neurons[i].hotness_ratio() >= min_hotness)
                    .collect();
                // Sort by hotness ratio descending
                hot.sort_by(|&a, &b| {
                    layer.neurons[b]
                        .hotness_ratio()
                        .partial_cmp(&layer.neurons[a].hotness_ratio())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                hot
            })
            .collect()
    }

    /// Get hot neuron indices as a flat boolean mask per layer
    pub fn hot_mask(&self, min_hotness: f64) -> Vec<Vec<bool>> {
        self.layers
            .iter()
            .map(|layer| {
                layer
                    .neurons
                    .iter()
                    .map(|n| n.hotness_ratio() >= min_hotness)
                    .collect()
            })
            .collect()
    }

    /// Summary statistics
    pub fn summary(&self) -> ProfileSummary {
        let mut total_neurons = 0;
        let mut total_hot = 0;
        let mut hot_ratios = Vec::new();

        for layer in &self.layers {
            for neuron in &layer.neurons {
                total_neurons += 1;
                let ratio = neuron.hotness_ratio();
                if ratio >= 0.5 {
                    total_hot += 1;
                }
                hot_ratios.push(ratio);
            }
        }

        hot_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let hot_fraction = if total_neurons == 0 {
            0.0
        } else {
            total_hot as f64 / total_neurons as f64
        };

        ProfileSummary {
            total_samples: self.total_samples,
            total_neurons,
            hot_neurons_at_50pct: total_hot,
            hot_fraction,
            median_hotness: hot_ratios.get(total_neurons / 2).copied().unwrap_or(0.0),
            p90_hotness: hot_ratios
                .get((total_neurons as f64 * 0.9) as usize)
                .copied()
                .unwrap_or(0.0),
        }
    }

    /// Export as JSON (hot neuron index for runtime use)
    pub fn export_hot_index(&self, min_hotness: f64) -> HotNeuronIndex {
        HotNeuronIndex {
            version: 1,
            threshold: self.threshold,
            min_hotness,
            total_samples: self.total_samples,
            layers: self
                .layers
                .iter()
                .map(|layer| {
                    let hot_indices: Vec<usize> = (0..layer.neurons.len())
                        .filter(|&i| layer.neurons[i].hotness_ratio() >= min_hotness)
                        .collect();
                    HotLayer {
                        layer_idx: layer.layer_idx,
                        hot_indices,
                        n_ff: layer.neurons.len(),
                    }
                })
                .collect(),
        }
    }
}

/// Summary of activation profiling
#[derive(Debug)]
pub struct ProfileSummary {
    pub total_samples: usize,
    pub total_neurons: usize,
    pub hot_neurons_at_50pct: usize,
    pub hot_fraction: f64,
    pub median_hotness: f64,
    pub p90_hotness: f64,
}

impl std::fmt::Display for ProfileSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Activation Profile Summary")?;
        writeln!(f, "  Samples:       {}", self.total_samples)?;
        writeln!(f, "  Total neurons: {}", self.total_neurons)?;
        writeln!(
            f,
            "  Hot (>50%):    {} ({:.1}%)",
            self.hot_neurons_at_50pct,
            self.hot_fraction * 100.0
        )?;
        writeln!(f, "  Median hotness: {:.3}", self.median_hotness)?;
        writeln!(f, "  P90 hotness:    {:.3}", self.p90_hotness)?;
        Ok(())
    }
}

/// Serializable hot neuron index for runtime use
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HotNeuronIndex {
    pub version: u32,
    pub threshold: f32,
    pub min_hotness: f64,
    pub total_samples: usize,
    pub layers: Vec<HotLayer>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HotLayer {
    pub layer_idx: usize,
    pub hot_indices: Vec<usize>,
    pub n_ff: usize,
}

impl HotNeuronIndex {
    /// Save to JSON file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<()> {
        if let Some(parent) = path.as_ref().parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load from JSON file
    pub fn load<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let index: Self = serde_json::from_str(&json)?;
        Ok(index)
    }

    /// GPU memory needed for hot neuron weights (bytes)
    pub fn gpu_memory_estimate(&self, head_dim: usize) -> usize {
        let mut total = 0;
        for layer in &self.layers {
            // Hot gate weights: hot_indices.len() * head_dim * 4 bytes (f32)
            total += layer.hot_indices.len() * head_dim * 4;
            // Hot up weights
            total += layer.hot_indices.len() * head_dim * 4;
            // Hot down weights: head_dim * hot_indices.len() * 4 bytes
            total += head_dim * layer.hot_indices.len() * 4;
        }
        total
    }
}

/// Thread-safe activation recorder for hooking into the forward pass
#[derive(Clone)]
pub struct ActivationRecorder {
    inner: Arc<Mutex<ActivationProfile>>,
}

impl ActivationRecorder {
    pub fn new(profile: ActivationProfile) -> Self {
        Self {
            inner: Arc::new(Mutex::new(profile)),
        }
    }

    /// Record gate activations for a layer
    pub fn record(&self, layer_idx: usize, gate_activations: &[f32]) {
        let mut profile = self.inner.lock().unwrap();
        profile.record_layer(layer_idx, gate_activations);
    }

    /// Finish one input sample
    pub fn finish_sample(&self) {
        let mut profile = self.inner.lock().unwrap();
        profile.finish_sample();
    }

    /// Get a snapshot of the current profile
    pub fn snapshot(&self) -> ActivationProfile {
        self.inner.lock().unwrap().clone()
    }
}

/// Result of a profiling run over a prompt set.
#[derive(Debug, Clone)]
pub struct ProfilingRun {
    pub profile: ActivationProfile,
    pub hot_index: HotNeuronIndex,
    pub prompts_processed: usize,
}

/// Built-in smoke prompts used when no prompt files are provided.
pub fn default_profile_prompts() -> Vec<String> {
    vec![
        "The capital of France is".to_string(),
        "Rust is a systems programming language that".to_string(),
        "Explain in one sentence what a GPU is.".to_string(),
        "List three uses for a local language model.".to_string(),
    ]
}

/// Load prompts from one or more newline-delimited text files.
pub fn load_prompts_from_files(prompt_files: &[PathBuf]) -> Result<Vec<String>> {
    let mut prompts = Vec::new();

    for prompt_file in prompt_files {
        let contents = std::fs::read_to_string(prompt_file)?;
        prompts.extend(
            contents
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(ToOwned::to_owned),
        );
    }

    Ok(prompts)
}

/// Run activation profiling over a set of prompts using the provided inference context.
pub fn run_activation_profiling(
    ctx: &mut crate::model::InferenceContext,
    prompts: &[String],
    layer_limit: Option<usize>,
    threshold: f32,
    min_hotness: f64,
    max_samples: usize,
) -> Result<ProfilingRun> {
    if prompts.is_empty() {
        return Err(anyhow!("no prompts provided for activation profiling"));
    }

    let layer_dims = ctx.profiling_layer_dims(layer_limit);
    if layer_dims.is_empty() {
        return Err(anyhow!("no layers available for activation profiling"));
    }

    let recorder = ActivationRecorder::new(ActivationProfile::new_with_dims(&layer_dims, threshold));
    ctx.set_activation_recorder(recorder.clone());

    let mut prompts_processed = 0;
    let sample_budget = max_samples.max(1);

    for prompt in prompts {
        if prompts_processed >= sample_budget {
            break;
        }

        let input_ids = ctx.tokenizer().encode(prompt);
        if input_ids.is_empty() {
            continue;
        }

        ctx.reset();
        ctx.forward(&input_ids)?;
        recorder.finish_sample();
        prompts_processed += 1;
    }

    ctx.clear_activation_recorder();

    if prompts_processed == 0 {
        return Err(anyhow!("no non-empty prompts were processed during activation profiling"));
    }

    let profile = recorder.snapshot();
    let hot_index = profile.export_hot_index(min_hotness);

    Ok(ProfilingRun {
        profile,
        hot_index,
        prompts_processed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuron_stats() {
        let mut profile = ActivationProfile::new(2, 4, 0.5);

        // Layer 0: neuron 0 and 1 are hot, 2 and 3 are cold
        profile.record_layer(0, &[1.0, 0.8, 0.1, 0.2]);
        profile.record_layer(0, &[1.2, 0.9, 0.0, 0.3]);
        profile.record_layer(0, &[0.9, 0.7, 0.2, 0.1]);
        profile.finish_sample();
        profile.finish_sample();
        profile.finish_sample();

        // Neuron 0: always hot (magnitude > 0.5)
        assert_eq!(profile.layers[0].neurons[0].hot_count, 3);
        assert_eq!(profile.layers[0].neurons[0].hotness_ratio(), 1.0);

        // Neuron 2: never hot (magnitude < 0.5)
        assert_eq!(profile.layers[0].neurons[2].hot_count, 0);
        assert_eq!(profile.layers[0].neurons[2].hotness_ratio(), 0.0);

        // Neuron 1: always hot
        assert_eq!(profile.layers[0].neurons[1].hotness_ratio(), 1.0);

        // Neuron 3: never hot
        assert_eq!(profile.layers[0].neurons[3].hotness_ratio(), 0.0);
    }

    #[test]
    fn test_hot_neurons() {
        let mut profile = ActivationProfile::new(1, 6, 0.5);
        profile.record_layer(0, &[1.0, 0.0, 0.9, 0.1, 0.8, 0.2]);
        profile.record_layer(0, &[1.1, 0.1, 0.8, 0.0, 0.9, 0.3]);
        profile.finish_sample();
        profile.finish_sample();

        let hot = profile.hot_neurons(0.9);
        // Neurons 0, 2, 4 should be hot (100% hotness)
        assert_eq!(hot[0].len(), 3);
        assert!(hot[0].contains(&0));
        assert!(hot[0].contains(&2));
        assert!(hot[0].contains(&4));
    }

    #[test]
    fn test_hot_mask() {
        let mut profile = ActivationProfile::new(1, 4, 0.5);
        profile.record_layer(0, &[1.0, 0.0, 0.8, 0.1]);
        profile.finish_sample();

        let mask = profile.hot_mask(0.5);
        assert_eq!(mask[0], vec![true, false, true, false]);
    }

    #[test]
    fn test_export_index() {
        let mut profile = ActivationProfile::new(2, 4, 0.5);
        profile.record_layer(0, &[1.0, 0.0, 0.8, 0.1]);
        profile.record_layer(1, &[0.0, 1.0, 0.1, 0.9]);
        profile.finish_sample();

        let index = profile.export_hot_index(0.5);
        assert_eq!(index.layers[0].hot_indices, vec![0, 2]);
        assert_eq!(index.layers[1].hot_indices, vec![1, 3]);
    }

    #[test]
    fn test_summary() {
        let mut profile = ActivationProfile::new(1, 4, 0.5);
        for _ in 0..10 {
            profile.record_layer(0, &[1.0, 0.0, 0.8, 0.1]);
            profile.finish_sample();
        }

        let summary = profile.summary();
        assert_eq!(summary.total_samples, 10);
        assert_eq!(summary.total_neurons, 4);
        assert_eq!(summary.hot_neurons_at_50pct, 2); // neurons 0 and 2
        assert!((summary.hot_fraction - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_gpu_memory_estimate() {
        let index = HotNeuronIndex {
            version: 1,
            threshold: 0.5,
            min_hotness: 0.5,
            total_samples: 100,
            layers: vec![HotLayer {
                layer_idx: 0,
                hot_indices: vec![0, 1, 2], // 3 hot neurons
                n_ff: 10,
            }],
        };
        // 3 hot * 128 * 4 (gate) + 3 * 128 * 4 (up) + 128 * 3 * 4 (down) = 4608
        let mem = index.gpu_memory_estimate(128);
        assert_eq!(mem, 4608);
    }

    #[test]
    fn test_variable_layer_width_profile() {
        let mut profile = ActivationProfile::new_with_dims(&[2, 4], 0.5);
        profile.record_layer(0, &[1.0, 0.1, 9.9]);
        profile.record_layer(1, &[0.1, 0.9, 0.2, 0.8, 9.9]);
        profile.finish_sample();

        assert_eq!(profile.layers[0].neurons.len(), 2);
        assert_eq!(profile.layers[1].neurons.len(), 4);
        assert_eq!(profile.layers[0].neurons[0].hot_count, 1);
        assert_eq!(profile.layers[1].neurons[1].hot_count, 1);
        assert_eq!(profile.layers[1].neurons[3].hot_count, 1);
    }
}
