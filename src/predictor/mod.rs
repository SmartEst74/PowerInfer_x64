//! Tiny MLP predictor for hot neuron classification
//!
//! The predictor takes top-k activations from previous layer and predicts
//! which neurons in the next layer will be hot. It's a small neural network
//! (~50K parameters) trained offline.

use ndarray::{Array1, Array2, Array3};
use std::path::Path;

/// Configuration for the predictor network
#[derive(Debug, Clone)]
pub struct PredictorConfig {
    pub input_dim: usize,      // top-k activations (e.g., 256)
    pub hidden_dim: usize,     // e.g., 512
    pub output_dim: usize,     // neuron blocks (e.g., 128 blocks × 64 neurons = 8192)
    pub n_layers: usize,       // 2
}

/// The predictor model: MLP with ReLU activations
pub struct Predictor {
    config: PredictorConfig,
    // Layer weights: (output_dim, input_dim) for linear1, etc.
    weights1: Array2<f32>,
    bias1: Array1<f32>,
    weights2: Array2<f32>,
    bias2: Array1<f32>,
}

impl Predictor {
    /// Load trained predictor from embedded weights
    pub fn from_embedded() -> Self {
        // In real implementation, weights are compiled into binary via include_bytes!
        // For now, return dummy model
        let config = PredictorConfig {
            input_dim: 256,
            hidden_dim: 512,
            output_dim: 128,  // 128 blocks
            n_layers: 2,
        };
        
        // Placeholder: these would be actual trained weights
        let weights1 = Array2::zeros((config.hidden_dim, config.input_dim));
        let bias1 = Array1::zeros(config.hidden_dim);
        let weights2 = Array2::zeros((config.output_dim, config.hidden_dim));
        let bias2 = Array1::zeros(config.output_dim);
        
        Self { config, weights1, bias1, weights2, bias2 }
    }

    /// Run forward pass: predict hot blocks from activations
    /// 
    /// # Arguments
    /// - `activations`: top-k activation values from previous layer (normalized)
    /// 
    /// # Returns
    /// Binary probability vector over blocks (sigmoid)
    pub fn predict(&self, activations: &[f32]) -> Array1<f32> {
        if activations.len() != self.config.input_dim {
            eprintln!("Warning: predictor input dim mismatch, expected {}, got {}", 
                self.config.input_dim, activations.len());
            return Array1::zeros(self.config.output_dim);
        }
        
        let x = Array1::from_vec(activations.to_vec());
        
        // Layer 1: x @ W1.T + b1, then ReLU
        let mut h = self.weights1.dot(&x) + &self.bias1;
        h.map_inplace(|v| *v = v.max(0.0));
        
        // Layer 2: h @ W2.T + b2, then sigmoid
        let mut logits = self.weights2.dot(&h) + &self.bias2;
        logits.map_inplace(|v| *v = 1.0 / (1.0 + (-*v).exp()));
        
        logits
    }

    /// Get thresholded prediction at given confidence level
    pub fn predict_hot(&self, activations: &[f32], threshold: f32) -> Vec<usize> {
        let probs = self.predict(activations);
        probs.iter()
            .enumerate()
            .filter(|(_, p)| **p >= threshold)
            .map(|(i, _)| i)
            .collect()
    }
}

/// Training utilities (for offline training of predictor)
pub mod train {
    use super::*;
    use rand::seq::SliceRandom;
    
    /// Training dataset
    pub struct PredictorDataset {
        pub samples: Vec<(Array1<f32>, Vec<usize>)>, // (activations, hot_blocks)
    }
    
    /// Train predictor from profiler data
    /// 
    /// # Arguments
    /// - `profiler_output`: JSONL of NeuronStats from `powerinfer-profile`
    /// - `block_size`: Number of neurons per block (e.g., 64)
    /// 
    /// # Returns
    /// Trained Predictor
    pub fn train_from_profiler<P: AsRef<Path>>(
        _profiler_output: P,
        _block_size: usize,
    ) -> Result<Predictor, Box<dyn std::error::Error>> {
        // 1. Load profiler data and convert to (activation, hot_label) pairs
        // 2. For each token position:
        //    - Extract top-256 activations from previous layer
        //    - Build binary label vector for current layer blocks
        // 3. Train MLP with binary cross-entropy loss
        // 4. Validate on held-out set
        // 5. Serialize weights
        
        // Placeholder - full implementation in later phase
        Ok(Predictor::from_embedded())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_predictor_forward() {
        let pred = Predictor::from_embedded();
        let dummy_activations = vec![0.1f32; 256];
        let probs = pred.predict(&dummy_activations);
        assert_eq!(probs.len(), 128);
        // All probs should be in [0,1]
        assert!(probs.iter().all(|p| (0.0..=1.0).contains(p)));
    }
    
    #[test]
    fn test_predict_hot() {
        let pred = Predictor::from_embedded();
        // Since weights are zeros, all probs = 0.5
        let dummy = vec![0.0; 256];
        let hot = pred.predict_hot(&dummy, 0.5);
        assert_eq!(hot.len(), 128); // all blocks >= 0.5
        let hot_lower = pred.predict_hot(&dummy, 0.6);
        assert!(hot_lower.is_empty());
    }
}
