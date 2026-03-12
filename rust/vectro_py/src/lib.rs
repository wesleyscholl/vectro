use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use ndarray::{Array1, Array2};
use vectro_lib::{Embedding, EmbeddingDataset};
use vectro_lib::search::{SearchIndex, QuantizedIndex};
use std::collections::HashMap;

/// Python wrapper for Embedding
#[pyclass]
struct PyEmbedding {
    inner: Embedding,
}

#[pymethods]
impl PyEmbedding {
    #[new]
    fn new(id: String, vector: PyReadonlyArray1<f32>) -> Self {
        let vector_vec = vector.as_array().to_vec();
        Self {
            inner: Embedding::new(id, vector_vec),
        }
    }

    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }

    #[getter]
    fn vector(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f32>>> {
        let array = Array1::from(self.inner.vector.clone());
        Ok(array.into_pyarray(py).to_owned())
    }

    fn __repr__(&self) -> String {
        format!("PyEmbedding(id='{}', dim={})", self.inner.id, self.inner.vector.len())
    }
}

/// Python wrapper for EmbeddingDataset
#[pyclass]
struct PyEmbeddingDataset {
    inner: EmbeddingDataset,
}

#[pymethods]
impl PyEmbeddingDataset {
    #[new]
    fn new() -> Self {
        Self {
            inner: EmbeddingDataset::new(),
        }
    }

    fn add_embedding(&mut self, embedding: &PyEmbedding) {
        self.inner.add(embedding.inner.clone());
    }

    fn add_vector(&mut self, id: String, vector: PyReadonlyArray1<f32>) {
        let vector_vec = vector.as_array().to_vec();
        let embedding = Embedding::new(id, vector_vec);
        self.inner.add(embedding);
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn get_embedding(&self, index: usize) -> Option<PyEmbedding> {
        self.inner.embeddings.get(index).map(|e| PyEmbedding { inner: e.clone() })
    }

    fn get_vectors(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f32>>> {
        if self.inner.is_empty() {
            return Ok(Array2::zeros((0, 0)).into_pyarray(py).to_owned());
        }
        
        let dim = self.inner.embeddings[0].vector.len();
        let mut array = Array2::zeros((self.inner.len(), dim));
        
        for (i, embedding) in self.inner.embeddings.iter().enumerate() {
            for (j, &value) in embedding.vector.iter().enumerate() {
                array[[i, j]] = value;
            }
        }
        
        Ok(array.into_pyarray(py).to_owned())
    }

    fn get_ids(&self) -> Vec<String> {
        self.inner.embeddings.iter().map(|e| e.id.clone()).collect()
    }

    fn __len__(&self) -> usize {
        self.len()
    }

    fn __repr__(&self) -> String {
        format!("PyEmbeddingDataset(size={})", self.inner.len())
    }
}

/// Python wrapper for SearchIndex
#[pyclass]
struct PySearchIndex {
    inner: SearchIndex,
    id_to_index: HashMap<String, usize>,
}

#[pymethods]
impl PySearchIndex {
    #[staticmethod]
    fn from_dataset(dataset: &PyEmbeddingDataset) -> PyResult<Self> {
        let index = SearchIndex::from_dataset(&dataset.inner.embeddings);
        
        // Build ID->index mapping
        let mut id_to_index = HashMap::new();
        for (idx, embedding) in dataset.inner.embeddings.iter().enumerate() {
            id_to_index.insert(embedding.id.clone(), idx);
        }
        
        Ok(Self { inner: index, id_to_index })
    }

    fn search_vector(&self, py: Python<'_>, query: PyReadonlyArray1<f32>, top_k: usize) -> PyResult<Py<PyTuple>> {
        let query_vec = query.as_array().to_vec();
        let results = self.inner.top_k(&query_vec, top_k);
        
        let mut indices = Vec::new();
        let mut similarities = Vec::new();
        
        // The results are (id, similarity) pairs, we need to convert to indices
        for (id, similarity) in results {
            // Find the index of this ID in the original dataset
            // Note: This is inefficient but works for the demo
            // In production, we'd want to store ID->index mapping
            if let Some(index) = self.find_id_index(id) {
                indices.push(index);
                similarities.push(similarity);
            }
        }
        
        let indices_array: &PyArray1<usize> = Array1::from(indices).into_pyarray(py);
        let similarities_array: &PyArray1<f32> = Array1::from(similarities).into_pyarray(py);
        
        Ok(PyTuple::new(py, &[indices_array.as_ref(), similarities_array.as_ref()]).into())
    }

    fn batch_search(&self, py: Python<'_>, queries: PyReadonlyArray2<f32>, top_k: usize) -> PyResult<Py<PyList>> {
        let queries_array = queries.as_array();
        let mut all_results = Vec::new();
        
        for query_row in queries_array.outer_iter() {
            let query_vec = query_row.to_vec();
            let results = self.inner.top_k(&query_vec, top_k);
            
            let mut indices = Vec::new();
            let mut similarities = Vec::new();
            
            for (id, similarity) in results {
                if let Some(index) = self.find_id_index(id) {
                    indices.push(index);
                    similarities.push(similarity);
                }
            }
            
            let indices_array: &PyArray1<usize> = Array1::from(indices).into_pyarray(py);
            let similarities_array: &PyArray1<f32> = Array1::from(similarities).into_pyarray(py);
            let result_tuple = PyTuple::new(py, &[indices_array.as_ref(), similarities_array.as_ref()]);
            
            all_results.push(result_tuple);
        }
        
        Ok(PyList::new(py, all_results).into())
    }

    fn __repr__(&self) -> String {
        // We can't access private fields, so use a simpler representation
        format!("PySearchIndex")
    }
}

impl PySearchIndex {
    fn find_id_index(&self, target_id: &str) -> Option<usize> {
        self.id_to_index.get(target_id).copied()
    }
}

/// Python wrapper for QuantizedIndex
#[pyclass]
struct PyQuantizedIndex {
    inner: QuantizedIndex,
    id_to_index: HashMap<String, usize>,
}

#[pymethods]
impl PyQuantizedIndex {
    #[staticmethod]
    fn from_dataset(dataset: &PyEmbeddingDataset) -> PyResult<Self> {
        let index = QuantizedIndex::from_dataset(&dataset.inner.embeddings);
        
        // Build ID->index mapping
        let mut id_to_index = HashMap::new();
        for (idx, embedding) in dataset.inner.embeddings.iter().enumerate() {
            id_to_index.insert(embedding.id.clone(), idx);
        }
        
        Ok(Self { inner: index, id_to_index })
    }

    fn search_vector(&self, py: Python<'_>, query: PyReadonlyArray1<f32>, top_k: usize) -> PyResult<Py<PyTuple>> {
        let query_vec = query.as_array().to_vec();
        let results = self.inner.top_k(&query_vec, top_k);
        
        let mut indices = Vec::new();
        let mut similarities = Vec::new();
        
        for (id, similarity) in results {
            if let Some(index) = self.find_id_index(id) {
                indices.push(index);
                similarities.push(similarity);
            }
        }
        
        let indices_array: &PyArray1<usize> = Array1::from(indices).into_pyarray(py);
        let similarities_array: &PyArray1<f32> = Array1::from(similarities).into_pyarray(py);
        
        Ok(PyTuple::new(py, &[indices_array.as_ref(), similarities_array.as_ref()]).into())
    }

    fn compression_ratio(&self) -> f32 {
        // Estimate compression ratio: f32 (4 bytes) vs u8 (1 byte) per dimension
        // Plus some overhead for quantization tables
        4.0 // Simplified estimate
    }

    fn memory_usage_bytes(&self) -> usize {
        // Simplified calculation since we can't access private fields
        // This would need proper getter methods on QuantizedIndex
        1024 // Placeholder
    }

    fn __repr__(&self) -> String {
        format!("PyQuantizedIndex(ratio={:.2}x)", self.compression_ratio())
    }
}

impl PyQuantizedIndex {
    fn find_id_index(&self, target_id: &str) -> Option<usize> {
        self.id_to_index.get(target_id).copied()
    }
}

/// Compression utilities
#[pyfunction]
fn compress_embeddings(py: Python<'_>, vectors: PyReadonlyArray2<f32>, ids: Option<Vec<String>>) -> PyResult<Py<PyTuple>> {
    let vectors_array = vectors.as_array();
    let mut dataset = EmbeddingDataset::new();
    
    for (i, vector_row) in vectors_array.outer_iter().enumerate() {
        let id = ids.as_ref().and_then(|ids| ids.get(i).cloned())
                   .unwrap_or_else(|| format!("vec_{}", i));
        let vector_vec = vector_row.to_vec();
        dataset.add(Embedding::new(id, vector_vec));
    }
    
    // Create both regular and quantized indices
    let search_index = SearchIndex::from_dataset(&dataset.embeddings);
    let quantized_index = QuantizedIndex::from_dataset(&dataset.embeddings);
    
    // Build ID->index mapping
    let mut id_to_index = HashMap::new();
    for (idx, embedding) in dataset.embeddings.iter().enumerate() {
        id_to_index.insert(embedding.id.clone(), idx);
    }
    
    let py_search_index = PySearchIndex { 
        inner: search_index, 
        id_to_index: id_to_index.clone()
    };
    let py_quantized_index = PyQuantizedIndex { 
        inner: quantized_index, 
        id_to_index
    };
    
    Ok(PyTuple::new(py, &[
        py_search_index.into_py(py),
        py_quantized_index.into_py(py)
    ]).into())
}

/// Quality analysis utilities
#[pyfunction]
fn analyze_compression_quality(
    original: PyReadonlyArray2<f32>,
    compressed_index: &PyQuantizedIndex,
    num_samples: Option<usize>
) -> PyResult<HashMap<String, f32>> {
    let samples = num_samples.unwrap_or(100);
    let original_array = original.as_array();
    let mut total_similarity = 0.0f32;
    let mut max_similarity = 0.0f32;
    let mut min_similarity = 1.0f32;
    
    let actual_samples = samples.min(original_array.nrows());
    
    for i in 0..actual_samples {
        let query = original_array.row(i).to_vec();
        let results = compressed_index.inner.top_k(&query, 1);
        
        if let Some((_, similarity)) = results.first() {
            total_similarity += similarity;
            max_similarity = max_similarity.max(*similarity);
            min_similarity = min_similarity.min(*similarity);
        }
    }
    
    let avg_similarity = total_similarity / actual_samples as f32;
    let compression_ratio = compressed_index.compression_ratio();
    
    let mut analysis = HashMap::new();
    analysis.insert("average_similarity".to_string(), avg_similarity);
    analysis.insert("max_similarity".to_string(), max_similarity);
    analysis.insert("min_similarity".to_string(), min_similarity);
    analysis.insert("compression_ratio".to_string(), compression_ratio);
    analysis.insert("memory_savings_percent".to_string(), (1.0 - 1.0/compression_ratio) * 100.0);
    analysis.insert("samples_analyzed".to_string(), actual_samples as f32);
    
    Ok(analysis)
}

/// Performance benchmarking utilities
#[pyfunction]
fn benchmark_search_performance(
    index: &PySearchIndex,
    queries: PyReadonlyArray2<f32>,
    top_k: usize,
    num_runs: Option<usize>
) -> PyResult<HashMap<String, f32>> {
    use std::time::Instant;
    
    let runs = num_runs.unwrap_or(10);
    let queries_array = queries.as_array();
    let mut total_time = 0.0;
    let mut successful_queries = 0;
    
    for _ in 0..runs {
        for query_row in queries_array.outer_iter() {
            let start = Instant::now();
            let query_vec = query_row.to_vec();
            let _results = index.inner.top_k(&query_vec, top_k);
            let duration = start.elapsed();
            total_time += duration.as_secs_f32() * 1000.0; // Convert to milliseconds
            successful_queries += 1;
        }
    }
    
    let avg_latency_ms = if successful_queries > 0 {
        total_time / successful_queries as f32
    } else {
        0.0
    };
    
    let queries_per_second = if avg_latency_ms > 0.0 {
        1000.0 / avg_latency_ms
    } else {
        0.0
    };
    
    let mut benchmark = HashMap::new();
    benchmark.insert("average_latency_ms".to_string(), avg_latency_ms);
    benchmark.insert("queries_per_second".to_string(), queries_per_second);
    benchmark.insert("successful_queries".to_string(), successful_queries as f32);
    benchmark.insert("total_runs".to_string(), (runs * queries_array.nrows()) as f32);
    
    Ok(benchmark)
}

/// Main Python module
#[pymodule]
fn vectro_py(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyEmbedding>()?;
    m.add_class::<PyEmbeddingDataset>()?;
    m.add_class::<PySearchIndex>()?;
    m.add_class::<PyQuantizedIndex>()?;
    m.add_function(wrap_pyfunction!(compress_embeddings, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_compression_quality, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_search_performance, m)?)?;
    
    // Add version info
    m.add("__version__", "1.1.0")?;
    m.add("__author__", "Wesley Scholl")?;
    m.add("__description__", "Python bindings for Vectro+ high-performance vector compression and search")?;
    
    Ok(())
}