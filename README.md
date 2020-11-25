# UnifiedEmbeddingModel
Implementation of unified embedding model from Embedding-based Retrieval in Facebook Search (https://arxiv.org/abs/2006.11632).

# Unified Embedding Model Architecture

<img src="https://github.com/liyinxiao/LambdaRankNN/blob/master/assets/model.png" width=750 />

# Example with dummy data
```
python3 main.py
```
Dummy Data
```
query_inputs = np.random.rand(100, query_input_size)
positive_document_inputs = np.random.rand(100, document_input_size)
negative_document_inputs = np.random.rand(100, document_input_size)
```