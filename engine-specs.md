# TensorRT Engine Specs

## FP32
### 1. I/O format and node nomenclature
Type   | Name             | Shape              | DataType
------ | ---------------- | ------------------ | --------
INPUT  | data_freq_time   | [-1, 256, 256, 1]  | FLOAT
INPUT  | data_dm_time     | [-1, 256, 256, 1]  | FLOAT
OUTPUT | dense_3          | [-1, 2]            | FLOAT

### 2. Inference (Colab T4 GPU)
- Inputs: dynamic, float32  
- Outputs: `dense_3` → shape `(29, 2)`, dtype `float32`  
- Input specs: `{'data_freq_time': ([-1, 256, 256, 1], float32), 'data_dm_time': ([-1, 256, 256, 1], float32)}`
- CPU time: 380 ms user, 118 ms sys (497 ms total)  
- Wall time: 534 ms

### 3. Build time (Colab T4 GPU)
- 2 min 12 s

---

## FP16
### 1. I/O format and node nomenclature
Type   | Name             | Shape              | DataType
------ | ---------------- | ------------------ | --------
INPUT  | data_freq_time   | [-1, 256, 256, 1]  | FLOAT
INPUT  | data_dm_time     | [-1, 256, 256, 1]  | FLOAT
OUTPUT | dense_3          | [-1, 2]            | FLOAT

### 2. Inference (Colab T4 GPU)
- Inputs: dynamic, float32  
- Outputs: `dense_3` → shape `(29, 2)`, dtype `float32`  
- Input specs: `{'data_freq_time': ([-1, 256, 256, 1], float32), 'data_dm_time': ([-1, 256, 256, 1], float32)}`
- CPU time: 211 ms user, 51.5 ms sys (263 ms total)  
- Wall time: 265 ms

### 3. Build time (Colab T4 GPU)
- 5 min 40 s

---

## INT8
### 1. I/O format and node nomenclature
Type   | Name             | Shape              | DataType
------ | ---------------- | ------------------ | --------
INPUT  | data_freq_time   | [-1, 256, 256, 1]  | FLOAT
INPUT  | data_dm_time     | [-1, 256, 256, 1]  | FLOAT
OUTPUT | dense_3          | [-1, 2]            | FLOAT

### 2. Inference (Colab T4 GPU)
- Inputs: dynamic, float32  
- Outputs: `dense_3`  
- Input specs: `{'data_freq_time': ([-1, 256, 256, 1], float32), 'data_dm_time': ([-1, 256, 256, 1], float32)}`
- CPU time: 37.1 ms user, 8.9 ms sys (46 ms total)  
- Wall time: 47.3 ms

### 3. Build time (Colab T4 GPU)
- 6 min 10 s