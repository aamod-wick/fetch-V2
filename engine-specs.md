# FP 32
## 1. I/O FORMAT AND NODE NOMENCLATURE 
Input 'data_freq_time' with dynamic shape and dtype float32
Input 'data_dm_time' with dynamic shape and dtype float32
Output 'dense_3' with dynamic shape and dtype float32
Type       | Name                 | Shape                | DataType
----------------------------------------------------------------------
INPUT      | data_freq_time       | [-1, 256, 256, 1]    | FLOAT
INPUT      | data_dm_time         | [-1, 256, 256, 1]    | FLOAT
OUTPUT     | dense_3              | [-1, 2]              | FLOAT
## 2. INFERENCE TIME COLAB T4 GPU  
Input 'data_freq_time' with dynamic shape and dtype float32
Input 'data_dm_time' with dynamic shape and dtype float32
Output 'dense_3' with dynamic shape and dtype float32
Input specs: {'data_freq_time': ([-1, 256, 256, 1], dtype('float32')), 'data_dm_time': ([-1, 256, 256, 1], dtype('float32'))}
Outputs: dict_keys(['dense_3'])
  dense_3: shape=(29, 2), dtype=float32
CPU times: user 380 ms, sys: 118 ms, total: 497 ms
Wall time: 534 ms
## 3. BUILD TIME COLAB T4 GPU   
2 MIN 12 seconds
-----
# FP 16
## 1. I/O FORMAT AND NODE NOMENCLATURE 
Input 'data_freq_time' with dynamic shape and dtype float32
Input 'data_dm_time' with dynamic shape and dtype float32
Output 'dense_3' with dynamic shape and dtype float32
Type       | Name                 | Shape                | DataType
----------------------------------------------------------------------
INPUT      | data_freq_time       | [-1, 256, 256, 1]    | FLOAT
INPUT      | data_dm_time         | [-1, 256, 256, 1]    | FLOAT
OUTPUT     | dense_3              | [-1, 2]              | FLOAT
## 2. INFERENCE TIME COLAB T4 GPU  
Input 'data_freq_time' with dynamic shape and dtype float32
Input 'data_dm_time' with dynamic shape and dtype float32
Output 'dense_3' with dynamic shape and dtype float32
Input specs: {'data_freq_time': ([-1, 256, 256, 1], dtype('float32')), 'data_dm_time': ([-1, 256, 256, 1], dtype('float32'))}
Outputs: dict_keys(['dense_3'])
  dense_3: shape=(29, 2), dtype=float32
CPU times: user 211 ms, sys: 51.5 ms, total: 263 ms
Wall time: 265 ms
## 3. BUILD TIME COLAB T4 GPU   
5 MIN 40 seconds
--------
# INT 8
## 1. I/O FORMAT AND NODE NOMENCLATURE 
Input 'data_freq_time' with dynamic shape and dtype float32
Input 'data_dm_time' with dynamic shape and dtype float32
Output 'dense_3' with dynamic shape and dtype float32
Type       | Name                 | Shape                | DataType
----------------------------------------------------------------------
INPUT      | data_freq_time       | [-1, 256, 256, 1]    | FLOAT
INPUT      | data_dm_time         | [-1, 256, 256, 1]    | FLOAT
OUTPUT     | dense_3              | [-1, 2]              | FLOAT
## 2. INFERENCE TIME COLAB T4 GPU  
Input 'data_freq_time' with dynamic shape and dtype float32
Input 'data_dm_time' with dynamic shape and dtype float32
Output 'dense_3' with dynamic shape and dtype float32
Input specs: {'data_freq_time': ([-1, 256, 256, 1], dtype('float32')), 'data_dm_time': ([-1, 256, 256, 1], dtype('float32'))}
CPU times: user 37.1 ms, sys: 8.9 ms, total: 46 ms
Wall time: 47.3 ms

## 3. BUILD TIME COLAB T4 GPU   
6 MIN 10 seconds