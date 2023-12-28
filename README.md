# Simple CUDA code for 2 matrix multiplication.

## installation
It's assume that you have gpu compatible torch and numpy installed. 

## run
- numpy version
```
python py-np.py
```
- pytorch version
```
python py-torch.py
```
- pycuda version:
```
cd pycuda
pip install pycuda
python py-cuda.py
```
- pybind version
```
cd pybind
pip install -e .
python py-pybind.py
```