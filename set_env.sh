export PYTHONPATH=$PYTHONPATH:$(pwd)
module load cuda/8.0.61 cudnn/8.0-v6.0

# PyTorch
# module load pytorch/0.2.0
export PATH=/z/home/mbanani/sw/miniconda2/bin:$PATH

# tensorflow for TensorBoard
module load numpy/1.12.0 gflags/2.2.1 tensorflow/1.0.0
module rm   numpy/1.12.0
