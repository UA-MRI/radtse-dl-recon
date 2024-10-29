docker run -u $(id -u):$(id -g) --gpus all -v /clusterhome/:/clusterhome -v /clusterscratch/:/clusterscratch -it finufft_pyradrecon bash
