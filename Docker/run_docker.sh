docker run -u $(id -u):$(id -g) --gpus all -v /clusterhome/:/clusterhome -v /clusterscratch/:/clusterscratch -it radtse_dl_recon bash
