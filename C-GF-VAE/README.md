# GF-VAE
A Flow-based Variational Autoencoder for Molecule Generation

the code can be separated into two parts, skeleton of the flow part is from paper MoFlow, which you may approach at https://github.com/calvin-zcx/moflow, thanks for his great job. The detailed code is shown in model folder.

I used this to train: 

python data_preprocess.py --data_name hmdb 


### TRAINING
python train_model.py  --data_name hmdb  --batch_size  256  --learning_rate 0.001 --max_epochs 400 --gpu 0  --debug True  --save_dir=results/hmdb_new  --b_n_flow 10  --b_hidden_ch 512,512  --a_n_flow 38  --a_hidden_gnn 256  --a_hidden_lin  512,64   --mask_row_size_list 1 --mask_row_stride_list 1  --noise_scale 0.6  --b_conv_lu 2  2>&1

#### CREATES THE RECONSTRUCTIONS
python generate.py --model_dir results/hmdb_new -snapshot model_snapshot_epoch_400 --gpu 0 --data_name hmdb --hyperparams-path moflow-params.json --batch-size 256 --reconstruct  2>&1


