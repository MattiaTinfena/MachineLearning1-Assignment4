# MachineLearning1-Assignment4
## Neural Networks and Autoencoders

Collection of MATLAB and Python experiments covering regression, classification and unsupervised representation learning with neural networks.

### Experiments

| Experiment | Implementation |
| --- | --- |
| Body-fat regression | MATLAB fitting network trained with Levenberg–Marquardt. |
| Glass classification | MATLAB pattern-recognition network trained with scaled conjugate gradients. |
| MNIST representation | Autoencoder with a two-dimensional latent space, trained on two randomly selected digit classes. |
| Breast-cancer classification | TensorFlow/Keras dense network with feature scaling and early stopping. |

### MATLAB workflow

Use MATLAB with the neural-network functionality required by `fitnet`, `patternnet` and `trainAutoencoder`, including the `bodyfat_dataset` and `glass_dataset` example datasets.

Set the current folder to `Task 1-2` and run:

```matlab
main
```

Keep the bundled MNIST files in that directory. The script produces training-performance, regression/classification and latent-space plots.

### Python workflow

```bash
git clone https://github.com/MattiaTinfena/MachineLearning1-Assignment4.git
cd MachineLearning1-Assignment4
python3 -m venv .venv
source .venv/bin/activate
python -m pip install jupyter numpy pandas matplotlib seaborn scikit-learn tensorflow
cd "Task 3"
jupyter notebook Task3.ipynb
```

Run the notebook cells in order. The notebook reads the included `breast_cancer/wdbc.data` file relative to `Task 3/`.

### Reproducibility

The experiments include random data partitions, network initialisations and digit-class selection. Fix the corresponding random seeds when comparing configurations. The breast-cancer task is an educational benchmark, not a clinical tool.
