# Recurrent Mixture Density Networks for Human-Robot Interaction

## Requirements

Install the requirements in [`requirements.txt`](requirements.txt) by running

```bash
pip install -r requirements.txt
```

Clone the repo `https://github.com/souljaboy764/phd_utils` and follow the installation instructions in its README. This repository has the datasets to be used already preprocessed.

## Training

### Buetepage
We use the same configuration for all of the Buetepage datasets, which can be trained using the following command:

```bash
python train_rmdvae.py --results path/to/buetepage_results --epochs 400 --num-components 3 --dataset DATASET --hidden-sizes 40 20 --latent-dim 5
```

Where `DATASET` is one of `BuetepageHH`, `BuetepagePepper` or `BuetepageYumi` (make sure the names match with the class names in [`dataset.py`](dataset.py)).

### NuiSI

Once the models are trained on the Buetepage dataset, we initialize the models for the NuiSI datasets with the pretrained weights trained on the Buetepage dataset.

```bash
python train_rmdvae.py --results path/to/nuisi_results --ckpt path/to/buetepage_checkpoint.pth --epochs 400 --num-components 3 --dataset DATASET --hidden-sizes 40 20 --latent-dim 5
```

### Handovers

For the comparison in the paper, we use a mix of both bimanual and unimanual robot-to-human handovers. This is in the `HandoverHH` class in [`dataset.py`](dataset.py) where no scaling is applied to the data.

```bash
python train_rmdvae.py --results path/to/handover_hh_results --epochs 400 --num-components 3 --dataset HandoverHH --hidden-sizes 80 40 --latent-dim 10
```

For executing handover behaviours on the Kobo robot, the class `HandoverKobo` should be given as an argument instead of `HandoverHH` in the above command.

For the model that performs unimanual handovers with the Pepper, the following should be run:

```bash
python train_rmdvae.py --results path/to/handover_pepper_results --epochs 400 --num-components 3 --dataset UnimanualPepper --hidden-sizes 40 20 --latent-dim 5
```

## Testing

The output of the below testing code is the Mean squared prediction error and standard deviation for each interaction in the dataset of the model that is being evaluated. To run the testing, simply run:

```bash
python test.py --ckpt /path/to_ckpt
```

To visualize the latent space learned by the mdoel, run the python notebook [`mse_plotter.ipynb`](mse_plotter.ipynb) which plots the first 3 latent coordinates as well as the Gaussians corresponding to the HMM state.
