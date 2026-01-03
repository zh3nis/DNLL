# DNLL

Discriminative Negative Log-Likelihood (DNLL) for Deep Linear Discriminant Analysis.
This repo accompanies the paper `Deep_LDA_Revisited.pdf` in this folder.

DNLL augments the LDA log-likelihood with a discriminative penalty that discourages
regions where multiple classes are simultaneously likely. For class score functions
`delta_c(x)` (unnormalized log-density or log-joint scores), the loss is:

```
L(x, y) = -delta_y(x) + lambda * sum_c exp(delta_c(x))
```

## Contents

- `dnll.py`: LDA head (`LDAHead`) and DNLL loss (`DNLLLoss`/`dnll_loss`).
- `lambda_reg_sweep.py`: CIFAR-10/100 and FashionMNIST experiments with a
  sweep over `lambda_reg` values.
- `DNLL.ipynb`: exploratory notebook (training and analysis).
- `cifar100_model.pth`: example checkpoint.
- `plots/`: output folder for sweep metrics/plots.

## Quickstart

Install dependencies (PyTorch + torchvision) and run a simple training loop with DNLL:

```python
import torch
from dnll import LDAHead, DNLLLoss

C, D = 10, 9
head = LDAHead(C=C, D=D, covariance_type="spherical")
loss_fn = DNLLLoss(lambda_reg=0.1)

z = torch.randn(32, D)
y = torch.randint(0, C, (32,))
scores = head(z)
loss = loss_fn(scores, y)
loss.backward()
```

## Lambda-reg sweep

Run a sweep on CIFAR-10 (downloads via torchvision):

```bash
python lambda_reg_sweep.py --dataset cifar10 --runs 5 --epochs 100
```

Other datasets:

```bash
python lambda_reg_sweep.py --dataset cifar100
python lambda_reg_sweep.py --dataset fashionmnist
```

Results are saved as JSON in `plots/` by default.

## Paper

See `Deep_LDA_Revisited.pdf` for full details on the motivation, derivation, and
experimental setup.

<!--
## Citation

If you use this code, please cite the paper:

```bibtex
@article{tezekbayev2025deeplda,
  title  = {Deep Linear Discriminant Analysis Revisited},
  author = {Tezekbayev, Maxat and Takhanov, Rustem and Bolatov, Arman and Assylbekov, Zhenisbek},
  year   = {2025},
  note   = {Manuscript, see Deep_LDA_Revisited.pdf}
}
```
-->

## License

MIT License. See `LICENSE`.
