# visda2019-multisource

## Prerequisites

You may need a machine with 4 GPUs and PyTorch v1.1.0 for Python 3.

## Training

Repeat the following procedures 4 times.

### Train the end-to-end adaptation module

1. Go to the `Adapt` folder

2. Train source only models

  ```bash experiments/<DOMAIN>/<NET>/train.sh```

Where `<DOMAIN>` is clipart or painting, `<NET>` is the network (e.g. `senet154`)

3. Train domain adaptation module

  ```bash experiments/<DOMAIN>/<NET>_<phase_id>/train.sh``` 

### Extract features

1. Copy the adaptation models to the folder `ExtractFeat/experiments/<phase_id>/<DOMAIN>/<NET>/snapshot`

2. Extract features by running the scripts

  ```bash experiments/<phase_id>/<DOMAIN>/scripts/<NET>.sh```

3. Copy the features from ```experiments/<phase_id>/<DOMAIN>/<NET>/<NET>_<source_and_target_domains>/result``` to ```dataset/visda2019/pkl_test/<phase_id>/<DOMAIN>/<NET>```

### Train the feature fusion based adaptation module
1. Go to the `FeatFusionTest` folder

2. Train feature fusion based adaptation module

  ```bash experiments/<phase_id>/<DOMAIN>/train.sh```

3. Copy the pseudo labels file to ```Adapt/experiments/<DOMAIN>/<NET>_<next_phase_id>``` for the next adaptation.

## Acknowledgements
Thanks to the domain adaptation community and the contributers of the pytorch ecosystem.

Pytorch pretrained-models [link](https://github.com/Cadene/pretrained-models.pytorch) and [link](https://github.com/lukemelas/EfficientNet-PyTorch)
