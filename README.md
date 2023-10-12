## Gem_Classification
A simple pytorch_lightning template for classification

### Usage

#### Basic Command

```python
# training
python main.py --model resnet50 --num_classes 25

# testing
python main.py --test True --resume 'path/weight' --regen False --model resnet50
```

see `config.py` for more arguments

#### Customize your model and dataset

modify  `model/model_interface.py`  and  `data/data_interface.py`  to use your own model and dataset. 

### Notice

- This code do not use data in `data/data55032/archive_test.zip`
- `--regen`  means to regenerate a new train_list, which leads to a new training dataset
