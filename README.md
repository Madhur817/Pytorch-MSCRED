# Pytorch-MSCRED

This is using PyTorch to implement MSCRED

Original paper:
[http://in.arxiv.org/abs/1811.08055](http://in.arxiv.org/abs/1811.08055)

Author's demo code in TensorFlow：
[https://github.com/7fantasysz/MSCRED](https://github.com/7fantasysz/MSCRED)

This project is converted to Pytorch through the above tensorFlow, the specific process is as follows：
- First convert the time series data to image matrices

  > python ./utils/matrix_generator.py

- Then train the model and generate the corresponding test set reconstructed matrices

  > python main.py

- Final evaluation model，results is generated in `outputs` Folder

  > python ./utils/evaluate.py