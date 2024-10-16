If you want to view the contents of the weights at `pt/unimol_exp1s4/pt.pt`, you can run the following code,

```
import torch
pt = torch.load('fragnet/exps/pt/unimol_exp1s4/pt.pt')
print(pt)
```
I have given the content of pt in `pt/unimol_exp1s4/pt.pt.data`.


Similarly for the file at, `ft/pnnl_full/fragnet_hpdl_exp1s_h4pt4_10`

```
import torch
ft = torch.load('fragnet/exps/ft/pnnl_full/fragnet_hpdl_exp1s_h4pt4_10/ft_100.pt')
print(pt)
```

The content of ft is in `ft/pnnl_full/fragnet_hpdl_exp1s_h4pt4_10/ft_100.pt.data`