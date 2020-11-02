Graph-to-Graph Transformers
=================

Self-attention models, such as Transformer, have been hugely successful in a wide
range of natural language processing (NLP) tasks, especially when combined with
language-model pre-training, such as BERT.

We propose ["Graph-to-Graph Transformer"](https://github.com/alirezamshi/G2GTr) and 
["Recursive Non-Autoregressive Graph-to-Graph Transformer for Dependency Parsing with Iterative Refinement"](https://arxiv.org/abs/2003.13118)
to generalize vanilla Transformer to encode graph structure, and builds the desired
output graph.

**Note** : To use G2GTr model to transition-based dependency parsing, please refer to [G2GTr](https://github.com/alirezamshi/G2GTr) repository.

Contents
---------------

- [Installation](#installation)
- [Quick Start](#othertasks)
- [Data Pre-processing and Initial Parser](#datapreprocessingandinitialparser)
- [Training](#training)
- [Evaluation](#evaluation)
- [Predict Raw Sentences](#predictraw)
- [Citations](#citations)

<a name="installation"/>  

Installation
--------------  
Following packages should be included in your environment:

- Python >= 3.7
- PyTorch >= 1.4.0
- Transformers(huggingface) = 2.4.1
  
The easier way is to run the following command:  

``` python 
conda env create -f environment.yml
conda activate rngtr
```

<a name="othertasks"/>  

Quick Start
-------------

Graph-to-Graph Transformer architecture is general and can be applied to
any NLP tasks which interacts with graphs. To use our implementation in your
task, you just need to add `BertGraphModel` class to your code to encode
both token-level and graph-level information. Here is a sample usage:

```python
#Loading BertGraphModel and initialize it with available BERT models.
import torch
from parser.utils.graph import initialize_bertgraph,BertGraphModel
# inputing unlabelled graph with label size 5, and Layer Normalization of key
# you can load other BERT pre-trained models too.
encoder = initialize_bertgraph('bert-base-cased',layernorm_key=True,layernorm_value=False,
             input_label_graph=False,input_unlabel_graph=True,label_size=5)

#sample input
input = torch.tensor([[1,2],[3,4]])
graph = torch.tensor([ [[1,0],[0,1]],[[0,1],[1,0]] ])
graph_rel = torch.tensor([[0,1],[3,4]])
output = encoder(input_ids=input,graph_arc=graph,graph_rel=graph_rel)
print(output[0].shape)
## torch.Size([2, 2, 768])

# inputting labelled graph
encoder = initialize_bertgraph('bert-base-cased',layernorm_key=True,layernorm_value=False,
             input_label_graph=True,input_unlabel_graph=False,label_size=5)

#sample input
input = torch.tensor([[1,2],[3,4]])
graph = torch.tensor([ [[2,0],[0,3]],[[0,1],[4,0]] ])
output = encoder(input_ids=input,graph_arc=graph,)
print(output[0].shape)
## torch.Size([2, 2, 768])
```
If you just want to use `BertGraphModel` in your research, you can just import it
from our repository:
```python
from parser.utils.graph import BertGraphModel,BertGraphConfig
config = BertGraphConfig(YOUR-CONFIG)
config.add_graph_par(GRAPH-CONFIG)
encoder = BertGraphModel(config)
```

<a name="datapreprocessingandinitialparser"/>  

Data Pre-processing and Initial Parser 
-----------------  

### Dataset Preparation  

We evaluated our model on [UD Treebanks](https://universaldependencies.org/), English 
and Chinese [Penn Treebanks](https://catalog.ldc.upenn.edu/LDC99T42), 
and [CoNLL 2009 Shared Task](https://www.aclweb.org/anthology/W09-1201/). 
In following sections, we prepare datasets and their evaluation scripts.

#### Penn Treebanks
English Penn Treebank can be downloaded from [english](https://catalog.ldc.upenn.edu/LDC99T42) and
[chinese](https://catalog.ldc.upenn.edu/LDC2005T01) under LDC license. For English 
Penn Treebank, replace gold POS tags with Stanford POS tagger with following command in 
[this repository](https://github.com/shuoyangd/hoolock):  
```
bash scripts/postag.sh ${data_dir}/ptb3-wsj-[train|dev|dev.proj|test].conllx
```

#### CoNLL 2009 Treebanks
You can download Treebanks from [here](https://catalog.ldc.upenn.edu/LDC2012T03) under
LDC license. We use predicted POS tags provided by organizers.

#### UD Treebanks
You can find required Treebanks from [here](https://universaldependencies.org/).
(use version 2.3)

### Initial Parser
As mentioned in our paper, you can use any initial parser to produce dependency graph. 
Here we use [Biaffine Parser](https://arxiv.org/abs/1611.01734) for Penn Treebanks, and German Corpus. We also apply our
model to ouput prediction of [UDify parser](https://arxiv.org/abs/1904.02099) for UD Treebanks.  
**Biaffine Parser**: To prepare biaffine initial parser, we use [this repository](https://github.com/yzhangcs/parser) 
to produce output predictions.  
**UDify Parser**: For UD Treebanks, we use [UDify repository](https://github.com/Hyperparticle/udify)
to produce required initial dependency graph.  
Alternatively, you can easily run the following command file to produce all required outputs:  
```
bash job_scripts/udify_dataset.bash
```

<a name="training"/>  

Training
-------------

To train your own model, you can easily fill out the script in `job_scripts` directory, 
and run it. Here is the list of sample scripts:  

Model | Script 
--- | ---  
Syntactic Transformer | `baseline.bash` | 
Any initial parser+RNGTr | `rngtr.bash` |
Empty+RNGTr | `empty_rngtr.bash` |

<a name="evaluation"/>  

Evaluation
-------------

First you should download official scripts from [UD](https://universaldependencies.org/conll18/evaluation.html), 
[Penn Treebaks](https://depparse.uvt.nl/), and [German](https://ufal.mff.cuni.cz/conll2009-st/eval-data.html). Then,
run the following command:  
```
bash job_scripts/predict.bash
```

To replicate `refinement analysis` and `error analysis` results, you should use 
[MaltEval](http://www.maltparser.org/malteval.html) tools.

<a name="predictraw"/>  

Predict Raw Sentences
--------------------- 

You can also predict dependency graphs of raw texts with a pre-trained model with ```predict.bash``` file. Just set ```input_type``` to ```raw```. Then, put all your sentences in a .txt file, and the output will be in CoNNL format.

Citations
-------------

<a name="citations"/>  

If you use this code for your research, please cite these works as:
```
@misc{mohammadshahi2020recursive,
      title={Recursive Non-Autoregressive Graph-to-Graph Transformer for Dependency Parsing with Iterative Refinement}, 
      author={Alireza Mohammadshahi and James Henderson},
      year={2020},
      eprint={2003.13118},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
```
@misc{mohammadshahi2020graphtograph,
      title={Graph-to-Graph Transformer for Transition-based Dependency Parsing}, 
      author={Alireza Mohammadshahi and James Henderson},
      year={2020},
      eprint={1911.03561},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
Have a question not listed here? Open [a GitHub Issue](https://github.com/idiap/g2g-transformer/issues) or 
send us an [email](alireza.mohammadshahi@idiap.ch).
