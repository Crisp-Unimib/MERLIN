[![Pypi Downloads](https://img.shields.io/pypi/dm/MERLIN.svg?label=Pypi%20downloads)](https://pypi.org/project/MERLIN/)

<!-- [![DOI:10.1016/j.inffus.2021.11.016](http://img.shields.io/badge/DOI-10.1016/j.inffus.2021.11.016-blue.svg)](https://doi.org/10.1016/j.inffus.2021.11.016) -->
<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hb4KN0SYxdj9SaExqqFGmAXAIyUnBVnA?usp=sharing) -->

[![Stars](https://img.shields.io/github/stars/Crisp-Unimib/MERLIN?style=social)](https://github.com/Crisp-Unimib/MERLIN)
[![Watchers](https://img.shields.io/github/watchers/Crisp-Unimib/MERLIN?style=social)](https://github.com/Crisp-Unimib/MERLIN)

# MERLIN

![](/img/MERLIN.jpg)

**_MERLIN is a global, model-agnostic, contrastive explainer for any tabular or text classifier_**. It provides contrastive explanations of how the behaviour of two machine learning models differs.

Imagine we have a machine learning classifier, let's say M1, and wish to understand how -and to what extent- it differs from a second model M2.
MERLIN aims at answering to the following questions:

1. Can we estimate to what extent M2 classifies data coherently to the predictions made by the M1 model?
2. Why do the criteria used by M1 result in class _c_, but M2 does not use the same criteria to classify as _c_?
3. Can we use natural language to explain the differences between models making them more comprehensible to final users?

For details and citations, see the [references' section](#References).

## Install

MERLIN is available on [PyPi](https://pypi.org/project/MERLINXAI/). Simply run:

```
pip install merlinxai
```

Or clone the repository and run:

```
pip install .
```

The PyEDA package is required but has not been added to the dependencies.
This is due to installation errors on Windows. If you are on Linux or Mac, you
should be able to install it by running:

```
pip3 install pyeda
```

However, if you are on Windows, we found that the best way to install is through
Christophe Gohlke's [pythonlibs page](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyeda).
For further information, please consult the official PyEDA
[installation documentation](https://pyeda.readthedocs.io/en/latest/install.html).

To produce the PDF files, a Graphviz installation is also required.
Full documentation on how to install Graphviz on any platform is available
[here](https://graphviz.org/download/).

## Input

MERLIN takes as input the _"feature data"_ (can be training or test, tabular or free text) and the corresponding _"labels"_ predicted by the classifier. This means you don't need to wrap MERLIN within your code at all!
As optional parameters, the user can specify:

- the coverage of the dataset to be used (default is 100%); otherwise, a sampling procedure is used;
- the surrogate type to be used (decision tree or rulefit);
- a set of hyperparameters to be used for creating the most accurate surrogate models;
- the size of the test set to measure the fidelity of the surrogates.

## MERLIN on tabular data

In this example, we apply MERLIN on a tabular dataset named _Occupancy_, which revolves around predicting occupancy in an office room based on sensor measurements of light, temperature, humidity, and CO2 levels.
In this case, M1 is responsible for classifying instances during the daytime, while M2 handles instances during the nighttime.

```
from merlin import MERLIN

exp = MERLIN(X_left, predicted_labels_left,
             X_right, predicted_labels_right,
             data_type='tabular', surrogate_type='sklearn',
             save_path=f'results/',)

exp.run_trace()
```

### BDD2Text

The BDD2Text for _Occupancy_ reveals that one path has not changed between M1 and M2: a high level of light, in the 4th quartile, means that the room is well-lit and is the best indicator for showing whether it is occupied or not.

There is also one added path in M2: at nighttime, having the light variable in the 3rd quartile now leads to a positive classification, which was not true in M1. During the daytime, the light in this 3rd quartile would not have been sufficient to classify a data instance positively, but it is so during nighttime.

```
exp.run_explain()
exp.explain.BDD2Text()
```

&nbsp;
&nbsp;

<p align="center">
<img src="/img/bdd2text.png" width="700" >
</p>
&nbsp;
&nbsp;

### Get Rules

The NLE shows the differences between the two models. However, a user might also wish to see example instances in the datasets where these rules apply.

To do so, MERLIN provides the _get_rule_examples_ function, which requires the user to specify a rule to be applied and the number of examples to show.

```
exp.data_manager['left'].get_rule_examples(rule, n_examples=5)
```

&nbsp;
&nbsp;

<p align="center">
<img src="/img/get_examples.PNG" width="700" >
</p>
&nbsp;
&nbsp;

## MERLIN on text data

The same process can also be applied to text classifiers. For example, in the _20newsgroups_ dataset, one might closely look at class _atheism_ as for this class, the number of deleted paths is higher than the added ones.

### BDD2Text

The NLE for _atheism_ shows the presence of the word _bill_ leads the retrained classifier M2 to assign the label _atheism_ to a specific record, whilst the presence of such a feature was not a criterion for the previous classifier M1.
Conversely, the explanation shows that M1 used the feature _keith_ to assign the label, whilst M2 discarded this rule.

Both terms refer to the name of the posts' authors: _Bill_'s posts are only contained within the dataset used to retrain whilst _Keith_'s ones are more frequent in the initial dataset rather than the second one (dataset taken from _Jin, P., Zhang, Y., Chen, X., & Xia, Y. Bag-of-embeddings for text classification. In IJCAI-2016_).

Finally, M2 discarded the rule _having political atheist_ that was sufficient for M1 for classifying the instance.

&nbsp;
&nbsp;

<p align="center">
<img src="/img/bdd2text_atheism.PNG" width="700" >
</p>
&nbsp;
&nbsp;

## Tutorials and Usage

A complete example of MERLIN usage is provided in the notebook ["MERLIN Demo"](/MERLIN%20Demo.ipynb) inside of the main repository folder. A notebook example with ML model training is also available in this repository, which can also be accessed in this [Google Colab notebook](https://colab.research.google.com/drive/1hb4KN0SYxdj9SaExqqFGmAXAIyUnBVnA?usp=sharing).

## References

A citation for MERLIN will be released soon.

MERLIN generalizes the approach proposed in _Malandri, L., Mercorio, F., Mezzanzanica, M., Nobani, N., & Seveso, A. (2022). ContrXT: Generating contrastive explanations from any text classifier. Information Fusion, 81, 103-115._ [(bibtex)](https://scholar.googleusercontent.com/scholar.bib?q=info:0m4K2oHziA8J:scholar.google.com/&output=citation&scisdr=Cm3RQ6UsEMDigjKD5sU:AGlGAw8AAAAAZJGF_sX6i_Yv-u1e4Uchy_LnXps&scisig=AGlGAw8AAAAAZJGF_olHzQufUAHR9c2EorlOe2s&scisf=4&ct=citation&cd=-1&hl=en)
