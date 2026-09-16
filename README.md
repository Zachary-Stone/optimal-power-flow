# AI for Optimal Power Flow
This tutorial introduces **AI for AC Optimal Power Flow (AC-OPF)** using a small 5-bus power network and PyTorch-based neural networks. The goal is to learn the full workflow: formulate a basic AC-OPF problem, generate or load solver labels, train ML approximators, evaluate them with OPF-aware metrics, and understand the limits of using ML for grid operation.

**Version released in 2023**

* Jorge Montalvo, CCAI, jorge@climatechange.ai
* Utkarsha Agwan, uagwan@berkeley.edu
* Panos Moutis, panay1ot1s@climatechange.ai

**Version revised and released in 2024**

* Enming Liang, enming.cityu@gmail.com
* Advising comments from Priya L. Donti (MIT) and Minghua Chen (CityU, HK)

**Version revised and released in 2026**

* Enming Liang, enming.cityu@gmail.com
* Advising comments from the CCAI Virtual Summer School Organizing Committee and Minghua Chen (CUHK-SZ)

## Access this tutorial

We recommend executing this notebook in a Colab environment to gain access to GPUs and to manage all necessary dependencies. <a target="_blank" href="https://colab.research.google.com/github/climatechange-ai-tutorials/optimal-power-flow/blob/main/AI_for_Optimal_Power_Flow.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Contribute to this tutorial

Please refer to these [GitHub instructions](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project#about-forking) to open a pull request via the "fork and pull request" workflow. 

Pull requests will be reviewed by members of the Climate Change AI Tutorials team for relevance, accuracy, and conciseness.

## Codebase refactor

The reusable Python package is being extracted from the notebook. The planned
module boundaries and implementation sequence are recorded in
[`REFACTORING_PROPOSAL.md`](REFACTORING_PROPOSAL.md). Package metadata,
configuration, and source-package skeleton are available now; experiment
workflows will be added incrementally.

Local development uses Poetry once `poetry.lock` has been generated:

```bash
poetry install
cd src
poetry run python main.py
```

The notebook's existing `requirements.txt` remains the dependency reference
for its Colab-oriented workflow. Runtime downloads, outputs, and checkpoints
are configured beneath `artifacts/` and are not committed to Git.

## Climate Change AI Tutorials
Check out the [tutorials page](https://www.climatechange.ai/tutorials?) on our website for a full list of tutorials demonstrating how AI can be used to tackle problems related to climate change.

## License
Usage of this tutorial is subject to the MIT License.

## Cite

### Plain Text
Montalvo, J., Agwan, U., Moutis, P., Liang, E. (2026). AI for Optimal Power Flow [Tutorial]. In Climate Change AI Summer School. Climate Change AI. https://doi.org/10.5281/zenodo.21827296

### BibTeX

```
@misc{montalvo2026ai,
  title={AI for Optimal Power Flow},
  author={Montalvo, Jorge and Agwan, Utkarsha and Moutis, Panos and Liang, Enming},
  year={2026},
  organization={Climate Change AI},
  type={Tutorial},
  doi={https://doi.org/10.5281/zenodo.21827296},
  booktitle={Climate Change AI Summer School},
  howpublished={\url{https://github.com/climatechange-ai-tutorials/optimal-power-flow}}
}
```
