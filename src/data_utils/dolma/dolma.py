# Copyright 2024 Allen Institute for AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research"""

import gzip
import json
import os
from pathlib import Path
from typing import List

import datasets

logger = datasets.logging.get_logger(__name__)


_DESCRIPTION = """\
Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research
"""

_URL_LISTS = {
    "v1_7": "urls/v1_7.txt",
}
_VERSIONS = {
    "v1_7": "1.7.0",
}
_DATES = {
    "v1_7": "(Apr 2024)",
}
_BASE_URL = "https://olmo-data.org"

_DATA_DIR = os.environ.get("DOLMA_DATA_DIR", None)

_CITATION = """\
@article{dolma,
  title = {{Dolma: An Open Corpus of Three Trillion Tokens for Language Model Pretraining Research}},
  author = {
    Luca Soldaini and Rodney Kinney and Akshita Bhagia and Dustin Schwenk and David Atkinson and
    Russell Authur and Ben Bogin and Khyathi Chandu and Jennifer Dumas and Yanai Elazar and
    Valentin Hofmann and Ananya Harsh Jha and Sachin Kumar and Li Lucy and Xinxi Lyu and Ian Magnusson and
    Jacob Morrison and Niklas Muennighoff and Aakanksha Naik and Crystal Nam and Matthew E. Peters and
    Abhilasha Ravichander and Kyle Richardson and Zejiang Shen and Emma Strubell and Nishant Subramani and
    Oyvind Tafjord and Evan Pete Walsh and Hannaneh Hajishirzi and Noah A. Smith and Luke Zettlemoyer and
    Iz Beltagy and Dirk Groeneveld and Jesse Dodge and Kyle Lo
},
  year = {2024},
  journal={arXiv preprint},
}
"""


class Dolma(datasets.GeneratorBasedBuilder):
    """Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research"""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=name,
            version=_VERSIONS[name],
            description=f"{_DESCRIPTION} {_DATES[name]}",
        )
        for name in _URL_LISTS.keys()
    ]

    DEFAULT_CONFIG_NAME = "v1_7"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    # "metadata": datasets.Value("string"),
                    "added": datasets.Value("string"),
                    "created": datasets.Value("string"),
                    "source": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        # path = dl_manager.download(_URL_LISTS[self.config.name])
        
        subset_files = list(Path('src/data_utils/dolma/dolma_v1_7_sample').iterdir())

        # with open(path, mode="rt", encoding="utf-8") as f:  # type: ignore[no-untyped-call]
        #     subset_urls = f.read().splitlines()

        # if _DATA_DIR is not None:
        #     subset_files = [os.path.join(_DATA_DIR, url.replace(_BASE_URL, "").lstrip("/")) for url in subset_urls]
        # else:
        #     subset_files = dl_manager.download(subset_urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,  # type: ignore[assignment]
                gen_kwargs={"files": subset_files},
            )
        ]

    def _generate_examples(self, files: List[str]):
        """This function returns the examples in the raw (text) form."""
        idx = 0
        for fn in files:
            logger.info("generating examples from = %s", fn)
            try:
                with gzip.open(fn, mode="rt", encoding="utf-8") as f:
                    for line in f:
                        row = json.loads(line)
                        yield idx, {
                            "id": row["id"],
                            "text": row["text"],
                            "added": row.get("added", ""),
                            "created": row.get("created", ""),
                            "source": row.get("source", ""),
                        }
                        idx += 1
            except:
                logger.info("error from = %s", fn)