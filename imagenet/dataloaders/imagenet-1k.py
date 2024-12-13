import os

import datasets

from imagenet.dataloaders.classes import IMAGENET2012_CLASSES


_CITATION = """\
@article{imagenet15russakovsky,
    Author = {Olga Russakovsky and Jia Deng and Hao Su and Jonathan Krause and Sanjeev Satheesh and Sean Ma and Zhiheng Huang and Andrej Karpathy and Aditya Khosla and Michael Bernstein and Alexander C. Berg and Li Fei-Fei},
    Title = { {ImageNet Large Scale Visual Recognition Challenge} },
    Year = {2015},
    journal   = {International Journal of Computer Vision (IJCV)},
    doi = {10.1007/s11263-015-0816-y},
    volume={115},
    number={3},
    pages={211-252}
}
"""

_HOMEPAGE = "https://image-net.org/index.php"

_DESCRIPTION = """\
ILSVRC 2012, commonly known as 'ImageNet' is an image dataset organized according to the WordNet hierarchy. Each meaningful concept in WordNet, possibly described by multiple words or word phrases, is called a "synonym set" or "synset". There are more than 100,000 synsets in WordNet, majority of them are nouns (80,000+). ImageNet aims to provide on average 1000 images to illustrate each synset. Images of each concept are quality-controlled and human-annotated. In its completion, ImageNet hopes to offer tens of millions of cleanly sorted images for most of the concepts in the WordNet hierarchy. ImageNet 2012 is the most commonly used subset of ImageNet. This dataset spans 1000 object classes and contains 1,281,167 training images, 50,000 validation images and 100,000 test images
"""

_DATA_URL = {
    "train": [f"/data/furuya/train_images_{i}.tar.gz" for i in range(5)],
    "val": ["/data/furuya/val_images.tar.gz"],
    "test": ["/data/furuya/test_images.tar.gz"],
}


class Imagenet1k(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    DEFAULT_WRITER_BATCH_SIZE = 1000

    def _info(self):
        assert len(IMAGENET2012_CLASSES) == 1000
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(names=list(IMAGENET2012_CLASSES.values())),
                }
            ),
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        archives = dl_manager.download(_DATA_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "archives": [dl_manager.iter_archive(archive) for archive in archives["train"]],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "archives": [dl_manager.iter_archive(archive) for archive in archives["val"]],
                    "split": "validation",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "archives": [dl_manager.iter_archive(archive) for archive in archives["test"]],
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, archives, split):
        """Yields examples."""
        idx = 0
        for archive in archives:
            for path, file in archive:
                if path.endswith(".JPEG"):
                    if split != "test":
                        # image filepath format: <IMAGE_FILENAME>_<SYNSET_ID>.JPEG
                        root, _ = os.path.splitext(path)
                        _, synset_id = os.path.basename(root).rsplit("_", 1)
                        label = IMAGENET2012_CLASSES[synset_id]
                    else:
                        label = -1
                    ex = {"image": {"path": path, "bytes": file.read()}, "label": label}
                    yield idx, ex
                    idx += 1