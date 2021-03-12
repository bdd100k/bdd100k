"""Label interators."""

from typing import Dict, List

from tqdm import tqdm

from .typing import DictAny
from .utils import init


class ImageLabelIterator:
    """Iterate through image dataset labels."""

    def __init__(
        self,
        mode: str = "det",
        ignore_as_class: bool = False,
        remove_ignore: bool = False,
    ) -> None:
        """Intialize the image label iterator."""
        self.ignore_as_class = ignore_as_class
        self.remove_ignore = remove_ignore
        self.coco, self.ignore_map, self.attr_id_dict = init(
            mode=mode, ignore_as_class=ignore_as_class
        )

        self.image_id = 0
        self.object_id = 0

        self.global_instance_id = 1

    def image_iteration(
        self, labels_per_image: DictAny  # pylint: disable=unused-argument
    ) -> None:
        """Actions for the image iteration."""
        self.image_id += 1

    def object_iteration(self, labels_per_object: DictAny) -> None:
        """Actions for the object iteration."""
        self.object_id += 1
        category_name = str(labels_per_object["category"])
        if category_name not in self.attr_id_dict:
            if self.ignore_as_class:
                category_name = "ignored"
                labels_per_object["category_ignored"] = False
            else:
                category_name = self.ignore_map[category_name]
                labels_per_object["category_ignored"] = True
        else:
            labels_per_object["category_ignored"] = False
        labels_per_object["category_id"] = self.attr_id_dict[category_name]

    def after_iteration(self) -> DictAny:
        """Actions after the iteration."""
        return self.coco

    def __call__(self, labels: List[List[DictAny]]) -> DictAny:
        """Executes iterations."""
        assert len(labels) > 0
        for labels_per_image in tqdm(labels[0]):
            if "labels" not in labels_per_image:
                continue
            self.image_iteration(labels_per_image)
            for labels_per_object in labels_per_image["labels"]:
                self.object_iteration(labels_per_object)
        return self.after_iteration()


class VideoLabelIterator(ImageLabelIterator):
    """Iterate through video dataset labels."""

    def __init__(
        self,
        mode: str = "det",
        ignore_as_class: bool = False,
        remove_ignore: bool = False,
    ) -> None:
        """Initialize the video label iterator."""
        super().__init__(mode, ignore_as_class, remove_ignore)
        self.video_id = 0
        self.instance_id_maps: Dict[str, int] = dict()

    def video_iteration(
        self,
        labels_per_video: List[DictAny],  # pylint: disable=unused-argument
    ) -> None:
        """Actions for the video iteration."""
        self.video_id += 1
        self.global_instance_id = 1
        self.instance_id_maps = dict()

    def object_iteration(self, labels_per_object: DictAny) -> None:
        """Actions for the object iteration."""
        super().object_iteration(labels_per_object)
        bdd100k_id = str(labels_per_object["id"])
        if bdd100k_id in self.instance_id_maps:
            labels_per_object["instance_id"] = self.instance_id_maps[
                bdd100k_id
            ]
        else:
            labels_per_object["instance_id"] = self.global_instance_id
            self.global_instance_id += 1
            self.instance_id_maps[bdd100k_id] = labels_per_object[
                "instance_id"
            ]

    def __call__(self, labels: List[List[DictAny]]) -> DictAny:
        """Executes iterations."""
        for labels_per_video in tqdm(labels):
            self.video_iteration(labels_per_video)
            for labels_per_image in labels_per_video:
                self.image_iteration(labels_per_image)
                for labels_per_object in labels_per_image["labels"]:
                    self.object_iteration(labels_per_object)
        return self.after_iteration()
