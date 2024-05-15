# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import os
from glob import glob
import os.path as osp
from .bases import BaseImageDataset


class CUHK_SYSU(BaseImageDataset):
    dataset_dir = "CUHK-SYSU"

    def __init__(self, root='', **kwargs):
        super(CUHK_SYSU, self).__init__()
        self.root = root
        self.train_path = os.path.join(self.root, self.dataset_dir)

        self._check_before_run()
        train = self.process_train(self.train_path)

        self.train = train
        self.query = train
        self.gallery = train
        self.mix_dataset = train
        ###'num_mix_pids

        self.num_mix_pids, self.num_mix_imgs, self.num_mix_cams,self.num_train_vids = self.get_imagedata_info(self.mix_dataset)
        print(self.num_mix_pids)
        # self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)

        print("=> CUHK-SYSU loaded")
        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  mix      | {:5d} | {:8d} | {:9d}".format(self.num_mix_pids, self.num_mix_imgs, self.num_mix_cams))

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.train_path):
            raise RuntimeError("'{}' is not available".format(self.train_path))

        # super().__init__(train, [], [], **kwargs)

    def process_train(self, train_path):
        data = []
        img_paths = glob(os.path.join(train_path, "*.png"))
        pid_container = set()
        for img_path in img_paths:
            split_path = img_path.split('/')[-1].split('_') # p00001_n01_s00001_hard0.png
            # pid = self.dataset_name + "_" + split_path[0][1:]
            pid = int(split_path[0][1:])
            camid = int(split_path[2][1:])
            pid_container.add(pid)
            # camid = self.dataset_name + "_" + split_path[2][1:]
            # data.append([img_path, pid, camid,0])
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        for img_path in img_paths:
            split_path = img_path.split('/')[-1].split('_') # p00001_n01_s00001_hard0.png
            # pid = self.dataset_name + "_" + split_path[0][1:]
            pid = int(split_path[0][1:])
            # camid = int(split_path[2][1:])
            camid = 0
            pid = pid2label[pid]
            # camid = self.dataset_name + "_" + split_path[2][1:]
            data.append([img_path, pid, camid,0])
        return data