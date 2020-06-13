#!/usr/bin/python3
# -*-coding:utf-8-*-
import os
import numpy as np
import json
import random
import decord
from config import Configuration


class BatchGenerator(object):
    def __init__(self, config):
        self._config = config
        self._label_mapping = dict()
        self._samples = self.collect_samples()

        self._train_samples, self._val_samples = self.split_train_val(train_val_ratio=5.0)
        self._train_batch_amount = len(self._train_samples) // self._config.batch_size
        self._val_batch_amount = len(self._val_samples) // self._config.batch_size
        print('# of training samples = {}'.format(len(self._train_samples)))
        print('# of validation samples = {}'.format(len(self._val_samples)))
        print('# of training batches = {}'.format(self._train_batch_amount))
        print('# of validation batches = {}'.format(self._val_batch_amount))

        self._train_batch_index, self._val_batch_index = 0, 0

    def collect_samples(self):
        annotation_files = [f for f in os.listdir(self._config.trainval_set_dir) if f.endswith('.json') and 'status' not in f]
        old_pattern = '郑思维-黄雅琼vs德差波尔-沙西丽'
        new_pattern = 'zsw-hyq-vs-debe-sxi'

        samples = list()
        for f_name in annotation_files:
            labels_info = json.load(open(os.path.join(self._config.trainval_set_dir, f_name)))
            for item in labels_info:
                video_name = item['video']
                label = item['label']

                video_full_path = os.path.join(
                    video_name.split('\\')[0],
                    video_name.split('\\')[1])

                video_full_path = video_full_path.replace(old_pattern, new_pattern)

                if not os.path.exists(os.path.join(self._config.trainval_set_dir, video_full_path)):
                    continue

                samples.append(dict(
                    video_path=video_full_path,
                    label=label
                ))

        print('Total samples amount = {}'.format(len(samples)))
        return samples

    def split_train_val(self, train_val_ratio=5.0):
        labels = list(set([s['label'] for s in self._samples]))
        for i, label in enumerate(labels):
            self._label_mapping[label] = i

        # dump to local disk for reference
        json.dump(self._label_mapping, open(self._config.mapping_path, 'w'), ensure_ascii=False, indent=True)

        random.shuffle(self._samples)
        batch_amount = len(self._samples) // self._config.batch_size
        train_batch_amount = int(batch_amount * train_val_ratio / (1.0 + train_val_ratio))
        split_index = train_batch_amount * self._config.batch_size

        train_samples = self._samples[0:split_index]
        val_samples = self._samples[split_index:]
        return train_samples, val_samples

    def next_train_batch(self):
        input_batch = np.zeros(shape=(self._config.batch_size,
                                      self._config.time_dimen,
                                      self._config.frame_height,
                                      self._config.frame_width,
                                      self._config.frame_channels))
        gt_batch = np.zeros(shape=(self._config.batch_size, self._config.ncls))

        for b_idx, sample in enumerate(self._train_samples[
                                       self._train_batch_index * self._config.batch_size:
                                       (1+self._train_batch_index) * self._config.batch_size]):
            video_path = os.path.join(self._config.trainval_set_dir, sample['video_path'])
            label = sample['label']
            video = decord.VideoReader(video_path)
            assert len(video) == self._config.time_dimen
            for t_idx, frame in enumerate(video):
                frame = frame.asnumpy()     # (height, width, channels)
                assert frame.shape[0] == self._config.frame_height
                assert frame.shape[1] == self._config.frame_width
                input_batch[b_idx][t_idx] = frame

            gt_batch[b_idx][self._label_mapping[label]] = 1.0

        self._train_batch_index += 1
        return input_batch, gt_batch

    def next_val_batch(self):
        input_batch = np.zeros(
            shape=(
                self._config.batch_size,
                self._config.time_dimen,
                self._config.frame_height,
                self._config.frame_width,
                self._config.frame_channels))
        gt_batch = np.zeros(shape=(self._config.batch_size, self._config.ncls))

        for b_idx, sample in enumerate(self._val_samples[
                                       self._val_batch_index * self._config.batch_size:
                                       (1+self._val_batch_index) * self._config.batch_size]):
            video_path = os.path.join(self._config.trainval_set_dir, sample['video_path'])
            label = sample['label']

            video = decord.VideoReader(video_path)
            assert len(video) == self._config.time_dimen
            for t_idx, frame in enumerate(video):
                frame = frame.asnumpy()     # (height, width, channels)
                assert frame.shape[0] == self._config.frame_height
                assert frame.shape[1] == self._config.frame_width
                input_batch[b_idx][t_idx] = frame

            gt_batch[b_idx][self._label_mapping[label]] = 1.0

        self._val_batch_index += 1
        return input_batch, gt_batch

    @property
    def train_batch_amount(self):
        return self._train_batch_amount

    @property
    def val_batch_amount(self):
        return self._val_batch_amount

    def reset_validation_batches(self):
        self._val_batch_index = 0

    def reset_training_batches(self):
        random.shuffle(self._train_samples)
        self._train_batch_index = 0


if __name__ == '__main__':
    batch_generator = BatchGenerator(
        config=Configuration()
    )

    for _ in range(batch_generator.train_batch_amount):
        train_batch, train_gt = batch_generator.next_train_batch()

    for _ in range(batch_generator.val_batch_amount):
        val_batch, val_gt = batch_generator.next_val_batch()
        print(val_batch.shape, val_gt.shape)
