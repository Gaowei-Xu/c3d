#!/usr/bin/python3
# -*-coding:utf-8-*-
import os


class Configuration(object):
    def __init__(self):
        self._trainval_set_dir = '../dataset/'
        self._mapping_path = '../mapping.json'

        self._batch_size = 64

        self._time_dimen = 12
        self._frame_height = 90
        self._frame_width = 160
        self._frame_channels = 3

        self._ncls = 18

        self._learning_rate = 0.0002

        self._max_epoch = 250
        self._train_summary_root_dir = '../train/'
        self._dump_model_para_root_dir = '../models/'
        self._save_every_epoch = 1

        self._selected_model_name = 'epoch_7_train_loss_348.611500_val_loss_342.913800.ckpt'

    @property
    def trainval_set_dir(self):
        return self._trainval_set_dir

    @property
    def mapping_path(self):
        return self._mapping_path

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def time_dimen(self):
        return self._time_dimen

    @property
    def frame_height(self):
        return self._frame_height

    @property
    def frame_width(self):
        return self._frame_width

    @property
    def frame_channels(self):
        return self._frame_channels

    @property
    def ncls(self):
        return self._ncls

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def max_epoch(self):
        return self._max_epoch

    @property
    def train_summary_root_dir(self):
        return self._train_summary_root_dir

    @property
    def dump_model_para_root_dir(self):
        return self._dump_model_para_root_dir

    @property
    def save_every_epoch(self):
        return self._save_every_epoch

    @property
    def selected_model_name(self):
        return self._selected_model_name
