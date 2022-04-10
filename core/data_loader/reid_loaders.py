import sys
sys.path.append('../')

from .dataset import *
from .loader import *

import torchvision.transforms as transforms
from tools import *


class ReIDLoaders:

    def __init__(self, config):

        # resize --> flip --> pad+crop --> colorjitor(optional) --> totensor+norm --> rea (optional)
        transform_train = [
            transforms.Resize(config.image_size, interpolation=3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop(config.image_size)]
        if config.use_colorjitor: # use colorjitor
            transform_train.append(transforms.ColorJitter(brightness=0.25, contrast=0.15, saturation=0.25, hue=0))
        transform_train.extend([transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        if config.use_rea: # use rea
            transform_train.append(RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]))
        self.transform_train = transforms.Compose(transform_train)

        # resize --> totensor --> norm
        self.transform_test = transforms.Compose([
            transforms.Resize(config.image_size, interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.datasets = ['market', 'p_duke', 'part_reid', 'duke', 'part_ilids', 'occ_reid', 'msmt', 'njust_win', 'njust_spr', 'non', 'njust_both', 'wildtrack', 'occ_duke']

        # dataset
        self.market_path = config.market_path
        self.duke_path = config.duke_path
        self.occ_duke_path = config.occ_duke_path
        self.combine_all = config.combine_all
        self.train_dataset = config.train_dataset
        self.test_dataset = config.test_dataset
        for a_train_dataset in self.train_dataset:
            assert a_train_dataset in self.datasets

        # batch size
        self.p = config.p
        self.k = config.k

        # load
        self._load()


    def _load(self):

        '''init train dataset'''
        train_samples = self._get_train_samples(self.train_dataset)
        self.train_iter = self._get_uniform_iter(train_samples, self.transform_train, self.p, self.k)

        '''init test dataset'''
        if self.test_dataset == 'market':
            self.market_query_samples, self.market_gallery_samples = self._get_test_samples('market')
            self.market_query_loader = self._get_loader(self.market_query_samples, self.transform_test, 32)
            self.market_gallery_loader = self._get_loader(self.market_gallery_samples, self.transform_test, 32)
        elif self.test_dataset == 'duke':
            self.duke_query_samples, self.duke_gallery_samples = self._get_test_samples('duke')
            self.duke_query_loader = self._get_loader(self.duke_query_samples, self.transform_test, 32)
            self.duke_gallery_loader = self._get_loader(self.duke_gallery_samples, self.transform_test, 32)
        elif self.test_dataset == 'occ_duke':
            self.duke_query_samples, self.duke_gallery_samples = self._get_test_samples('occ_duke')
            self.duke_query_loader = self._get_loader(self.duke_query_samples, self.transform_test, 32)
            self.duke_gallery_loader = self._get_loader(self.duke_gallery_samples, self.transform_test, 32)


    def _get_train_samples(self, train_dataset):
        '''get train samples, support multi-dataset'''
        samples_list = []
        for a_train_dataset in train_dataset:
            if a_train_dataset == 'market':
                samples = Samples4Market(self.market_path, relabel=True, combineall=self.combine_all).train
            elif a_train_dataset == 'duke':
                samples = Samples4Duke(self.duke_path, relabel=True, combineall=self.combine_all).train
            elif a_train_dataset == 'occ_duke':
                samples = Samples4Duke(self.occ_duke_path, relabel=True, combineall=self.combine_all).train
            elif a_train_dataset == 'p_duke':
                samples = Samples4pDuke(self.p_duke_path, relabel=True, combineall=self.combine_all).train
            elif a_train_dataset == 'msmt':
                samples = Samples4MSMT17(self.msmt_path, relabel=True, combineall=self.combine_all).train
            elif 'njust' in a_train_dataset:
                assert a_train_dataset in ['njust_win', 'njust_spr', 'njust_both']
                season = a_train_dataset.split('_')[1]
                samples = Samples4NJUST365(self.njust_path, relabel=True, combineall=self.combine_all, season=season).train
            samples_list.append(samples)
        if len(train_dataset) > 1:
            samples = combine_samples(samples_list)
            samples = PersonReIDSamples._relabels(None, samples, 1)
            PersonReIDSamples._show_info(None, samples, samples, samples, name=str(train_dataset))
        return samples

    def _get_test_samples(self, test_dataset):
        if test_dataset == 'market':
            market = Samples4Market(self.market_path, relabel=True, combineall=self.combine_all)
            query_samples = market.query
            gallery_samples = market.gallery
        elif test_dataset == 'duke':
            duke = Samples4Duke(self.duke_path, relabel=True, combineall=self.combine_all)
            query_samples = duke.query
            gallery_samples = duke.gallery
        elif test_dataset == 'occ_duke':
            duke = Samples4Duke(self.occ_duke_path, relabel=True, combineall=self.combine_all)
            query_samples = duke.query
            gallery_samples = duke.gallery
        elif test_dataset == 'p_duke':
            duke = Samples4pDuke(self.p_duke_path, relabel=True, combineall=self.combine_all)
            query_samples = duke.query
            gallery_samples = duke.gallery

        return query_samples, gallery_samples

    def _get_uniform_iter(self, samples, transform, p, k):
        '''
        load person reid data_loader from images_folder
        and uniformly sample according to class
        '''
        dataset = PersonReIDDataSet(samples, transform=transform)
        loader = data.DataLoader(dataset, batch_size=p * k, num_workers=8, drop_last=False, sampler=ClassUniformlySampler(dataset, class_position=1, k=k))
        iters = IterLoader(loader)
        return iters

    def _get_random_iter(self, samples, transform, batch_size):
        dataset = PersonReIDDataSet(samples, transform=transform)
        loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=True)
        iters = IterLoader(loader)
        return iters

    def _get_random_loader(self, samples, transform, batch_size):
        dataset = PersonReIDDataSet(samples, transform=transform)
        loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=True)
        return loader

    def _get_loader(self, samples, transform, batch_size):
        dataset = PersonReIDDataSet(samples, transform=transform)
        loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=False)
        return loader
