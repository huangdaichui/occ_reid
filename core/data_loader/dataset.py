import numpy as np
from PIL import Image
import copy
import os
import copy
from prettytable import PrettyTable
from easydict import EasyDict
import random

def os_walk(folder_dir):
    for root, dirs, files in os.walk(folder_dir):
        files = sorted(files, reverse=True)
        dirs = sorted(dirs, reverse=True)
        return root, dirs, files


class PersonReIDSamples:

    def _relabels(self, samples, label_index):
        '''
        reorder labels
        map labels [1, 3, 5, 7] to [0,1,2,3]
        '''
        ids = []
        for sample in samples:
            ids.append(sample[label_index])
        # delete repetitive elments and order
        ids = list(set(ids))
        ids.sort()
        # reorder
        for sample in samples:
            sample[label_index] = ids.index(sample[label_index])
        return samples

    def _load_images_path(self, folder_dir):
        '''
        :param folder_dir:
        :return: [(path, identiti_id, camera_id)]
        '''
        samples = []
        root_path, _, files_name = os_walk(folder_dir)
        for file_name in files_name:
            if '.jpg' in file_name:
                identi_id, camera_id = self._analysis_file_name(file_name)
                samples.append([root_path + file_name, identi_id, camera_id])
        return samples

    def _load_images_path_pduke_t(self, folder_dir):
        '''
        :param folder_dir:
        :return: [(path, identiti_id, camera_id)]
        '''
        samples = []
        root_path, _, files_name = os_walk(folder_dir)
        for file_name in files_name:
            if '.jpg' in file_name:
                identi_id, _ = self._analysis_file_name_pduke(file_name)
                samples.append([root_path + file_name, identi_id, 0])
        return samples

    def _load_images_path_pduke_tt(self, folder_dir):
        '''
        :param folder_dir:
        :return: [(path, identiti_id, camera_id)]
        '''
        samples = []
        root_path, _, files_name = os_walk(folder_dir)
        for file_name in files_name:
            if '.jpg' in file_name:
                identi_id, _ = self._analysis_file_name_pduke(file_name)
                samples.append([root_path + file_name, identi_id, 1])
        return samples

    def _analysis_file_name(self, file_name):
        '''
        :param file_name: format like 0844_c3s2_107328_01.jpg
        :return: 0844, 3
        '''
        split_list = file_name.replace('.jpg', '').replace('c', '').replace('s', '_').split('_')
        identi_id, camera_id = int(split_list[0]), int(split_list[1])
        return identi_id, camera_id

    def _analysis_file_name_pduke(self, file_name):
        '''
        :param file_name: format like 0844_c3s2_107328_01.jpg
        :return: 0844, 3
        '''
        split_list = file_name.replace('.jpg', '').split('_')
        identi_id, _ = int(split_list[0]), int(split_list[1])
        return identi_id, _

    def _load_jpg_path1(self, folder_dir):

        samples = []
        root_path, _, files_name = os_walk(folder_dir)
        for file_name in files_name:
            if '.jpg' in file_name:
                identi_id, camera_id = self._analysis_jpg_name1(file_name)
                samples.append([root_path + file_name, identi_id, camera_id])
        return samples

    def _load_jpg_path0(self, folder_dir):

        samples = []
        root_path, _, files_name = os_walk(folder_dir)
        for file_name in files_name:
            if '.jpg' in file_name:
                identi_id, camera_id = self._analysis_jpg_name0(file_name)
                samples.append([root_path + file_name, identi_id, camera_id])
        return samples

    def _load_jpg_path_only_name0(self, folder_dir):

        samples = []
        root_path, _, files_name = os_walk(folder_dir)
        for file_name in files_name:
            if '.jpg' in file_name:
                identi_id, camera_id = self._analysis_jpg_name_only_id0(file_name)
                samples.append([root_path + file_name, identi_id, camera_id])
        return samples





    def _analysis_jpg_name1(self, file_name):
        split_list = file_name.replace('.jpg', '').split('_')
        identi_id, camera_id = int(split_list[0]), 1
        return identi_id, camera_id

    def _analysis_jpg_name0(self, file_name):
        split_list = file_name.replace('.jpg', '').split('_')
        identi_id, camera_id = int(split_list[0]), 0
        return identi_id, camera_id

    def _analysis_jpg_name_only_id1(self, file_name):
        split_list = file_name.replace('.jpg', '')
        identi_id, camera_id = int(split_list[0]), 1
        return identi_id, camera_id

    def _analysis_jpg_name_only_id3(self, file_name):
        split_list = file_name.replace('.tif', '').split('_')
        identi_id, camera_id = int(split_list[0]), 0
        return identi_id, camera_id

    def _analysis_jpg_name_only_id4(self, file_name):
        split_list = file_name.replace('.tif', '').split('_')
        identi_id, camera_id = int(split_list[0]), 1
        return identi_id, camera_id

    def _analysis_jpg_name_only_id0(self, file_name):
        split_list = file_name.replace('.jpg', '')
        identi_id, camera_id = int(split_list[0]), 0
        return identi_id, camera_id

    def _show_info(self, train, query, gallery, name=None):
        def analyze(samples):
            pid_num = len(set([sample[1] for sample in samples]))
            cid_num = len(set([sample[2] for sample in samples]))
            sample_num = len(samples)
            return sample_num, pid_num, cid_num

        train_info = analyze(train)
        query_info = analyze(query)
        gallery_info = analyze(gallery)

        # please kindly install prettytable: ```pip install prettyrable```
        table = PrettyTable(['set', 'images', 'identities', 'cameras'])
        table.add_row([self.__class__.__name__ if name is None else name, '', '', ''])
        table.add_row(['train', str(train_info[0]), str(train_info[1]), str(train_info[2])])
        table.add_row(['query', str(query_info[0]), str(query_info[1]), str(query_info[2])])
        table.add_row(['gallery', str(gallery_info[0]), str(gallery_info[1]), str(gallery_info[2])])
        print(table)


class Samples4Market(PersonReIDSamples):
    '''
    Market Dataset
    '''
    def __init__(self, market_path, relabel=True, combineall=False):

        # parameters
        self.market_path = market_path
        self.relabel = relabel
        self.combineall = combineall

        # paths of train, query and gallery
        train_path = os.path.join(self.market_path, 'bounding_box_train/')
        query_path = os.path.join(self.market_path, 'query/')
        gallery_path = os.path.join(self.market_path, 'bounding_box_test/')

        # load
        train = self._load_images_path(train_path)
        query = self._load_images_path(query_path)
        gallery = self._load_images_path(gallery_path)
        if self.combineall:
            train += copy.deepcopy(query) + copy.deepcopy(gallery)

        # reorder person identities
        if self.relabel:
            train = self._relabels(train, 1)
        self.train, self.query, self.gallery = train, query, gallery
        self._show_info(train, query, gallery)

class Samples4Duke(PersonReIDSamples):
    '''
    Duke dataset
    '''
    def __init__(self, market_path, relabel=True, combineall=False):

        # parameters
        self.market_path = market_path
        self.relabel = relabel
        self.combineall = combineall

        # paths of train, query and gallery
        train_path = os.path.join(self.market_path, 'bounding_box_train/')
        query_path = os.path.join(self.market_path, 'query/')
        gallery_path = os.path.join(self.market_path, 'bounding_box_test/')

        # load
        train = self._load_images_path(train_path)
        query = self._load_images_path(query_path)
        gallery = self._load_images_path(gallery_path)
        if self.combineall:
            train += copy.deepcopy(query) + copy.deepcopy(gallery)
        # reorder person identities
        if self.relabel:
            train = self._relabels(train, 1)
        self.train, self.query, self.gallery = train, query, gallery
        self._show_info(train, query, gallery)

    def _analysis_file_name(self, file_name):
        '''
        :param file_name: format like 0002_c1_f0044158.jpg
        :return:
        '''
        split_list = file_name.replace('.jpg', '').replace('c', '').split('_')
        identi_id, camera_id = int(split_list[0]), int(split_list[1])
        return identi_id, camera_id

class Samples4pDuke(PersonReIDSamples):
    '''
    Duke dataset
    '''
    def __init__(self, market_path, relabel=True, combineall=False):

        # parameters
        self.market_path = market_path
        self.relabel = relabel
        self.combineall = combineall

        # paths of train, query and gallery
        train_path = os.path.join(self.market_path, 'train/new/')
        query_path = os.path.join(self.market_path, 'test/occluded_body_images/')
        gallery_path = os.path.join(self.market_path, 'test/whole_body_images/')

        # load
        train = self._load_images_path_pduke_t(train_path)
        query = self._load_images_path_pduke_tt(query_path)
        gallery = self._load_images_path_pduke_t(gallery_path)
        if self.combineall:
            train += copy.deepcopy(query) + copy.deepcopy(gallery)
        # reorder person identities
        if self.relabel:
            train = self._relabels(train, 1)
        self.train, self.query, self.gallery = train, query, gallery
        self._show_info(train, query, gallery)

    def _analysis_file_name(self, file_name):
        '''
        :param file_name: format like 0002_c1_f0044158.jpg
        :return:
        '''
        split_list = file_name.replace('.jpg', '').replace('c', '').split('_')
        identi_id, camera_id = int(split_list[0]), int(split_list[1])
        return identi_id, camera_id




class Samples4Partial(PersonReIDSamples):

    def __init__(self, partial_reid_path, relabel=True, combineall=False):
        # parameters
        self.partial_reid_path = partial_reid_path
        self.relabel = relabel
        self.combineall = combineall

        # path of query and gallery
        query_path = os.path.join(self.partial_reid_path, 'occluded_body_images/')
        gallery_path = os.path.join(self.partial_reid_path, 'whole_body_images/')

        # load
        query = self._load_jpg_path1(query_path)
        gallery = self._load_jpg_path0(gallery_path)
        self.query, self.gallery = query, gallery

class Samples4Partial_ilids(PersonReIDSamples):

    def __init__(self, partial_reid_path, relabel=True, combineall=False):
        # parameters
        self.partial_reid_path = partial_reid_path
        self.relabel = relabel
        self.combineall = combineall

        # path of query and gallery
        query_path = os.path.join(self.partial_reid_path, 'Probe/')
        gallery_path = os.path.join(self.partial_reid_path, 'Gallery/')

        # load
        query = self._load_jpg_path_only_name0(query_path)
        gallery = self._load_jpg_path_only_name1(gallery_path)
        self.query, self.gallery = query, gallery


class Samples4WildTrack(PersonReIDSamples):

    def __init__(self, folder_dir):
        '''
        :param folder_dir:
        :return: [(path, identiti_id, camera_id)]
        '''
        samples = []
        root_path, id_paths, _ = os_walk(folder_dir)
        for id_path in id_paths:
            # print(root_path, id_path)
            _, _, files_name = os_walk(os.path.join(root_path, id_path))
            for file_name in files_name:
                if '.png' or '.jpg' in file_name:
                    camera_id = self._analysis_file_name(file_name)
                    img_path = os.path.join(root_path, id_path, file_name)
                    # size = Image.open(img_path).size
                    samples.append([img_path, int(id_path), int(camera_id)])

        random.seed('wildtrack')
        self.query_samples, self.gallery_samples = random.choices(samples, k=1000), samples
        self._show_info(samples, self.query_samples, self.gallery_samples)

    def _analysis_file_name(self, file_name):
        camera_id = file_name.replace('.png', '').split('_')[0]
        return camera_id

    # def _vis_size_distri(self, samples):
    #     pixels = [sample[3][1]*sample[3][0] for sample in samples]
    #     plt.figure()
    #     plt.hist(pixels, 500, range=[0,200000])
    #     plt.savefig('./pixel_distri.png')


def combine_samples(samples_list):
    '''combine more than one samples (e.g. market.train and duke.train) as a samples'''
    all_samples = []
    max_pid, max_cid = 0, 0
    for samples in samples_list:
        for a_sample in samples:
            img_path = a_sample[0]
            pid = max_pid + a_sample[1]
            cid = max_cid + a_sample[2]
            all_samples.append([img_path, pid, cid])
        max_pid = max([sample[1] for sample in all_samples])
        max_cid = max([sample[2] for sample in all_samples])
    return all_samples





class PersonReIDDataSet:

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):

        this_sample = copy.deepcopy(self.samples[index])

        this_sample[0] = self._loader(this_sample[0])
        if self.transform is not None:
            this_sample[0] = self.transform(this_sample[0])
        this_sample[1] = np.array(this_sample[1])

        return this_sample

    def __len__(self):
        return len(self.samples)

    def _loader(self, img_path):
        return Image.open(img_path).convert('RGB')
