import torch
import os
import numpy as np
import re

from src.config import cfg 
from src.data import make_gallery_loader, make_query_loader
from src.modeling import build_model
from src.data.transforms import build_transforms


def Singleton(cls):
    _instance = {}

    def _singleton(*args, **kargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kargs)
        return _instance[cls]

    return _singleton


@Singleton
class ReidModel(object):
    """
    """
    def __init__(self, config = cfg, config_file="configs/softmax_triplet.yml"):
        """ initialize a ReidModel object 
        :param:
            :config_file: a relative path of yml file, which specify the model type, model weight path etc.
        """
        # build kernel
        config.merge_from_file(os.path.join(os.path.dirname(__file__), config_file))
        self.__cfg = config
        self.__kernel = None
        self.__build_kernel(config)

        # load or extract gallery features
        self.__gallery_image_features = None
        self.__gallery_image_paths = None
        self.__load_gallery_features()

        self.__transform = build_transforms(config)

    def __build_kernel(self, config):
        """ build reid algorighm kernel object. only for extracting image features
        :param:
            config: CfgNode object; specifiy model structure, weight path etc.
        :return: reid algorighm kernel object
        """
        kernel = build_model(config)
        kernel.load_param(config.TEST.WEIGHT)
        kernel.eval()
        kernel.cuda()
        self.__kernel = kernel

    def __update_gallery_feature(self, feature_dir="./Features"):
        """ forece to update the gallery feature in case of gallery change
        :param:
            :feature_path: archive path of gallery features: 
        """
        # update features in memory
        gallery_loader = make_gallery_loader(self.__cfg)
        gallery_image_features, gallery_image_paths = self.__extract_gallery_features(gallery_loader)
        self.__gallery_image_features = gallery_image_features
        self.__gallery_image_paths = gallery_image_paths

        # update features archived
        os.makedirs(feature_dir, exist_ok=True)
        feature_path = os.path.join(feature_dir, "feautures.pt")
        torch.save(gallery_image_features, feature_path)

        image_list_path = os.path.join(feature_dir, "image_paths.txt")
        with open(image_list_path, 'w') as f:
            for image_path in gallery_image_paths:
                f.write(image_path+'\n')

    def __load_gallery_features(self, feature_dir="./Features"):
        """ load gallery features if exists, otherwise extract gallery features and store it.
        :param:
            :feature_path: archive path of gallery features
        """
        feature_path = os.path.join(feature_dir, "feautures.pt")
        image_list_path = os.path.join(feature_dir, "image_paths.txt")

        if os.path.exists(feature_path) and os.path.exists(image_list_path):
            self.__gallery_image_features = torch.load(feature_path)
            with open(image_list_path) as f:
                self.__gallery_image_paths = f.readlines()
                self.__gallery_image_paths = [image_path.rstrip('\n') for image_path in self.__gallery_image_paths]
        else:
            self.__update_gallery_feature(feature_dir=feature_dir)


    def __extract_single_feature(self, image):
        """ extract feature for a single image. usually for query image
        :param:
            :image: PIL.Image.Image object; image to process.
        """
        image = image.convert('RGB')
        image = self.__transform(image)
        image = image.unsqueeze(0)              # 从(c,h,w) _ (1,c,h,w)适配模型的输入
        image = image.to(self.__cfg.MODEL.DEVICE)
        with torch.no_grad():
            feature = self.__kernel(image)
            feature = torch.nn.functional.normalize(feature, dim=1, p=2)            # nomralization for compute cosine distance
            return feature

    def __extract_gallery_features(self, dataloader):
        features = []
        image_paths = []
        g_pids = []
        g_camids = []

        with torch.no_grad():
            for batch in dataloader:
                data, pids, camids, img_paths = batch
                data = data.to(self.__cfg.MODEL.DEVICE)
                feat = self.__kernel(data)
                features.append(feat)
                image_paths += img_paths
                g_pids.extend(pids)
                g_camids.extend(camids)
            
            self.__g_pids = g_pids
            self.__g_camids = g_camids
            features = torch.cat(features, dim=0)
            features = torch.nn.functional.normalize(features, dim=1, p=2)
            return features, image_paths


    def __compute_distance_matrix(self, qf, gf):
        """ compute the distance between query feature and gallery features
        :param:
            qf: query feature
            gf: gallery feature
        :return: distmat: a [1xn] matrix, matrix[0][i] indicates the distance between query feature and ith gallery feature.
        """
        # nomalize the query feature and gallery feature
        print("The query feature is normalized")

        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)

        return distmat

    def get_gallery_size(self):
        return len(self.__gallery_image_paths)

    def get_similar_image_rank_list(self, query, max_length=10):
        """ get a rank list of similary images of input image.
        :param:
            query: image to match the gallery images.
        :return: a list of image path, the distance between query image and each in the list ascends.
        image path format like "/path/0001_c0_f0123456.jpg"
        """
        query_feature = self.__extract_single_feature(query)
        distmat = self.__compute_distance_matrix(query_feature, self.__gallery_image_features)
        rank_list = np.argsort(distmat.cpu(), axis=1)
        rank_list = rank_list.squeeze()
        return [self.__gallery_image_paths[index] for index in rank_list[:max_length]]

    def evaluate_performance_on_origin_dataset(self):
        """ to verify the implementation is correct
        """
        from src.utils.eval_reid import eval_func

        # extrack all query features
        query_loader = make_query_loader(self.__cfg)
        query_image_features = []
        query_image_paths = []
        q_pids = []
        q_camids = []
        with torch.no_grad():
            for batch in query_loader:
                data, pids, camids, img_paths = batch
                data = data.to(self.__cfg.MODEL.DEVICE)
                feat = self.__kernel(data)
                query_image_features.append(feat)
                query_image_paths += img_paths
                q_pids.extend(pids)
                q_camids.extend(camids)

            query_image_features = torch.cat(query_image_features, dim=0)
            query_image_features = torch.nn.functional.normalize(query_image_features, dim=1, p=2)

        # calculate distance matrix
        distmat = self.__compute_distance_matrix(query_image_features, self.__gallery_image_features)
        cmc, mAP = eval_func(distmat.cpu(), np.asarray(q_pids), np.asarray(self.__g_pids), np.asarray(q_camids), np.asarray(self.__g_camids))
        return cmc[0], mAP