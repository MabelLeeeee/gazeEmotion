import os
import logging
import time
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from ptgaze.common import Camera, Face, FacePartsName
from torch.autograd import Variable
from PIL import Image
from typing import List
from omegaconf import DictConfig
from skimage.transform import resize

from models import create_model, VGG
from src.head_pose_estimation import HeadPoseNormalizer, LandmarkEstimator
from src.creat_tf import create_transform
from src.utils import get_3d_face_model


logger = logging.getLogger(__name__)

#
# parser = argparse.ArgumentParser(description='Retinaface')
# parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
#                     type=str, help='Trained state_dict file path to open')
# parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
# parser.add_argument('--origin_size', default=True, type=str, help='Whether use origin image size to evaluate')
# parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str, help='Dir to save txt results')
# parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
# parser.add_argument('--dataset_folder', default='./data/demo', type=str, help='dataset path')
# parser.add_argument('--confidence_threshold', default=0.2, type=float, help='confidence_threshold')# 0.02
# parser.add_argument('--top_k', default=5000, type=int, help='top_k')
# parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
# parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
# parser.add_argument('-s', '--save_image', action="store_true", default=False, help='show detection results')
# parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
# args = parser.parse_args(args=[])

#class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # print('Missing keys:{}'.format(len(missing_keys)))
    # print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    # print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    # print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    # print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


class GazeEstimator:
    EYE_KEYS = [FacePartsName.REYE, FacePartsName.LEYE]

    def __init__(self, config: DictConfig):
        self._config = config

        self._face_model3d = get_3d_face_model(config)

        self.camera = Camera(config.gaze_estimator.camera_params)
        self._normalized_camera = Camera(
            config.gaze_estimator.normalized_camera_params)

        self._landmark_estimator = LandmarkEstimator(config)
        self._head_pose_normalizer = HeadPoseNormalizer(
            self.camera, self._normalized_camera,
            self._config.gaze_estimator.normalized_camera_distance)
        self._gaze_estimation_model = self._load_model()
        self._transform = create_transform(config)
        self.emotion_model()

    def emotion_model(self):
        # device= torch.device("cpu" if args.cpu else "cuda")
        device = torch.device("cuda")
        torch.set_grad_enabled(False)
        # # 检测模型
        # cfg = None
        # if args.network == "mobile0.25":
        #     cfg = cfg_mnet
        # elif args.network == "resnet50":
        #     cfg = cfg_re50
        # net and model
        # detection_model = RetinaFace(cfg=cfg, phase='test')
        # detection_model = load_model(detection_model, args.trained_model,
        #                              args.cpu)  # cpu:defalt = False, trained_model:default='./weights/Resnet50_Final.pth'
        # # detection_model = load_model(detection_model, args.trained_model,
        # #                              False)  # cpu:defalt = False, trained_model:default='./weights/Resnet50_Final.pth'
        # detection_model.to(device)
        # detection_model.eval()
        print('Detection model finished')
        # 情绪模型
        self.emotion_model = VGG('VGG19')
        current_file = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file)
        parent_dir = os.path.dirname(current_dir)
        checkpoint = torch.load(os.path.join(parent_dir, './weights/PrivateTest_model.t7'))
        self.emotion_model.load_state_dict(checkpoint['net'])
        self.emotion_model.to(device)
        self.emotion_model.eval()
        print('Emotion model finished')

    def _load_model(self) -> torch.nn.Module:
        model = create_model(self._config)
        checkpoint = torch.load(self._config.gaze_estimator.checkpoint,
                                map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.to(torch.device(self._config.device))
        model.eval()
        return model

    def detect_faces(self, image: np.ndarray) -> List[Face]:
        return self._landmark_estimator.detect_faces(image)


    def estimate_emos(self, image: np.ndarray, face: Face):
        self._face_model3d.estimate_head_pose(face, self.camera)
        self._face_model3d.compute_3d_pose(face)
        self._face_model3d.compute_face_eye_centers(face, self._config.mode)
        # device = torch.device("cpu" if args.cpu else "cuda")
        device = torch.device("cuda")
        torch.set_grad_enabled(False)

        cut_size = 44
        transform_test = T.Compose([
            T.TenCrop(cut_size),
            T.Lambda(lambda crops: torch.stack([T.ToTensor()(crop) for crop in crops])), ])
        # emotion predictions##########################################
        emotions = []
        bbox = face.bbox
        x1y1 = bbox[0]
        x2y2 = bbox[1]
        if x1y1[0] > 0:
            x = int(x1y1[0])
        else:
            x = 1
        y = int(x1y1[1])
        w = int(x2y2[0]) - int(x1y1[0])
        h = int(x2y2[1]) - int(x1y1[1])

        raw_img = image[y:y + h, x:x + w]
        gray = rgb2gray(raw_img)
        if gray.size > 0:
            gray = resize(gray, (48, 48)).astype(np.uint8)
        else:
            gray = np.zeros((48, 48))
        img = gray[:, :, np.newaxis]

        img = np.concatenate((img, img, img), axis=2)

        img = np.clip(img, 0, 255).astype(np.uint8)
        if img.shape[:2] == (1, 1):
            img = img.squeeze()
        img = Image.fromarray(img)
        inputs = transform_test(img)

        ncrops, c, h_resize, w_resize = np.shape(inputs)

        inputs = inputs.view(-1, c, h_resize, w_resize)
        inputs = inputs.cuda()
        inputs = Variable(inputs, volatile=True)
        a = time.time()
        outputs = self.emotion_model(inputs)
        b = time.time()
        # print(b-a)

        outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

        score = F.softmax(outputs_avg)
        _, emo_predicted = torch.max(outputs_avg.data, 0)
        emotions.append(emo_predicted)
        face.normalized_emo_prediction = emo_predicted
        face.denormalize_emo_prediction()
        # print(int(emo_predicted.cpu().numpy()))
        class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        # cv2.putText(image, class_names[int(emo_predicted.cpu().numpy())], (x - 20, y - 20), cv2.FONT_HERSHEY_PLAIN, 2,
        #             (0, 255, 0), 1)
        return emo_predicted


    def estimate_gaze(self, image: np.ndarray, face: Face) -> None:
        self._face_model3d.estimate_head_pose(face, self.camera)
        self._face_model3d.compute_3d_pose(face)
        self._face_model3d.compute_face_eye_centers(face, self._config.mode)

        if self._config.mode == 'MPIIGaze':
            for key in self.EYE_KEYS:
                eye = getattr(face, key.name.lower())
                self._head_pose_normalizer.normalize(image, eye)
            L = self._run_mpiigaze_model(face)
        elif self._config.mode == 'MPIIFaceGaze':
            self._head_pose_normalizer.normalize(image, face)
            self._run_mpiifacegaze_model(face)
        elif self._config.mode == 'ETH-XGaze':
            self._head_pose_normalizer.normalize(image, face)
            self._run_ethxgaze_model(face)
        else:
            raise ValueError

    @torch.no_grad()
    def _run_mpiigaze_model(self, face: Face) -> None:
        import time
        time_start = time.time()
        images = []
        head_poses = []
        for key in self.EYE_KEYS:
            eye = getattr(face, key.name.lower())
            image = eye.normalized_image
            normalized_head_pose = eye.normalized_head_rot2d
            if key == FacePartsName.REYE:
                image = image[:, ::-1].copy()
                normalized_head_pose *= np.array([1, -1])
            image = self._transform(image)
            images.append(image)
            head_poses.append(normalized_head_pose)
        images = torch.stack(images)
        head_poses = np.array(head_poses).astype(np.float32)
        head_poses = torch.from_numpy(head_poses)

        device = torch.device(self._config.device)
        images = images.to(device)
        head_poses = head_poses.to(device)
        # faceLandmarks.append(face.landmarks)
        predictions = self._gaze_estimation_model(images, head_poses)
        predictions = predictions.cpu().numpy()

        for i, key in enumerate(self.EYE_KEYS):
            eye = getattr(face, key.name.lower())
            eye.normalized_gaze_angles = predictions[i]
            if key == FacePartsName.REYE:
                eye.normalized_gaze_angles *= np.array([1, -1])
            eye.angle_to_vector()
            eye.denormalize_gaze_vector()

        time_end = time.time()
        # time = time_end - time_start
        # f = 1 / time
        # if self.config.demo.use_camera:
        #     print(f, '帧/s')
        return face.landmarks

    @torch.no_grad()
    def _run_mpiifacegaze_model(self, face: Face) -> None:
        import time
        time_start = time.time()
        image = self._transform(face.normalized_image).unsqueeze(0)

        device = torch.device(self._config.device)
        image = image.to(device)
        prediction = self._gaze_estimation_model(image)
        prediction = prediction.cpu().numpy()

        face.normalized_gaze_angles = prediction[0]
        face.angle_to_vector()
        face.denormalize_gaze_vector()
        time_end = time.time()
        # time = time_end - time_start
        # f = 1 / time
        # print(f, '帧/s')

    @torch.no_grad()
    def _run_ethxgaze_model(self, face: Face) -> None:
        import time
        time_start = time.time()
        image = self._transform(face.normalized_image).unsqueeze(0)

        device = torch.device(self._config.device)
        image = image.to(device)
        prediction = self._gaze_estimation_model(image)
        prediction = prediction.cpu().numpy()

        face.normalized_gaze_angles = prediction[0]
        face.angle_to_vector()
        face.denormalize_gaze_vector()
        time_end = time.time()
        # time = time_end - time_start
        # f = 1 / time
        # print(f, '帧/s')

