# 在S:\miniconda3\envs\emotion\Lib\site-packages\ptgaze\common\visualizer.py 的Visualizer类中 添加以下方法

def emo_text(self,
             bbox: np.ndarray,
             emo_predicted: int = 6,
             color: Tuple[int, int, int] = (0, 255, 0),
             lw: int = 1) -> None:
    assert self.image is not None
    assert bbox.shape == (2, 2)
    bbox = np.round(bbox).astype(int).tolist()
    x1y1 = bbox[0]
    x2y2 = bbox[1]
    if x1y1[0] > 0:
        x = int(x1y1[0])
    else:
        x = 1
    y = int(x1y1[1])
    # cv2.rectangle(self.image, tuple(bbox[0]), tuple(bbox[1]), color, lw)
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    class_names0 = ['Stress', 'Stress', 'Anxiety', 'Happy', 'Depression', 'Anxiety', 'Neutral']
    # 在原始图像上绘制文本
    cv2.putText(self.image, class_names[int(emo_predicted)], (x - 20, y - 20), cv2.FONT_HERSHEY_PLAIN, 2,
                (255, 0, 0), 2)


# 在S:\miniconda3\envs\emotion\Lib\site-packages\ptgaze\common\face_parts.py FaceParts类中 添加以下方法

    def denormalize_emo_prediction(self) -> None:
        # Here emo_prediction is tensor vector,
        self.emo_prediction = int(self.normalized_emo_prediction.cpu().numpy())
