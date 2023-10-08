# Copyright (C) 2023 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np
from ultralytics import YOLO


class ModelHandler:
    def __init__(self, labels):
        self.model = YOLO('/opt/nuclio/best.pt')
        self.labels = labels

    def infer(self, image):
        image = np.array(image)

        detections = self.model.predict(image)

        results = []

        for result in detections[0]:
            labels = int(result.boxes.cls.cpu().numpy()[0])
            scores =  result.boxes.conf.cpu().numpy()[0]
            xtl, ytl, xbr, ybr = result.boxes.xyxy[0].cpu().numpy().astype(int)

            results.append({
                "confidence": str(scores),
                "label": self.labels.get(labels, "unknown"),
                "points": [int(xtl), int(ytl), int(xbr), int(ybr)],
                "type": "rectangle",
            })

        return results
