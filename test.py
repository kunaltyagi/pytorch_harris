from pathlib import Path

import cv2 as cv
import numpy as np

import depthai as dai

if __name__ == "__main__":
    pipeline = dai.Pipeline()
    # Source
    camera = pipeline.createColorCamera()
    camera.setPreviewSize(300, 300)
    camera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camera.setInterleaved(False)
    # Ops
    detection = pipeline.createNeuralNetwork()
    blob_path = Path(__file__).parent / "out" / "model.blob"
    detection.setBlobPath(f"{blob_path.as_posix()}")
    # Link Camera -> Model
    camera.preview.link(detection.input)
    # Link Model Output -> Host
    x_out = pipeline.createXLinkOut()
    x_out.setStreamName("custom")
    image = pipeline.createXLinkOut()
    image.setStreamName("rgb")
    camera.preview.link(image.input)
    detection.out.link(x_out.input)

    device = dai.Device(pipeline)

    frame_buffer = device.getOutputQueue(name="custom", maxSize=4)
    image = device.getOutputQueue(name="rgb", maxSize=4)

    while True:
        frame = frame_buffer.get()
        img = image.get().getCvFrame()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (3, 3), 0)
        v = np.median(blurred)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        # edged = cv.Canny(image, lower, upper)
        # Model output
        layer = np.array(frame.getFirstLayerFp16())
        # deal with inf
        layer[layer == np.Inf] = 255
        # Reshape
        layer = np.array(layer, dtype=np.uint8)
        shape = (300, 300, 1)
        frame_data = layer.reshape(shape)
        cv.imshow("Image", frame_data)
        cv.imshow("CV", img)
        if cv.waitKey(1) == ord("q"):
            break
