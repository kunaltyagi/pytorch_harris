from pathlib import Path

import cv2 as cv
import numpy as np

import depthai as dai

def find_edges(gray):
    blurred = cv.GaussianBlur(gray, (3, 3), 0)

    v = np.median(blurred)
    sigma = 0.33

    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv.Canny(image, lower, upper)
    return edged

def find_features(img, gray):
    dst = cv.cornerHarris(gray,2,3,0.04)
    dst = cv.dilate(dst,None)
    img[dst>0.01*dst.max()]=[0,0,255]

    return img

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
        # Model output
        layer = np.array(frame.getFirstLayerFp16())
        # deal with inf
        layer[layer == np.Inf] = 255
        # Reshape
        layer = np.array(layer, dtype=np.uint8)
        shape = (2, 300, 300)
        frame_data = layer.reshape(shape)
        edges = frame_data[0, :,:] * 5  # for visual appeal
        corners = frame_data[1, :,:]

        cv.imshow("Image", img)
        cv.imshow("Device Corners", corners)
        cv.imshow("Device Edges", edges)

        # edged = find_edges(gray)
        # cv.imshow("Host Edges", edged)
        corner = find_features(img, gray)
        cv.imshow("Corner", corner)

        if cv.waitKey(1) == ord("q"):
            break
