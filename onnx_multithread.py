import os
os.environ["OMP_NUM_THREADS"] = "1" #  Very important to reduce CPU usage. If you are trying to hit max FPS possible, you may need to up this number.
import threading
import cv2
import onnxruntime
from queue import Queue
import torchvision.transforms as transforms
import numpy as np
import time
import psutil, os
import sys

# Config variables
num_threads = 2 # Number of Python threads to use (using ~1 more than needed to achieve wanted fps yields lower CPU usage)
queue_max_size = num_threads + 4 # Optimize for best CPU usage, Memory, and Latency. A max size is needed to not create a potential memory leak.
video_src = 'path/to/video or UVC port number'
model_path = 'path/to/the/model.onnx'
interval = 1  # FPS print update rate
visualize_output = False 
low_priority = True # Set process priority to low
print_fps = True 
limit_fps = False # Do not enable alongside visualize_output
limited_fps = 60

# Init variables
frames = 0
queues = []
threads = []
model_output = np.zeros((22, 2))
output_queue = Queue(maxsize=queue_max_size) 
start_time = time.time()

for _ in range(num_threads):
    queue = Queue(maxsize=queue_max_size)
    queues.append(queue)

opts = onnxruntime.SessionOptions()
opts.inter_op_num_threads = 1
opts.intra_op_num_threads = 1 
opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.optimized_model_filepath = ''
ort_session = onnxruntime.InferenceSession(model_path, opts, providers=['CPUExecutionProvider'])

if low_priority:
    process = psutil.Process(os.getpid()) # Set process priority to low
    try:
        sys.getwindowsversion()
    except AttributeError:
        process.nice(0) # UNIX: 0 low 10 high
        process.nice()
    else:
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS) # Windows
        process.nice()
        # See https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getpriorityclass#return-value for values

def visualize(frame, model_output):
    width, height, _ = frame.shape
    for point in model_output:
        x, y = point
        cv2.circle(frame, (int(x * width), int(y * height)), 1, (0, 255, 0), -1) 

    if cv2.waitKey(10) == 27:
        raise Exception("OpenCV Close Triggered.")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def run_model(input_queue, output_queue, session):
    while True:
        frame = input_queue.get()
        if frame is None:
            break

        to_tensor = transforms.ToTensor() # Prepare image for model
        img_tensor = to_tensor(frame)
        img_tensor.unsqueeze_(0)
        img_np = img_tensor.numpy()

        ort_inputs = {session.get_inputs()[0].name: img_np} # Run model
        model_output = session.run(None, ort_inputs)

        model_output = model_output[1] # Format output
        model_output = np.reshape(model_output, (22, 2))

        output_queue.put((frame, model_output)) # Put outputs into queue


def run_onnx_model(queues, frame):
    for i in range(len(queues)):
        if not queues[i].full():
            queues[i].put(frame)
            break

def stop_onnx_model_threads(queues):
    for queue in queues:
        queue.put(None)

def get_combined_output(output_queue):
    combined_image_stream = []
    combined_data_stream = []
    
    while not output_queue.empty():
        frame, model_output = output_queue.get()
        combined_image_stream.append(frame)
        combined_data_stream.append(model_output)
    
    return combined_image_stream, combined_data_stream


for i in range(num_threads): # init threads
    thread = threading.Thread(target=run_model, args=(queues[i], output_queue, ort_session), name=f"Thread {i}")
    threads.append(thread)
    thread.start()

cap = cv2.VideoCapture(video_src)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (112, 112)) # Resize frame before sending to queue (likely only beneficial if downsizing)

    run_onnx_model(queues, frame)
    if not output_queue.empty():
        frames += 1
        
        if time.time() - start_time > interval:
            fps = frames / (time.time() - start_time)
            print(f"FPS: {fps:.2f}")
            frames = 0
            start_time = time.time()

        frame, model_output = output_queue.get()
        if visualize_output and not np.allclose(model_output, 0):
            visualize(frame, model_output)

    if visualize_output:
        cv2.imshow("Model Visualization", frame)
        
    if limit_fps:
        time.sleep(1/limited_fps)

cap.release()
stop_onnx_model_threads(queues)
combined_image_stream, combined_data_stream = get_combined_output(output_queue)
