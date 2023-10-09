import os
os.environ["OMP_NUM_THREADS"] = "1" #  very important to reduce CPU usage
import threading
import cv2
import onnxruntime
from queue import Queue
import numpy as np
import time
import psutil, os
import sys
# Config variables
num_threads = 4 # Number of python threads to use (using ~1 more than needed to acheive wanted fps yeilds lower cpu usage)
queue_max_size = 2 # Optimize for best CPU usage, Memory, and Latency. A maxsize is needed to not create a potential memory leak.
video_src = 0
model_path = 'EFV2300K45E100P2.onnx'
interval = 1  # FPS print update rate
visualize_output = True
low_priority = True # set process priority to low
print_fps = True 
limit_fps = False # do not enable along side visualize_output
limited_fps = 30

# Init variables
frames = 0
queues = []
threads = []
frame_index = 0
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
    process = psutil.Process(os.getpid()) # set process priority to low
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
    cv2.imshow("Model Visualization", frame)
    if cv2.waitKey(20) == 27:
        raise Exception("OpenCV Close Triggered.")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def run_model(input_queue, output_queue, session):
    while True:
        frame, frame_index = input_queue.get()
        if frame is None:
            break

        framenorm = frame # run model example code
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = np.expand_dims(frame_gray, axis=2)
        frame_gray_batch = np.expand_dims(frame_gray, axis=0)
        frame_tensor = np.transpose(frame_gray_batch, (0, 3, 1, 2)).astype(np.float32) / 255.0
        frame = np.array(frame_tensor)

        ort_inputs = {session.get_inputs()[0].name: frame} # Run model
        model_output = session.run(None, ort_inputs)

        model_output = model_output[0]
        model_output = model_output[0]

        output_queue.put((frame_index, framenorm, model_output)) # Output model data


def run_onnx_model(queues, frame, frame_index):
    for i in range(len(queues)):
        if not queues[i].full():
            queues[i].put((frame, frame_index))
            break


def stop_onnx_model_threads(queues):
    for queue in queues:
        queue.put(None)

for i in range(num_threads): # init threads
    thread = threading.Thread(target=run_model, args=(queues[i], output_queue, ort_session), name=f"Thread {i}")
    threads.append(thread)
    thread.start()

cap = cv2.VideoCapture(video_src)

frame_list = []
number_dict = {}

def store_frame(number, frame, array):
    if len(frame_list) >= num_threads +1:
        # Remove the oldest frame and number
        oldest_data = frame_list.pop(0)
        oldest_number = oldest_data[0]
        if oldest_number in number_dict:
            del number_dict[oldest_number]
    
    data = (number, frame, array)
    frame_list.append(data)
    number_dict[number] = data

def get_frame(number):
    data = number_dict.get(number, None)
    if data is not None:
        return data[1]  # Return the frame from the data tuple
    else:
        return None

def get_array(number):
    data = number_dict.get(number, None)
    if data is not None:
        return data[2]  # Return the array from the data tuple
    else:
        return None

fc = 0
e = 1
last_frame_index = 0
while cap.isOpened():
    ret, frame = cap.read()
  
    frame = cv2.flip(frame, 0)
    if not ret:
        break

    frame = cv2.resize(frame, (256, 256)) # resize frame before sending to queue (likely only benificial if downsizing)

    run_onnx_model(queues, frame, frame_index)
    frame_index += 1

    if not output_queue.empty():
        frame_indo, frame, model_output = output_queue.get()
    #    frame_indo = frame_indo - num_threads
        store_frame(frame_indo, frame, model_output)
        fc += 1

        latest_frame_number = fc  # Keep track of the latest frame number
        requested_number = fc - 1  # Request the frame one step behind
        frame = get_frame(requested_number)
        requested_array = get_array(requested_number)

        if frame is not None and requested_array is not None:
            cv2.imshow("Requested Frame", frame)
            cv2.waitKey(1)
            

        else:
            fc -= 1
            if abs(fc - frame_indo) > 2:
                fc = frame_indo
            pass

        if fc >= 1000:  # Reset at 1000 to avoid memory leak
            frame_index = 0
            fc = 0

        frames += 1
        if time.time() - start_time > interval:
            fps = frames / (time.time() - start_time)
            print(f"FPS: {fps:.2f}")
            frames = 0
            start_time = time.time()

        for i in range(len(model_output)):            # Clip values between 0 - 1
                    model_output[i] = max(min(model_output[i], 1), 0) 

    if limit_fps:
        time.sleep(1/limited_fps)


cap.release()
stop_onnx_model_threads(queues)
raise AssertionError
