# onnx-multithreader
Run ONNX models in a multithreaded environment for lower CPU usage

Increasing `inter_op_num_threads` for more FPS did not satisfy my usage requirements so I wrote this program to run 2 models and interweave the output for more efficient utilization.
In my testing, this method provided a 40x improvement in CPU usage for higher FPS needs, and can even be ~4 times more efficient than a normal ONNX inference script.


Found this repo useful? Give it a star!
