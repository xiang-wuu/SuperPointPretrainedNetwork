import pycuda.driver as cuda
import numpy as np
import tensorrt as trt
import os
import sys
import time

# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()


def build_engine(engine_file_path):
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            context = engine.create_execution_context()
            return engine, context
    else:
        print('engine file not found')
        sys.exit(0)


def infer(engine, context, host_input):
    # initialize TensorRT engine and parse ONNX model
    # engine, context = build_engine(PLAN_FILE_PATH)
    # get sizes of input and output and allocate memory required for input data and for output data
    output_shapes = []
    host_outputs = []
    device_outputs = []
    for binding in engine:
        if engine.binding_is_input(binding):  # we expect only one input
            input_shape = engine.get_binding_shape(binding)
            # print(input_shape)
            input_size = trt.volume(
                input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
            device_input = cuda.mem_alloc(input_size)
        else:  # and one output
            output_shape = engine.get_binding_shape(binding)
            output_shapes.append(output_shape)
            # print(output_shape)
            # create page-locked memory buffers (i.e. won't be swapped to disk)
            host_output = cuda.pagelocked_empty(trt.volume(
                output_shape) * engine.max_batch_size, dtype=np.float32)
            host_outputs.append(host_output)
            device_output = cuda.mem_alloc(host_output.nbytes)
            device_outputs.append(device_output)

    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()

    # print(host_input.shape)
    cuda.memcpy_htod_async(device_input, host_input, stream)

    # run inference
    context.execute_async(bindings=[int(device_input), int(
        device_outputs[0]), int(
        device_outputs[1])], stream_handle=stream.handle)

    # for i in len(host_outputs):
    cuda.memcpy_dtoh_async(host_outputs[0], device_outputs[0], stream)
    cuda.memcpy_dtoh_async(host_outputs[1], device_outputs[1], stream)

    stream.synchronize()

    # print(host_output.reshape(output_shape).shape)
    return host_outputs[0].reshape(output_shapes[0]), host_outputs[1].reshape(output_shapes[1])
