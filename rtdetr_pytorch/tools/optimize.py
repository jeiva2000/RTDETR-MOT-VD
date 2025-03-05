import onnxruntime as rt

sess_options = rt.SessionOptions()

# Set graph optimization level
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

# To enable model serialization after graph optimization set this
sess_options.optimized_model_filepath = "optimized_model_extend_first_model.onnx"

session = rt.InferenceSession("../rtdetr_101vd_6x.onnx", sess_options)
