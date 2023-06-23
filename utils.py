import torch

def export_onnx(model, input_shape, device='cuda', output_path="./lpc_pytorch.onnx"):
    model.eval()
    x = torch.randn(input_shape[0], input_shape[1], input_shape[2], input_shape[3], device=device)

    # Export the model
    torch.onnx.export(model,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    output_path,   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    #opset_version=10,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'] # the model's output names
                    # dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                    #               'output' : {0 : 'batch_size'}}
                    )
