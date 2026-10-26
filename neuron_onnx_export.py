import numpy as np
from onnx import helper, TensorProto, checker
import onnx

def create_formal_neuron_onnx(num_inputs, activation_type="relu", output_path="formal_neuron.onnx"):
    # 定义输入输出
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [num_inputs]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1]
    )

    # 随机初始化权重、偏置
    w_data = np.random.randn(num_inputs).astype(np.float32) * 0.1
    b_data = np.array([0.1], dtype=np.float32)

    w_init = helper.make_tensor("W", TensorProto.FLOAT, [num_inputs], w_data)
    b_init = helper.make_tensor("B", TensorProto.FLOAT, [1], b_data)

    # 计算节点：点积 + 加偏置
    dot_node = helper.make_node("Dot", ["input", "W"], ["dot_out"])
    add_node = helper.make_node("Add", ["dot_out", "B"], ["z"])

    # 激活函数
    act_map = {
        "relu": helper.make_node("Relu", ["z"], ["output"]),
        "sigmoid": helper.make_node("Sigmoid", ["z"], ["output"]),
        "tanh": helper.make_node("Tanh", ["z"], ["output"]),
        "linear": helper.make_node("Identity", ["z"], ["output"])
    }
    act_node = act_map[activation_type]

    # 构建计算图
    graph = helper.make_graph(
        nodes=[dot_node, add_node, act_node],
        name="formal_neuron_graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[w_init, b_init]
    )

    # 生成模型，opset11
    model = helper.make_model(
        graph,
        producer_name="Formal Neuron",
        opset_imports=[helper.make_opsetid("", 11)]
    )
    checker.check_model(model)
    onnx.save(model, output_path)
    print(f"✅ ONNX模型已保存：{output_path}")
    return model


if __name__ == "__main__":
    create_formal_neuron_onnx(
        num_inputs=3,
        activation_type="relu",
        output_path="formal_neuron.onnx"
    )
