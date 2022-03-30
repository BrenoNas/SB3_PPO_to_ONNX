from stable_baselines3 import PPO
import torch

class OnnxablePolicy(torch.nn.Module):
  def __init__(self, policy):
      super(OnnxablePolicy, self).__init__()
      self.policy = policy

  def forward(self, observation):
    return self.policy.get_distribution(observation).get_actions(deterministic=True)


path = "Models\PPO.zip"
model = PPO.load(path)
model.policy.to("cpu")
onnxable_model = OnnxablePolicy(model.policy)
dummy_input = torch.randn(1, 4)

torch.onnx.export(onnxable_model,
                dummy_input, 
                "Models\ONXXmodel.onnx",
                opset_version=9,
                input_names = ['Input'],   
                output_names = ['Output'])