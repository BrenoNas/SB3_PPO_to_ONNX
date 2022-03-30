import numpy as np
import gym
import onnx
import onnxruntime as ort

onnx_path = "Models\ONXXmodel.onnx"
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
ort_sess = ort.InferenceSession(onnx_path)

env = gym.make('CartPole-v1')

obs = env.reset()
for i in range(10000):
    action = ort_sess.run(None, {'Input': np.array(obs.astype(np.float32), ndmin=2)})
    obs, reward, done, info = env.step(action[0][0])
    env.render()
    if done:
      obs = env.reset()

