# Summary


- [ ] Inherit from [`base.model.BaseModel`](./model.md#minimal-compatible-model) and implement `forward` function with [ONNX compatibility](./model.md#design-for-onnx-deployment) in mind.

- [ ] Implement training class inherited from [`BaseTrainer`](./training.md#write-a-trainer) to customize training logic in each epoch.

- [ ] Create `config.json` with the [structure](./training.md#complete-training-configuration) consistent with your model and training classes.

- [ ] Run [`scripts/train.py --debug`](./training.md#validate-train-and-override-settings) to test `onnx.export`.

- [ ] Run [`scripts/train.py -c config.json`](./training.md#validate-train-and-override-settings) to save logs and best models.

- [ ] Implement all inference hooks described in [Inference](./inference.md).

- [ ] Create `config.json` for inference to specify [path to trained model and testing dataset](./inference.md#inference-configuration).

- [ ] Have fun !
