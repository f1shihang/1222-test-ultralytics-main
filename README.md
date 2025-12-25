# 1222-test-ultralytics-main
### 修改Ultralytics源码：按类别可视化GT/TP/FP/FN结果
#### 1. 替换源码文件
将自定义的 `val.py` 文件替换到Ultralytics源码的指定路径：ultralytics/models/yolo/detect/val.py

#### 2. 调用可视化函数
在替换后的 `val.py` 文件中，找到**第388行**，通过修改 `target_cls` 参数指定需要可视化的类别，调用 `_dump_cls_tp_fp_fn` 函数即可按类别输出GT/TP/FP/FN结果：
```python
self._dump_cls_tp_fp_fn(predn, pbatch, target_cls=10, iou_thres=0.45)

若需保存多个类别的可视化结果，多次调用上述代码即可（每次修改 target_cls 为对应类别 ID）；
