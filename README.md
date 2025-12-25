# 1222-test-ultralytics-main
修改ultralytics源码，按类别进行GT/TP/FP/FN结果的可视化
使用其中的val.py替换yolo源码中ultralytics/models/yolo/detect的val.py
使用时进需要将388行的代码中的target_cls进行替换即可，想要保存多个类别就多调用几次
self._dump_cls_tp_fp_fn(predn, pbatch, target_cls=10, iou_thres=0.45)
