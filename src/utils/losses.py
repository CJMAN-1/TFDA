import torch
import torch.nn.functional as F

class Losses():
    def __init__(self):
        pass

    def CrossEntropy2d(self, predict, target, class_weight=None):
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))

        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != 255)
        target = target[target_mask]
        if not target.data.dim():
            return torch.zeros(1)
        predict = predict.permute(0,2,3,1).contiguous() # contiguous : premute를해도 접근하는 인덱스만 바뀌지 실제 메모리에서 위치는 안바뀌는데 contiguous는 실제 메모리위치를 인접하게 바꿔줌. view같은 함수를 쓸때 메모리가 연속하지않으면 오류가 난다고함.
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c).contiguous()

        loss = F.cross_entropy(predict, target, weight=class_weight, reduction='mean')

        return loss