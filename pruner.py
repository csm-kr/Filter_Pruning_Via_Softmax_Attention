import torch
import torch.nn as nn
import torch.nn.functional as F


def prune(model, newmodel, p=0.5):
    print('Pruning...')

    linear_cnt = 0
    for [m0, m1] in zip(model.modules(), newmodel.modules()):

        if isinstance(m0, nn.Conv2d):

            if m0.kernel_size == (3, 3):
                # check original and new model weight size
                original_channel_size = m0.weight.data.size(0)
                new_channel_size = m1.weight.data.size(0)

                if original_channel_size > 1 and original_channel_size != 3:
                    assert original_channel_size * p == new_channel_size, "Puruned percent is not allowed"

                    # get attention for softmax higher than 0.5 percentage scores
                    # ------------------------------softmax------------------------------
                    scores, indices = torch.softmax(F.adaptive_avg_pool3d(m0.weight.data, 1).squeeze(), dim=-1).sort(descending=True)  # 차원이 1이라서 -1 가능
                    indices = indices[:new_channel_size]

                    # assign new weight and bias
                    m1.weight.data = m0.weight.data[indices.tolist()].clone()  # [64, 1, 3, 3] --> [32, 1, 3, 3]
                    m1.bias.data = m0.bias.data[indices.tolist()].clone()      # [64] --> [32]
                # continue
        elif isinstance(m0, nn.BatchNorm2d):
            assert isinstance(m1, nn.BatchNorm2d), "There should not be bn layer here."

            original_channel_size = m0.weight.data.size(0)
            new_channel_size = m1.weight.data.size(0)

            if original_channel_size > 1:
                assert original_channel_size * p == new_channel_size, "Puruned percent is not allowed"

                # get attention for softmax higher than 0.5 percentage scores
                scores, indices = torch.softmax(m0.weight.data.squeeze(), dim=-1).sort(
                    descending=True)
                indices = indices[:new_channel_size]

                m1.weight.data = m0.weight.data[indices.tolist()].clone()
                m1.bias.data = m0.bias.data[indices.tolist()].clone()
                m1.running_mean = m0.running_mean[indices.tolist()].clone()
                m1.running_var = m0.running_var[indices.tolist()].clone()
            # continue
        elif isinstance(m0, nn.Linear):
            linear_cnt += 1
            # if linear_cnt <= 15:
            if linear_cnt == 1:
                original_channel_size = m0.weight.data.size(1)
                new_channel_size = m1.weight.data.size(1)
                if original_channel_size > 1:
                    assert original_channel_size * p == new_channel_size, "Puruned percent is not allowed"

                    # get attention for softmax higher than 0.5 percentage scores
                    scores, indices = torch.softmax(torch.mean(m0.weight.data, dim=0), dim=-1).sort(
                        descending=True)
                    indices = indices[:new_channel_size]

                    m1.weight.data = m0.weight.data[:, indices.tolist()].clone()
                    m1.bias.data = m0.bias.data.clone()
            # continue
            else:
                print('pass')
    print('End Pruning!')



