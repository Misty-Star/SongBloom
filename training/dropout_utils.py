"""训练期条件 dropout 辅助函数。"""


def apply_condition_dropouts(cfg_dropout, att_dropout, attributes):
    attributes = cfg_dropout(attributes)
    attributes = att_dropout(attributes)
    return attributes
