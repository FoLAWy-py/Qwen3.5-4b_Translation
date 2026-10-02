from witrans_tools.train import AnswerCollator


def test_padding_keeps_answer_mask_and_does_not_train_prompt():
    rows = [{"input_ids": [1, 2, 3, 9], "attention_mask": [1, 1, 1, 1], "labels": [-100, -100, 3, 9]},
            {"input_ids": [1, 4, 9], "attention_mask": [1, 1, 1], "labels": [-100, 4, 9]}]
    batch = AnswerCollator(9)(rows)
    assert batch["labels"].tolist() == [[-100, -100, 3, 9], [-100, 4, 9, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]
